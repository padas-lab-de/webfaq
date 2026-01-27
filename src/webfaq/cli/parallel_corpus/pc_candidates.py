import click
import os
import json
import torch
import re
import numpy as np
from tqdm import tqdm
from webfaq.utils import *
from webfaq.config import *


THRESHOLD_LABSE_SIMILARITY = 0.8
LIMIT_QUESTIONS_SIMILARITY = 0.7


def generate_candidates(
    scheme_host: str,
    languages: List[str],
    scheme_host_offsets: List[List[Dict[str, int]]],
    dataset_name: str,
):
    """
    Given a 'scheme_host' string, a set of languages, and a threshold,
    returns candidate documents (plus scores) that match cross-lingually.
    """
    # Initialize documents dictionary
    documents = {}

    # For each language, open the corresponding file and add relevant docs
    for language, _offsets in zip(languages, scheme_host_offsets):
        language_path = os.path.join(DATASETS_FOLDER, dataset_name, "results", language)

        # Skip if language has less than 100 scheme_hosts
        consider_language = language in LANGUAGES_100_ORIGINS

        # Load offsets
        faq_offsets = [offset["faq_offset"] for offset in _offsets]
        if consider_language:
            labels_offsets = [offset["labels_offset"] for offset in _offsets]
            flags_offsets = [offset["flags_offset"] for offset in _offsets]
        embeddings_offsets = [offset["embeddings_offset"] for offset in _offsets]

        # Initialize set of questions
        set_questions = set()

        with open(os.path.join(language_path, "faq.jsonl"), "r") as faq_file, open(
            os.path.join(language_path, "embeddings.jsonl"), "r"
        ) as embeddings_file:

            if consider_language:
                labels_file = open(os.path.join(language_path, "labels.jsonl"), "r")
                flags_file = open(os.path.join(language_path, "flags.jsonl"), "r")

            for i, (faq_offset, embeddings_offset) in enumerate(
                zip(faq_offsets, embeddings_offsets)
            ):

                faq_file.seek(faq_offset)
                if consider_language:
                    if i >= len(labels_offsets) or i >= len(flags_offsets):
                        continue
                    labels_file.seek(labels_offsets[i])
                    flags_file.seek(flags_offsets[i])
                embeddings_file.seek(embeddings_offset)

                while True:
                    faq_line = faq_file.readline().strip()
                    if consider_language:
                        labels_line = labels_file.readline().strip()
                        flags_line = flags_file.readline().strip()
                    embeddings_line = embeddings_file.readline().strip()

                    # Break at end of file
                    if not faq_line:
                        break

                    try:
                        faq_document = json.loads(faq_line)
                        url = faq_document["url"]
                        _scheme_host = faq_document["scheme_host"]
                        question = faq_document["question"]
                        answer = faq_document["answer"]
                    except json.JSONDecodeError as e:
                        click.echo(
                            f"Skipping invalid JSON line in {language}/faq.jsonl: {faq_line} ({e})"
                        )
                        continue

                    # Break if scheme_host changes
                    if _scheme_host != scheme_host:
                        break

                    # Skip if question is already in documents
                    if question in set_questions:
                        continue
                    set_questions.add(question)

                    topic = "-"
                    question_type = "-"
                    if consider_language:
                        try:
                            labels_document = json.loads(labels_line)
                            topic = labels_document["topic"]
                            question_type = labels_document["question_type"]
                        except json.JSONDecodeError as e:
                            click.echo(
                                f"Skipping invalid JSON line in {language}/labels.jsonl: {labels_line} ({e})"
                            )
                            continue

                        try:
                            flags_document = json.loads(flags_line)
                            filter = flags_document["filter"]
                            near_duplicate_similarity = flags_document[
                                "near_duplicate_similarity"
                            ]
                        except json.JSONDecodeError as e:
                            click.echo(
                                f"Skipping invalid JSON line in {language}/flags.jsonl: {flags_line.strip()} ({e})"
                            )
                            continue

                        # Check if filtered
                        if filter:
                            continue
                        if near_duplicate_similarity >= LIMIT_QUESTIONS_SIMILARITY:
                            continue

                    try:
                        embeddings_document = json.loads(embeddings_line)
                        embedding = embeddings_document["embedding"]
                    except json.JSONDecodeError as e:
                        click.echo(
                            f"Skipping invalid JSON line in {language}/embeddings.jsonl: {embeddings_line} ({e})"
                        )
                        continue

                    # Detect language code in URL path
                    # match = re.match(f"^{scheme_host}" + r"/([a-z]{2})/", url)
                    # if match:
                    #     if language == map_iso_code(match.group(1)):
                    #         url_path = url[len(scheme_host) + 4:]
                    #     else:
                    #         continue
                    # else:
                    #     url_path = "-"
                    url_path = "-"

                    # Add to the index for language
                    if not url_path in documents:
                        documents[url_path] = {}
                    if not language in documents[url_path]:
                        documents[url_path][language] = []
                    documents[url_path][language].append(
                        {
                            "url": url,
                            "question": question,
                            "answer": answer,
                            "topic": topic,
                            "question_type": question_type,
                            "embedding": embedding,
                        }
                    )

            if consider_language:
                labels_file.close()
                flags_file.close()

    # Collate all URL paths that are not aligned
    _url_paths = list(documents.keys())
    for url_path in _url_paths:
        if url_path == "-":
            continue

        _languages = list(documents[url_path].keys())
        if len(_languages) > 1:
            continue
        for language in _languages:
            # if not "-" in documents:
            #     documents["-"] = {}
            # if not language in documents["-"]:
            #     documents["-"][language] = []
            # documents["-"][language].extend(documents[url_path][language])
            del documents[url_path][language]
            if len(documents[url_path]) == 0:
                del documents[url_path]

    # Generate candidates
    candidates = []
    for url_path in sorted(documents.keys()):

        _languages = list(documents[url_path].keys())
        if len(_languages) < 2:
            continue

        embeddings_dict = {}
        for language in sorted(_languages):

            # Skip if no documents
            if len(documents[url_path][language]) == 0:
                continue

            # Stack embeddings and copy to GPU
            if not language in embeddings_dict:
                embeddings = np.stack(
                    [
                        document["embedding"]
                        for document in documents[url_path][language]
                    ]
                )
                embeddings = torch.Tensor(embeddings).cuda()
                norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                embeddings_dict[language] = norm

            for _language in sorted(_languages):
                if _language <= language:
                    continue

                # Skip if no documents
                if len(documents[url_path][_language]) == 0:
                    continue

                # Stack embeddings and copy to GPU
                _embeddings = np.stack(
                    [
                        document["embedding"]
                        for document in documents[url_path][_language]
                    ]
                )
                _embeddings = torch.Tensor(_embeddings).cuda()
                _norm = torch.nn.functional.normalize(_embeddings, p=2, dim=1)
                embeddings_dict[_language] = _norm

                # Compute cosine similarity
                similarity_matrix = (
                    torch.matmul(
                        embeddings_dict[language], embeddings_dict[_language].T
                    )
                    .cpu()
                    .numpy()
                )

                # Store maxima for each column
                max_similarity_columns = np.max(similarity_matrix, axis=0)

                # Filter candidates
                for i in range(similarity_matrix.shape[0]):
                    j = np.argmax(similarity_matrix[i, :])
                    if similarity_matrix[i, j] < THRESHOLD_LABSE_SIMILARITY:
                        continue
                    # Skip if it is row-wise or column-wise not the maximum similarity
                    if similarity_matrix[i, j] < max_similarity_columns[j]:
                        continue
                    candidate = {
                        "scheme_host": scheme_host,
                        "similarity": float(similarity_matrix[i, j]),
                        "languages": (language, _language),
                        "urls": (
                            documents[url_path][language][i]["url"],
                            documents[url_path][_language][j]["url"],
                        ),
                        "questions": (
                            documents[url_path][language][i]["question"],
                            documents[url_path][_language][j]["question"],
                        ),
                        "answers": (
                            documents[url_path][language][i]["answer"],
                            documents[url_path][_language][j]["answer"],
                        ),
                        "topics": (
                            documents[url_path][language][i]["topic"],
                            documents[url_path][_language][j]["topic"],
                        ),
                        "question_types": (
                            documents[url_path][language][i]["question_type"],
                            documents[url_path][_language][j]["question_type"],
                        ),
                    }
                    candidates.append(candidate)

        # Free memory
        del embeddings_dict
        torch.cuda.empty_cache()

    return candidates


@click.command()
@click.argument("from_index", type=int, default=0)
@click.argument("to_index", type=int, default=-1)
def pc_candidates(from_index: int, to_index: int):
    """
    Generate candidate pairs for parallel corpus extraction.
    """
    # Assert CUDA is available
    assert torch.cuda.is_available(), "CUDA is not available"

    # Read host languages file
    scheme_hosts_languages_path = os.path.join(
        DATASETS_FOLDER, "faqs", "scheme_hosts_languages.json"
    )

    # Initialize results path
    results_path = os.path.join(DATASETS_FOLDER, "faqs")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Directory not found: {results_path}")
    if not os.path.isdir(results_path):
        raise NotADirectoryError(f"Path is not a directory: {results_path}")

    # Initialize candidates and offsets
    candidates = []
    offsets = {}

    # Loop over all languages
    for language in sorted(os.listdir(results_path)):
        language_path = os.path.join(results_path, language)
        if not os.path.isdir(language_path):
            continue

        # Load offsets
        with open(os.path.join(language_path, "offsets.json"), "r") as offsets_file:
            _offsets = json.load(offsets_file)
            offsets[language] = _offsets

    with open(scheme_hosts_languages_path, "r") as file:
        scheme_hosts_languages = json.load(file)

        with tqdm(total=len(scheme_hosts_languages), mininterval=10) as pbar:

            # Loop over all hosts
            for i, (scheme_host, languages) in enumerate(
                scheme_hosts_languages.items()
            ):
                if to_index > 0 and i >= to_index:
                    break

                # Update progress bar
                pbar.update(1)

                if i < from_index:
                    continue

                for language in languages:
                    if language not in offsets:
                        click.echo(f"Language {language} not found in offsets")
                        continue

                # Get offsets for this scheme + host
                languages = [
                    language
                    for language in sorted(languages)
                    if scheme_host in offsets[language]
                ]
                scheme_host_offsets = [
                    offsets[language][scheme_host]
                    for language in languages
                    if scheme_host in offsets[language]
                ]

                if len(languages) < 2:
                    continue

                # Generate candidates
                candidates.extend(
                    generate_candidates(
                        scheme_host, languages, scheme_host_offsets, dataset_name
                    )
                )

    # Save candidates to file
    candidates_path = os.path.join(
        DATASETS_FOLDER,
        dataset_name,
        "results",
        f"candidates_{from_index}_{to_index}.jsonl",
    )
    click.echo(f"Save candidates to: {candidates_path}")

    # Write candidates to file
    hist_similarities = {}
    with open(candidates_path, "w") as file:
        for candidate in candidates:
            s = int(candidate["similarity"] * 100) / 100.0
            hist_similarities[s] = hist_similarities.get(s, 0) + 1
            file.write(json.dumps(candidate) + "\n")

    # Compute statistics
    click.echo("Histogram of similarities:")
    for s, count in sorted(hist_similarities.items()):
        click.echo(f"{s:.2f}: {count}")

    click.echo("Done")
