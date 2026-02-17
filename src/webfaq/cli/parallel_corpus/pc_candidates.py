import gzip
import json
import os
import re

import click
import numpy as np
import torch
from tqdm import tqdm

from webfaq.config import *
from webfaq.utils import *

THRESHOLD_LABSE_SIMILARITY = 0.8
THRESHOLD_QUESTION_ANSWER_SIMILARITY = 0.5


def generate_candidates(
    origin: str,
    languages: List[str],
    origin_offsets: List[List[Dict[str, int]]],
):
    """
    Given a 'origin' string, a set of languages, and a threshold,
    returns candidate documents (plus scores) that match cross-lingually.
    """
    # Initialize documents dictionary
    documents = {}

    # For each language, open the corresponding file and add relevant docs
    for language, _offsets in zip(languages, origin_offsets):
        language_path = os.path.join(DATASETS_FOLDER, "faqs", language)

        # Load offsets
        faq_offsets = [offset["faq_offset"] for offset in _offsets]
        # semantic_similarity_offsets = [offset["semantic_similarity_offset"] for offset in _offsets]
        # semantic_similarity_offsets = [offset["semantic_similarity_file_offset"] for offset in _offsets]
        semantic_similarity_offsets = []
        for offset in _offsets:
            assert (
                "semantic_similarity_offset" in offset
                or "semantic_similarity_file_offset" in offset
            )
            if "semantic_similarity_offset" in offset:
                semantic_similarity_offsets.append(offset["semantic_similarity_offset"])
            else:
                semantic_similarity_offsets.append(
                    offset["semantic_similarity_file_offset"]
                )
        labse_embeddings_offsets = [
            offset["labse_embeddings_offset"] for offset in _offsets
        ]

        # Initialize set of questions
        set_questions = set()

        for filename in sorted(os.listdir(language_path)):
            if not filename.startswith("faqs_sorted_") or not filename.endswith(
                ".jsonl.gz"
            ):
                continue

            faq_path = os.path.join(language_path, filename)
            semantic_similarity_path = os.path.join(
                language_path, filename.replace("faqs_sorted_", "semantic_similarity_")
            )
            labse_embeddings_path = os.path.join(
                language_path, filename.replace("faqs_sorted_", "labse_embeddings_")
            )
            with gzip.open(faq_path, "rt", encoding="UTF-8") as faq_file, gzip.open(
                semantic_similarity_path, "rt", encoding="UTF-8"
            ) as semantic_similarity_file, gzip.open(
                labse_embeddings_path, "rt", encoding="UTF-8"
            ) as labse_embeddings_file:

                for i, (
                    faq_offset,
                    semantic_similarity_offset,
                    labse_embeddings_offset,
                ) in enumerate(
                    zip(
                        faq_offsets,
                        semantic_similarity_offsets,
                        labse_embeddings_offsets,
                    )
                ):

                    faq_file.seek(faq_offset)
                    semantic_similarity_file.seek(semantic_similarity_offset)
                    labse_embeddings_file.seek(labse_embeddings_offset)

                    while True:
                        faq_line = faq_file.readline().strip()
                        semantic_similarity_line = (
                            semantic_similarity_file.readline().strip()
                        )
                        labse_embeddings_line = labse_embeddings_file.readline().strip()

                        # Break at end of file
                        if not faq_line:
                            break

                        try:
                            faq_document = json.loads(faq_line)
                            url = faq_document["url"]
                            _origin = faq_document["origin"]
                            question = faq_document["question"]
                            answer = faq_document["answer"]
                        except json.JSONDecodeError as e:
                            click.echo(
                                f"Skipping invalid JSON line in {faq_path}: {faq_line} ({e})"
                            )
                            continue

                        # Break if origin changes
                        if _origin != origin:
                            break

                        # Skip if question is already in documents
                        if question in set_questions:
                            continue
                        set_questions.add(question)

                        try:
                            semantic_similarity_document = json.loads(
                                semantic_similarity_line
                            )
                            score = semantic_similarity_document["score"]
                        except json.JSONDecodeError as e:
                            click.echo(
                                f"Skipping invalid JSON line in {semantic_similarity_path}: {semantic_similarity_line.strip()} ({e})"
                            )
                            continue

                        # Check if filtered
                        if score < THRESHOLD_QUESTION_ANSWER_SIMILARITY:
                            continue

                        try:
                            labse_embeddings_document = json.loads(
                                labse_embeddings_line
                            )
                            embedding = labse_embeddings_document["embedding"]
                        except json.JSONDecodeError as e:
                            click.echo(
                                f"Skipping invalid JSON line in {labse_embeddings_path}: {labse_embeddings_line.strip()} ({e})"
                            )
                            continue

                        # Add to the index for language
                        if not language in documents:
                            documents[language] = []
                        documents[language].append(
                            {
                                "url": url,
                                "question": question,
                                "answer": answer,
                                "embedding": embedding,
                            }
                        )

    # Generate candidates
    candidates = []
    _languages = list(documents.keys())
    if len(_languages) < 2:
        return candidates

    embeddings_dict = {}
    for language in sorted(_languages):

        # Skip if no documents
        if len(documents[language]) == 0:
            continue

        # Stack embeddings and copy to GPU
        if not language in embeddings_dict:
            embeddings = np.stack(
                [document["embedding"] for document in documents[language]]
            )
            embeddings = torch.Tensor(embeddings).cuda()
            norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings_dict[language] = norm

        for _language in sorted(_languages):
            if _language <= language:
                continue

            # Skip if no documents
            if len(documents[_language]) == 0:
                continue

            # Stack embeddings and copy to GPU
            _embeddings = np.stack(
                [document["embedding"] for document in documents[_language]]
            )
            _embeddings = torch.Tensor(_embeddings).cuda()
            _norm = torch.nn.functional.normalize(_embeddings, p=2, dim=1)
            embeddings_dict[_language] = _norm

            # Compute cosine similarity
            similarity_matrix = (
                torch.matmul(embeddings_dict[language], embeddings_dict[_language].T)
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
                    "origin": origin,
                    "similarity": float(similarity_matrix[i, j]),
                    "languages": (language, _language),
                    "urls": (
                        documents[language][i]["url"],
                        documents[_language][j]["url"],
                    ),
                    "questions": (
                        documents[language][i]["question"],
                        documents[_language][j]["question"],
                    ),
                    "answers": (
                        documents[language][i]["answer"],
                        documents[_language][j]["answer"],
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
    origins_languages_path = os.path.join(DATASETS_FOLDER, "origins_languages.json")

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

        for filename in sorted(os.listdir(language_path)):
            if not filename.startswith("offsets_") or not filename.endswith(
                ".jsonl.gz"
            ):
                continue

            # Load offsets
            offsets_path = os.path.join(language_path, filename)
            with gzip.open(offsets_path, "rt", encoding="UTF-8") as offsets_file:
                _offsets = json.load(offsets_file)
                if language not in offsets:
                    offsets[language] = _offsets
                else:
                    for origin, origin_offsets in _offsets.items():
                        if origin not in offsets[language]:
                            offsets[language][origin] = origin_offsets
                        else:
                            offsets[language][origin].extend(origin_offsets)

    with open(origins_languages_path, "r") as file:
        origins_languages = json.load(file)

        with tqdm(total=len(origins_languages), mininterval=10) as pbar:

            # Loop over all hosts
            for i, (origin, languages) in enumerate(origins_languages.items()):
                if to_index > 0 and i >= to_index:
                    break

                click.echo()
                click.echo(f"Processing origin {i}: {origin}")

                # Update progress bar
                pbar.update(1)

                if i < from_index:
                    continue

                for language in languages:
                    if language not in offsets:
                        click.echo(f"Language {language} not found in offsets")
                        continue

                # Skip if any language has more than 10,000 URLs
                if any([v > 10_000 for v in languages.values()]):
                    click.echo(
                        f"Skipping origin {origin} with more than 10,000 URLs:\n{languages}"
                    )
                    continue

                # Get offsets for this origin
                languages = [
                    language
                    for language in sorted(languages)
                    if origin in offsets[language]
                ]
                origin_offsets = [
                    offsets[language][origin]
                    for language in languages
                    if origin in offsets[language]
                ]

                # Skip if less than 2 languages - should not happen
                if len(languages) < 2:
                    continue

                # Generate candidates
                candidates.extend(
                    generate_candidates(origin, languages, origin_offsets)
                )

    # Save candidates to file
    candidates_path = os.path.join(
        DATASETS_FOLDER,
        f"candidates_{from_index}_{to_index}.jsonl",
    )
    os.makedirs(os.path.dirname(candidates_path), exist_ok=True)
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
