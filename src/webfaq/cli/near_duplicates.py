import click
import os
import sys
import json
import torch
import gzip
from tqdm import tqdm
from webfaq.utils import *
from webfaq.config import *


THRESHOLD_QUESTION_ANSWER_SIMILARITY = 0.5
# # LIMIT_QUESTIONS_SIMILARITY = 0.8
# LIMIT_QUESTIONS_SIMILARITY = 0.7


def blockwise_matmul(matrix, block_size=4_096):
    # Allocate result on CPU to avoid GPU Out-Of-Memory Error
    n, _ = matrix.shape
    result = torch.zeros((n, n), device="cpu")

    # Perform block-wise multiplication
    for i in range(0, n, block_size):
        # Compute only upper triangle, as it's symmetric
        for j in range(i, n, block_size):
            block_result = torch.matmul(
                matrix[i : i + block_size], matrix[j : j + block_size].T
            )

            # Move block result to CPU
            result[i : i + block_size, j : j + block_size] = block_result.cpu()

            # Mirror the block to save computation
            if i != j:
                result[j : j + block_size, i : i + block_size] = block_result.T.cpu()

    return result


@click.command()
@click.argument("dataset_name", type=str)
def near_duplicates(dataset_name: str):
    """
    Compute near-duplicate questions for the extracted Q&A pairs.
    """
    # Initialize results path
    results_path = os.path.join(DATASETS_FOLDER, dataset_name, "results")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Directory not found: {results_path}")
    if not os.path.isdir(results_path):
        raise NotADirectoryError(f"Path is not a directory: {results_path}")

    # Loop over all languages
    for language in sorted(os.listdir(results_path)):
        language_path = os.path.join(results_path, language)
        if not os.path.isdir(language_path):
            continue

        if not language in LANGUAGES_100_SCHEME_HOSTS:
            continue

        click.echo(f"Language: {language}")

        if language != "eng":
            continue

        # if language <= "deu":
        #     continue

        # Initialize flags and list of ids
        flags = []

        # Initialize questions set
        set_questions = set()

        num_lines = count_lines(os.path.join(language_path, "faq.jsonl"))
        with tqdm(total=num_lines, mininterval=10, file=sys.stdout) as pbar:

            # Load Q&A pairs with labels
            with open(
                os.path.join(language_path, "faq.jsonl"), "r"
            ) as faq_file, gzip.open(
                os.path.join(language_path, "retrieval_embeddings.jsonl.gz"), "r"
            ) as retrieval_embeddings_file:

                ids = []
                set_filtered_ids = set()
                dict_near_duplicate_similarity = {}
                current_ids = []
                current_questions = []
                current_question_embeddings = []
                current_scheme_host = None
                for faq_line, retrieval_embeddings_line in zip(
                    faq_file, retrieval_embeddings_file
                ):

                    # Update progress bar
                    pbar.update(1)

                    try:
                        faq_document = json.loads(faq_line)
                        id = faq_document["id"]
                        scheme_host = faq_document["scheme_host"]
                        question = faq_document["question"]
                        answer = faq_document["answer"]

                        # Add to list of ids
                        ids.append(id)
                    except json.JSONDecodeError as e:
                        click.echo(
                            f"Skipping invalid JSON line in {language}/faq.jsonl: {faq_line.strip()} ({e})"
                        )
                        ids.append("")
                        continue

                    # Check for exact duplicates
                    if question in set_questions:
                        set_filtered_ids.add(id)
                        continue
                    set_questions.add(question)

                    try:
                        retrieval_embeddings_document = json.loads(
                            retrieval_embeddings_line
                        )
                    except json.JSONDecodeError as e:
                        click.echo(
                            f"Skipping invalid JSON line in {language}/retrieval_embeddings.jsonl.gz: {retrieval_embeddings_line.strip()} ({e})"
                        )
                        continue

                    # Check if question and answer embeddings are close enough
                    question_embedding = torch.tensor(
                        retrieval_embeddings_document["question_embedding"]
                    )
                    answer_embedding = torch.tensor(
                        retrieval_embeddings_document["answer_embedding"]
                    )
                    similarity = torch.nn.functional.cosine_similarity(
                        question_embedding, answer_embedding, dim=0
                    ).item()
                    if similarity < THRESHOLD_QUESTION_ANSWER_SIMILARITY:
                        click.echo(
                            f"Similarity: {similarity} - {question} - {answer}",
                            err=True,
                        )
                        set_filtered_ids.add(id)
                        continue

                    # Add to batch of scheme host
                    if (
                        current_scheme_host is None
                        or current_scheme_host == scheme_host
                    ):
                        current_ids.append(id)
                        current_questions.append(question)
                        norm = torch.nn.functional.normalize(
                            question_embedding, p=2, dim=0
                        ).view(1, -1)
                        current_question_embeddings.append(norm)

                    else:
                        # Skip if current questions is empty
                        if len(current_questions) > 0:
                            # Check for near-duplicate questions
                            question_embeddings_matrix = torch.cat(
                                current_question_embeddings, dim=0
                            ).cuda()
                            with torch.amp.autocast("cuda"):
                                similarity_values = blockwise_matmul(
                                    question_embeddings_matrix
                                ).numpy()

                            for i, (_id, _question) in enumerate(
                                zip(current_ids, current_questions)
                            ):
                                similarity_values_row = similarity_values[i]
                                similarity_values_row[i] = 0
                                dict_near_duplicate_similarity[_id] = max(
                                    similarity_values_row
                                )
                                # if any(similarity_values_row > LIMIT_QUESTIONS_SIMILARITY):
                                #     set_filtered_ids.add(_id)
                                #     _i = list(similarity_values_row >= LIMIT_QUESTIONS_SIMILARITY).index(True)
                                #     click.echo(f"Duplicate question ({language}): {_question} - {current_questions[_i]} - {similarity_values_row[_i]}", err=True)

                            # Reset batch
                            current_ids = [id]
                            current_questions = [question]
                            norm = torch.nn.functional.normalize(
                                question_embedding, p=2, dim=0
                            ).view(1, -1)
                            current_question_embeddings = [norm]

                        # Update flags dictionary
                        for _id in ids:
                            flags.append(
                                {
                                    "id": _id,
                                    "filter": _id in set_filtered_ids,
                                    "near_duplicate_similarity": float(
                                        dict_near_duplicate_similarity.get(_id, 0)
                                    ),
                                }
                            )
                        ids = []
                        set_filtered_ids = set()
                        dict_near_duplicate_similarity = {}

                    # Update current scheme host
                    current_scheme_host = scheme_host

                    # if scheme_host > "http://15icc.org":
                    #     break

                # Skip if current questions is empty
                if len(current_questions) > 0:
                    # Check for near-duplicate questions
                    question_embeddings_matrix = torch.cat(
                        current_question_embeddings, dim=0
                    ).cuda()
                    with torch.amp.autocast("cuda"):
                        similarity_values = blockwise_matmul(
                            question_embeddings_matrix
                        ).numpy()

                    for i, (_id, _question) in enumerate(
                        zip(current_ids, current_questions)
                    ):
                        similarity_values_row = similarity_values[i]
                        similarity_values_row[i] = 0
                        dict_near_duplicate_similarity[_id] = max(similarity_values_row)
                        # if any(similarity_values_row > LIMIT_QUESTIONS_SIMILARITY):
                        #     set_filtered_ids.add(_id)
                        #     _i = list(similarity_values_row >= LIMIT_QUESTIONS_SIMILARITY).index(True)
                        #     click.echo(f"Duplicate question ({language}): {_question} - {current_questions[_i]} - {similarity_values_row[_i]}", err=True)

                    # Reset batch
                    current_ids = [id]
                    current_questions = [question]
                    norm = torch.nn.functional.normalize(
                        question_embedding, p=2, dim=0
                    ).view(1, -1)
                    current_question_embeddings = [norm]

                # Update flags dictionary
                for _id in ids:
                    flags.append(
                        {
                            "id": _id,
                            "filter": _id in set_filtered_ids,
                            "near_duplicate_similarity": float(
                                dict_near_duplicate_similarity.get(_id, 0)
                            ),
                        }
                    )
                ids = []
                set_filtered_ids = set()
                dict_near_duplicate_similarity = {}

        # Save flags in file
        flags_path = os.path.join(language_path, "flags.jsonl")
        with open(flags_path, "w") as flags_file:
            for flag in flags:
                flags_file.write(json.dumps(flag) + "\n")

    click.echo("Done")
