import gzip
import json
import os
import time

import click
import pandas as pd
from tqdm import tqdm

from webfaq.config import *
from webfaq.utils import *

RATE_LIMIT_PER_HOUR = 128
THRESHOLD_MIN_EXAMPLES = 100


@click.command()
@click.argument("temp_folder", type=str, default=TEMP_FOLDER)
def prepare_base(temp_folder: str):
    """
    Preparing the dataset of extracted Q&A pairs to be pushed to HuggingFace.
    """
    # Initialize results path
    results_path = os.path.join(DATASETS_FOLDER, "faqs")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Directory not found: {results_path}")
    if not os.path.isdir(results_path):
        raise NotADirectoryError(f"Path is not a directory: {results_path}")

    # Loop over all languages
    for language in sorted(os.listdir(results_path)):
        language_path = os.path.join(results_path, language)
        if not os.path.isdir(language_path):
            continue

        click.echo(f"Language: {language}")

        # Check if output path already exists
        output_path = os.path.join(temp_folder, "upload", f"{language}.parquet")
        if os.path.exists(output_path):
            click.echo(
                f"Skipping language {language} as output file already exists: {output_path}"
            )
            continue

        # Initialize results list
        results = []

        for filename in sorted(os.listdir(language_path)):
            if not filename.startswith("faqs_sorted_") or not filename.endswith(
                ".jsonl.gz"
            ):
                continue

            click.echo(f"Reading file: {filename}")

            # Check if semantic similarity scores file exists
            semantic_similarity_path = os.path.join(
                language_path, filename.replace("faqs_sorted_", "semantic_similarity_")
            )
            if not os.path.exists(semantic_similarity_path):
                raise FileNotFoundError(
                    f"Semantic similarity scores file not found: {semantic_similarity_path}"
                )

            consider_labels = language in LANGUAGES_100_ORIGINS

            # Check if labels file exists
            labels_path = os.path.join(
                language_path, filename.replace("faqs_sorted_", "labels_")
            )
            if consider_labels:
                if not os.path.exists(labels_path):
                    raise FileNotFoundError(f"Labels file not found: {labels_path}")

            # Check if nfqa labels file exists
            nfqa_labels_path = os.path.join(
                language_path, filename.replace("faqs_sorted_", "nfqa_")
            )
            if consider_labels:
                if not os.path.exists(nfqa_labels_path):
                    raise FileNotFoundError(
                        f"NFQA labels file not found: {nfqa_labels_path}"
                    )

            with gzip.open(
                os.path.join(language_path, filename), "rt", encoding="UTF-8"
            ) as faq_file, gzip.open(
                semantic_similarity_path, "rt", encoding="UTF-8"
            ) as semantic_similarity_file:

                if consider_labels:
                    labels_file = gzip.open(labels_path, "rt", encoding="UTF-8")
                    nfqa_labels_file = gzip.open(
                        nfqa_labels_path, "rt", encoding="UTF-8"
                    )
                else:
                    labels_file = [None] * 1_000_000  # Dummy list
                    nfqa_labels_file = [None] * 1_000_000  # Dummy list

                for (
                    faq_line,
                    semantic_similarity_line,
                    labels_line,
                    nfqa_labels_line,
                ) in tqdm(
                    zip(
                        faq_file,
                        semantic_similarity_file,
                        labels_file,
                        nfqa_labels_file,
                    )
                ):

                    try:
                        faq_document = json.loads(faq_line)

                        semantic_similarity_document = json.loads(
                            semantic_similarity_line
                        )
                        if not faq_document["id"] == semantic_similarity_document["id"]:
                            click.echo(
                                f"ID mismatch between FAQ and semantic similarity: {faq_document['id']} != {semantic_similarity_document['id']}"
                            )
                            continue

                        topic = None
                        if not labels_line is None:
                            labels_document = json.loads(labels_line)
                            if "label" in labels_document:
                                topic = labels_document["label"]
                            if (
                                topic
                                and not faq_document["id"] == labels_document["id"]
                            ):
                                click.echo(
                                    f"ID mismatch between FAQ and labels: {faq_document['id']} != {labels_document['id']}"
                                )
                                continue

                        question_type = None
                        if not nfqa_labels_line is None:
                            nfqa_labels_document = json.loads(nfqa_labels_line)
                            if "label" in nfqa_labels_document:
                                question_type = nfqa_labels_document["label"]
                            if (
                                question_type
                                and not faq_document["id"] == nfqa_labels_document["id"]
                            ):
                                click.echo(
                                    f"ID mismatch between FAQ and NFQA labels: {faq_document['id']} != {nfqa_labels_document['id']}"
                                )
                                continue
                    except json.JSONDecodeError as e:
                        click.echo(
                            f"Skipping invalid JSON line in {language}/faq.jsonl: {faq_line.strip()} ({e})"
                        )
                        continue

                    # Append to list
                    result = {
                        "id": faq_document["id"],
                        "origin": faq_document["origin"],
                        "url": faq_document["url"],
                        "question": faq_document["question"],
                        "answer": faq_document["answer"],
                        "semantic_similarity_score": semantic_similarity_document[
                            "score"
                        ],
                    }
                    if topic:
                        result["topic"] = topic
                    if question_type:
                        result["question_type"] = question_type

                    # Append to list
                    results.append(result)

                # Close files
                if consider_labels:
                    labels_file.close()  # type: ignore
                    nfqa_labels_file.close()  # type: ignore

        if len(results) < THRESHOLD_MIN_EXAMPLES:
            click.echo(
                f"Skipping language {language} due to insufficient examples ({len(results)} < {THRESHOLD_MIN_EXAMPLES})"
            )
            continue
        click.echo(f"Total examples for language {language}: {len(results)}")

        # Write results to Parquet file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(results)
        df.to_parquet(output_path, engine="pyarrow", index=False)
        click.echo(f"Wrote Parquet file: {output_path}")

    click.echo("Done")
