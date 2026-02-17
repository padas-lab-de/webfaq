import gzip
import json
import os

import click
import torch
from tqdm import tqdm
from transformers import pipeline

from webfaq.config import *
from webfaq.utils import *

BATCH_SIZE = 64


def concat_qa(example):
    text = f"{example['question']} ### {example['answer']}"
    if "title" in example and example["title"]:
        text += f" ### Title: {example['title']}"
    if "description" in example and example["description"]:
        text += f" ### Description: {example['description']}"
    return text


@click.command()
def label():
    """
    Compute the topic labels for the extracted Q&A pairs.
    """
    # Instantiate Embedding Model
    pretrained_model_name = "michaeldinzinger/xlm-roberta-base-qa-topic-classification"  # "AliSalman29/nfqa-multilingual-classifier"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "text-classification",
        model=pretrained_model_name,
        truncation=True,
        max_length=512,
        device=device,
    )
    click.echo(f"Loading model: {pretrained_model_name}")

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

        if language not in LANGUAGES_100_ORIGINS:
            click.echo(f"Skipping language {language}")
            continue

        click.echo(f"Language: {language}")

        for filename in sorted(os.listdir(language_path)):
            if not filename.startswith("faqs_sorted_") or not filename.endswith(
                ".jsonl.gz"
            ):
                continue

            click.echo(f"Reading file: {filename}")

            # Check if labels already exist
            labels_path = os.path.join(
                language_path, filename.replace("faqs_sorted_", "labels_")
            )
            if os.path.exists(labels_path):
                click.echo(f"Labels already exist: {labels_path}")
                continue

            # Load Q&A pairs
            ids = []
            texts = []

            with gzip.open(
                os.path.join(language_path, filename), "rt", encoding="UTF-8"
            ) as file:

                for line in tqdm(file):
                    try:
                        document = json.loads(line)
                        ids.append(document["id"])
                        texts.append(concat_qa(document))
                    except json.JSONDecodeError as e:
                        click.echo(
                            f"Skipping invalid JSON line: {line.strip()} ({e})",
                            err=True,
                        )
                        ids.append("")
                        texts.append("")

            # Compute labels for concatenated texts
            labels = []
            for start_idx in tqdm(range(0, len(texts), BATCH_SIZE)):
                batch_texts = texts[start_idx : start_idx + BATCH_SIZE]
                results = pipe(batch_texts, truncation=True, max_length=512)

                for result in results:  # type: ignore
                    label = result["label"]  # type: ignore
                    labels.append(label)

            # Save labels to file
            click.echo(f"Writing file {labels_path}")
            with gzip.open(labels_path, "wt", encoding="UTF-8") as file:
                for i in tqdm(range(len(ids))):
                    document = {
                        "id": ids[i],
                        "label": labels[i],
                    }
                    file.write(json.dumps(document) + "\n")

    click.echo("Done")
