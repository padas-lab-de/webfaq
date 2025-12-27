import click
import os
import json
from webfaq.config import *


@click.command()
@click.argument("dataset_name", type=str)
def offsets(dataset_name: str):
    """
    Compute the offsets for each combination of scheme and host in the extracted Q&A pairs.
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

        click.echo(f"Language: {language}")

        # if language == "eng":
        #     continue

        consider_labels = language in LANGUAGES_100_SCHEME_HOSTS

        # Load Q&A pairs
        offsets = {}
        with open(os.path.join(language_path, "faq.jsonl"), "r") as faq_file, open(
            os.path.join(language_path, "embeddings.jsonl"), "r"
        ) as embeddings_file:

            if consider_labels:
                labels_file = open(os.path.join(language_path, "labels.jsonl"), "r")
                flags_file = open(os.path.join(language_path, "flags.jsonl"), "r")

            current_scheme_host = None
            # counter = 0
            while True:
                # Track offsets
                faq_offset = faq_file.tell()
                if consider_labels:
                    labels_offset = labels_file.tell()
                    flags_offset = flags_file.tell()
                embeddings_offset = embeddings_file.tell()

                # Read lines
                faq_line = faq_file.readline()
                _ = embeddings_file.readline()
                if consider_labels:
                    _ = labels_file.readline()
                    _ = flags_file.readline()

                # Check if EOF
                if not faq_line:
                    break

                # Extract URL and scheme_host
                try:
                    faq_document = json.loads(faq_line)
                    scheme_host = faq_document["scheme_host"]
                except json.JSONDecodeError as e:
                    click.echo(
                        f"Skipping invalid JSON line in faq.jsonl: {faq_line.strip()} ({e})"
                    )
                    continue

                # Check if scheme_host are the same as the previous one
                if current_scheme_host != scheme_host:
                    # Update current scheme_host
                    current_scheme_host = scheme_host

                    if not current_scheme_host in offsets:
                        offsets[current_scheme_host] = []
                    _offsets = {
                        "faq_offset": faq_offset,
                        "embeddings_offset": embeddings_offset,
                    }
                    if consider_labels:
                        _offsets["labels_offset"] = labels_offset
                        _offsets["flags_offset"] = flags_offset
                    offsets[current_scheme_host].append(_offsets)

                # counter += 1
                # if counter > 10:
                #     break

            if consider_labels:
                labels_file.close()
                flags_file.close()

        # Save offsets to file
        offsets_path = os.path.join(language_path, "offsets.json")
        click.echo(f"Writing file {offsets_path}")
        with open(offsets_path, "w") as file:
            json.dump(offsets, file, indent=2)

    click.echo("Done")
