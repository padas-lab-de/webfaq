import gzip
import json
import os

import click
from tqdm import tqdm

from webfaq.config import *
from webfaq.utils import *


@click.command()
def offsets():
    """
    Compute the offsets for each combination of scheme and host in the extracted Q&A pairs.
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

        for filename in sorted(os.listdir(language_path)):
            if not filename.startswith("faqs_sorted_") or not filename.endswith(
                ".jsonl.gz"
            ):
                continue

            click.echo(f"Reading file: {filename}")

            # Check if offsets already exist
            offsets_path = os.path.join(
                language_path, filename.replace("faqs_sorted_", "offsets_")
            )
            if os.path.exists(offsets_path):
                click.echo(f"Offsets already exist: {offsets_path}")
                continue

            # Load Q&A pairs
            offsets = {}

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

                with tqdm() as pbar:

                    current_origin = None
                    # counter = 0
                    while True:
                        # Update progress bar
                        pbar.update(1)

                        # Track offsets
                        faq_offset = faq_file.tell()
                        semantic_similarity_offset = semantic_similarity_file.tell()
                        labse_embeddings_offset = labse_embeddings_file.tell()

                        # Read lines
                        faq_line = faq_file.readline()
                        _ = semantic_similarity_file.readline()
                        _ = labse_embeddings_file.readline()

                        # Check if EOF
                        if not faq_line:
                            break

                        # Extract URL and origin
                        try:
                            faq_document = json.loads(faq_line)
                            origin = faq_document["origin"]
                        except json.JSONDecodeError as e:
                            click.echo(
                                f"Skipping invalid JSON line in faq.jsonl: {faq_line.strip()} ({e})"
                            )
                            continue

                        # Check if origin are the same as the previous one
                        if current_origin != origin:
                            # Update current origin
                            current_origin = origin

                            if not current_origin in offsets:
                                offsets[current_origin] = []
                            _offsets = {
                                "faq_offset": faq_offset,
                                "semantic_similarity_offset": semantic_similarity_offset,
                                "labse_embeddings_offset": labse_embeddings_offset,
                            }
                            offsets[current_origin].append(_offsets)

            # Save offsets to file
            offsets_path = os.path.join(
                language_path, filename.replace("faqs_sorted_", "offsets_")
            )
            click.echo(f"Writing file {offsets_path}")
            with gzip.open(offsets_path, "wt", encoding="UTF-8") as file:
                json.dump(offsets, file, indent=2)

    click.echo("Done")
