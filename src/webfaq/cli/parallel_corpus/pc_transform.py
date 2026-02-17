import gzip
import json
import os
from glob import glob

import click

from webfaq.config import *

THRESHOLD_SIMILARITY = 0.9  # For 0.946 precision
THRESHOLD_BITEXTS = 100


@click.command()
@click.argument("filename_pattern", type=str)
def pc_transform(filename_pattern: str):
    """
    Transform the scored parallel candidates and save them as bitexts.
    """
    # Load the JSONL files
    scores_paths = glob(os.path.join(DATASETS_FOLDER, filename_pattern))
    click.echo(f"Found {len(scores_paths)} scores files")

    # Initialize bitexts
    bitexts = {}

    for scores_path in scores_paths:
        click.echo(f"Scores file: {scores_path}")

        with open(scores_path, "r") as file:

            for line in file:
                try:
                    document = json.loads(line)
                    origin = document["origin"]
                    similarity = document["similarity"]
                    languages = document["languages"]
                    questions = document["questions"]
                    answers = document["answers"]

                except json.JSONDecodeError as e:
                    click.echo(
                        f"Skipping invalid JSON line in {scores_path}: {line.strip()} ({e})"
                    )
                    continue

                # Validity checks
                assert len(languages) == 2
                assert len(questions) == 2
                assert len(answers) == 2

                # Skip if similarity is below threshold
                if similarity < THRESHOLD_SIMILARITY:
                    continue

                # Skip if questions or answers are identical
                if questions[0] == questions[1] or answers[0] == answers[1]:
                    continue

                result = {
                    "language1": languages[0],
                    "language2": languages[1],
                    "origin": origin,
                    "labse_similarity": similarity,
                    "question1": questions[0],
                    "question2": questions[1],
                    "answer1": answers[0],
                    "answer2": answers[1],
                    "details": {
                        "urls": document["urls"],
                    },
                }

                # Add to bitexts
                languages = tuple(sorted(languages))
                if not languages in bitexts:
                    bitexts[languages] = []
                bitexts[languages].append(result)

    # Save bitexts to file
    bitexts_eng = []
    bitext_other = []
    for languages, scored_documents in sorted(bitexts.items()):
        if len(scored_documents) < THRESHOLD_BITEXTS:
            click.echo(
                f'Skipping language combination "{languages}" with less than {THRESHOLD_BITEXTS} bitexts ({len(scored_documents)})'
            )
            continue

        # Sort bitexts by first URL
        scored_documents = sorted(
            scored_documents, key=lambda x: x["details"]["urls"][0]
        )

        if languages[0] == "eng" or languages[1] == "eng":
            bitexts_eng.extend(scored_documents)
        else:
            bitext_other.extend(scored_documents)

    # Save English bitexts
    bitexts_path = os.path.join(DATASETS_FOLDER, "bitexts", "eng.jsonl.gz")
    os.makedirs(os.path.dirname(bitexts_path), exist_ok=True)
    click.echo(f"Saving English bitexts: {bitexts_path}")
    with gzip.open(bitexts_path, "wt", encoding="UTF-8") as file:
        for scored_document in bitexts_eng:
            file.write(json.dumps(scored_document) + "\n")

    # Save other bitexts
    bitexts_path = os.path.join(DATASETS_FOLDER, "bitexts", "other.jsonl.gz")
    os.makedirs(os.path.dirname(bitexts_path), exist_ok=True)
    click.echo(f"Saving other bitexts: {bitexts_path}")
    with gzip.open(bitexts_path, "wt", encoding="UTF-8") as file:
        for scored_document in bitext_other:
            file.write(json.dumps(scored_document) + "\n")

    click.echo("Done")
