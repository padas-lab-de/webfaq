import click
import os
import json
from glob import glob
from webfaq.config import *


THRESHOLD_SIMILARITY = 0.9  # For 0.946 precision
THRESHOLD_BITEXTS = 4_000


@click.command()
@click.argument("dataset_name", type=str)
@click.argument("filename_pattern", type=str)
def pc_transform(dataset_name: str, filename_pattern: str):
    """
    Transform the scored parallel candidates and save them as bitexts.
    """
    # Load the JSONL files
    scores_paths = glob(
        os.path.join(DATASETS_FOLDER, dataset_name, "results", filename_pattern)
    )
    click.echo(f"Found {len(scores_paths)} scores files")

    # Initialize bitexts
    bitexts = {}

    for scores_path in scores_paths:
        click.echo(f"Scores file: {scores_path}")

        with open(scores_path, "r") as file:

            for line in file:
                try:
                    document = json.loads(line)
                    similarity = document["similarity"]
                    score = document["score"]
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

                # Add to bitexts
                languages_str = f"{languages[0]}_{languages[1]}"
                if not languages_str in bitexts:
                    bitexts[languages_str] = []
                bitexts[languages_str].append(
                    {
                        "origin": document["scheme_host"],
                        "labse_similarity": similarity,
                        # "score": score,
                        # "languages": languages,
                        "question1": questions[0],
                        "question2": questions[1],
                        "answer1": answers[0],
                        "answer2": answers[1],
                        "details": {
                            "urls": document["urls"],
                            "topics": document["topics"],
                            "question_types": document["question_types"],
                        },
                    }
                )

    # Save bitexts to file
    for languages_str, scored_documents in bitexts.items():
        if len(scored_documents) < THRESHOLD_BITEXTS:
            click.echo(
                f'Skipping language combination "{languages_str}" with less than {THRESHOLD_BITEXTS} bitexts ({len(scored_documents)})'
            )
            continue

        # Sort bitexts by first URL
        scored_documents = sorted(
            scored_documents, key=lambda x: x["details"]["urls"][0]
        )

        bitexts_path = os.path.join(
            DATASETS_FOLDER, dataset_name, "bitexts", f"{languages_str}.jsonl"
        )
        os.makedirs(os.path.dirname(bitexts_path), exist_ok=True)

        click.echo(f'Saving language combination "{languages_str}": {bitexts_path}')
        with open(bitexts_path, "w") as file:
            for scored_document in scored_documents:
                file.write(json.dumps(scored_document) + "\n")
