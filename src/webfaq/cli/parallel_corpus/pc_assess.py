import click
import os
import json
from glob import glob
from webfaq.config import *


THRESHOLD_SCORES = [80, 85, 90]


@click.command()
@click.argument("dataset_name", type=str)
@click.argument("filename_pattern", type=str)
def pc_assess(dataset_name: str, filename_pattern: str):
    """
    Assess the scored parallel candidates.
    """
    click.echo("Assess the scored parallel candidates...")

    # Create HF datasets
    count_combinations = {}
    count_questions_identical = 0
    count_answer_identical = 0
    count_questions_answer_identical = 0
    hist_similarities = {}
    hist_scores = {}

    # Load the JSONL files
    scores_paths = glob(
        os.path.join(DATASETS_FOLDER, dataset_name, "results", filename_pattern)
    )
    click.echo(f"Found {len(scores_paths)} scores files")

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

                # Count the combinations
                combination = tuple(sorted(languages))
                count_combinations[combination] = (
                    count_combinations.get(combination, 0) + 1
                )

                # Check if the questions are identical
                if questions[0] == questions[1]:
                    count_questions_identical += 1

                # Check if the answers are identical
                if answers[0] == answers[1]:
                    count_answer_identical += 1

                # Check if the questions and answers are identical
                if questions[0] == questions[1] and answers[0] == answers[1]:
                    count_questions_answer_identical += 1

                # Count the similarity and score values
                s = int(similarity * 100) / 100.0
                if 0.8 <= s < 1.0:
                    if not s in hist_similarities:
                        hist_similarities[s] = []
                    if score >= 0:
                        hist_similarities[s].append(score)
                hist_scores[score] = hist_scores.get(score, 0) + 1

    # Sort combinations by count and scores by key
    count_combinations = dict(
        sorted(count_combinations.items(), key=lambda item: item[1], reverse=True)
    )
    hist_similarities = dict(
        sorted(hist_similarities.items(), key=lambda item: item[0])
    )
    hist_scores = dict(sorted(hist_scores.items(), key=lambda item: item[0]))

    # Print the results
    click.echo("")
    click.echo(f"Count all: {sum(count_combinations.values())}")
    click.echo(f"Count questions identical: {count_questions_identical}")
    click.echo(f"Count answers identical: {count_answer_identical}")
    click.echo(
        f"Count questions and answers identical: {count_questions_answer_identical}"
    )
    click.echo(f"")
    click.echo(f"Count combinations:")
    for combination, count in count_combinations.items():
        click.echo(f"  {combination}: {count}")
    click.echo(f"Histogram similarities:")
    for _s, list_scores in hist_similarities.items():
        if len(list_scores) == 0:
            continue
        mean_scores = sum(list_scores) / len(list_scores)
        click.echo(f"  {_s:3.2f}: {len(list_scores):6d}   -   Mean: {mean_scores:3.2f}")
    click.echo(f"Histogram scores:")
    for score, count in hist_scores.items():
        click.echo(f"  {score:3d}: {count}")

    for threshold_score in THRESHOLD_SCORES:
        click.echo("")
        click.echo(f"Score: {threshold_score}")

        num_total_parallel = 0
        num_total = 0
        for similarity, list_scores in hist_similarities.items():
            num_total_parallel += len(
                [score for score in list_scores if score >= threshold_score]
            )
            num_total += len(list_scores)
        click.echo(
            f"Number of scores above {threshold_score}: {num_total_parallel} / {num_total} ({num_total_parallel / num_total:.3f}%)"
        )

        n_extract = 0
        n_correct = 0
        for threshold_similarity in range(99, 79, -1):
            threshold_similarity = threshold_similarity / 100.0

            for score in hist_similarities[threshold_similarity]:
                n_extract += 1
                if score >= threshold_score:
                    n_correct += 1

            precision = n_correct / n_extract
            recall = n_correct / num_total_parallel
            f1_score = 2 * precision * recall / (precision + recall)
            click.echo(
                f"S: {threshold_similarity:3.2f}  P: {precision:.3f}  R: {recall:.3f}  F1: {f1_score:.3f}"
            )

    click.echo("Done")
