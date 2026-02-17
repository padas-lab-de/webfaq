import gzip
import json
import os
from collections import defaultdict

import click
import numpy as np
from tqdm import tqdm

from webfaq.config import *
from webfaq.utils import *


@click.command()
def statistics():
    """
    Compute statistics for the extracted Q&A dataset:
    - Number of languages with less than 100 hosts
    - Number of Q&A pairs (per language and overall)
    - Number of Q&A pairs per topic (per language and overall)
    - Number of Q&A pairs per question type (per language and overall)
    - Number of hosts (per language and overall)
    - Average question and answer length (per language and overall)

    Additionally: Map of combinations of scheme and host with the corresponding set of languages.
    """
    # Initialize results path
    results_path = os.path.join(DATASETS_FOLDER, "faqs")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Directory not found: {results_path}")
    if not os.path.isdir(results_path):
        raise NotADirectoryError(f"Path is not a directory: {results_path}")

    # Initialize statistics
    set_languages = set()
    qa_pairs = {}
    origins = {}
    origins_languages = defaultdict(dict)
    question_lengths = {}
    answer_lengths = {}
    semantic_similarity_scores = {}
    qa_pairs_per_topic = {}
    qa_pairs_per_question_type = {}

    # Loop over all languages
    language_paths = []
    for language in sorted(os.listdir(results_path)):
        language_path = os.path.join(results_path, language)
        if not os.path.isdir(language_path):
            continue
        language_paths.append((language, language_path))

    for language, language_path in sorted(language_paths):

        click.echo(f"Language: {language}")

        # Update statistics
        set_languages.add(language)

        # Initialize language statistics
        qa_pairs[language] = 0
        origins[language] = set()
        question_lengths[language] = []
        answer_lengths[language] = []
        semantic_similarity_scores[language] = []
        qa_pairs_per_topic[language] = {}
        qa_pairs_per_question_type[language] = {}
        for filename in sorted(os.listdir(language_path)):
            if not filename.startswith("faqs_sorted_") or not filename.endswith(
                ".jsonl.gz"
            ):
                continue

            click.echo(f"Reading file: {filename}")

            semantic_similarity_path = os.path.join(
                language_path, filename.replace("faqs_sorted_", "semantic_similarity_")
            )
            labels_path = os.path.join(
                language_path, filename.replace("faqs_sorted_", "labels_")
            )
            nfqa_labels_path = os.path.join(
                language_path, filename.replace("faqs_sorted_", "nfqa_")
            )
            with gzip.open(
                os.path.join(language_path, filename), "rt", encoding="UTF-8"
            ) as faq_file, gzip.open(
                semantic_similarity_path, "rt", encoding="UTF-8"
            ) as semantic_similarity_file:

                if language in LANGUAGES_100_ORIGINS:
                    labels_file = gzip.open(labels_path, "rt", encoding="UTF-8")
                    nfqa_labels_file = gzip.open(
                        nfqa_labels_path, "rt", encoding="UTF-8"
                    )
                else:
                    # Create None generators
                    labels_file = (None for _ in range(1_000_000))
                    nfqa_labels_file = (None for _ in range(1_000_000))

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
                        id_faq = faq_document["id"]
                        origin = faq_document["origin"]
                        question = faq_document["question"]
                        answer = faq_document["answer"]
                    except json.JSONDecodeError as e:
                        click.echo(
                            f"Skipping invalid JSON line in {language}/{filename}: {faq_line.strip()} ({e})"
                        )
                        continue

                    try:
                        semantic_similarity_document = json.loads(
                            semantic_similarity_line
                        )
                        id_semantic_similarity = semantic_similarity_document["id"]
                        assert (
                            id_faq == id_semantic_similarity
                        ), f"ID mismatch: {id_faq} != {id_semantic_similarity}"
                        semantic_similarity_score = semantic_similarity_document[
                            "score"
                        ]
                    except json.JSONDecodeError as e:
                        click.echo(
                            f"Skipping invalid JSON line in {semantic_similarity_path}: {semantic_similarity_line.strip()} ({e})"
                        )
                        continue

                    # Update statistics
                    qa_pairs[language] += 1
                    origins[language].add(origin)
                    origins_languages[origin][language] = (
                        origins_languages[origin].get(language, 0) + 1
                    )
                    semantic_similarity_scores[language].append(
                        float(semantic_similarity_score)
                    )
                    question_lengths[language].append(len(question))
                    answer_lengths[language].append(len(answer))

                    if labels_line is not None:
                        try:
                            labels_document = json.loads(labels_line)
                            id_labels = labels_document["id"]
                            assert (
                                id_faq == id_labels
                            ), f"ID mismatch: {id_faq} != {id_labels}"
                            topic = labels_document["label"]
                        except json.JSONDecodeError as e:
                            click.echo(
                                f"Skipping invalid JSON line in {labels_path}: {labels_line.strip()} ({e})"
                            )
                            continue

                        # Update statistics
                        qa_pairs_per_topic[language][topic] = (
                            qa_pairs_per_topic[language].get(topic, 0) + 1
                        )

                    if nfqa_labels_line is not None:
                        try:
                            nfqa_labels_document = json.loads(nfqa_labels_line)
                            id_nfqa_labels = nfqa_labels_document["id"]
                            assert (
                                id_faq == id_nfqa_labels
                            ), f"ID mismatch: {id_faq} != {id_nfqa_labels}"
                            question_type = nfqa_labels_document["label"]
                        except json.JSONDecodeError as e:
                            click.echo(
                                f"Skipping invalid JSON line in {nfqa_labels_path}: {nfqa_labels_line.strip()} ({e})"
                            )
                            continue

                        # Update statistics
                        qa_pairs_per_question_type[language][question_type] = (
                            qa_pairs_per_question_type[language].get(question_type, 0)
                            + 1
                        )

            # Close files
            if language in LANGUAGES_100_ORIGINS:
                labels_file.close()
                nfqa_labels_file.close()

        # Keep only distribution of semantic similarity scores
        _scores = np.array(semantic_similarity_scores[language])
        if len(_scores) == 0:
            semantic_similarity_scores[language] = (0.0, 0.0, 0.0, 0.0)
        else:
            semantic_similarity_scores[language] = (
                _scores.min(),
                _scores.max(),
                _scores.mean(),
                _scores.std(),
            )

    # Output results
    _origins_languages = {
        _origin: {
            _language: origins_languages[_origin][_language]
            for _language in sorted(list(origins_languages[_origin].keys()))
        }
        for _origin in origins_languages
        if len(origins_languages[_origin]) > 1
    }

    # Save results to file
    origins_languages_path = os.path.join(DATASETS_FOLDER, "origins_languages.json")
    os.makedirs(os.path.dirname(origins_languages_path), exist_ok=True)
    click.echo(f"Writing file {origins_languages_path}")
    with open(origins_languages_path, "w") as file:
        json.dump(_origins_languages, file, indent=4)

    # Transform statistics
    _statistics = {
        "num_languages": len(qa_pairs),
        "languages": sorted(list(set_languages)),
        "qa_pairs_overall": sum(qa_pairs.values()),
        "qa_pairs_per_language": qa_pairs,
        "semantic_similarity_scores_per_language": semantic_similarity_scores,
        "qa_pairs_per_topic_overall": {
            topic: sum(
                [
                    qa_pairs_per_topic[language].get(topic, 0)
                    for language in qa_pairs_per_topic
                ]
            )
            for topic in set().union(
                *[qa_pairs_per_topic[language] for language in qa_pairs_per_topic]
            )
        },
        "qa_pairs_per_topic_and_language": qa_pairs_per_topic,
        "qa_pairs_per_question_type_overall": {
            question_type: sum(
                [
                    qa_pairs_per_question_type[language].get(question_type, 0)
                    for language in qa_pairs_per_question_type
                ]
            )
            for question_type in set().union(
                *[
                    qa_pairs_per_question_type[language]
                    for language in qa_pairs_per_question_type
                ]
            )
        },
        "qa_pairs_per_question_type_and_language": qa_pairs_per_question_type,
        "origins": {language: len(origins[language]) for language in origins},
    }

    # Save results to file
    statistics_path = os.path.join(DATASETS_FOLDER, "statistics.json")
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)
    click.echo(f"Writing file {statistics_path}")
    with open(statistics_path, "w") as file:
        json.dump(_statistics, file, indent=4)

    click.echo("Done")
