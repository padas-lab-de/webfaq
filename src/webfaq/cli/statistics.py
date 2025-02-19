import click
import os
import json
from tqdm import tqdm
from webfaq.utils import *
from webfaq.config import *


@click.command()
@click.argument("dataset_name", type=str)
def statistics(dataset_name: str):
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
    results_path = os.path.join(DATASETS_FOLDER, dataset_name, "results")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Directory not found: {results_path}")
    if not os.path.isdir(results_path):
        raise NotADirectoryError(f"Path is not a directory: {results_path}")
    
    # Initialize statistics
    set_languages = set()
    set_languages_ge_100_scheme_hosts = set()
    set_languages_l_100_scheme_hosts = set()
    qa_pairs = {}
    qa_pairs_per_topic = {}
    qa_pairs_per_question_type = {}
    qa_pairs_filtered_07 = {}
    qa_pairs_unfiltered_07 = {}
    qa_pairs_filtered_075 = {}
    qa_pairs_unfiltered_075 = {}
    qa_pairs_filtered_08 = {}
    qa_pairs_unfiltered_08 = {}
    scheme_hosts = {}
    question_lengths = {}
    answer_lengths = {}
    scheme_hosts_languages_07 = {}

    # Loop over all languages
    for language in sorted(os.listdir(results_path)):
        language_path = os.path.join(results_path, language)
        if not os.path.isdir(language_path):
            continue

        click.echo(f"Language: {language}")

        # Update statistics
        set_languages.add(language)
        consider_language = language in LANGUAGES_100_SCHEME_HOSTS
        if consider_language:
            set_languages_ge_100_scheme_hosts.add(language)
        else:
            set_languages_l_100_scheme_hosts.add(language)

        # Initialize language statistics
        qa_pairs[language] = 0
        scheme_hosts[language] = set()
        question_lengths[language] = []
        answer_lengths[language] = []
        if consider_language:
            qa_pairs_per_topic[language] = {}
            qa_pairs_per_question_type[language] = {}

        # Load Q&A pairs with labels
        num_lines = count_lines(os.path.join(language_path, "faq.jsonl"))
        with tqdm(total=num_lines, mininterval=10) as pbar:

            with open(os.path.join(language_path, "faq.jsonl"), "r") as faq_file:
                
                if consider_language:
                    labels_file = open(os.path.join(language_path, "labels.jsonl"), "r")
                    flags_file = open(os.path.join(language_path, "flags.jsonl"), "r")
                else:
                    # Create None generators
                    labels_file = (None for _ in range(num_lines))
                    flags_file = (None for _ in range(num_lines))
                
                for faq_line, labels_line, flags_line in zip(faq_file, labels_file, flags_file):

                    # Update progress bar
                    pbar.update(1)

                    try:
                        faq_document = json.loads(faq_line)
                        scheme_host = faq_document["scheme_host"]
                        question = faq_document["question"]
                        answer = faq_document["answer"]
                    except json.JSONDecodeError as e:
                        click.echo(f"Skipping invalid JSON line in {language}/faq.jsonl: {faq_line.strip()} ({e})")
                        continue

                    # Update statistics
                    qa_pairs[language] += 1
                    scheme_hosts[language].add(scheme_host)
                    question_lengths[language].append(len(question))
                    answer_lengths[language].append(len(answer))

                    if consider_language:
                        try:
                            labels_document = json.loads(labels_line)
                            topic = labels_document["topic"]
                            question_type = labels_document["question_type"]
                        except json.JSONDecodeError as e:
                            click.echo(f"Skipping invalid JSON line in {language}/labels.jsonl: {labels_line.strip()} ({e})")
                            continue

                        # Update statistics
                        qa_pairs_per_topic[language][topic] = qa_pairs_per_topic[language].get(topic, 0) + 1
                        qa_pairs_per_question_type[language][question_type] = qa_pairs_per_question_type[language].get(question_type, 0) + 1

                        try:
                            flags_document = json.loads(flags_line)
                            filter = flags_document["filter"]
                            near_duplicate_similarity = flags_document["near_duplicate_similarity"]
                        except json.JSONDecodeError as e:
                            click.echo(f"Skipping invalid JSON line in {language}/flags.jsonl: {labels_line.strip()} ({e})")
                            continue

                        # Update statistics
                        if filter:
                            qa_pairs_filtered_07[language] = qa_pairs_filtered_07.get(language, 0) + 1
                            qa_pairs_filtered_075[language] = qa_pairs_filtered_075.get(language, 0) + 1
                            qa_pairs_filtered_08[language] = qa_pairs_filtered_08.get(language, 0) + 1
                        else:
                            if near_duplicate_similarity >= 0.7:
                                qa_pairs_filtered_07[language] = qa_pairs_filtered_07.get(language, 0) + 1
                            else:
                                qa_pairs_unfiltered_07[language] = qa_pairs_unfiltered_07.get(language, 0) + 1

                            if near_duplicate_similarity >= 0.75:
                                qa_pairs_filtered_075[language] = qa_pairs_filtered_075.get(language, 0) + 1
                            else:
                                qa_pairs_unfiltered_075[language] = qa_pairs_unfiltered_075.get(language, 0) + 1

                            if near_duplicate_similarity >= 0.8:
                                qa_pairs_filtered_08[language] = qa_pairs_filtered_08.get(language, 0) + 1
                            else:
                                qa_pairs_unfiltered_08[language] = qa_pairs_unfiltered_08.get(language, 0) + 1

                            # Update scheme_hosts_languages if not filtered
                            if not scheme_host in scheme_hosts_languages_07:
                                scheme_hosts_languages_07[scheme_host] = {}
                            scheme_hosts_languages_07[scheme_host][language] = scheme_hosts_languages_07[scheme_host].get(language, 0) + 1

        # Close files
        if consider_language:
            labels_file.close()
            flags_file.close()

    # Initialize file paths
    statistics_path = os.path.join(results_path, "statistics.json")
    scheme_hosts_languages_path = os.path.join(results_path, "scheme_hosts_languages.json")

    # Transform statistics
    _statistics = {
        "num_languages": len(set_languages),
        "languages": sorted(list(set_languages)),
        "num_languages_ge_100_scheme_hosts": len(set_languages_ge_100_scheme_hosts),
        "num_languages_l_100_scheme_hosts": len(set_languages_l_100_scheme_hosts),
        "languages_ge_100_scheme_hosts": sorted(list(set_languages_ge_100_scheme_hosts)),
        "languages_l_100_scheme_hosts": sorted(list(set_languages_l_100_scheme_hosts)),
        "qa_pairs_overall": sum(qa_pairs.values()),
        "qa_pairs_filtered/unfiltered_overall (0.7)": (sum(qa_pairs_filtered_07.values()), sum(qa_pairs_unfiltered_07.values())),
        "qa_pairs_filtered/unfiltered_overall (0.75)": (sum(qa_pairs_filtered_075.values()), sum(qa_pairs_unfiltered_075.values())),
        "qa_pairs_filtered/unfiltered_overall (0.8)": (sum(qa_pairs_filtered_08.values()), sum(qa_pairs_unfiltered_08.values())),
        "qa_pairs_per_language": qa_pairs,
        "qa_pairs_filtered/unfiltered_per_language (0.7)": {language: (qa_pairs_filtered_07.get(language, 0), qa_pairs_unfiltered_07.get(language, 0)) for language in sorted(set(qa_pairs_filtered_07).union(qa_pairs_unfiltered_07))},
        "qa_pairs_filtered/unfiltered_per_language (0.75)": {language: (qa_pairs_filtered_075.get(language, 0), qa_pairs_unfiltered_075.get(language, 0)) for language in sorted(set(qa_pairs_filtered_075).union(qa_pairs_unfiltered_075))},
        "qa_pairs_filtered/unfiltered_per_language (0.8)": {language: (qa_pairs_filtered_08.get(language, 0), qa_pairs_unfiltered_08.get(language, 0)) for language in sorted(set(qa_pairs_filtered_08).union(qa_pairs_unfiltered_08))},
        "qa_pairs_per_topic_overall": {topic: sum([qa_pairs_per_topic[language].get(topic, 0) for language in qa_pairs_per_topic]) for topic in set().union(*[qa_pairs_per_topic[language] for language in qa_pairs_per_topic])},
        "qa_pairs_per_topic_and_language": qa_pairs_per_topic,
        "qa_pairs_per_question_type_overall": {question_type: sum([qa_pairs_per_question_type[language].get(question_type, 0) for language in qa_pairs_per_question_type]) for question_type in set().union(*[qa_pairs_per_question_type[language] for language in qa_pairs_per_question_type])},
        "qa_pairs_per_question_type_and_language": qa_pairs_per_question_type,
        "scheme_hosts": {language: len(scheme_hosts[language]) for language in scheme_hosts},
        "question_lengths_overall": {"min": min([min(question_lengths[language]) for language in question_lengths]), "max": max([max(question_lengths[language]) for language in question_lengths]), "avg": sum([sum(question_lengths[language]) for language in question_lengths]) / sum([len(question_lengths[language]) for language in question_lengths])},
        "question_lengths_per_language": {language: {"min": min(question_lengths[language]), "max": max(question_lengths[language]), "avg": sum(question_lengths[language]) / len(question_lengths[language])} for language in question_lengths},
        "answer_lengths_overall": {"min": min([min(answer_lengths[language]) for language in answer_lengths]), "max": max([max(answer_lengths[language]) for language in answer_lengths]), "avg": sum([sum(answer_lengths[language]) for language in answer_lengths]) / sum([len(answer_lengths[language]) for language in answer_lengths])},
    }
    _scheme_hosts_languages_07 = {_scheme_host: {_language: scheme_hosts_languages_07[_scheme_host][_language] for _language in sorted(list(scheme_hosts_languages_07[_scheme_host].keys()))} for _scheme_host in scheme_hosts_languages_07 if len(scheme_hosts_languages_07[_scheme_host]) > 1}

    # Save results to file
    click.echo(f"Writing file {statistics_path}")
    with open(statistics_path, "w") as file:
        json.dump(_statistics, file, indent=4)
    click.echo(f"Writing file {scheme_hosts_languages_path}")
    with open(scheme_hosts_languages_path, "w") as file:
        json.dump(_scheme_hosts_languages_07, file, indent=4)

    click.echo("Done")
