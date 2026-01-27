import click
import json
import os
import gzip
import pandas as pd
import time
from tqdm import tqdm
from webfaq.utils import *
from webfaq.config import *


RATE_LIMIT_PER_HOUR = 128
THRESHOLD_MIN_EXAMPLES = 100


@click.command()
@click.argument("temp_folder", type=str, default=TEMP_FOLDER)
def upload_base(temp_folder: str):
    """
    Pushing the dataset of extracted Q&A pairs to HuggingFace.
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

        # Initialize results list
        results = []

        for filename in sorted(os.listdir(language_path)):
            if not filename.startswith("faqs_sorted_") or not filename.endswith(".jsonl.gz"):
                continue

            click.echo(f"Reading file: {filename}")

            # Check if labels file exists
            labels_path = os.path.join(language_path, filename.replace("faqs_sorted_", "labels_"))
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Labels file not found: {labels_path}")
            
            # Check if semantic similarity scores already exist
            semantic_similarity_path = os.path.join(language_path, filename.replace("faqs_sorted_", "semantic_similarity_"))
            if os.path.exists(semantic_similarity_path):
                click.echo(f"Semantic similarity scores already exist: {semantic_similarity_path}")
                continue

            with gzip.open(os.path.join(language_path, filename), "rt", encoding="UTF-8") as faq_file, \
                 gzip.open(labels_path, "rt", encoding="UTF-8") as labels_file, \
                 gzip.open(semantic_similarity_path, "rt", encoding="UTF-8") as semantic_similarity_file:
                
                for faq_line, labels_line, semantic_similarity_line in tqdm(zip(faq_file, labels_file, semantic_similarity_file)):

                    try:
                        faq_document = json.loads(faq_line)

                        labels_document = json.loads(labels_line)
                        topic = None
                        if "topic" in labels_document:
                            topic = labels_document["topic"]
                        if topic and not faq_document["id"] == labels_document["id"]:
                            click.echo(f"ID mismatch between FAQ and labels: {faq_document['id']} != {labels_document['id']}")
                            continue

                        semantic_similarity_document = json.loads(semantic_similarity_line)
                        if not faq_document["id"] == semantic_similarity_document["id"]:
                            click.echo(f"ID mismatch between FAQ and semantic similarity: {faq_document['id']} != {semantic_similarity_document['id']}")
                            continue
                    except json.JSONDecodeError as e:
                        click.echo(f"Skipping invalid JSON line in {language}/faq.jsonl: {faq_line.strip()} ({e})")
                        continue

                    # Append to list
                    result = {
                        "id": faq_document["id"],
                        "origin": faq_document["origin"],
                        "url": faq_document["url"],
                        "question": faq_document["question"],
                        "answer": faq_document["answer"],
                        "semantic_similarity_score": labels_document["score"],
                    }
                    if topic:
                        result["topic"] = topic

                    # Append to list
                    results.append(result)

        if len(results) < THRESHOLD_MIN_EXAMPLES:
            click.echo(f"Skipping language {language} due to insufficient examples ({len(results)} < {THRESHOLD_MIN_EXAMPLES})")
            continue
        click.echo(f"Total examples for language {language}: {len(results)}")

        # Write results to Parquet file
        output_path = os.path.join(temp_folder, "upload", f"{language}.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_parquet(output_path, engine="pyarrow", index=False)
        click.echo(f"Wrote Parquet file: {output_path}")

    click.echo("Done")
