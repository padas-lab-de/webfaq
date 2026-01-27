import click
import os
import json
import gzip
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from webfaq.utils import *
from webfaq.config import *


@click.command()
def semantic_similarity():
    """
    Compute the semantic similarity scores between questions and answers.
    """
    # Initialize embeddings path
    embeddings_path = os.path.join("/root/bulk/webfaq", "faqs")
    # embeddings_path = os.path.join(DATASETS_FOLDER, "faqs")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Directory not found: {embeddings_path}")
    if not os.path.isdir(embeddings_path):
        raise NotADirectoryError(f"Path is not a directory: {embeddings_path}")

    # Loop over all languages
    for language in sorted(os.listdir(embeddings_path)):
        language_path = os.path.join(embeddings_path, language)
        if not os.path.isdir(language_path):
            continue

        # if language < "spa":
        #     continue

        click.echo(f"Language: {language}")

        for filename in sorted(os.listdir(language_path)):
            if not filename.startswith("jina_embeddings_") or not filename.endswith(".jsonl.gz"):
                continue

            click.echo(f"Reading file: {filename}")

            # Check if semantic similarity scores already exist
            semantic_similarity_path = os.path.join(language_path, filename.replace("jina_embeddings_", "semantic_similarity_"))
            if os.path.exists(semantic_similarity_path):
                click.echo(f"Semantic similarity scores already exist: {semantic_similarity_path}")
                continue

            # Load Q&A pairs
            ids = []
            cos_sim_scores = []

            with gzip.open(os.path.join(language_path, filename), "rt", encoding="UTF-8") as file:

                for line in tqdm(file):
                    try:
                        document = json.loads(line)
                        ids.append(document["id"])
                        question_embedding = document["question_embedding"]
                        answer_embedding = document["answer_embedding"]

                        cos_sim = cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(answer_embedding).reshape(1, -1))[0][0]
                        cos_sim = float(np.float16(cos_sim))
                        cos_sim_scores.append(cos_sim)
                    except json.JSONDecodeError as e:
                        click.echo(f"Skipping invalid JSON line: {line.strip()} ({e})", err=True)
                        ids.append("")
                        cos_sim_scores.append(0.0)

            # Save semantic similarity scores to file
            click.echo(f"Writing file {semantic_similarity_path}")
            with gzip.open(semantic_similarity_path, "wt", encoding="UTF-8") as file:
                for id, score in tqdm(zip(ids, cos_sim_scores), total=len(ids)):
                    document = {
                        "id": id,
                        "score": score,
                    }
                    file.write(json.dumps(document) + "\n")

    click.echo("Done")
