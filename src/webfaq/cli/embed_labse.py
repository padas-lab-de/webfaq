import click
import os
import json
import gzip
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from webfaq.utils import *
from webfaq.config import *


BATCH_SIZE = 50_000


class LaBSE_Wrapper:
    def __init__(self):
        self.model = SentenceTransformer("LaBSE")

    def encode(self, sentences):
        return self.model.encode(
            sentences, show_progress_bar=True, convert_to_numpy=True
        )

    def get_sentence_embedding_dimension(self):
        return self.model.get_sentence_embedding_dimension()


@click.command()
def embed_labse():
    """
    Compute the vector embeddings for the extracted Q&A pairs.
    """
    # Instantiate LaBSE for cross-lingual similarity search with multilingual sentence embeddings
    model = LaBSE_Wrapper()

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

        # if language != "jpn":  # < "jpn"
        #     click.echo(f"Skipping language: {language}")
        #     continue

        click.echo(f"Language: {language}")

        for filename in sorted(os.listdir(language_path)):
            if not filename.startswith("faqs_sorted_") or not filename.endswith(".jsonl.gz"):
                continue

            click.echo(f"Reading file: {filename}")

            # Check if embeddings already exist
            embeddings_path = os.path.join(language_path, filename.replace("faqs_sorted_", "labse_embeddings_"))
            if os.path.exists(embeddings_path):
                click.echo(f"Embeddings already exist: {embeddings_path}")
                # continue

                ids = []
                embeddings = []
                with gzip.open(embeddings_path, "rt", encoding="UTF-8") as file:
                    for line in tqdm(file):
                        data = json.loads(line)
                        ids.append(data["id"])
                        embedding = np.array(data["embedding"], dtype=np.float16)
                        embeddings.append(embedding)

            else:

                # Load Q&A pairs
                ids = []
                questions = []
                answers = []

                with gzip.open(os.path.join(language_path, filename), "rt", encoding="UTF-8") as file:

                    for line in tqdm(file):
                        try:
                            document = json.loads(line)
                            ids.append(document["id"])
                            questions.append(document["question"])
                            answers.append(document["answer"])
                        except json.JSONDecodeError as e:
                            click.echo(f"Skipping invalid JSON line: {line.strip()} ({e})", err=True)
                            ids.append("")
                            questions.append("")
                            answers.append("")

                # Compute embeddings for questions and answers
                embeddings = []
                for i_start in range(0, len(questions), BATCH_SIZE):
                    i_end = min(i_start + BATCH_SIZE, len(questions))
                    sentences = [
                        f"{q} {a}".strip()
                        for q, a in zip(questions[i_start:i_end], answers[i_start:i_end])
                    ]
                    for embedding in model.encode(sentences):
                        embedding = embedding.astype(np.float16)
                        embeddings.append(embedding)

            # Save embeddings to file
            click.echo(f"Writing file {embeddings_path}")
            with gzip.open(embeddings_path, "wt", encoding="UTF-8") as file:
                for i in range(len(ids)):
                    document = {
                        "id": ids[i],
                        "embedding": embeddings[i].tolist(),
                    }
                    file.write(json.dumps(document) + "\n")

    click.echo("Done")
