import click
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from webfaq.utils import *
from webfaq.config import *


BATCH_SIZE = 1_000_000


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
@click.argument("dataset_name", type=str)
def embed_labse(dataset_name: str):
    """
    Compute the vector embeddings for the extracted Q&A pairs.
    """
    # Instantiate LaBSE for cross-lingual similarity search with multilingual sentence embeddings
    model = LaBSE_Wrapper()

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

        # Check if embeddings already exist
        embeddings_path = os.path.join(language_path, "embeddings.jsonl")
        if os.path.exists(embeddings_path):
            click.echo(f"Embeddings already exist: {embeddings_path}")
            continue

        # Count number of Q&A pairs
        num_lines = count_lines(os.path.join(language_path, "faq.jsonl"))

        # Load Q&A pairs
        ids = []
        questions = []
        answers = []
        with tqdm(total=num_lines, mininterval=10) as pbar:

            with open(os.path.join(language_path, "faq.jsonl"), "r") as file:

                for line in file:
                    try:
                        document = json.loads(line)
                        ids.append(document["id"])
                        questions.append(document["question"])
                        answers.append(document["answer"])
                    except json.JSONDecodeError as e:
                        click.echo(f"Skipping invalid JSON line: {line.strip()} ({e})")
                        ids.append("")
                        questions.append("")
                        answers.append("")

                    # Update progress bar
                    pbar.update(1)

        # Compute embeddings for questions and answers
        embeddings = []
        for i_start in range(0, len(questions), BATCH_SIZE):
            i_end = min(i_start + BATCH_SIZE, len(questions))
            sentences = [
                f"{q} {a}".strip()
                for q, a in zip(questions[i_start:i_end], answers[i_start:i_end])
            ]
            embeddings.extend(model.encode(sentences))

        # Save embeddings to file
        click.echo(f"Writing file {embeddings_path}")
        with tqdm(total=len(embeddings), mininterval=10) as pbar:

            with open(embeddings_path, "w") as file:

                for i in range(len(embeddings)):
                    document = {
                        "id": ids[i],
                        "embedding": embeddings[i].tolist(),
                    }
                    file.write(json.dumps(document) + "\n")

                    # Update progress bar
                    pbar.update(1)

    click.echo("Done")
