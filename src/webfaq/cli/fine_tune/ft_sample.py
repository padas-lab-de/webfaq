import click
import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from webfaq.utils import *
from webfaq.config import *


LANGUAGES = {
    "ara": 64_000,
    "dan": 64_000,
    "deu": 192_000,
    "eng": 256_000,
    "fas": 128_000,
    "fra": 192_000,
    "hin": 64_000,
    "ind": 64_000,
    "ita": 192_000,
    "jpn": 192_000,
    "kor": 64_000,
    "nld": 192_000,
    "pol": 128_000,
    "por": 128_000,
    "rus": 192_000,
    "spa": 192_000,
    "swe": 64_000,
    "tur": 64_000,
    "vie": 64_000,
    "zho": 64_000,
}


@click.command()
@click.argument("dataset_name", type=str)
@click.argument("repo_id", type=str)
def ft_sample(dataset_name: str, repo_id: str):
    """
    Sample a train dataset for fine-tuning with WebFAQ.
    """
    # Initialize the random number generator
    random.seed(42)

    # Iterate over languages
    for language, limit in LANGUAGES.items():
        click.echo(f"Language: {language}")

        # Load datasets
        list_qrels = []
        for d in tqdm(load_dataset(repo_id, f"{language}-qrels")["train"], desc="Mapping"):
            list_qrels.append((d["query-id"], d["corpus-id"]))

        dict_queries = {}
        for d in tqdm(load_dataset(repo_id, f"{language}-queries")["queries"], desc="Mapping"):
            dict_queries[d["_id"]] = d["text"]

        dict_corpus = {}
        for d in tqdm(load_dataset(repo_id, f"{language}-corpus")["corpus"], desc="Mapping"):
            dict_corpus[d["_id"]] = d["text"]

        # Randomly sample subsets
        i = random.randint(0, len(list_qrels) - 1)
        if i + limit > len(list_qrels):
            list_qrels_train = list_qrels[i:] + list_qrels[:i + limit - len(list_qrels)]
        else:
            list_qrels_train = list_qrels[i:i + limit]

        # Sample train dataset
        train_dataset = []
        with tqdm(total=len(list_qrels_train), mininterval=10, desc="Train") as pbar:

            for query_id, corpus_id in list_qrels_train:
                query = dict_queries[query_id]
                positive = dict_corpus[corpus_id]

                train_dataset.append({
                    "query": query,
                    "positive": positive
                    # Using in-batch negatives
                })

                # Update progress bar
                pbar.update(1)

        # Store train and eval datasets
        train_path = os.path.join(DATASETS_FOLDER, dataset_name, "fine_tune", language, "train.jsonl")
        os.makedirs(os.path.dirname(train_path), exist_ok=True)

        with open(train_path, "w") as file:
            for d in train_dataset:
                file.write(json.dumps(d) + "\n")

    click.echo("Done")
