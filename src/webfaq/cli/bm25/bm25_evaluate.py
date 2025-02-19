import click
import json
import numpy as np
import os
import pandas as pd

from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from webfaq.utils import get_task_name, get_ndcg_10, get_optimal_scores, get_retrieval_scores


@click.command()
@click.argument("index_path", type=str)
@click.argument("data_path", type=str)
@click.argument("save_path", type=str)
@click.argument("task-name", type=click.Choice(["webfaq", "miracl", "tydi", "miraclhn"]))
@click.argument("prebuilt-index", type=bool, default=False)
def bm25_evaluate(index_path: str, data_path: str, save_path: str, task_name: str, prebuilt_index: bool):
    """
    Computes the BM25 score on the given index using the given queries.

    :param index_path: Path to index files.
    :param data_path: Path to the folder where queries and qrels are stored.
    :param save_path: Path to store the performance.
    :param task_name: The name of the task to evaluate.
    :param prebuilt_index: Whether the given index path is a Lucene prebuilt index or not.
    """

    os.makedirs(save_path, exist_ok=True)
    queries = pd.read_csv(f"{data_path}queries.tsv", sep="\t", header=None, names=["id", "text"])
    qrels = pd.read_csv(f"{data_path}qrels.tsv", sep="\t")
    task_name = get_task_name(task_name)
    if prebuilt_index:
        searcher = LuceneSearcher.from_prebuilt_index(index_path)
    else:
        searcher = LuceneSearcher(index_path)
    ndcgs = []

    for idx in tqdm(range(len(queries))):
        query = queries.iloc[idx]["text"]
        hits = searcher.search(query)
        rel_docs = qrels.loc[qrels["query-id"] == queries.iloc[idx]["id"]]
        scores = get_retrieval_scores(rel_docs, hits)
        opt_scores = get_optimal_scores(str(queries.iloc[idx]["id"]), rel_docs)
        ndcgs.append(get_ndcg_10(opt_scores, scores, str(queries.iloc[idx]["id"])))

    with open(f"{save_path}{task_name}_bm25.json", "w") as file:
        json.dump({"mteb_dataset_name": task_name, "test": {"ndcg_at_10": np.mean(ndcgs)}}, file)
        click.echo(f'NDCG@10 : {np.mean(ndcgs)}')
