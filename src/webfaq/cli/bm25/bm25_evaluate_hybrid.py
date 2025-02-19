import click
import json
import numpy as np
import os
import pandas as pd

from mteb.evaluation.evaluators import RetrievalEvaluator
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from webfaq.utils import get_task_name, get_optimal_scores, get_eval_task, get_ndcg_10


def _merge_scores(hits_bm25: list, roberta_scores: dict):
    """
    Merges the scores of documents retrieved by BM25 and XLM-RoBERTa. If both methods retrieved a document, the scores
    are summed. Otherwise, the score is set to zero.

    :param hits_bm25: The list of documents retrieved by BM25.
    :param roberta_scores: The documents retrieved by XLM-RoBERTa.
    :return: A dictionary containing the merged scores sorted from highest to lowest.
    """
    merged_scores = {}
    for idx in range(len(hits_bm25)):
        doc_id = hits_bm25[idx].docid
        if doc_id in roberta_scores.keys():
            merged_scores[doc_id] = 1.1 * roberta_scores[doc_id] + hits_bm25[idx].score
        else:
            merged_scores[doc_id] = 0
    for doc_id in roberta_scores.keys():
        if doc_id not in merged_scores.keys():
            merged_scores[doc_id] = 0
    return dict(sorted(merged_scores.items(), key=lambda x: x[1], reverse=True))


def _get_retrieval_scores(hits_bm25: list, roberta_scores: dict, rel_docs: pd.Series):
    """
    Merges the scores of documents retrieved by BM25 and XLM-RoBERTa and returns the relevance scores of the top ten
    documents.

    :param hits_bm25: The list of documents retrieved by BM25.
    :param roberta_scores: The documents retrieved by XLM-RoBERTa.
    :param rel_docs: A pandas series containing the relevant documents and their scores.
    :return: A dictionary containing the document ids and relevance scores.
    """
    scores = {}
    seen_docs = []
    merged_scores = _merge_scores(hits_bm25, roberta_scores)

    for doc_id in merged_scores.keys():
        if len(scores) < 10:
            if doc_id not in seen_docs:
                seen_docs.append(doc_id)
                if doc_id in rel_docs["corpus-id"].values:
                    score = rel_docs.loc[rel_docs['corpus-id'] == doc_id]['score'].item()
                    scores[str(doc_id)] = score
                else:
                    scores[str(doc_id)] = 0
        else:
            break

    return scores


def _get_ndcgs(queries: pd.DataFrame, qrels: pd.DataFrame, roberta_scores: dict, searcher: LuceneSearcher):
    """
    Calculates the nDCG per query of a hybrid method combining retrieval results of dense and sparse models.

    :param queries: The queries used to retrieve documents.
    :param qrels: The relevance judgements for each query.
    :param roberta_scores: The documents retrieved by XLM-RoBERTa.
    :param searcher: The LuceneSearcher used to retrieve documents for BM25.
    :return: The list of calculated nDCG scores.
    """
    ndcgs = []
    for idx in tqdm(range(len(queries))):
        query = queries.iloc[idx]["text"]
        query_id = str(queries.iloc[idx]["id"])
        hits_bm25 = searcher.search(query, k=1000)
        rel_docs = qrels.loc[qrels["query-id"] == queries.iloc[idx]["id"]]
        opt_scores = get_optimal_scores(query_id, rel_docs)
        roberta = roberta_scores[query_id]
        scores = _get_retrieval_scores(hits_bm25, roberta, rel_docs)
        ndcgs.append(get_ndcg_10(opt_scores, scores, query_id))
    return ndcgs


@click.command()
@click.argument("index_path", type=str)
@click.argument("data_path", type=str)
@click.argument("save_path", type=str)
@click.argument("task-name", type=click.Choice(["webfaq", "miracl", "tydi", "miraclhn"]))
def bm25_evaluate_hybrid(index_path: str, data_path: str, save_path: str, task_name: str):
    """
    Evaluates retrieval performance of a hybrid setting combining retrieval results of BM25 and XLM-RoBERTa.

    :param index_path: Path to the BM25 index folders.
    :param data_path: Path to the data folder.
    :param save_path: Path to store results.
    :param task_name: The name of the task to evaluate.
    """
    task = get_eval_task(task_name)
    eval_split = "dev" if "miracl" in task_name else "test"
    task_name_mteb = get_task_name(task_name)
    eval_langs = task.metadata.eval_langs
    model = SentenceTransformer("anonymous202501/xlm-roberta-base-msmarco-webfaq", trust_remote_code=True)

    for lang in eval_langs:
        click.echo(f"Evaluating on language {lang}")
        task.data_loaded = False
        eval_lang = lang if task_name == "webfaq" else eval_langs[lang][0]
        language = eval_lang.split("-")[0]
        task.metadata.eval_langs = [lang]
        os.makedirs(f"{save_path}{language}", exist_ok=True)
        queries = pd.read_csv(f"{data_path}{language}/queries.tsv", sep="\t", header=None, names=["id", "text"])
        qrels = pd.read_csv(f"{data_path}{language}/qrels.tsv", sep="\t")
        task.load_data()
        searcher = LuceneSearcher(f"{index_path}{language}")
        retriever = RetrievalEvaluator(model)
        roberta_scores = retriever(task.corpus[eval_split], task.queries[eval_split])
        ndcgs = _get_ndcgs(queries, qrels, roberta_scores, searcher)

        with open(f"{save_path}{language}/{task_name_mteb}_hybrid.json", "w") as file:
            json.dump({"mteb_dataset_name": task_name_mteb, "test": {"ndcg_at_10": np.mean(ndcgs)}}, file)
            click.echo(f'NDCG@10 : {np.mean(ndcgs)}')
