import iso639
import pandas as pd
from iso639 import LanguageNotFoundError
from mteb.tasks.retrieval.multilingual import (
    MIRACLRetrieval,
    MIRACLRetrievalHardNegatives,
    MrTidyRetrieval,
    WebFAQRetrieval,
)
from pytrec_eval import RelevanceEvaluator


def count_lines(filepath):
    with open(filepath, "r") as f:
        return sum(1 for _ in f)


def map_iso_code(iso_alpha_2: str):
    """
    Maps the given ISO alpha-2 country code to its corresponding alpha-3 version.

    :param iso_alpha_2: An ISO alpha-2 country code.
    :return: The ISO alpha-3 country code.
    """
    if len(iso_alpha_2) == 3:
        return iso_alpha_2
    try:
        language = iso639.Language.from_part1(iso_alpha_2)
        return language.part3
    except LanguageNotFoundError as e:
        # click.echo(f"LanguageNotFoundError: {e}", err=True)
        return None


def get_ndcg_10(opt_scores: dict, scores: dict, query_id: str):
    """
    Calculates the NDCG@10 for the given optimal scores and scores of the retrieved documents.

    :param opt_scores: The optimal retrieval results for the current query.
    :param scores: The actual retrieval results for the current query.
    :param query_id: The query id.
    :return The calculated NDCG value.
    """
    evaluator = RelevanceEvaluator(opt_scores, {"ndcg_cut.10"})
    scores = {query_id: scores}
    res = evaluator.evaluate(scores)[query_id]
    return res["ndcg_cut_10"]


def get_optimal_scores(query_id: str, rel_docs: pd.Series):
    """
    Creates a dictionary containing the retrieval scores for the given query for each relevant document.

    :param query_id: The query id.
    :param rel_docs: A pandas series containing the relevant documents and their scores.
    :return: A dictionary containing the ids of relevant documents together with their score.
    """
    qrels = {query_id: {}}
    for idx in range(len(rel_docs)):
        row = rel_docs.iloc[idx]
        qrels[query_id][str(row["corpus-id"])] = row["score"].item()
    return qrels


def get_retrieval_scores(rel_docs: pd.Series, hits: list):
    """
    Returns the relevance scores for the top 10 retrieved documents.

    :param rel_docs: A pandas series containing the relevant documents and their scores.
    :param hits: A list of retrieved documents.
    :return: A dictionary containing the document ids and relevance scores.
    """
    scores = {}
    seen_docs = []

    for hit in hits:
        if len(scores) < 10 and hit.docid not in seen_docs:
            seen_docs.append(hit.docid)
            if hit.docid in rel_docs["corpus-id"].values:
                score = rel_docs.loc[rel_docs["corpus-id"] == hit.docid]["score"].item()
                scores[str(hit.docid)] = score
            else:
                scores[str(hit.docid)] = 0
    return scores


def get_task_name(task_name: str):
    """
    Returns the task name under which the results are saved based on the passed string.

    :param task_name: The task name representation.
    :return: The task name.
    """
    if task_name == "webfaq":
        return "WebFAQ"
    elif task_name == "miracl":
        return "MIRACLRetrieval"
    elif task_name == "miraclhn":
        return "MIRACLRetrievalHardNegatives"
    elif task_name == "tydi":
        return "MrTidyRetrieval"
    else:
        raise NotImplementedError(f"Task {task_name} is not supported!")


def get_eval_task(task_name: str):
    """
    Returns the correct evaluation task for the given task name.

    :param task_name: The name of the evaluation task.
    :return: The evaluation task.
    """
    if task_name == "webfaq":
        return WebFAQ()
    elif task_name == "miracl":
        return MIRACLRetrieval()
    elif task_name == "miraclhn":
        return MIRACLRetrievalHardNegatives()
    elif task_name == "tydi":
        return MrTydiRetrieval()
    else:
        raise NotImplementedError(f"Task {task_name} is not supported!")
