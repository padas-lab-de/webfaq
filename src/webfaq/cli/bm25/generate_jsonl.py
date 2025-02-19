import click
import os
import pandas as pd

from webfaq.mteb.tasks import WebFAQ, MrTydiRetrieval, MIRACLRetrieval, MIRACLRetrievalHardNegatives


def _get_eval_task(task_name: str):
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


@click.command()
@click.argument("eval-lang", type=str)
@click.argument("save-path", type=str)
@click.argument("task-name", type=click.Choice(["webfaq", "miracl", "tydi", "miraclhn"]))
def bm25_generate_jsonl(eval_lang: str, save_path: str, task_name: str):
    """
    Transforms the HuggingFace dataset into a JSONL file for BM25 index creation.

    :param eval_lang: The language for which the dataset should be built.
    :param save_path: The path to store the dataset files.
    :param task_name: The name of the task used to load the dataset.
    """

    task = _get_eval_task(task_name)
    task.metadata.eval_langs = [eval_lang]
    task.load_data()
    queries, docs, qrels = [], [], []
    eval_split = "dev" if "miracl" in task_name else "test"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}corpus/", exist_ok=True)

    for query_id in task.queries[eval_split]:
        queries.append([query_id, task.queries[eval_split][query_id]])

    df = pd.DataFrame(queries, columns=["id", "document"])
    df.to_csv(f"{save_path}queries.tsv", sep="\t", index=False, header=False)

    for doc_id in task.corpus[eval_split]:
        docs.append([doc_id, task.corpus[eval_split][doc_id]["text"]])

    df = pd.DataFrame(docs, columns=["id", "contents"])
    df.to_json(f"{save_path}corpus/corpus.jsonl", orient="records", lines=True)

    for query_id in task.relevant_docs[eval_split]:
        for doc_id in task.relevant_docs[eval_split][query_id]:
            qrels.append([query_id, doc_id, task.relevant_docs[eval_split][query_id][doc_id]])

    df = pd.DataFrame(qrels, columns=["query-id", "corpus-id", "score"])
    df.to_csv(f"{save_path}qrels.tsv", sep="\t", index=False)
