from __future__ import annotations

import logging

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_LANGS = {
    "arabic": ["ara-Arab"],
    "bengali": ["ben-Beng"],
    "english": ["eng-Latn"],
    "finnish": ["fin-Latn"],
    "indonesian": ["ind-Latn"],
    "japanese": ["jpn-Jpan"],
    "korean": ["kor-Kore"],
    "russian": ["rus-Cyrl"],
    "swahili": ["swa-Latn"],
    "telugu": ["tel-Telu"],
    "thai": ["tha-Thai"],
}
_EVAL_SPLIT = "test"

logger = logging.getLogger(__name__)


def _load_data_retrieval(
    path: str, lang: str, splits: str, cache_dir: str = None, revision: str = None
):
    corpus = {split: {} for split in splits}
    queries = {split: {} for split in splits}
    relevant_docs = {split: {} for split in splits}

    split = _EVAL_SPLIT

    qrels_data = datasets.load_dataset(
        path,
        name=f"{lang}-qrels",
        cache_dir=cache_dir,
        revision=revision,
        trust_remote_code=True,
    )[split]

    for row in qrels_data:
        query_id = row["query-id"]
        doc_id = row["corpus-id"]
        score = row["score"]
        if query_id not in relevant_docs[split]:
            relevant_docs[split][query_id] = {}
        relevant_docs[split][query_id][doc_id] = score

    corpus_data = datasets.load_dataset(
        path,
        name=f"{lang}-corpus",
        cache_dir=cache_dir,
        revision=revision,
        trust_remote_code=True,
    )["train"]

    for row in corpus_data:
        doc_id = row["_id"]
        doc_title = row["title"]
        doc_text = row["text"]
        corpus[split][doc_id] = {"title": doc_title, "text": doc_text}

    queries_data = datasets.load_dataset(
        path,
        name=f"{lang}-queries",
        cache_dir=cache_dir,
        revision=revision,
        trust_remote_code=True,
    )[split]

    for row in queries_data:
        query_id = row["_id"]
        query_text = row["text"]
        queries[split][query_id] = query_text

    queries = queries
    logger.info("Loaded %d %s Queries.", len(queries), split.upper())

    return corpus, queries, relevant_docs


class MrTydiRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MrTidyRetrieval",
        description="Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations.",
        reference="https://huggingface.co/datasets/castorini/mr-tydi",
        dataset={
            "path": "mteb/mrtidy",
            "revision": "fc24a3ce8f09746410daee3d5cd823ff7a0675b7",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2023-11-01", "2024-05-15"),
        form=None,
        domains=["Encyclopaedic"],
        task_subtypes=[],
        license="cc-by-sa-3.0",
        socioeconomic_status=None,
        annotations_creators="human-annotated",
        dialect=[],
        text_creation=None,
        sample_creation="found",
        n_samples=None,
        avg_character_length=None,
        bibtex_citation="""@article{mrtydi,
              title={{Mr. TyDi}: A Multi-lingual Benchmark for Dense Retrieval}, 
              author={Xinyu Zhang and Xueguang Ma and Peng Shi and Jimmy Lin},
              year={2021},
              journal={arXiv:2108.08787},
        }""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data_retrieval(
            path=self.metadata_dict["dataset"]["path"],
            lang=self.metadata.eval_langs[0],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True