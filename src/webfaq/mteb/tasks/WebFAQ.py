import datasets
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


_LANGUAGES = [
    "ara-Arab",
    "dan-Latn",
    "deu-Latn",
    "eng-Latn",
    "fas-Arab",
    "fra-Latn",
    "hin-Deva",
    "ind-Latn",
    "ita-Latn",
    "jpn-Hani",
    "kor-Hani",
    "nld-Latn",
    "pol-Latn",
    "por-Latn",
    "rus-Cyrl",
    "spa-Latn",
    "swe-Latn",
    "tur-Latn",
    "vie-Latn",
    "zho-Hani",
]


class WebFAQ(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WebFAQ",
        dataset={
            "path": "anonymous202501/webfaq-retrieval",
            "revision": "b4320165e039fec2ed05f4b1ba74e4e9376da070",
        },
        description="WebFAQ uses FAQ pages scraped from microdata and json-ld content of a diverse set of webpages.",
        reference="https://www.fim.uni-passau.de",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        """
        Loads the different split of the dataset (queries/corpus/relevants)
        """
        if self.data_loaded:
            return

        self.language = self.metadata.eval_langs[0].split("-")[0]
        self.relevant_docs = {}
        self.queries = {}
        self.corpus = {}

        data_qrels = datasets.load_dataset(
            self.metadata_dict["dataset"]["path"],
            f"{self.language}-qrels",
            split=self.metadata.eval_splits[0],
            revision=self.metadata_dict["dataset"].get("revision", None),
        )
        data_queries = datasets.load_dataset(
            self.metadata_dict["dataset"]["path"],
            f"{self.language}-queries",
            split="queries",
            revision=self.metadata_dict["dataset"].get("revision", None),
        )
        data_corpus = datasets.load_dataset(
            self.metadata_dict["dataset"]["path"],
            f"{self.language}-corpus",
            split="corpus",
            revision=self.metadata_dict["dataset"].get("revision", None),
        )

        self.relevant_docs = {
            self.metadata.eval_splits[0]: {
                d["query-id"]: {d["corpus-id"]: int(d["score"])} for d in data_qrels
            }
        }
        set_query_ids = set(self.relevant_docs[self.metadata.eval_splits[0]].keys())
        self.queries = {
            self.metadata.eval_splits[0]: {
                d["_id"]: d["text"] for d in data_queries if d["_id"] in set_query_ids
            }
        }
        self.corpus = {
            self.metadata.eval_splits[0]: {
                d["_id"]: {"title": d["title"], "text": d["text"]} for d in data_corpus
            }
        }

        self.data_loaded = True
