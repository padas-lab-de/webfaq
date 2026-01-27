import click
import os
import json
import gzip
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel
from webfaq.utils import *
from webfaq.config import *


BATCH_SIZE = 50_000


def _construct_document(doc):
    if isinstance(doc, str):
        return doc.strip()
    elif "title" in doc:
        return f'{doc["title"]} {doc["text"].strip()}'
    else:
        return doc["text"].strip()


class JinaEmbeddingsV3Wrapper(torch.nn.Module):
    def __init__(
        self,
        pretrained_model_name="jinaai/jina-embeddings-v3",
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        return self.encoder.encode(sentences, *args, task="retrieval.query", **kwargs)

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        _sentences = [_construct_document(sentence) for sentence in sentences]
        return self.encoder.encode(
            _sentences, *args, task="retrieval.passage", **kwargs
        )

    def get_instructions(self):
        return [
            self.encoder._task_instructions[x]
            for x in ["retrieval.query", "retrieval.passage"]
        ]

    def forward(self, *args, **kwargs):
        task_id = self.encoder._adaptation_map["retrieval.passage"]
        num_examples = kwargs["input_ids"].shape[0]
        adapter_mask = torch.full(
            (num_examples,), task_id, dtype=torch.int32, device=self.encoder.device
        )
        return self.encoder.forward(*args, adapter_mask=adapter_mask, **kwargs)

    @property
    def device(self):
        return self.encoder.device

    @staticmethod
    def has_instructions():
        return True


@click.command()
def embed_jina():
    """
    Compute the vector embeddings for the extracted Q&A pairs.
    """
    # Instantiate Embedding Model
    pretrained_model_name = "jinaai/jina-embeddings-v3"
    click.echo(f"Loading model: {pretrained_model_name}")
    model = JinaEmbeddingsV3Wrapper(pretrained_model_name)

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

        click.echo(f"Language: {language}")

        for filename in sorted(os.listdir(language_path)):
            if not filename.startswith("faqs_sorted_") or not filename.endswith(".jsonl.gz"):
                continue

            click.echo(f"Reading file: {filename}")

            # Check if embeddings already exist
            embeddings_path = os.path.join(language_path, filename.replace("faqs_sorted_", "jina_embeddings_"))
            if os.path.exists(embeddings_path):
                click.echo(f"Embeddings already exist: {embeddings_path}")
                continue

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
            question_embeddings = []
            answer_embeddings = []
            for i_start in tqdm(range(0, len(questions), BATCH_SIZE)):
                i_end = min(i_start + BATCH_SIZE, len(questions))
                question_embeddings.extend(
                    model.encode_queries(questions[i_start:i_end], truncate_dim=512).astype(np.float16)
                )
                answer_embeddings.extend(
                    model.encode_corpus(answers[i_start:i_end], truncate_dim=512).astype(np.float16)
                )

            # Save embeddings to file
            click.echo(f"Writing file {embeddings_path}")
            with gzip.open(embeddings_path, "wt", encoding="UTF-8") as file:
                for i in tqdm(range(len(ids))):
                    document = {
                        "id": ids[i],
                        "question_embedding": question_embeddings[i].tolist(),
                        "answer_embedding": answer_embeddings[i].tolist(),
                    }
                    file.write(json.dumps(document) + "\n")

    click.echo("Done")
