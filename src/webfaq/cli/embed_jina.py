import click
import os
import json
import gzip
import torch
from tqdm import tqdm
from transformers import AutoModel
from webfaq.utils import *
from webfaq.config import *


BATCH_SIZE = 1_000_000


def _construct_document(doc):
    if isinstance(doc, str):
        return doc
    elif 'title' in doc:
        return f'{doc["title"]} {doc["text"].strip()}'
    else:
        return doc['text'].strip()


class JinaEmbeddingsV3Wrapper(torch.nn.Module):
    def __init__(
            self,
            pretrained_model_name='jinaai/jina-embeddings-v3',
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        return self.encoder.encode(sentences, *args, task='retrieval.query', **kwargs)

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        _sentences = [_construct_document(sentence) for sentence in sentences]
        return self.encoder.encode(_sentences, *args, task='retrieval.passage', **kwargs)

    def get_instructions(self):
        return [self.encoder._task_instructions[x] for x in ['retrieval.query', 'retrieval.passage']]

    def forward(self, *args, **kwargs):
        task_id = self.encoder._adaptation_map['retrieval.passage']
        num_examples = kwargs['input_ids'].shape[0]
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
@click.argument("dataset_name", type=str)
def embed_jina(dataset_name: str):
    """
    Compute the vector embeddings for the extracted Q&A pairs.
    """
    # Instantiate Embedding Model
    pretrained_model_name = "jinaai/jina-embeddings-v3"
    click.echo(f"Loading model: {pretrained_model_name}")
    model = JinaEmbeddingsV3Wrapper(pretrained_model_name)

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
        embeddings_path = os.path.join(language_path, "retrieval_embeddings.jsonl.gz")
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
        question_embeddings = []
        answer_embeddings = []
        for i_start in range(0, len(questions), BATCH_SIZE):
            i_end = min(i_start + BATCH_SIZE, len(questions))
            question_embeddings.extend(model.encode_queries(questions[i_start:i_end], truncate_dim=512))
            answer_embeddings.extend(model.encode_corpus([{"text": a} for a in answers[i_start:i_end]], truncate_dim=512))

        # Save embeddings to file
        click.echo(f"Writing file {embeddings_path}")
        str_dump = ""
        for i in range(len(ids)):
            document = {
                "id": ids[i],
                "question_embedding": question_embeddings[i].tolist(),
                "answer_embedding": answer_embeddings[i].tolist(),
            }
            str_dump += json.dumps(document) + "\n"
        with gzip.open(embeddings_path, "wt", encoding="utf-8") as file:
            file.write(str_dump)

    click.echo("Done")
