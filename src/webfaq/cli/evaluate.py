import click
import json
import os
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from webfaq.cli.embed_jina import JinaEmbeddingsV3Wrapper
from webfaq.utils import *
from webfaq.globals import *
from webfaq.config import *


@click.command()
@click.argument("pretrained_model_name", type=str)
@click.argument("task_name", type=click.Choice(["webfaq", "miracl", "tydi"]))
def evaluate(pretrained_model_name: str, task_name: str):
    """
    Compute the vector embeddings for the extracted Q&A pairs of a specific language and evaluate retrieval performance.

    :param pretrained_model_name: A SentenceTransformer compatible model path.
    :param task: The name of the task to use.
    """
    task = get_eval_task(task_name)
    eval_split = "dev" if "miracl" in task_name else "test"
    eval_langs = task.metadata.eval_langs

    # Instantiate Embedding Model
    click.echo(f"Loading model: {pretrained_model_name}")
    if pretrained_model_name == "jinaai/jina-embeddings-v3":
        model = JinaEmbeddingsV3Wrapper(pretrained_model_name)
    elif pretrained_model_name == "intfloat/multilingual-e5-large-instruct":
        model = E5MultilingualWrapper(pretrained_model_name)
    else:
        model = SentenceTransformer(pretrained_model_name, trust_remote_code=True)

    # Loop through individual languages of the dataset to avoid loading all language datasets at once
    for lang in eval_langs:
        click.echo(f"Evaluating on language {lang}")
        task.data_loaded = False
        eval_lang = lang if task_name == "webfaq" else eval_langs[lang][0]
        language = eval_lang.split("-")[0]
        task.metadata.eval_langs = [lang]

        # Initialize output folder
        model_short = pretrained_model_name.split("/")[0]
        output_folder = os.path.join(
            TEMP_FOLDER, f"evaluation/{language}/{model_short}"
        )
        os.makedirs(output_folder, exist_ok=True)

        # Run evaluation
        evaluation = MTEB(tasks=[task])
        evaluation.run(
            model,
            eval_splits=[eval_split],
            output_folder=output_folder,
            overwrite_results=True,
            encode_kwargs={"batch_size": 16},
        )

        # Read JSON file
        results_path = os.path.join(output_folder, f"{task.metadata_dict['name']}.json")
        with open(results_path, "r") as file:
            data = json.load(file)
            ndcg_at_10 = data[eval_split]["ndcg_at_10"]
            click.echo(f'NDCG@10 for task {task.metadata_dict["name"]}: {ndcg_at_10}')

    click.echo("Done")
