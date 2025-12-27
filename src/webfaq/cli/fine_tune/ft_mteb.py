import click
import os
import json
from sentence_transformers import SentenceTransformer
from mteb import MTEB
from mteb.tasks.retrieval.eng import *

from webfaq.config import *


@click.command()
@click.argument("pretrained_model_name", type=str)
def ft_mteb(pretrained_model_name: str):
    """
    Run the MTEB evaluation of the fine-tuned model.
    """
    # Instantiate Embedding Model
    name = pretrained_model_name.split("/")[-1]
    click.echo(f"Loading model: {pretrained_model_name} ({name})")
    # model = JinaEmbeddingsV3Wrapper(pretrained_model_name)
    model = SentenceTransformer(pretrained_model_name)

    # Initialize results dictionary
    results = {}

    # Load results from file if exists
    model_output_folder = os.path.join(TEMP_FOLDER, "evaluation", name)
    results_path = os.path.join(model_output_folder, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as file:
            results = json.load(file)

    # Initialize tasks
    tasks = [
        ArguAna(),
        ClimateFEVER(),
        CQADupstackAndroidRetrieval(),
        CQADupstackEnglishRetrieval(),
        CQADupstackGamingRetrieval(),
        CQADupstackGisRetrieval(),
        CQADupstackMathematicaRetrieval(),
        CQADupstackPhysicsRetrieval(),
        CQADupstackProgrammersRetrieval(),
        CQADupstackStatsRetrieval(),
        CQADupstackTexRetrieval(),
        CQADupstackUnixRetrieval(),
        CQADupstackWebmastersRetrieval(),
        CQADupstackWordpressRetrieval(),
        DBPedia(),
        FEVER(),
        FiQA2018(),
        HotpotQA(),
        MSMARCO(),
        NFCorpus(),
        NQ(),
        QuoraRetrieval(),
        SCIDOCS(),
        SciFact(),
        Touche2020(),
        TRECCOVID(),
    ]

    for task in tasks:
        task_name = str(task).split("(")[0]
        click.echo(f"Task: {task_name}")

        eval_split = "test" if task_name != "MSMARCO" else "dev"

        # Initialize output folder
        task_output_folder = os.path.join(model_output_folder, task_name)
        os.makedirs(task_output_folder, exist_ok=True)

        # Run evaluation
        evaluation = MTEB(tasks=[task])
        evaluation.run(
            model,
            eval_splits=[eval_split],
            output_folder=task_output_folder,
            overwrite_results=True,
            encode_kwargs={"batch_size": 4},
        )

        # Read JSON file
        language_results_path = os.path.join(task_output_folder, f"{task_name}.json")
        with open(language_results_path, "r") as file:
            data = json.load(file)
            ndcg_at_10 = data[eval_split]["ndcg_at_10"]
            click.echo(f"NDCG@10 for task {task_name}: {ndcg_at_10}")
            mrr_at_100 = data[eval_split]["mrr_at_100"]
            click.echo(f"MRR@100 for task {task_name}: {mrr_at_100}")
            recall_at_100 = data[eval_split]["recall_at_100"]
            click.echo(f"Recall@100 for task {task_name}: {recall_at_100}")
            results[task_name] = data[eval_split]

        # Save results to file
        with open(results_path, "w") as file:
            json.dump(results, file, indent=4)

    click.echo("Done")
