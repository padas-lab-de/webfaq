import click
import os
import json
from sentence_transformers import SentenceTransformer
from mteb import MTEB
from mteb.tasks.Retrieval.eng.ArguAnaRetrieval import ArguAna
from mteb.tasks.Retrieval.eng.ClimateFEVERRetrieval import ClimateFEVER
from mteb.tasks.Retrieval.eng.CQADupstackAndroidRetrieval import CQADupstackAndroidRetrieval
from mteb.tasks.Retrieval.eng.CQADupstackEnglishRetrieval import CQADupstackEnglishRetrieval
from mteb.tasks.Retrieval.eng.CQADupstackGamingRetrieval import CQADupstackGamingRetrieval
from mteb.tasks.Retrieval.eng.CQADupstackGisRetrieval import CQADupstackGisRetrieval
from mteb.tasks.Retrieval.eng.CQADupstackMathematicaRetrieval import CQADupstackMathematicaRetrieval
from mteb.tasks.Retrieval.eng.CQADupstackPhysicsRetrieval import CQADupstackPhysicsRetrieval
from mteb.tasks.Retrieval.eng.CQADupstackProgrammersRetrieval import CQADupstackProgrammersRetrieval
from mteb.tasks.Retrieval.eng.CQADupstackStatsRetrieval import CQADupstackStatsRetrieval
from mteb.tasks.Retrieval.eng.CQADupstackTexRetrieval import CQADupstackTexRetrieval
from mteb.tasks.Retrieval.eng.CQADupstackUnixRetrieval import CQADupstackUnixRetrieval
from mteb.tasks.Retrieval.eng.CQADupstackWebmastersRetrieval import CQADupstackWebmastersRetrieval
from mteb.tasks.Retrieval.eng.CQADupstackWordpressRetrieval import CQADupstackWordpressRetrieval
from mteb.tasks.Retrieval.eng.DBPediaRetrieval import DBPedia
from mteb.tasks.Retrieval.eng.FEVERRetrieval import FEVER
from mteb.tasks.Retrieval.eng.FiQA2018Retrieval import FiQA2018
from mteb.tasks.Retrieval.eng.HotpotQARetrieval import HotpotQA
from mteb.tasks.Retrieval.eng.MSMARCORetrieval import MSMARCO
from mteb.tasks.Retrieval.eng.NFCorpusRetrieval import NFCorpus
from mteb.tasks.Retrieval.eng.NQRetrieval import NQ
from mteb.tasks.Retrieval.eng.QuoraRetrieval import QuoraRetrieval
from mteb.tasks.Retrieval.eng.SCIDOCSRetrieval import SCIDOCS
from mteb.tasks.Retrieval.eng.SciFactRetrieval import SciFact
from mteb.tasks.Retrieval.eng.Touche2020Retrieval import Touche2020
from mteb.tasks.Retrieval.eng.TRECCOVIDRetrieval import TRECCOVID
# from webfaq.mteb.tasks.WebFAQ import WebFAQ
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
        with open(results_path, 'r') as file:
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
        TRECCOVID()
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
            encode_kwargs={"batch_size": 4}
        )

        # Read JSON file
        language_results_path = os.path.join(task_output_folder, f"{task_name}.json")
        with open(language_results_path, 'r') as file:
            data = json.load(file)
            ndcg_at_10 = data[eval_split]['ndcg_at_10']
            click.echo(f'NDCG@10 for task {task_name}: {ndcg_at_10}')
            mrr_at_100 = data[eval_split]['mrr_at_100']
            click.echo(f'MRR@100 for task {task_name}: {mrr_at_100}')
            recall_at_100 = data[eval_split]['recall_at_100']
            click.echo(f'Recall@100 for task {task_name}: {recall_at_100}')
            results[task_name] = data[eval_split]

        # Save results to file
        with open(results_path, 'w') as file:
            json.dump(results, file, indent=4)

    click.echo("Done")
