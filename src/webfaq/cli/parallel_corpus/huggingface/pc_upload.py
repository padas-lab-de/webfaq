import gzip
import os
import time
from glob import glob

import click
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

from webfaq.cli.parallel_corpus.huggingface.readme import readme_template
from webfaq.config import *

RATE_LIMIT_PER_HOUR = 128


@click.command()
@click.argument("repo_id", type=str)
def pc_upload(repo_id: str):
    """
    Pushing the cross-lingual dataset of extracted Q&A pairs to HuggingFace.
    """
    # Login to HuggingFace
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    # Create the repository if it does not exist
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # List all directories in the temporary folder
    filenames = glob(os.path.join(DATASETS_FOLDER, "bitexts", "*.jsonl.gz"))
    filenames = sorted(filenames)
    click.echo(f"Found {len(filenames)} bitext files")

    # Create README information
    format_dataset_info = ""
    format_configs = ""

    # Upload files to the repository
    for file_path in filenames:
        click.echo(f"File: {file_path}")
        filename = file_path.split("/")[-1].split(".")[0]
        # file_path = os.path.join(DATASETS_FOLDER, "bitexts", filename)

        def count_lines(file_path):
            with gzip.open(file_path, "rt", encoding="UTF-8") as f:
                return sum(1 for _ in f)

        num_examples = count_lines(file_path)
        click.echo(f"Number of examples: {num_examples}")

        format_dataset_info += f"  - config_name: {filename}\n"
        format_dataset_info += f"    features:\n"
        format_dataset_info += f"      - name: language1\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: language2\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: origin\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: labse_similarity\n"
        format_dataset_info += f"        dtype: float64\n"
        format_dataset_info += f"      - name: question1\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: question2\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: answer1\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: answer2\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: details\n"
        format_dataset_info += f"        sequence:\n"
        format_dataset_info += f"          - name: urls\n"
        format_dataset_info += f"            dtype: string\n"
        format_dataset_info += f"    splits:\n"
        format_dataset_info += f"      - name: default\n"
        format_dataset_info += f"        num_bytes: {os.path.getsize(file_path)}\n"
        format_dataset_info += f"        num_examples: {num_examples}\n"
        format_configs += f"  - config_name: {filename}\n"
        format_configs += f"    data_files:\n"
        format_configs += f"      - split: default\n"
        format_configs += f"        path: data/{filename}.jsonl.gz\n"

        # Upload JSONL to the repository
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"data/{filename}.jsonl.gz",
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Throttle uploads to respect rate limits
        click.echo(
            f"Waiting for {3600 / RATE_LIMIT_PER_HOUR:.2f} seconds to respect rate limits..."
        )
        time.sleep(3600 / RATE_LIMIT_PER_HOUR)

    # Remove trailing newline
    format_dataset_info = format_dataset_info.strip("\n")
    format_configs = format_configs.strip("\n")

    # Create README.md
    readme_path = os.path.join(TEMP_FOLDER, "bitexts", "README.md")
    os.makedirs(os.path.dirname(readme_path), exist_ok=True)
    readme_text = readme_template.format(
        format_dataset_info,
        format_configs,
    )
    with open(readme_path, "w") as f:
        f.write(readme_text)
    click.echo(f"Created README.md at {readme_path}")

    # Upload README.md to the repository
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    # Print information
    click.echo(f"Total number of language combinations: {len(filenames)}")
