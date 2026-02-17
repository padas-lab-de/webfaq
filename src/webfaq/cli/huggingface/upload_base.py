import gzip
import json
import os
import time
from glob import glob

import click
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from tqdm import tqdm

from webfaq.cli.huggingface.readme import readme_template
from webfaq.config import *
from webfaq.utils import *

RATE_LIMIT_PER_HOUR = 128


@click.command()
@click.argument("repo_id", type=str)
@click.argument("temp_folder", type=str, default=TEMP_FOLDER)
def upload_base(repo_id: str, temp_folder: str):
    """
    Uploading the prepared dataset of extracted Q&A pairs to HuggingFace.
    """
    # Login to HuggingFace
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    # Create the repository if it does not exist
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # List all directories in the temporary folder
    language_paths = glob(os.path.join(temp_folder, "upload", "*.parquet"))
    language_paths = sorted(language_paths)
    click.echo(f"Found {len(language_paths)} languages")

    # Create README information
    format_language = ""
    format_dataset_info = ""
    format_configs = ""

    # Upload files to the repository
    for language_path in language_paths:
        language = language_path.split("/")[-1].split(".")[0]

        click.echo(f"Language: {language}")

        consider_labels = language in LANGUAGES_100_ORIGINS

        df = pd.read_parquet(language_path, engine="pyarrow")
        num_examples = len(df)

        format_language += f"- {language}\n"
        format_dataset_info += f"  - config_name: {language}\n"
        format_dataset_info += f"    features:\n"
        format_dataset_info += f"      - name: id\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: origin\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: url\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: question\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: answer\n"
        format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"      - name: semantic_similarity_score\n"
        format_dataset_info += f"        dtype: float\n"
        if consider_labels:
            format_dataset_info += f"      - name: topic\n"
            format_dataset_info += f"        dtype: string\n"
            format_dataset_info += f"      - name: question_type\n"
            format_dataset_info += f"        dtype: string\n"
        format_dataset_info += f"    splits:\n"
        format_dataset_info += f"      - name: default\n"
        format_dataset_info += f"        num_bytes: {os.path.getsize(language_path)}\n"
        format_dataset_info += f"        num_examples: {num_examples}\n"
        format_configs += f"  - config_name: {language}\n"
        format_configs += f"    data_files:\n"
        format_configs += f"      - split: default\n"
        format_configs += f"        path: data/{language}.parquet\n"

        # Upload Parquet file to the repository
        api.upload_file(
            path_or_fileobj=language_path,
            path_in_repo=f"data/{language}.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Throttle uploads to respect rate limits
        click.echo(
            f"Waiting for {3600 / RATE_LIMIT_PER_HOUR:.2f} seconds to respect rate limits..."
        )
        time.sleep(3600 / RATE_LIMIT_PER_HOUR)

    # Remove trailing newline
    format_language = format_language.strip("\n")
    format_dataset_info = format_dataset_info.strip("\n")
    format_configs = format_configs.strip("\n")

    # Create README.md
    readme_path = os.path.join(temp_folder, "upload", "README.md")
    readme_text = readme_template.format(
        format_language,
        format_dataset_info,
        format_configs,
    )
    with open(readme_path, "w") as f:
        f.write(readme_text)

    # Upload README.md to the repository
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
