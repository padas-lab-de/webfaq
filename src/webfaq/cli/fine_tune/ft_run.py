import click
import os
import json
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.readers import InputExample
from webfaq.utils import *
from webfaq.config import *


BATCH_SIZE = 128
LEARNING_RATE = 1e-5


@click.command()
@click.argument("dataset_name", type=str)
def ft_run(dataset_name: str):
    """
    Fine-tune a model on the WebFAQ dataset.
    """
    # Initialize the random number generator
    random.seed(42)

    # Initialize fine tune path
    fine_tune_path = os.path.join(DATASETS_FOLDER, dataset_name, "fine_tune")
    if not os.path.exists(fine_tune_path):
        raise FileNotFoundError(f"Directory not found: {fine_tune_path}")
    if not os.path.isdir(fine_tune_path):
        raise NotADirectoryError(f"Path is not a directory: {fine_tune_path}")
    
    # Initialize batched train and eval dataset
    train_examples_batched = []

    for language in sorted(os.listdir(fine_tune_path)):
        language_path = os.path.join(fine_tune_path, language)
        if not os.path.isdir(language_path):
            continue

        click.echo(f"Language: {language}")

        # Initialize batch
        batch = []

        # Load train dataset
        num_lines = count_lines(os.path.join(language_path, "train.jsonl"))
        with tqdm(total=num_lines, mininterval=10, desc="Train") as pbar:

            with open(os.path.join(language_path, "train.jsonl"), "r") as file:
                
                for line in file:

                    # Update progress bar
                    pbar.update(1)
                    
                    try:
                        document = json.loads(line)
                        query = document["query"]
                        positive = document["positive"]
                    except json.JSONDecodeError as e:
                        click.echo(f"Skipping invalid JSON line in {language}/train.jsonl: {line.strip()} ({e})")
                        continue

                    # Add to batch
                    batch.append(InputExample(texts=[query, positive]))
                    if len(batch) >= BATCH_SIZE:
                        random.shuffle(batch)
                        train_examples_batched.append(batch)
                        batch = []

    # Load pre-trained model
    model = SentenceTransformer("xlm-roberta-base")

    # Initialize the MultipleNegativesRankingLoss
    loss = MultipleNegativesRankingLoss(model=model)

    # Shuffle datasets
    random.shuffle(train_examples_batched)

    # Flatten batches and create datasets
    train_examples = []
    for batch in train_examples_batched:
        for example in batch:
            train_examples.append(example)

    output_dir = os.path.join(MODELS_FOLDER, dataset_name, "xlm-roberta-base-msmarco-ft")
    os.makedirs(output_dir, exist_ok=True)

    # Train the model
    train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=BATCH_SIZE)
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=1,
        warmup_steps=10,
        use_amp=True,
        checkpoint_path=output_dir,
        checkpoint_save_steps=len(train_dataloader) / 4,
        optimizer_params={"lr": LEARNING_RATE},
    )

    click.echo("Done")