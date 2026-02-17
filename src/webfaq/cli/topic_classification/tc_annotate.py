import gzip
import json
import os
import re
import time

import click
import dotenv
import openai
from tqdm import tqdm

from webfaq.config import *

LIMIT = 2_000
# LIMIT = 10
THRESHOLD = 100


SYSTEM_PROMPT = """Please classify the following Q&A pair into one of the following topics:
1) Products and Commercial Services
2) Traveling and Hospitality
3) Healthcare Services, Wellness and Lifestyle
4) Entertainment, Recreation and Leisure
5) Employment, Education and Training
6) Banking, Financial Services and Insurance
7) Legal Services, Regulations and Government
8) General Information and Other

Please answer only with the number of the topic. Note that the title and description are optional and may not be present.

Q&A pair: ¿Que restaurantes llevan comida domicilio en Los Cabos ### En el App de DiDifood encontraras mas de null restaurantes con entrega a domicilio en Los Cabos ### Title: Comida rápida, Los 10 Mejores Restaurantes a Domicilio en Cabo San Lucas Cerca De Mí | Uber Eats
Label: 2

Q&A pair: चौघड़िया का क्या अर्थ है? ### चौघड़िया शब्द दो शब्दों का मेल है - चो, अर्थात् चार, और घड़िया, यानी घड़ी। हिंदू समय के अनुसार, प्रत्येक घड़ी, 24 मिनट के बराबर है। सूर्योदय से सूर्यास्त तक 30 घड़ी होती हैं जिन्हें 8 से विभाजित किया गया है। इसलिए, दिन में 8 चौघड़िया मुहूर्त और 8 रात्रि चौघड़िया मुहूर्त होते हैं। एक चौघड़िया 4 घड़ी (लगभग 96 मिनट) के बराबर होता है। अतः, एक चौघड़िया लगभग 1.5 घंटे तक रहता है।
Label: 8

Q&A pair: Is DOT coin close to its All Time High price? ### DOT all time high price (ath) is €47.6. Its current price is €6.4. This means that the difference between Polkadot (DOT) All Time High price and DOT current price is -87%.
Label: 6

Q&A pair: Quel est le poids maximum pour un colis? ### Le poids maximum autorisé pour un colis Colissimo ou Chronopost est de 30 kg. ### Title: FAQ Chronopost : réponses à vos questions | Chronopost ### Description: Trouvez rapidement des réponses aux questions fréquentes sur les services Chronopost : livraison, suivi, envoi de colis, réclamations et bien plus.
Label: 1
"""

PROMPT = """Q&A pair: {text}
Label: """


def call_api(client, question, answer, title=None, description=None):
    text = question + " ### " + answer
    if title:
        text += " ### Title: " + title
    if description:
        text += " ### Description: " + description

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": PROMPT.format(text=text),
            },
        ],
    )

    return response.choices[0].message.content


@click.command()
def tc_annotate():
    """
    Annotate Q&A pairs with a topic label.
    """
    dotenv.load_dotenv()

    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], organization=os.environ["OPENAI_ORG_ID"]
    )

    # Initialize results path
    results_path = os.path.join(DATASETS_FOLDER, "faqs")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Directory not found: {results_path}")
    if not os.path.isdir(results_path):
        raise NotADirectoryError(f"Path is not a directory: {results_path}")

    # Loop over all languages
    annotated_qa_pairs = []
    for language in sorted(os.listdir(results_path)):
        language_path = os.path.join(results_path, language)
        if not os.path.isdir(language_path):
            continue

        click.echo(f"Language: {language}")

        # Load Q&A pairs
        qa_pairs = []
        for filename in sorted(os.listdir(language_path)):
            if not filename.startswith("faqs_sorted_") or not filename.endswith(
                ".jsonl.gz"
            ):
                continue

            click.echo(f"Reading file: {filename}")

            with gzip.open(os.path.join(language_path, filename), "rt") as file:

                previous_origin = None
                for line in file:
                    # Load JSON line
                    _dict = json.loads(line)
                    question = _dict["question"]
                    answer = _dict["answer"]
                    title = _dict["title"]
                    description = _dict["description"]

                    # Skip Q&A pairs with the same origin (scheme + host)
                    origin = _dict["origin"]
                    if origin == previous_origin:
                        continue
                    previous_origin = origin

                    # Append Q&A pair
                    qa_pairs.append((question, answer, title, description))

                    # Break if the limit plus buffer is reached
                    if len(qa_pairs) >= LIMIT + 10:
                        break

            # Break if the limit plus buffer is reached
            if len(qa_pairs) >= LIMIT + 10:
                break

        # Skip if the number of Q&A pairs is less than 100
        if len(qa_pairs) < min(THRESHOLD, LIMIT):
            click.echo("Skipping language due to insufficient number of Q&A pairs")
            continue

        retry = True
        num_retries = 0
        while retry and num_retries < 3:
            try:
                # Annotate Q&A pairs
                annotated_qa_pairs_language = []
                with tqdm(
                    total=len(qa_pairs), desc=f"Annotate Q&A pairs for {language}"
                ) as pbar:

                    for question, answer, title, description in qa_pairs:
                        # Annotate using an LLM
                        label = call_api(client, question, answer, title, description)

                        # Parse label
                        label = label.strip()
                        if re.match(r"^[1-8]$", label):
                            label = int(label)
                        elif re.match(r"^Label: [1-8]$", label):
                            label = int(label[-1])
                        elif re.match(r"^label: [1-8]$", label):
                            label = int(label[-1])
                        elif re.match(r"^[1-8]\)$", label):
                            label = int(label[0])
                        elif re.match(r"^[1-8]\)\s+.*", label):
                            label = int(label[0])
                        else:
                            click.echo(f"Invalid label: {label}", err=True)
                            continue

                        # Append annotated Q&A pair
                        annotated_qa_pairs_language.append(
                            {
                                "language": language,
                                "label": label,
                                "question": question,
                                "answer": answer,
                                "title": title,
                                "description": description,
                            }
                        )

                        # Update progress bar
                        pbar.update(1)

                        # Break if the limit is reached
                        if len(annotated_qa_pairs_language) >= LIMIT:
                            break

                retry = False
            except openai.RateLimitError as e:
                click.echo(f"OpenAI API error: {e}", err=True)
                retry = True
                num_retries += 1
                click.echo(f"Retrying [{num_retries}]...", err=True)

                # Wait before retrying
                time.sleep(60 * 60 * 2 ** (num_retries - 1))  # Exponential backoff

        # Append annotated Q&A pairs for the current language
        annotated_qa_pairs.extend(annotated_qa_pairs_language)

        # Save annotated Q&A pairs (after each language)
        annotations_path = os.path.join(RESOURCES_FOLDER, "tc_annotations.jsonl")
        os.makedirs(os.path.dirname(annotations_path), exist_ok=True)
        with open(annotations_path, "w") as file:
            for qa_pair in annotated_qa_pairs:
                file.write(json.dumps(qa_pair) + "\n")

    click.echo("Done")
