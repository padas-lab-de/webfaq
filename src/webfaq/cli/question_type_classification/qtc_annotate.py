import click
import os
import json
import openai
import dotenv
import re
from tqdm import tqdm
from webfaq.config import *


LIMIT = 1_000
# LIMIT = 10
THRESHOLD = 100


PROMPT = """Please classify the following questions based on their question word type:
1) What - Asking for information about something
2) When - Asking about time
3) Where - Asking in or at what place or position
4) Which - Asking about choice
5) Who/Whom/Whose - Asking what or which person
6) Why - Asking for reason
7) How - Asking about manner, extent or degree
8) Is, are, do, does - Questions expecting a yes or no answer
9) Can, could, will, would, may, might, shall, should - Questions expecting a yes or no answer
10) No Question Word - The provided sentence is not formed as a question

Please answer only with the number of the question word type.

Question: Soğucak Yaylası Nerededir?
Label: 3

Question: شرایط مالیات در سوئد چگونه است؟
Label: 1

Question: Can We Choose our Hotels?
Label: 9

Question: Kā atstāt atsauksmi
Label: 7

Question: {question}
Label: """


def get_label(client, question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": PROMPT.format(question=question)}],
    )

    return response.choices[0].message.content


@click.command()
@click.argument("dataset_name", type=str)
def qtc_annotate(dataset_name: str):
    """
    Annotate Q&A pairs with a question type label.
    """
    dotenv.load_dotenv()

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Initialize results path
    results_path = os.path.join(DATASETS_FOLDER, dataset_name, "results")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Directory not found: {results_path}")
    if not os.path.isdir(results_path):
        raise NotADirectoryError(f"Path is not a directory: {results_path}")

    # Loop over all languages
    annotated_questions = []
    for language in sorted(os.listdir(results_path)):
        language_path = os.path.join(results_path, language)
        if not os.path.isdir(language_path):
            continue

        click.echo(f"Language: {language}")

        # Load questions
        questions = []
        with open(os.path.join(language_path, "faq.jsonl"), "r") as file:

            previous_scheme_host = None
            for line in file:
                # Load JSON line
                _dict = json.loads(line)
                question = _dict["question"]

                # Skip Q&A pairs with the same scheme_host
                scheme_host = _dict["scheme_host"]
                if scheme_host == previous_scheme_host:
                    continue
                previous_scheme_host = scheme_host

                # Append question
                questions.append(question)

                # Break if the limit plus buffer is reached
                if len(questions) >= LIMIT + 10:
                    break

        # Skip if the number of Q&A pairs is less than 100
        if len(questions) < THRESHOLD and len(questions) < LIMIT:
            click.echo("Skipping language due to insufficient Q&A pairs")
            continue

        # Annotate Q&A pairs
        annotated_questions_language = []
        with tqdm(
            total=min(len(questions), LIMIT), desc=f"Annotate Q&A pairs for {language}"
        ) as pbar:

            for question in questions:
                # Annotate using an LLM
                label = get_label(client, question)

                # Parse label
                label = label.strip()
                if re.match(r"^([1-9]|10)$", label):
                    label = int(label)
                elif re.match(r"^Label: ([1-9]|10)$", label):
                    label = int(label.split(": ")[1])
                elif re.match(r"^label: ([1-9]|10)$", label):
                    label = int(label.split(": ")[1])
                elif re.match(r"^[1-9|10]\)$", label):
                    label = int(label[:-1])
                elif re.match(r"^[1-9|10]\)\s+.*", label):
                    label = int(label.split(")")[0])
                else:
                    click.echo(f"Invalid label: {label}", err=True)
                    continue

                # Append annotated Q&A pair
                annotated_questions_language.append(
                    {"language": language, "label": label, "question": question}
                )

                # Update progress bar
                pbar.update(1)

                # Break if the limit is reached
                if len(annotated_questions_language) >= LIMIT:
                    break

        # Append annotated Q&A pairs for the current language
        annotated_questions.extend(annotated_questions_language)

        # Save annotated Q&A pairs (after each language)
        annotations_path = os.path.join(
            RESOURCES_FOLDER, dataset_name, "qtc_annotations.jsonl"
        )
        os.makedirs(os.path.dirname(annotations_path), exist_ok=True)
        with open(annotations_path, "w") as file:
            for qa_pair in annotated_questions:
                file.write(json.dumps(qa_pair) + "\n")

    click.echo("Done")
