import click
import os
import json
import openai
import dotenv
import re
import random
from tqdm import tqdm
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from webfaq.utils import *
from webfaq.config import *


SCORING_SAMPLE_PROBABILITY = 0.001
MAX_WORKERS = 32


PROMPT = """Score the following translation from {source_lang} to {target_lang} on a
continuous scale from 0 to 100, where score of zero means "no meaning preserved" and
score of one hundred means "perfect meaning and grammar". Please provide a short,
concise answer.
{source_lang} source: "{source_seg}"
{target_lang} translation: "{target_seg}"
Score: """


def parse_output(output: str) -> int:
    # Split output into lines
    lines = output.split("\n")

    score = -1
    for line in lines:
        line = line.strip()
        if line == "":
            continue

        if (match := re.match("^([0-9]{1,3})$", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match("Score: ([0-9]{1,3})", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match("Score: \*\*([0-9]{1,3})", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match(".* ([0-9]{1,3}) out of 100", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match(".* \*\*([0-9]{1,3})\*\* out of 100", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match(".* \*\*([0-9]{1,3}) out of 100\*\*", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match(".* ([0-9]{1,3})/100", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match(".* \*\*([0-9]{1,3})/100\*\*", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match("I would score .* ([0-9]{1,3})", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match("I would score .* \*\*([0-9]{1,3})", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match("I would rate .* ([0-9]{1,3})", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match("I would rate .* \*\*([0-9]{1,3})", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match("I would give .* ([0-9]{1,3})", line)) is not None:
            score = int(match.group(1))
            break
        elif (match := re.match("I would give .* \*\*([0-9]{1,3})", line)) is not None:
            score = int(match.group(1))
            break

    if score == -1:
        click.echo(f"Invalid text:\n{output.strip()}", err=True)
    
    return score



def get_score(
        client,
        languages: Tuple[str, str],
        questions: Tuple[str, str],
        answers: Tuple[str, str]
) -> int:
    # Validity checks
    assert len(languages) == 2
    assert len(questions) == 2
    assert len(answers) == 2

    # Prepare segments
    seg_0 = f"{questions[0]} {answers[0]}"
    seg_1 = f"{questions[1]} {answers[1]}"

    # Get score from GPT-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {
                "role": "system",
                "content": PROMPT.format(source_lang=languages[0], target_lang=languages[1], source_seg=seg_0, target_seg=seg_1)
            }
        ]
    )
    output = response.choices[0].message.content.strip()

    # Parse output
    return parse_output(output)


def process_line(
        line: str,
        client: openai.OpenAI,
        pbar: tqdm
) -> Optional[Dict[str, Union[int, str]]]:
    
    # Update progress bar
    pbar.update(1)

    try:
        document = json.loads(line)
        similarity = document["similarity"]
        languages = document["languages"]
        questions = document["questions"]
        answers = document["answers"]

        # similarity < THRESHOLD_SIMILARITY
        if random.random() < SCORING_SAMPLE_PROBABILITY:
            # Score using an LLM
            score = get_score(client, languages, questions, answers)

        else:
            score = -2

    except json.JSONDecodeError as e:
        click.echo(f"Skipping invalid JSON line: {line.strip()} ({e})")
        return None

    return {
        "scheme_host": document["scheme_host"],
        "similarity": similarity,
        "score": score,
        "languages": languages,
        "urls": document["urls"],
        "questions": questions,
        "answers": answers,
        "topics": document["topics"],
        "question_types": document["question_types"],
    }


@click.command()
@click.argument("dataset_name", type=str)
@click.argument("filename_pattern", type=str)
def pc_score(dataset_name: str, filename_pattern: str):
    """
    Creating scores through automated evaluation of parallel sentences using OpenAI's GPT-4o-mini.
    """
    dotenv.load_dotenv()

    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    # Load the JSONL files
    paths = glob(os.path.join(DATASETS_FOLDER, dataset_name, "results", filename_pattern))
    click.echo(f"Found {len(paths)} candidates files")

    for path in sorted(paths):
        click.echo(f"Candidates file: {path}")

        # Initialize scored documents
        scored_documents = []

        num_samples = count_lines(path)
        with tqdm(total=num_samples) as pbar:

            with open(path, "r") as file:

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

                    futures = []
                    for line in file:
                        futures.append(executor.submit(process_line, line, client, pbar))

                    for future in as_completed(futures):
                        scored_document = future.result()
                        if scored_document is not None:
                            scored_documents.append(scored_document)

        # Save scores to file
        scores_path = path.replace("candidates_", "scores_")
        with open(scores_path, "w") as file:
            for scored_document in scored_documents:
                file.write(json.dumps(scored_document) + "\n")

    click.echo("Done")
