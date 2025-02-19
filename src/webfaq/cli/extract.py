import click
import os
import sys
import gzip
import json
import re
import fasttext
import html
import time
import emoji
import hashlib
from tqdm import tqdm
from collections import defaultdict
from urllib.parse import urlparse
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.html import HTMLTree
from webfaq.utils import *
from webfaq.config import *


MAX_WORKERS = 16
LANGID_CONFIDENCE_THRESHOLD = 0.63
TYPE_PREDICATES = [
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://schema.org/type",
]
KEEP_TYPES = [
    "http://schema.org/FAQPage",
    "https://schema.org/FAQPage",
    "http://schema.org/Question",
    "https://schema.org/Question",
    "http://schema.org/Answer",
    "https://schema.org/Answer",
]
URL_TYPES = [
    "http://schema.org/url",
    "https://schema.org/url",
    "http://schema.org/embedUrl",
    "https://schema.org/embedUrl",
    "http://schema.org/contentUrl",
    "https://schema.org/contentUrl",
]


def create_entities_file(input_path: str, entities_path: str):
    """
    The file is assumed to be grouped by graph URL (quads with the same graph URL are grouped together)
    """
    # Initialize entities dictionary
    entities = defaultdict(dict)

    # Count lines in input file
    num_lines = sum(1 for _ in gzip.open(input_path, "rt", encoding="utf-8"))

    with tqdm(total=num_lines, mininterval=10, file=sys.stdout) as pbar:

        with gzip.open(input_path, "rt", encoding="utf-8") as file:
            
            current_url = ""
            current_url_subjects_quads = defaultdict(list)
            for line in file:
                # Update progress bar
                pbar.update(1)

                # Parse quad line
                quad = _parse_quad_line(line.strip())
                if not quad:
                    continue

                if quad["graph"] != current_url:
                    # Loop over all subjects and create entities
                    for subject, quads in current_url_subjects_quads.items():
                        entity = _create_entity(quads)
                        if entity:
                            entities[current_url][subject] = entity

                    # Update the current URL
                    current_url = quad["graph"]
                    current_url_subjects_quads.clear()

                # Add the quad to the list
                current_url_subjects_quads[quad["subject"]].append(quad)

            # Create entities from the remaining quads
            for subject, quads in current_url_subjects_quads.items():
                entity = _create_entity(quads)
                if entity:
                    entities[current_url][subject] = entity

    # Sort the entities by graph URL
    click.echo("Sorting entities by graph URL")
    entities = dict(sorted(entities.items(), key=lambda item: item[0]))

    # Write the entities to the sorted file
    click.echo(f"Writing entities to file: {entities_path}")
    with gzip.open(entities_path, "wt", encoding="utf-8") as file:
        json.dump(entities, file, indent=2)


def _normalize_newlines(text):
    """
    Replace sequences of \n, \r\n, or whitespace-separated newlines
    with a single \n.
    """
    # Regex explanation:
    # - (\s*[\r\n]+): Matches any whitespace (\s*) followed by one or more newlines (\r\n or \n)
    # - Replace with a single newline (\n)
    normalized_text = re.sub(r'\\r\\n|\\n', r'\n', text).strip()
    while True:
        _normalized_text = re.sub(r'\s*\n+\s*', r'\n', normalized_text)
        if _normalized_text == normalized_text:
            break
        normalized_text = _normalized_text
    return normalized_text


def _parse_quad_line(line: str) -> Dict[str, Any]:
    line = line.strip()

    for url_type in URL_TYPES:
        if url_type in line:
            return None

    nquad_regex = re.compile(
        r'^(<[^>]+>|_:[a-zA-Z0-9]+)\s'   # Subject: URI in <...> or blank node starting with "_:"
        r'(<[^>]+>)\s'                   # Predicate: a URI in <...>
        r'(<[^>]+>|".*"(?:\^\^<[^>]+>)?(?:\@[^\s]+(?:\sclass=)?)?|_:[a-zA-Z0-9]+)\s'  # Object: URI, blank node, or literal string (including empty "")
        r'(<[^>]+>)\s\s\s\.$'                # URL (graph): a URI in <...>, followed by a dot
    )

    subject, predicate, object, url = "", "", "", ""
    match = nquad_regex.match(line)
    if not match:
        # click.echo(f"Invalid N-Quad line: {line}", err=True)
        return None
    
    groups = match.groups()
    if len(groups) != 4:
        # click.echo(f"Invalid N-Quad line: {line} -> {groups}", err=True)
        return None
    
    subject, predicate, object, url = groups

    # Unescape HTML entities and extract plain text
    # print("----")
    # print("Graph:", url)
    # print("Subject:", subject)
    # print("Predicate:", predicate)
    # print("Object:", object)

    if object.startswith("\""):
        literal_regex = re.compile(r'^"(.*)"(?:\^\^<[^>]+>)?(?:\@[^\s]+(?:\sclass=)?)?$')
        match = literal_regex.match(object)
        if not match:
            raise ValueError(f"Invalid literal object: {object}")
        literal_string = match.group(1).strip()
        # original_literal_string = literal_string
        literal_string = html.unescape(literal_string)
        literal_string = re.sub(r'\\t', ' ', literal_string)
        tree = HTMLTree.parse(literal_string)
        literal_string = extract_plain_text(
            tree,
            main_content=False,
            alt_texts=False,
            preserve_formatting=False,
            noscript=True
        )
        literal_string = re.sub(r"\\\"", "\"", literal_string)
        literal_string = _normalize_newlines(literal_string)
        object = literal_string

    return {
        "graph": url.strip("<>"),
        "subject": subject.strip("<>"),
        "predicate": predicate.strip("<>"),
        "object": object.strip("<>"),
    }


def _create_entity(
        quads: List[Dict[str, Any]]
) -> Dict[str, Any]:
    assert quads, "Quads list cannot be empty"
    graph_url = quads[0]["graph"]
    subject = quads[0]["subject"]
    predicate_type = None

    properties = defaultdict(list)
    for quad in quads:
        assert quad["graph"] == graph_url, "Quads must belong to the same graph"
        assert quad["subject"] == subject, "Quads must belong to the same subject"

        predicate = quad["predicate"]
        object = quad["object"]
        if predicate in TYPE_PREDICATES:
            predicate_type = object
        else:
            properties[predicate].append(object)
    
    if not predicate_type:
        # click.echo(f"Predicate type not found for entity: {subject} -> {list(properties.keys())}", err=True)
        return None
    
    if predicate_type not in KEEP_TYPES:
        # click.echo(f"Skipping entity with type: {graph_url} -> {predicate_type}", err=True)
        return None
    
    entity = {
        "type": predicate_type,
        "properties": properties,
    }
    return entity


def parse_entities_file(
        entities_file_path: str,
        faq_path: str,
        model: fasttext.FastText._FastText
):
    # Initialize results dictionary
    results = defaultdict(list)

    # Parse the entities file
    with gzip.open(entities_file_path, "rt", encoding="utf-8") as entities_file:

        click.echo(f"Reading entities from file: {entities_file_path}")
        start = time.time()
        entities = json.load(entities_file)
        click.echo(f"Read {len(entities)} entities in {time.time() - start:.2f} seconds from file: {entities_file_path}")

        with tqdm(total=len(entities), mininterval=10, file=sys.stdout) as pbar:

            ids_set = set()
            for graph_url, entity_map in entities.items():

                # Update progress bar
                pbar.update(1)

                faqpage_mainEntities = []
                questions = {}
                answers = {}
                for subject, entity in entity_map.items():

                    if entity["type"].lower() in ["http://schema.org/FAQPage".lower(), "https://schema.org/FAQPage".lower()]:
                        main_entities = []
                        for property, values in entity["properties"].items():
                            if property.lower() == "http://schema.org/mainEntity".lower():
                                main_entities.extend(values)
                            elif property.lower() == "https://schema.org/mainEntity".lower():
                                main_entities.extend(values)
                        if not main_entities:
                            # click.echo(f"Main entity not found for FAQPage: {graph_url} -> {subject}", err=True)
                            continue
                        faqpage_mainEntities.extend(main_entities)
                    elif entity["type"].lower() in ["http://schema.org/Question".lower(), "https://schema.org/Question".lower()]:
                        name = None
                        acceptedAnswer = None
                        for property, values in entity["properties"].items():
                            if property.lower() == "http://schema.org/name":
                                name = values
                            elif property.lower() == "https://schema.org/name":
                                name = values
                            elif property.lower() == "http://schema.org/acceptedAnswer".lower():
                                acceptedAnswer = values
                            elif property.lower() == "https://schema.org/acceptedAnswer".lower():
                                acceptedAnswer = values
                        if not name:
                            # click.echo(f"Question name not found: {graph_url} -> {subject}", err=True)
                            continue
                        if not acceptedAnswer:
                            # click.echo(f"Accepted answer not found: {graph_url} -> {subject}", err=True)
                            continue
                        # questions[subject] = (name[0], acceptedAnswer[0])
                        questions[name[0]] = (subject, acceptedAnswer[0])
                    elif entity["type"].lower() in ["http://schema.org/Answer".lower(), "https://schema.org/Answer".lower()]:
                        text = None
                        for property, values in entity["properties"].items():
                            if property.lower() == "http://schema.org/text":
                                text = values
                            elif property.lower() == "https://schema.org/text":
                                text = values
                        if not text:
                            # click.echo(f"Answer text not found: {graph_url} -> {subject}", err=True)
                            continue
                        answers[subject] = text[0]

                # for question_subject, (question, answer_subject) in questions.items():
                for question, (question_subject, answer_subject) in questions.items():
                    if not question_subject in faqpage_mainEntities:
                        # click.echo(f"Question not found in main entities: {graph_url} -> {question_subject}", err=True)
                        continue
                    if not answer_subject in answers:
                        # click.echo(f"Answer not found for question: {graph_url} -> {question_subject}", err=True)
                        continue
                    answer = answers[answer_subject]

                    # Skip if question or answer are empty
                    if not question or not answer:
                        continue

                    # Skip if answer contains a question mark
                    if "?" in answer:
                        continue

                    # Skip if answer contains too many newlines
                    if answer.count("\n") >= 10:
                        continue

                    # Skip if question is too short
                    if len(question) < 10:
                        continue

                    # Skip if answer is too short
                    if len(answer) < 30:
                        continue

                    # Add host
                    host = "-"
                    try:
                        parsed_url = urlparse(graph_url)
                        host = parsed_url.hostname
                    except Exception:
                        click.echo(f"{graph_url} cannot be parsed", err=True)

                    # Extract scheme and host
                    scheme = graph_url.split("://")[0]
                    _host = graph_url.split("://")[1].split("/")[0].split("?")[0].split("#")[0]

                    if not _host.lower().startswith(host):
                        continue
                    scheme_host = f"{scheme}://{host}"

                    # Remove Q: and A: prefixes
                    question = re.sub(r'^Q:\s+', '', question)
                    question = re.sub(r'^Question:\s+', '', question)
                    answer = re.sub(r'^A:\s+', '', answer)
                    answer = re.sub(r'^Answer:\s+', '', answer)

                    # Remove numbering prefixes
                    question = re.sub(r'^\d+\.\d*\s+', '', question)
                    answer = re.sub(r'^\d+\.\d*\s+', '', answer)

                    # Remove leading and trailing quotation marks
                    if question.startswith('"') and question.endswith('"'):
                        question = question[1:-1]
                    if answer.startswith('"') and answer.endswith('"'):
                        answer = answer[1:-1]
                    if question.startswith('“') and question.endswith('”'):
                        question = question[1:-1]
                    if answer.startswith('“') and answer.endswith('”'):
                        answer = answer[1:-1]
                    if question.startswith('‘') and question.endswith('’'):
                        question = question[1:-1]
                    if answer.startswith('‘') and answer.endswith('’'):
                        answer = answer[1:-1]
                    if question.startswith('‟') and question.endswith('”'):
                        question = question[1:-1]
                    if answer.startswith('‟') and answer.endswith('”'):
                        answer = answer[1:-1]
                    if question.startswith('„') and question.endswith('”'):
                        question = question[1:-1]
                    if answer.startswith('„') and answer.endswith('”'):
                        answer = answer[1:-1]
                    if question.startswith('„') and question.endswith('“'):
                        question = question[1:-1]
                    if answer.startswith('„') and answer.endswith('“'):
                        answer = answer[1:-1]

                    # Remove emojis
                    question = emoji.replace_emoji(question, "").strip()
                    answer = emoji.replace_emoji(answer, "").strip()
                    
                    # Detect language with fastText
                    text = question + ' ' + answer
                    text = text.replace('\n', ' ')
                    langid = model.predict(text, k=1)
                    iso_alpha_2 = langid[0][0].split('__label__')[1]
                    lang_code = map_iso_code(iso_alpha_2)
                    confidence = langid[1][0]
                    if confidence < LANGID_CONFIDENCE_THRESHOLD:
                        click.echo(f"Low confidence ({confidence}) for language {lang_code}: {text}", err=True)
                        continue
                    if not lang_code:
                        continue

                    # Compute hash ID for question-answer pair
                    id = hashlib.md5(text.encode("utf-8")).hexdigest()

                    # Skip if the question-answer pair is duplicated
                    if id in ids_set:
                        continue
                    ids_set.add(id)

                    result = {
                        "id": id,
                        "url": graph_url,
                        "scheme_host": scheme_host,
                        "language": lang_code,
                        "confidence": confidence,
                        "question": question,
                        "answer": answer,
                    }

                    results[lang_code].append(result)

    for lang_code in results:
        _faq_path = faq_path.format(lang_code)
        os.makedirs(os.path.dirname(_faq_path), exist_ok=True)

        # Open the file in append mode and write the results
        with open(_faq_path, "w", encoding="utf-8") as output_file:
            for result in results[lang_code]:
                output_file.write(json.dumps(result) + "\n")


def process(
        input_path: str,
        entities_path: str,
        faq_path: str,
        model: fasttext.FastText._FastText
):
    # Create entities file
    if not os.path.exists(entities_path):
        create_entities_file(input_path, entities_path)

    # Parse entities file
    parse_entities_file(entities_path, faq_path, model)
    click.echo(f"Processed file: {input_path}")


@click.command()
@click.argument("prefix", type=str)
def extract(prefix: str):
    """
    Extract the Q&A pairs from the FAQPage dump of Web Data Commons.
    """
    # Load fastText model
    path = os.path.join(MODELS_FOLDER, 'lid.176.bin')
    model = fasttext.load_model(path)

    files = []
    for filename in os.listdir(os.path.join(DATASETS_FOLDER, "downloads")):
        input_path = os.path.join(DATASETS_FOLDER, "downloads", filename)
        if os.path.isfile(input_path) and filename.startswith(prefix) and filename.endswith(".gz"):
            files.append(input_path)
    click.echo(f"Found {len(files)} files to process")

    for input_path in sorted(files):
        click.echo(f"Processing file: {input_path}")

        # Define file paths
        input_name = os.path.basename(input_path).replace(".gz", "")
        entities_path = os.path.join(DATASETS_FOLDER, prefix, "entities", f"entities_{input_name}.json.gz")
        faq_path = os.path.join(DATASETS_FOLDER, prefix, "results", "{}", input_name, "faq.jsonl")
        os.makedirs(os.path.dirname(entities_path), exist_ok=True)

        process(input_path, entities_path, faq_path, model)
    
    click.echo("Done")
