import click
import os
import sys
import gzip
import json
import re
import fasttext
import emoji
import hashlib
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import defaultdict
from urllib.parse import urlparse
from webfaq.utils import *
from webfaq.config import *


MAX_WORKERS = 16
LANGID_CONFIDENCE_THRESHOLD = 0.25  # 0.63
LANGID_CONFIDENCE_HINT_THRESHOLD = 0.12


def process_parquet_file(
    parquet_file_path: str,
    ids_set: Set[str],
    model: fasttext.FastText._FastText,
):
    # Initialize results dictionary
    results = defaultdict(list)

    # Read Parquet file
    try:
        pq_file = pq.ParquetFile(parquet_file_path)
    except Exception as e:
        click.echo(
            f"Exception caught when reading Parquet file {parquet_file_path}: {e}",
            err=True,
        )
        return

    for batch in pq_file.iter_batches(batch_size=10_000):
        # Convert batch to JSON list
        batch_list = batch.to_pylist()

        for row in tqdm(batch_list, total=len(batch_list), mininterval=10, file=sys.stdout):

            faq = row["faq"]
            if not faq:
                continue

            url = row["url"]
            url_scheme = row["url_scheme"]
            url_domain = row["url_domain"]
            url_subdomain = row["url_subdomain"]
            url_suffix = row["url_suffix"]
            origin = f"{url_scheme}://"
            if url_subdomain:
                origin += f"{url_subdomain}."
            origin += f"{url_domain}.{url_suffix}"
            title = row.get("title", "")
            description = row.get("description", "")
            keywords = row.get("keywords", "")
            author = row.get("author", "")
            warc_date = row.get("warc_date", "")

            for faq_dict in faq:
                # Extract fields
                question = faq_dict.get("question", "").strip()
                answer = faq_dict.get("answer", "").strip()
                # lang_code = faq_dict.get("language", "--").strip().lower()  # Too inaccurate

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

                # Remove Q: and A: prefixes
                question = re.sub(r"^Q:\s+", "", question)
                question = re.sub(r"^Question:\s+", "", question)
                answer = re.sub(r"^A:\s+", "", answer)
                answer = re.sub(r"^Answer:\s+", "", answer)

                # Remove numbering prefixes
                question = re.sub(r"^\d+\.\d*\s+", "", question)
                answer = re.sub(r"^\d+\.\d*\s+", "", answer)

                # Remove leading and trailing quotation marks
                if question.startswith('"') and question.endswith('"'):
                    question = question[1:-1]
                if answer.startswith('"') and answer.endswith('"'):
                    answer = answer[1:-1]
                if question.startswith("“") and question.endswith("”"):
                    question = question[1:-1]
                if answer.startswith("“") and answer.endswith("”"):
                    answer = answer[1:-1]
                if question.startswith("‘") and question.endswith("’"):
                    question = question[1:-1]
                if answer.startswith("‘") and answer.endswith("’"):
                    answer = answer[1:-1]
                if question.startswith("‟") and question.endswith("”"):
                    question = question[1:-1]
                if answer.startswith("‟") and answer.endswith("”"):
                    answer = answer[1:-1]
                if question.startswith("„") and question.endswith("”"):
                    question = question[1:-1]
                if answer.startswith("„") and answer.endswith("”"):
                    answer = answer[1:-1]
                if question.startswith("„") and question.endswith("“"):
                    question = question[1:-1]
                if answer.startswith("„") and answer.endswith("“"):
                    answer = answer[1:-1]

                # Remove emojis
                question = emoji.replace_emoji(question, "").strip()
                answer = emoji.replace_emoji(answer, "").strip()

                def get_lang_code_hints(segment: str) -> Optional[List[str]]:
                    segment = segment.group(1).lower()
                    lang_code_hints = []
                    if re.match(r"^[a-z]{2}-[a-z]{2}$", segment):
                        iso_alpha_2 = segment.split("-")[0]
                        iso_alpha_3 = map_iso_code(iso_alpha_2)
                        if iso_alpha_3:
                            lang_code_hints.append(iso_alpha_3)
                    elif re.match(r"^[a-z]{3}(-[a-z]{2})?$", segment):
                        lang_code_hints.append(segment.split("-")[0])
                    elif re.match(r"^[a-z]{2}$", segment):
                        iso_alpha_3 = CLDR_MAPPING.get(segment.upper(), None)
                        if iso_alpha_3:
                            lang_code_hints.extend(iso_alpha_3)
                        iso_alpha_3 = map_iso_code(segment)
                        if iso_alpha_3 and iso_alpha_3 not in lang_code_hints:
                            lang_code_hints.append(iso_alpha_3)
                    return lang_code_hints

                # Manual language assignment (if any)
                lang_code = LANG_CODE_ORIGIN_MAPPING.get(origin, None)
                confidence = 1.0

                if not lang_code:
                    # Detect language code
                    lang_code_hints = []
                    first_url_segment = re.match(r"^https?://[^/]+/([^/]+)/?", url)
                    if first_url_segment:
                        lang_code_hints = get_lang_code_hints(first_url_segment)
                    if not lang_code_hints:
                        second_url_segment = re.match(r"^https?://[^/]+/[^/]+/([^/]+)/?", url)
                        if second_url_segment:
                            lang_code_hints = get_lang_code_hints(second_url_segment)

                    # Detect language with fastText
                    text = question + " " + answer
                    if title:
                        text += " # " + title
                    if description:
                        text += " # " + description
                    text = text.replace("\n", " ").strip()
                    if not text:
                        continue
                    langid = model.predict(text, k=10)
                    candidates = {}
                    for i in range(len(langid[0])):
                        iso_alpha_2 = langid[0][i].split("__label__")[1]
                        lang_code = map_iso_code(iso_alpha_2)
                        confidence = langid[1][i]
                        candidates[lang_code] = confidence

                    # Use hint from URL if available and confident enough
                    lang_code = None
                    if lang_code_hints:
                        for lang_code_hint in lang_code_hints:
                            if lang_code_hint in candidates:
                                if candidates[lang_code_hint] >= LANGID_CONFIDENCE_HINT_THRESHOLD:
                                    lang_code = lang_code_hint
                                    break

                    # Fallback to best candidate
                    if not lang_code:
                        iso_alpha_2 = langid[0][0].split("__label__")[1]
                        lang_code = map_iso_code(iso_alpha_2)
                        confidence = langid[1][0]
                        if confidence < LANGID_CONFIDENCE_THRESHOLD:
                            click.echo(
                                f"Low confidence ({confidence}) for language {lang_code}: {text} ({url}; {langid}; {lang_code_hints})",
                                err=True,
                            )
                            continue

                # Skip if language code could not be determined
                if not lang_code:
                    continue

                # Compute hash ID for question-answer pair
                text = question + " " + answer
                text = text.replace("\n", " ").strip()
                _id = hashlib.md5(text.encode("UTF-8")).hexdigest()

                # Skip if the question-answer pair is duplicated
                if _id in ids_set:
                    continue
                ids_set.add(_id)

                result = {
                    "id": _id,
                    "url": url,
                    "origin": origin,
                    "language": lang_code,
                    "confidence": confidence,
                    "question": question,
                    "answer": answer,
                    "title": title,
                    "description": description,
                    "keywords": keywords,
                    "author": author,
                    "warc_date": warc_date,
                }

                results[lang_code].append(result)

    for lang_code in results:
        # Open the file in append mode and write the results
        faq_path = os.path.join(DATASETS_FOLDER, "faqs", lang_code, f"raw.jsonl.gz")
        os.makedirs(os.path.dirname(faq_path), exist_ok=True)
        with gzip.open(faq_path, "at", encoding="UTF-8") as output_file:
            for result in results[lang_code]:
                output_file.write(json.dumps(result) + "\n")


@click.command()
def extract():
    """
    Extract the Q&A pairs from OpenWebIndex datasets.
    """
    # Load fastText model
    model_path = os.path.join(MODELS_FOLDER, "lid.176.bin")
    model = fasttext.load_model(model_path)

    parquet_files = []
    for root, _, files in os.walk(OWI_DATASETS_FOLDER):
        for name in files:
            if name.endswith(".parquet"):
                parquet_files.append(os.path.join(root, name))
    click.echo(f"Found {len(parquet_files)} Parquet files to process")

    ids_set = set()
    for parquet_file in sorted(parquet_files):
        click.echo(f"Processing file: {parquet_file}")
        process_parquet_file(parquet_file, ids_set, model)

    click.echo("Done")
