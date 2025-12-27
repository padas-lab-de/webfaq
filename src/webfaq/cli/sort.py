import click
import os
import gzip
import json
from tqdm import tqdm
from webfaq.utils import *
from webfaq.config import *


THRESHOLD_FILE_SIZE = 10_000_000


@click.command()
def sort():
    """
    Sort and deduplicate the extracted JSONL files.
    """
    # Loop over all languages
    faqs_path = os.path.join(DATASETS_FOLDER, "faqs")
    if not os.path.exists(faqs_path):
        click.echo(f"No FAQs found in {faqs_path}", err=True)
        return

    # Instantiate an id set to avoid duplicates
    ids_set = set()

    for language in sorted(os.listdir(faqs_path)):
        language_path = os.path.join(faqs_path, language)
        if not os.path.isdir(language_path):
            continue

        click.echo(f"Language: {language}")

        # Read all JSONL files for the language
        all_faqs = {}
        for file_name in sorted(os.listdir(language_path)):
            if file_name != "raw.jsonl.gz":
                continue

            file_path = os.path.join(language_path, file_name)
            click.echo(f"Reading file: {file_name}")

            with gzip.open(file_path, "rt", encoding="UTF-8") as input_file:
                for line in input_file:

                    try:
                        faq = json.loads(line)
                    except Exception as e:
                        click.echo(f"Exception caught when reading line: {e}", err=True)
                        continue

                    # Avoid duplicates
                    _id = faq["id"]
                    if _id in ids_set:
                        continue
                    ids_set.add(_id)

                    sort_key = faq["url"] + "#" + _id
                    all_faqs[sort_key] = faq

        # Compare number of unique FAQs
        click.echo(f"Number of unique FAQs for {language}: {len(all_faqs)}")
        count_path = os.path.join(language_path, "count.txt")
        if os.path.exists(count_path):
            with open(count_path, "r", encoding="UTF-8") as count_file:
                recorded_count = int(count_file.read().strip())
                if recorded_count != len(all_faqs):
                    click.echo(
                        f"Warning: Recorded count {recorded_count} does not match unique FAQs {len(all_faqs)}",
                        err=True,
                    )

        # Sort the FAQs by URL and ID
        sorted_faqs = [all_faqs[key] for key in sorted(all_faqs.keys())]

        # Write to new JSONL files with the "sorted_" prefix
        index_file = 0
        output_path = os.path.join(language_path, f"faqs_sorted_{index_file}.jsonl.gz")
        output_file = gzip.open(output_path, "wt", encoding="UTF-8")

        count_file = 0
        for faq in tqdm(
            sorted_faqs,
            total=len(sorted_faqs),
            mininterval=10,
            desc=f"Writing sorted FAQs for {language}",
        ):
            output_file.write(json.dumps(faq) + "\n")
            count_file += 1

            if count_file >= THRESHOLD_FILE_SIZE:
                output_file.close()

                # Open a new file for the next batch of sorted FAQs
                index_file += 1
                output_path = os.path.join(
                    language_path, f"faqs_sorted_{index_file}.jsonl.gz"
                )
                output_file = gzip.open(output_path, "wt", encoding="UTF-8")
                count_file = 0

        output_file.close()

    click.echo("Done")
