import click
import os
import json
import heapq
import shutil
from webfaq.config import *


@click.command()
@click.argument("dataset_name", type=str)
@click.option("--extend", is_flag=True)
def merge(dataset_name: str, extend: bool):
    """
    Merge the extracted Q&A pairs from the FAQPage dump of Web Data Commons.
    """
    # Open all faq.jsonl files and write the results into one file
    results_path = os.path.join(DATASETS_FOLDER, dataset_name, "results")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Directory not found: {results_path}")
    if not os.path.isdir(results_path):
        raise NotADirectoryError(f"Path is not a directory: {results_path}")

    for language in sorted(os.listdir(results_path)):
        language_path = os.path.join(results_path, language)
        if not os.path.isdir(language_path):
            continue

        click.echo(f"Language: {language}")

        # Initialize faq path
        faq_path = os.path.join(results_path, language, "faq.jsonl")
        os.makedirs(os.path.dirname(faq_path), exist_ok=True)

        # Open all part files
        part_files = []
        lines = []
        documents = []
        heap = []
        for input_name in sorted(os.listdir(language_path)):
            input_path = os.path.join(language_path, input_name)
            if not os.path.isdir(input_path):
                continue

            part_files.append(
                open(os.path.join(input_path, "faq.jsonl"), "r", encoding="utf-8")
            )
            line = part_files[-1].readline()
            lines.append(line)
            document = json.loads(line)
            documents.append(document)
            heapq.heappush(heap, (document["url"], len(part_files) - 1))

        # Open existing faq file
        if extend and os.path.exists(faq_path):
            copy_faq_path = faq_path + ".bak"
            shutil.copyfile(faq_path, copy_faq_path)
            os.remove(faq_path)

            part_files.append(open(copy_faq_path, "r", encoding="utf-8"))
            line = part_files[-1].readline()
            lines.append(line)
            document = json.loads(line)
            documents.append(document)
            heapq.heappush(heap, (document["url"], len(part_files) - 1))

        # Skip if there are no files to merge
        if not part_files:
            continue

        with open(faq_path, "w", encoding="utf-8") as output_file:

            # Get next line from the file with the smallest value
            ids_set = set()
            while heap:
                _continue = False
                _, index = heapq.heappop(heap)
                line = lines[index]
                document = documents[index]
                id = document["id"]

                # Skip if the question-answer pair is duplicated
                if id in ids_set:
                    _continue = True
                else:
                    ids_set.add(id)

                # Write the line to the output file
                if not _continue:
                    output_file.write(line)

                # Read the next line from the same file
                next_line = part_files[index].readline()
                if next_line:
                    lines[index] = next_line
                    document = json.loads(next_line)
                    documents[index] = document
                    heapq.heappush(heap, (document["url"], index))

        # Close all files
        for file in part_files:
            file.close()

        # Remove all part directories
        for input_name in os.listdir(language_path):
            input_path = os.path.join(language_path, input_name)
            if not os.path.isdir(input_path):
                continue
            shutil.rmtree(input_path)

        # Remove the backup file
        if extend and os.path.exists(copy_faq_path):
            os.remove(copy_faq_path)

    click.echo("Done")
