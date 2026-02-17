> [!NOTE]
> The most recent code is in the branch v2. For all further information regarding WebFAQ 2.0, please have a look at our [HuggingFace repository](https://huggingface.co/datasets/michaeldinzinger/webfaq-v2).

# WebFAQ 2.0

WebFAQ 2.0 is a large-scale multilingual dataset of 198 million natural question–answer pairs across 108 languages, mined from structured FAQ pages on the web.

It is the successor of the original WebFAQ (v1) dataset (96M QAs, 75 languages): 👉 [https://huggingface.co/datasets/PaDaS-Lab/webfaq](https://huggingface.co/datasets/PaDaS-Lab/webfaq)

WebFAQ 2.0 significantly expands multilingual coverage, bilingual QA alignments, and introduces a new hard negatives dataset for training dense retrieval models .

## Install dependencies

There are two ways you can install the dependencies to run the code.

### Using Poetry (recommended)

If you have the [Poetry](https://python-poetry.org/) package manager for Python installed already, you can simply set up everything with:

```console
poetry install && poetry shell
```

After the installation of all dependencies, you will end up in a new shell with a loaded venv. In this shell, you can run the main `webfaq` command. You can exit the shell at any time with `exit`.

```console
webfaq --help
```

### Using Pip (alternative)

You can also create a venv yourself and use `pip` to install dependencies:

```console
python3 -m venv venv
source venv/bin/activate
pip install .
```

## Download `fastText` language detection model

You can download the pretrained language detection model distributed by the authors of [fastText](https://fasttext.cc/docs/en/language-identification.html):

```console
mkdir -p models
cd models
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

To use the model, install the `fasttext` package in your Python environment:

```console
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
```

## Development

### Run Code Formatting

To run the code formatting, you can use the following command:

```console
isort .
black .
```

The order of the commands is important. `isort` will sort the imports in the files, and `black` will format the code.

## Run extraction code

First, you need to download Open Web Index (OWI) datasets. You can download datasets via the [OWILIX](https://openwebsearcheu-public.pages.it4i.eu/owi-cli/index.html) command-line tool. After downloading datasets, they are located in the local cache (typically `~/.owi/public/main/`).

To extract the QA pairs from OWI datasets, run the following command:

```console
webfaq extract
webfaq sort
```

The extracted QA pairs will be stored in the `datasets/faqs/` directory.

To further embed the extracted QAs with LaBSE and Jina (v3), run the following commands:

```console
webfaq embed-labse
webfaq embed-jina
```

## License

This repository is licensed under the MIT License. See the LICENSE file for details.
