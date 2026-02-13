> [!NOTE]
> The most recent code is in the branch v2. For all further information regarding WebFAQ 2.0, please have a look at our [HuggingFace repository](https://huggingface.co/datasets/michaeldinzinger/webfaq-v2).

# WebFAQ

The **WebFAQ Q&A Dataset** is a broad-coverage corpus of **96 million** natural question-answer (QA) pairs in **75 languages**, gathered from FAQ pages on the web. It leverages structured [schema.org FAQPage](https://schema.org/FAQPage) annotations, making it a unique resource for large-scale Question Answering research. Each entry includes a question, the corresponding answer, and additional metadata such as language, topic, and question type.

This repository contains the code to extract the question-answer (QA) pairs from Web Data Commons (WDC) dumps, as well as related code to process and analyze the dataset.

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

## Download JDK 21

The code uses the `pyserini` library, which requires Java 21. You can install it on Linux with:

```console
sudo apt update
sudo apt install openjdk-21-jdk
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

## Run extraction code

First, you need to download the Web Data Commons (WDC) dumps. You can find the dumps [here](https://webdatacommons.org/structureddata/). A sample file is provided in the project directory ([FAQPage_sample.txt](FAQPage_sample.txt)).

To extract the QA pairs from the WDC dumps, run the following command:

```console
mkdir -p datasets/downloads
cp FAQPage_sample.txt datasets/downloads/
gzip datasets/downloads/FAQPage_sample.txt
webfaq extract FAQPage
webfaq merge FAQPage
```

The extracted QA pairs will be stored in the `datasets/FAQPage/` directory. `FAQPage` is the dataset name that has to be provided as an argument to the next commands.

To further embed the extracted QAs with LaBSE and Jina (v3), run the following commands:

```console
webfaq embed-labse FAQPage
webfaq embed-jina FAQPage
```

## Evaluation

### BM25

We use `pyserini` to build indexes for and evaluate a BM25 model. To convert the English WebFAQ dataset into the format
`pyserini` expects, use the following command:

```console
webfaq bm25 generate-jsonl eng-Latn path/to/store/dataset/ webfaq
```

To create an index on a dataset stored at `/webfaq/data/eng/`, run

```console
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input path/to/stored/dataset/ \
    --index /path/to/save/index/ \
    --generator DefaultLuceneDocumentGenerator \
    --threads 1 \
    --storePositions --storeDocvectors --storeRaw
```

Retrieval performance on the created index can then be evaluated using

```console
webfaq bm25 evaluate /webfaq/data/indexes/eng/ /webfaq/data/eng/ /webfaq/temp/evaluation/eng/
```

with results being stored in `/webfaq/temp/evaluation/eng/`.

### Dense Retrieval

To evaluate retrieval performance using a dense retrieval model compatible with the `sentence-transformers` library on
WebFAQ, use:

```console
webfaq evaluate <model_name> webfaq
```

### Hybrid Retrieval

In our setting, hybrid retrieval evaluates the combined retrieval results of BM25 and an in-domain pre-trained
XLM-RoBERTa model. The following command evaluates the hybrid setting:

```console
webfaq bm25 evaluate-hybrid <path/to/bm25/index/folders/> <path/to/bm25/dataset/> <path/to/save/results/> webfaq
```

As the script evaluates hybrid retrieval performance across all languages in the dataset, the BM25 index and dataset
folders are expected to contain subfolders with the respective language tag, e.g. `eng`, which contain the
language-specific files.

## License

This repository is licensed under the MIT License. See the LICENSE file for details.
