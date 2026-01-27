import json
import os
from typing import *

LANGUAGES_100_ORIGINS = [
    "ara",
    "aze",
    "ben",
    "bul",
    "cat",
    "ces",
    "dan",
    "deu",
    "ell",
    "eng",
    "est",
    "fas",
    "fin",
    "fra",
    "heb",
    "hin",
    "hrv",
    "hun",
    "ind",
    "isl",
    "ita",
    "jpn",
    "kat",
    "kaz",
    "kor",
    "lav",
    "lit",
    "mar",
    "msa",
    "nld",
    "nor",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "slv",
    "spa",
    "sqi",
    "srp",
    "swe",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzb",
    "vie",
    "zho",
]
QUESTION_TYPES = [
    "What",
    "When",
    "Where",
    "Which",
    "Who/Whom/Whose",
    "Why",
    "How",
    "Is, are, do, does",
    "Can, could, will, would, may, might, shall, should",
    "No Question Word",
]
TOPICS = [
    "Products and Commercial Services",
    "Traveling and Hospitality",
    "Healthcare Services, Wellness and Lifestyle",
    "Entertainment, Recreation and Leisure",
    "Employment, Education and Training",
    "Banking, Financial Services and Insurance",
    "Legal Services, Regulations and Government",
    "General Information and Other",
]

DATASETS_FOLDER = "datasets"
MODELS_FOLDER = "models"
OWI_DATASETS_FOLDER = os.path.expanduser("~/.owi/public/main/")
RESOURCES_FOLDER = "resources"
TEMP_FOLDER = "temp"

CLDR_MAPPING = json.loads(open(os.path.join(RESOURCES_FOLDER, "iso3166_to_iso639_2_cldr.json")).read())
LANG_CODE_ORIGIN_MAPPING = json.loads(open(os.path.join(RESOURCES_FOLDER, "language_code_origin_mapping.json")).read())