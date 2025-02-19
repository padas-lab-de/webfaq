import click
from webfaq.cli.extract import extract
from webfaq.cli.merge import merge
from webfaq.cli.evaluate import evaluate
from webfaq.cli.embed_labse import embed_labse
from webfaq.cli.embed_jina import embed_jina
from webfaq.cli.near_duplicates import near_duplicates
from webfaq.cli.offsets import offsets
from webfaq.cli.statistics import statistics
from webfaq.cli.topic_classification.tc_annotate import tc_annotate
from webfaq.cli.question_type_classification.qtc_annotate import qtc_annotate
from webfaq.cli.parallel_corpus.pc_candidates import pc_candidates
from webfaq.cli.parallel_corpus.pc_score import pc_score
from webfaq.cli.parallel_corpus.pc_assess import pc_assess
from webfaq.cli.parallel_corpus.pc_transform import pc_transform
from webfaq.cli.fine_tune.ft_sample import ft_sample
from webfaq.cli.fine_tune.ft_run import ft_run
from webfaq.cli.fine_tune.ft_mteb import ft_mteb
from webfaq.cli.bm25.generate_jsonl import bm25_generate_jsonl
from webfaq.cli.bm25.bm25_evaluate import bm25_evaluate
from webfaq.cli.bm25.bm25_evaluate_hybrid import bm25_evaluate_hybrid


@click.group()
def main():
    """
    WebFAQ

    Creating a Q&A dataset from FAQ-style web page annotations.
    """
    pass


@click.group()
def topic_classification():
    """
    Subgroup for the training of a multilingual topic classification model on Q&A pairs.
    """
    pass


@click.group()
def question_type_classification():
    """
    Subgroup for the training of a multilingual question type classification model on FAQ-style questions.
    """
    pass


@click.group()
def parallel_corpus():
    """
    Subgroup for the creation of a parallel corpus from Q&A pairs.
    """
    pass


@click.group()
def fine_tune():
    """
    Subgroup for fine-tuning a model on the WebFAQ dataset.
    """
    pass


@click.group()
def bm25():
    """
    Subgroup for evaluating BM25 on the WebFAQ dataset.
    """
    pass


main.add_command(extract)
main.add_command(merge)
main.add_command(evaluate)
main.add_command(embed_labse)
main.add_command(embed_jina)
main.add_command(near_duplicates)
main.add_command(offsets)
main.add_command(statistics)
main.add_command(topic_classification)
main.add_command(question_type_classification)
main.add_command(parallel_corpus)
main.add_command(fine_tune)
main.add_command(bm25)

topic_classification.add_command(tc_annotate, "annotate")

question_type_classification.add_command(qtc_annotate, "annotate")

parallel_corpus.add_command(pc_candidates, "candidates")
parallel_corpus.add_command(pc_score, "score")
parallel_corpus.add_command(pc_assess, "assess")
parallel_corpus.add_command(pc_transform, "transform")

fine_tune.add_command(ft_sample, "sample")
fine_tune.add_command(ft_run, "run")
fine_tune.add_command(ft_mteb, "mteb")

bm25.add_command(bm25_generate_jsonl, "generate-jsonl")
bm25.add_command(bm25_evaluate, "evaluate")
bm25.add_command(bm25_evaluate_hybrid, "evaluate-hybrid")


if __name__ == "__main__":
    main()