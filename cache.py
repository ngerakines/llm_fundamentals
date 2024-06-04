"""
cache.py

This file will cache the models and datasets used in this project.

"""

__usage__ = """
examples:

    Basic usage:

        ./.venv/bin/python cache.py

    If you've previously set offline flags, unset:

        TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 ./.venv/bin/python cache.py

    Using the rust-backed download accelerator:

        HF_HUB_ENABLE_HF_TRANSFER=1 ./.venv/bin/python cache.py
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
from datasets import load_dataset
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=__usage__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    models = [
        "distilbert/distilbert-base-uncased",
        "google-t5/t5-small",
        "MaartenGr/BERTopic_Wikipedia",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/sentence-t5-xxl",
        "tomaarsen/span-marker-xlm-roberta-base-fewnerd-fine-super",
        "tomaarsen/span-marker-bert-base-fewnerd-fine-super",
        "bert-base-cased",
        "xlm-roberta-base",
    ]
    datasets = ["billsum", "squad"]

    for model in models:
        if args.verbose > 0:
            print(f"Downloading model: {model}")
        snapshot_download(repo_id=model)

    # BUG: datasets does not use the same cache as snapshot_download.
    # for dataset in datasets:
    #     if args.verbose > 0:
    #         print(f"Downloading dataset: {dataset}")
    #     snapshot_download(repo_id=dataset, repo_type="dataset")

    load_dataset("squad", split="train[:10100]")
    load_dataset("billsum", split="train[:10000]")
