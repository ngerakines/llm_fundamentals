"""Train a model for generating summaries of large bodies of text.

Scenario:

    You are the assistant coach of a basketball team. The legal team was very
    impressed with your question and answer tool and wants you to create a
    tool that determines if any proposed bills are coming through that would
    impact the team.

    Using transformers, you create a model that has been trainied on
    legistlation. Using it, you can summarize incoming legislation so a legal
    team member can review it quickly.

Notes:

    This takes about 50 minutes to run (Nov 2023 MacBook Pro, M3 Max, 64gb).

"""

__usage__ = """
examples:

    Run and measure the model training:

        date && time ./.venv/bin/python coach4a.py

"""


# FROM https://huggingface.co/docs/transformers/en/tasks/summarization

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import numpy as np


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=__usage__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "--directory",
        type=str,
        default="team_legal",
        help="The directory that the trained model will be placed. (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="billsum",
        help="The dataset used to train the model. (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train[:10000]",
        help="The split argument to the dataset loader. (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google-t5/t5-small",
        help="The checkpoint used to base the trained model off of. (default: %(default)s)",
    )

    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.dataset_split)
    dataset = dataset.train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prefix = "summarize: "

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model)

    rouge = evaluate.load("rouge")

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.directory,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        save_total_limit=3,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
