"""Train a model for question answering using transformers.

Scenario:

    You are the assistant coach of a basketball team. The head coach is
    frustrated by the number of interuptions that they receive from players,
    marketing, and operations. They have instructed you to create a better way
    of answering repetative questions that allows them to focus on their work.

    Using transformers, you will create a model using an existing data set of
    questions and answers that the head coach has provided you. You will also
    extend it with your own questions and answers as they come in.

Notes:

    This takes about 15 minutes to run (Nov 2023 MacBook Pro, M3 Max, 64gb).

"""

__usage__ = """
examples:

    ./.venv/bin/python coach3a.py

"""

# FROM https://huggingface.co/docs/transformers/en/tasks/question_answering

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DefaultDataCollator,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)


def preprocess_function(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


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
        default="team_knowledge_base",
        help="The directory that the trained model will be placed. (default: %(default)s)",
    )

    args = parser.parse_args()

    # https://huggingface.co/datasets/rajpurkar/squad

    squad = load_dataset("squad", split="train[:10100]")
    squad = squad.train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    tokenized_squad = squad.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=squad["train"].column_names,
    )

    data_collator = DefaultDataCollator()

    model = AutoModelForQuestionAnswering.from_pretrained(
        "distilbert/distilbert-base-uncased"
    )

    training_args = TrainingArguments(
        output_dir=args.directory,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
