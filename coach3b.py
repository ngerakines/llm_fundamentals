"""Query a trained question answering model.

Scenario:

    You are the assistant coach of a basketball team. The head coach is
    frustrated by the number of interuptions that they receive from players,
    marketing, and operations. They have instructed you to create a better way
    of answering repetative questions that allows them to focus on their work.

    Using transformers, you will create a model using an existing data set of
    questions and answers that the head coach has provided you. You will also
    extend it with your own questions and answers as they come in.

"""

__usage__ = """
examples:

    ./.venv/bin/python coach3b.py

"""


# FROM https://huggingface.co/docs/transformers/en/tasks/question_answering

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
from transformers import pipeline


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
        help="The directory containing the trained model. (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="1500",
        help="The checkpoint to use. (default: %(default)s)",
    )
    parser.add_argument(
        "question",
        nargs="?",
        type=str,
        default="What was the name of the astrologer that visited Gautama's father?",
        help="The question to ask. (default: %(default)s)",
    )
    parser.add_argument(
        "context",
        nargs="?",
        type=str,
        default="According to this narrative, shortly after the birth of young prince Gautama, an astrologer named Asita visited the young prince's father, Suddhodana, and prophesied that Siddhartha would either become a great king or renounce the material world to become a holy man, depending on whether he saw what life was like outside the palace walls.",
        help="The context of the question (default: %(default)s)",
    )

    args = parser.parse_args()

    question_answerer = pipeline(
        "question-answering", model=f"./{args.directory}/checkpoint-{args.checkpoint}/"
    )

    print(question_answerer(question=args.question, context=args.context))
