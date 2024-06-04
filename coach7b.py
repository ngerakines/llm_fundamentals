"""Create a simple embeddings search database using using transformers.

Scenario:

    You are an assistant coach of a basketball team. The head coach has tasked
    you with doing more involved work on the relatedness of content across
    different sources. They want you to take all of the question and answer
    content, legal content, and real-time game-play messages and help and
    create search tools for them all.

    Using sentence_transformers, create a simple search tool for the head
    coach.

"""

__usage__ = """
examples:

    ./.venv/bin/python coach7b.py "Who is Michael Jordan"

"""

# FROM https://huggingface.co/blog/getting-started-with-embeddings

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import json
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=__usage__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="The model. (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--docs",
        type=str,
        default="team_messages.json",
        help="The file containing documents. (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--embeddings",
        type=str,
        default="team_messages_embeddings.csv",
        help="The file to write embeddings to. (default: %(default)s)",
    )
    parser.add_argument(
        "content",
        nargs="?",
        type=str,
        default="Did you see him slam dunk?",
        help="The message to process (default: %(default)s)",
    )

    args = parser.parse_args()

    model = SentenceTransformer(args.model)

    with open(args.docs, "r") as team_messages_file:
        docs = json.load(team_messages_file)

    docs_df = load_dataset("csv", data_files=args.embeddings)
    docs_torched = torch.from_numpy(docs_df["train"].to_pandas().to_numpy()).to(
        torch.float
    )

    content_embeddings = model.encode(args.content)

    content_torch = torch.FloatTensor(content_embeddings)

    hits = semantic_search(content_torch, docs_torched, top_k=5)

    if (
        len(hits) > 0
        and len(hits[0]) > 0
        and len(hits[0][0]) > 0
        and hits[0][0]["score"] > 0.2
    ):
        top_hit = hits[0][0]
        print("%.4f" % (top_hit["score"]), docs[top_hit["corpus_id"]])
    else:
        print("no results")
