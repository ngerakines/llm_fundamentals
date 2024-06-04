"""An simple example of sentence_transformers.

Scenario:

    You are the coach of a basketball team and you are watching what the
    oposing team is doing. Using sentence_transformers, you are comparing the
    instructions that the coach of the oposing team is giving to their players.
"""

__usage__ = """
examples:

    ./.venv/bin/python coach1.py "Pass the ball to John when he is near the basket."

    ./.venv/bin/python coach1.py -vv 'Take the shot!'

"""

# FROM https://huggingface.co/docs/hub/en/sentence-transformers

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
from sentence_transformers import SentenceTransformer, util

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
        default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        help="The model to be used for encoding the query and documents. (default: %(default)s)",
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        default="Pass the ball to John when he is near the basket.",
        help="The query to be processed. (default: %(default)s)",
    )
    args = parser.parse_args()

    docs = [
        "Pass the ball to someone else.",
        "Take a 2 point shot",
        "Take a 3 point shot",
    ]

    # Load the model
    model = SentenceTransformer(args.model)

    # Encode query and documents
    query_emb = model.encode(args.query)
    doc_emb = model.encode(docs)
    if args.verbose > 1:
        print(query_emb)
        print(doc_emb)

    # Compute dot score between query and all document embeddings
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    # Combine docs & scores
    doc_score_pairs = list(zip(docs, scores))

    # Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    for doc, score in doc_score_pairs:
        print(score, doc)
