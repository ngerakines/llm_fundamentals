"""Use Sentence Transformers to categorize game play calls.

Scenario:

    You are the coach of a basketball team. You want to collect more
    information about what plays are called and when. You've instructed your
    assistant to write down all of the calls that are made during the game and
    then categorize them. Using sentence_transformers, create buckets of
    typical game play calls and catagorize incoming game play calls against
    them.
"""

__usage__ = """
examples:

    ./.venv/bin/python coach2.py "make a 3 point shot"

    ./.venv/bin/python coach2.py "Dribble the ball."

    ./.venv/bin/python coach2.py "Give the ball to Jon"

    ./.venv/bin/python coach2.py -m sentence-transformers/sentence-t5-xxl "Give the ball to Jon"

    ./.venv/bin/python coach2.py -m sentence-transformers/sentence-t5-xxl "Dribble the ball."

    ./.venv/bin/python coach2.py -m sentence-transformers/all-MiniLM-L6-v2 "Dribble the ball."

"""

# FROM https://huggingface.co/docs/hub/en/sentence-transformers

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
from typing import List, Set, Dict, Tuple
from sentence_transformers import SentenceTransformer, util


def main(query: str, model_name: str):
    model = SentenceTransformer(model_name)

    library = {
        "TRAVEL": set(["Dribble the ball to the other side of the court."]),
        "PASS": set(
            [
                "Pass the ball to your teammate.",
                "Pass the ball to the player on your left.",
                "Pass the ball to the player on your right.",
            ]
        ),
        "SHOOT": set(
            [
                "Shoot the ball into the hoop.",
                "Shoot the ball into the basket.",
                "Atempt a 3 point shot.",
                "Attempt a 2 point shot.",
                "Attempt a free throw.",
                "Attempt to dunk the ball into the hoop.",
            ]
        ),
    }
    print(get_intent(query, library, model))


def collect_docs(library: Dict[str, Set[str]]) -> List[str]:
    docs = []
    for _, phrases in library.items():
        docs.extend(phrases)
    return docs


def get_intent(
    query: str, library: Dict[str, Set[str]], model: SentenceTransformer
) -> Set[Tuple[float, str, str]]:
    docs = collect_docs(library)

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    top_score = max(scores)

    doc_score_pairs = list(zip(docs, scores))

    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    results: Set[Tuple[float, str, str]] = set()

    for doc, score in doc_score_pairs:
        if score >= top_score:
            for intent, phrases in library.items():
                if doc in phrases:
                    results.add((score, intent, doc))
        else:
            break

    return results

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
    parser.add_argument("query", nargs="?", type=str, help="The query to be processed. (default: %(default)s)")

    args = parser.parse_args()
    main(args.query, args.model)
