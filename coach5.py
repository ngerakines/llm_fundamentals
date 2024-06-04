"""Use a categorization model to identify the topic of short messages.

Scenario:

    You are an assistant coach of a basketball team. You've been asked to help
    relay messages from different sources during the game. Messages vary by
    type and length, but also relevance. The head coach does not want you to
    relay messages from commentators, anaylists, or players that are not
    relevant to the game.

    Using BERTopic, you are categorizing messages as they are received. If they
    are not on topic, they will be disregarded.

"""

__usage__ = """
examples:

    ./.venv/bin/python coach5.py "Yeah, can you get me 2 tacos?"

    ./.venv/bin/python coach5.py "The Pelicans aren't going to know what hits them at this rate."

    ./.venv/bin/python coach5.py "Can you zoom in on the player's shoes?"
"""

# FROM https://huggingface.co/docs/hub/en/bertopic
# See Also: https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


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
        default="MaartenGr/BERTopic_Wikipedia",
        help="The model. (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="The embedding model. (default: %(default)s)",
    )
    parser.add_argument(
        "content",
        nargs="?",
        type=str,
        default="Michael Jordan is the greatest basketball player of all time.",
        help="The message to categorize (default: %(default)s)",
    )

    args = parser.parse_args()

    embedding_model = SentenceTransformer(args.embedding_model)

    topic_model = BERTopic.load(args.model, embedding_model=embedding_model)

    topic, prob = topic_model.transform(args.content)

    topic_pairs = list(zip(topic, prob))

    for topic, prob in topic_pairs:
        topic_label = topic_model.topic_labels_.get(topic, "unknown")
        print(f"{topic_label} {prob}")
