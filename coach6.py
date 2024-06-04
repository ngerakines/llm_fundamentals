"""Use an named-entity recognition model to identify key people, places, and things in messages.

Scenario:

    You are an assistant coach of a basketball team. The coach was impressed
    with your ability to categorize and filter messages during games. They want
    you to create an improved system for identifying key people, places, and
    things.

    Using span_maker, identify key people, places, and things in messages.

"""

__usage__ = """
examples:

    ./.venv/bin/python coach6.py "Yeah, can you get me 2 tacos?"

    ./.venv/bin/python coach6.py "The Pelicans aren't going to know what hits them at this rate."

    ./.venv/bin/python coach6.py "Can you zoom in on the player's shoes?"
"""

# FROM https://huggingface.co/docs/hub/en/span_marker
# See Also: https://github.com/tomaarsen/SpanMarkerNER

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from span_marker import SpanMarkerModel


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
        default="tomaarsen/span-marker-xlm-roberta-base-fewnerd-fine-super",
        help="The NER model. (default: %(default)s)",
    )
    parser.add_argument(
        "content",
        nargs="?",
        type=str,
        default="Michael Jordan is the greatest basketball player of all time.",
        help="The message to process (default: %(default)s)",
    )

    args = parser.parse_args()

    model = SpanMarkerModel.from_pretrained(args.model)
    entities = model.predict(args.content)

    if len(entities) == 0:
        print("no entities found")

    for entity in entities:
        print("%.3f" % (entity["score"]), entity["label"], entity["span"])
