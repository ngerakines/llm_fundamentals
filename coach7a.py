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

    ./.venv/bin/python coach7a.py

"""

# FROM https://huggingface.co/blog/getting-started-with-embeddings

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import json
from sentence_transformers import SentenceTransformer
import pandas as pd

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

    args = parser.parse_args()

    docs = [
        "An adventure is an exciting experience or undertaking that is typically bold, sometimes risky. Adventures may be activities with danger such as traveling, exploring, skydiving, mountain climbing, scuba diving, river rafting, or other extreme sports. Adventures are often undertaken to create psychological arousal or in order to achieve a greater goal, such as the pursuit of knowledge that can only be obtained by such activities.",
        "Basketball is a team sport in which two teams, most commonly of five players each, opposing one another on a rectangular court, compete with the primary objective of shooting a basketball through the defender's hoop, while preventing the opposing team from shooting through their own hoop. A field goal is worth two points, unless made from behind the three-point line, when it is worth three. After a foul, timed play stops and the player fouled or designated to shoot a technical foul is given one, two or three one-point free throws. The team with the most points at the end of the game wins, but if regulation play expires with the score tied, an additional period of play is mandated.",
        'Michael Jeffrey Jordan, also known by his initials MJ, is an American businessman and former professional basketball player. He played fifteen seasons in the National Basketball Association between 1984 and 2003, winning six NBA championships with the Chicago Bulls. He was integral in popularizing basketball and the NBA around the world in the 1980s and 1990s, becoming a global cultural icon. His profile on the NBA website states, "By acclamation, Michael Jordan is the greatest basketball player of all time."',
        "The National Basketball Association is a professional basketball league in North America composed of 30 teams. It is one of the major professional sports leagues in the United States and Canada and is considered the premier professional basketball league in the world.",
        "The Chicago Bulls are an American professional basketball team based in Chicago. The Bulls compete in the National Basketball Association as a member of the Central Division of the Eastern Conference. The team was founded on January 16, 1966, and played its first game during the 1966-67 NBA season. The Bulls play their home games at the United Center, an arena on Chicago's West Side.",
        "In basketball, the basketball court is the playing surface, consisting of a rectangular floor, with baskets at each end. Indoor basketball courts are almost always made of polished wood, usually maple, with 10 foot high rims on each basket. Outdoor surfaces are generally made from standard paving materials such as concrete or asphalt. International competitions may use glass basketball courts.",
        "A backboard is a piece of basketball equipment. It is a raised vertical board with an attached basket consisting of a net suspended from a hoop. It is made of a flat, rigid piece of, often Plexiglas or tempered glass which also has the properties of safety glass when accidentally shattered. It is usually rectangular as used in NBA, NCAA and international basketball. In recreational environments, a backboard may be oval or a fan-shape, particularly in non-professional games.",
        "A layup in basketball is a two-point shot attempt made by leaping from below, laying the ball up near the basket, and using one hand to bounce it off the backboard and into the basket. The motion and one-handed reach distinguish it from a jump shot. The layup is considered the most basic shot in basketball. When doing a layup, the player lifts the outside foot, or the foot away from the basket.",
        'A slam dunk, also simply known as dunk, is a type of basketball shot that is performed when a player jumps in the air, controls the ball above the horizontal plane of the rim, and scores by shoving the ball directly through the basket with one or both hands. It is a type of field goal that is worth two points. Such a shot was known as a "dunk shot" until the term "slam dunk" was coined by former Los Angeles Lakers announcer Chick Hearn.',
        'In basketball, a field goal is a basket scored on any shot or tap other than a free throw, worth two or three points depending on the location of the attempt on the basket. Uncommonly, a field goal can be worth other values such as one point in FIBA 3x3 basketball competitions or four points in the BIG3 basketball league. "Field goal" is the official terminology used by the National Basketball Association in their rule book, in their box scores and statistics, and in referees\' rulings. The same term is also the official wording used by the National Collegiate Athletic Association and high school basketball.',
        "The finger roll is a specialized type of basketball layup shot where the ball is rolled off the tips of the player's fingers. The advantage of the finger roll is that the ball can travel high in the air over a defender that might otherwise block a regular jump shot or dunk, while the spin applied by the rolling over the fingers will carry the ball to the basket off the backboard. The shot was pioneered by center Wilt Chamberlain in the 1960s.",
        "A shot clock is a countdown timer used in a variety of games and sports, indicating a set amount of time that a team may possess the object of play before attempting to score a goal. Shot clocks are used in several sports including basketball, water polo, canoe polo, lacrosse, poker, ringette, korfball, tennis, ten-pin bowling, and various cue sports. It is analogous with the play clock used in American and Canadian football, and the pitch clock used in baseball. This article deals chiefly with the shot clock used in basketball.",
    ]

    model = SentenceTransformer(args.model)

    docs_embeddings = model.encode(docs)

    docs_df = pd.DataFrame(docs_embeddings)

    docs_df.to_csv(args.embeddings, index=False)

    json_object = json.dumps(docs, indent=4)

    with open(args.docs, "w") as outfile:
        outfile.write(json_object)
