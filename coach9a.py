"""Query an llm through ollama.

Scenario:

    You are an assistant coach of a basketball team. Your coach has tasked you
    with providing helpful and thoughtful answers to questions about the team's
    star, Michael Jordan, during press conferences.

    Using chromadb and llama3, create a database of facts that can be queried
    and integrated into a prompt-based model to generate responses to questions
    about Michael Jordan.

"""

__usage__ = """
examples:

    ./.venv/bin/python coach9a.py

"""


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import argparse
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=__usage__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "-m",
        "--embedding-model",
        type=str,
        default="llama3",
        help="The model to be used for encoding the query and documents. (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--collection",
        type=str,
        default="michael_jordan_facts",
        help="The vector database collection. (default: %(default)s)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="team_facts.db",
        help="The vector database. (default: %(default)s)",
    )
    args = parser.parse_args()

    ollama_emb = OllamaEmbeddings(
        model=args.embedding_model,
    )

    chromadb_client = Chroma(
        persist_directory=f"{args.db}",
        collection_name=args.collection,
        embedding_function=ollama_emb,
    )

    ids, texts = zip(
        *[
            ("mj1", "Michael Jordan is a retired professional basketball player."),
            (
                "mj2",
                "In 1984 Jordan, a guard standing 6 feet 6 inches (1.98 meters), was drafted by the Chicago Bulls. He quickly became known as an exceptionally talented shooter and passer and a tenacious defender. In his first season (1984-85), he led the league in scoring and was named Rookie of the Year; after missing most of the following season with a broken foot, he returned to lead the NBA in scoring for seven consecutive seasons, averaging about 33 points per game. He was only the second player (after Wilt Chamberlain) to score 3,000 points in a single season (1986-87). Jordan was named the NBA's Most Valuable Player (MVP) five times (1988, 1991, 1992, 1996, 1998) and was also named Defensive Player of the Year in 1988.",
            ),
            (
                "mj3",
                "Jordan grew up in Wilmington, North Carolina, and entered the University of North Carolina at Chapel Hill in 1981. As a freshman, he made the winning basket against Georgetown in the 1982 national championship game. Jordan was named College Player of the Year in both his sophomore and junior years, leaving North Carolina after his junior year. He led the U.S. basketball team to Olympic gold medals in 1984 in Los Angeles and in 1992 in Barcelona, Spain. The players who competed in the latter Games became known as the Dream Team.",
            ),
            (
                "mj4",
                "In October 1993, after leading the Bulls to their third consecutive championship, Jordan retired briefly and pursued a career in professional baseball. He returned to basketball in March 1995. In the 1995-96 season Jordan led the Bulls to a 72-10 regular season record, the best in the history of the NBA (broken in 2015-16 by the Golden State Warriors). From 1996 to 1998 the Jordan-led Bulls again won three championships in a row, and each time Jordan was named MVP of the NBA finals. After the 1997-98 season Jordan retired again.",
            ),
            (
                "mj5",
                'During this time Jordan earned the nickname "Air Jordan" because of his extraordinary leaping ability and acrobatic maneuvers, and his popularity reached heights few athletes (or celebrities of any sort) have known. He accumulated millions of dollars from endorsements, most notably for his Nike Air Jordan basketball shoes.',
            ),
            (
                "mj6",
                "Jordan remained close to the sport, buying a share of the Washington Wizards in January 2000. He was also appointed president of basketball operations for the club. However, managing rosters and salary caps was not enough for Jordan, and in September 2001 he renounced his ownership and management positions with the Wizards in order to be a player on the team. His second return to the NBA was greeted with enthusiasm by the league, which had suffered declining attendance and television ratings since his 1998 retirement. After the 2002-03 season, Jordan announced his final retirement. He ended his career with 32,292 total points and a 30.1-points-per-game average, which was the best in league history at that time, as well as 2,514 steals, then the second most ever.",
            ),
            (
                "mj7",
                "In 2006 Jordan became minority owner and general manager of the NBA's Charlotte Bobcats (now known as the Charlotte Hornets). He bought a controlling interest in the team in 2010 and became the first former NBA player to become a majority owner of one of the league's franchises. Jordan sold his share in 2023.",
            ),
            (
                "mj8",
                "Jordan made a successful film, Space Jam (1996), in which he starred with animated characters Bugs Bunny and Daffy Duck. In 1996 the NBA named him one of the 50 greatest players of all time, and in 2009 he was elected to the Naismith Memorial Basketball Hall of Fame. He was awarded the Presidential Medal of Freedom in 2016.",
            ),
        ]
    )

    chromadb_client.add_texts(texts=list(texts), ids=list(ids))
