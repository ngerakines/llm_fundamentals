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

    ./.venv/bin/python coach9b.py

"""


import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate


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
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        default="What is Michael Jordan known for?",
        help="The message to process (default: %(default)s)",
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
    retriever = chromadb_client.as_retriever(search_kwargs={"k": 5})

    llm = Ollama(model="llama3", stop=["<|eot_id|>"])

    template = """
You are a helpful AI assistant.
Answer based on the context provided. 
context: {context}
input: {input}
answer:
"""
    prompt = PromptTemplate.from_template(template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    response = retrieval_chain.invoke({"input": args.query})

    if args.verbose > 1:
        print(response)
    else:
        print(response["answer"])
