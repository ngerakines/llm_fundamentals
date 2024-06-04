"""Query an llm through ollama.

Scenario:

    You are an assistant coach of a basketball team. Your coach has tasked you
    with providing helpful and concise short answers to questions during game time.

    Using llama3, query the model with a question and get a short one or two word
    answer that you can quickly give the coach.

"""

__usage__ = """
examples:

    ./.venv/bin/python coach8.py "What is capital of America?"

    ./.venv/bin/python coach8.py "Who is Michael Jordan"

"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


import argparse
from langchain_community.llms import Ollama
from langchain import PromptTemplate
from langchain_core._api.deprecation import suppress_langchain_deprecation_warning


def get_model_response(user_prompt, system_prompt):
    # NOTE: No f string and no whitespace in curly braces
    template = """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

    # Added prompt template
    prompt = PromptTemplate(
        input_variables=["system_prompt", "user_prompt"], template=template
    )

    # Modified invoking the model
    response = llm(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt))

    return response


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
        default="llama3",
        help="The model to be used for encoding the query and documents. (default: %(default)s)",
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        default="What is capital of America?",
        help="The query to process. (default: %(default)s)",
    )
    args = parser.parse_args()

    supported_models = ["llama3", "llama3:8b", "llama3:70b"]
    if args.model not in supported_models:
        raise ValueError(f"Unsupported model. Choose one of: {supported_models}")

    llm = Ollama(model=args.model, stop=["<|eot_id|>"])

    system_prompt = "Give a one or two word answers only."

    with suppress_langchain_deprecation_warning():
        print(get_model_response(args.query, system_prompt))
