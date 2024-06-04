"""Query a summary generation model.

Scenario:

    You are the assistant coach of a basketball team. The legal team was very
    impressed with your question and answer tool and wants you to create a
    tool that determines if any proposed bills are coming through that would
    impact the team.

    Using transformers, you create a model that has been trainied on
    legistlation. Using it, you can summarize incoming legislation so a legal
    team member can review it quickly.

"""

__usage__ = """
examples:

    Query the model:

        ./.venv/bin/python coach4b.py

"""

# FROM https://huggingface.co/docs/transformers/en/tasks/summarization

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
from transformers import pipeline

default_content = 'Marine and Hydrokinetic Renewable Energy Promotion Act of 2011 - Amends the Energy Independence and Security Act of 2007 to require the program of marine and hydrokinetic renewable energy technology research, development, demonstration, and commercial application to: (1) apply advanced systems engineering and system integration methods to identify critical interfaces and develop open standards for marine and hydrokinetic renewable energy; (2) transfer the resulting environmental data to industry stakeholders as public information through published interface definitions, standards, and demonstration projects; and (3) develop incentives for industry to comply with such standards.\n\nRequires the Secretary of Energy (DOE) to award competitive grants to support modifying or constructing four or more geographically dispersed marine and hydrokinetic renewable energy technology research, development, and demonstration test facilities for the demonstration of multiple technologies in actual operating environments. Requires the Secretary to give preference to existing facilities and National Marine Renewable Energy Research, Development, and Demonstration Centers. Renames such Centers as the "National Marine and Hydrokinetic Renewable Energy Research, Development, and Demonstration Centers" and expands their research and clearinghouse duties to include hydrokinetic as well as marine renewable energy research. Authorizes such Centers to serve as technology test facilities. Requires the Secretary to establish a marine-based energy device verification program to provide a bridge from the marine and hydrokinetic renewable energy capture device design and development efforts underway across the industry to commercial deployment of such devices. Requires the Secretary to establish a grant program to: (1) advance the development of marine and hydrokinetic renewable energy; (2) help fund the costs of environmental analysis affecting the deployment of marine hydrokinetic devices; (3) help eligible entities to collect the types of environmental data that are required when working in a public resource, monitor the impacts of demonstration projects, and make the resulting information available for dissemination to aid future projects; and (4) help fund the cost of advancing renewable marine and hydrokinetic technologies in ocean and riverine environments from demonstration projects to development and deployment. Authorizes appropriations for marine and hydrokinetic renewable energy technologies through FY2013.'

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
        default="team_legal",
        help="The directory containing the trained model. (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="2000",
        help="The checkpoint to use. (default: %(default)s)",
    )
    parser.add_argument(
        "content",
        nargs="?",
        type=str,
        default=default_content,
        help="The content to summarize (default: %(default)s)",
    )

    args = parser.parse_args()

    content = args.content
    if not content.startswith("summarize: "):
        content = "summarize: " + content

    summarizer = pipeline(
        "summarization", model=f"./{args.directory}/checkpoint-{args.checkpoint}/"
    )
    result = summarizer(content)
    result = next(iter(result), {})
    print(result.get("summary_text", "no summary found"))
