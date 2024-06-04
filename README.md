# llm_fundamentals

This repository contains research and experiments for AI / ML / CS / LLM topics.

# Setup

This project was developed against Python 3.12.2.

First, create a virtual environment:

    $ python3 -m venv ./.venv

Then, install the required packages:

    $ ./.venv/bin/pip install -r requirements.txt

macOS Users: You'll need to do a few things first.

* `brew install gfortran openblas`
* `export OPENBLAS="$(brew --prefix openblas)"`

# Usage

The `cache.py` script can be used to download and cache the models and datasets used by the project.

    $ ./.venv/bin/python3 cache.py -h

Once everything is cached, you can set some environment flags to ensure everything runs in offline mode.

    $ export HF_DATASETS_OFFLINE=1
    $ export TRANSFORMERS_OFFLINE=1

Each of the scripts has some usage documentation and help text to get you started, but generally work with defaults:

    $ ./.venv/bin/python3 coach1.py

# Tips

The tools `htop` and `nvtop` are pretty handy.

    $ brew install nvtop htop
