#!/bin/bash

export PROJECT_PATH="$(pwd)"
export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"
export PYTHON="$(pyenv which python 3.10.6)"
source ./.venv/bin/activate
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/.venv/lib/python3.10/site-packages/torch/lib"
uv run --project $PROJECT_PATH --active ./train.py --config ./configs/exp1 --data-dir ./dataset
