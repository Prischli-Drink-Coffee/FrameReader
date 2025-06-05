#!/bin/bash

if [ -f "./.venv/bin/activate" ]; then
    deactivate 2>/dev/null
fi

export PROJECT_PATH="$(pwd)/src"
export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/.venv/lib/python3.9/site-packages/torch/lib"

uv run ./src/pipeline/server.py