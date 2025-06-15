#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PROJECT_ROOT}/.venv/lib/python3.9/site-packages/torch/lib"
cd "$PROJECT_ROOT"
./.venv/bin/python -m src.utils.create_sql