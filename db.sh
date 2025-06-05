#!/bin/bash

# Get the absolute path of the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set Python path to include the src directory
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

# Set LD_LIBRARY_PATH if needed for torch
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PROJECT_ROOT}/.venv/lib/python3.9/site-packages/torch/lib"

# Change to project directory
cd "$PROJECT_ROOT"

# Run the server using the virtual environment's Python
./.venv/bin/python -m src.utils.create_sql