#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"
cd "$PROJECT_ROOT"
./.venv/bin/python -m src.pipeline.test