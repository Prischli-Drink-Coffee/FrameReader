#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PROJECT_ROOT}/.venv/lib/python3.9/site-packages/torch/lib"
cd "$PROJECT_ROOT"

check_and_install_ytdlp() {
    if command -v yt-dlp &> /dev/null; then
        echo "yt-dlp is already installed"
        return 0
    fi
    
    echo "yt-dlp not found. Installing..."
    
    if command -v pip &> /dev/null; then
        pip install yt-dlp
    elif [ -f "./.venv/bin/pip" ]; then
        ./.venv/bin/pip install yt-dlp
    else
        echo "Error: pip not found. Please install yt-dlp manually"
        exit 1
    fi
    
    if command -v yt-dlp &> /dev/null; then
        echo "yt-dlp installed successfully"
    else
        echo "Error: Failed to install yt-dlp"
        exit 1
    fi
}

check_and_install_ytdlp
mkdir -p logs
./.venv/bin/python -m src.pipeline.server