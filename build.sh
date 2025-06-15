#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "--- Building FrameReader Backend ---"

echo "Running db.sh to initialize database..."
if [ -f "$PROJECT_ROOT/db.sh" ]; then
    bash "$PROJECT_ROOT/db.sh"
    echo "db.sh executed successfully."
else
    echo "Warning: db.sh not found at $PROJECT_ROOT/db.sh. Skipping database initialization."
fi

echo "Building Docker image for backend..."
docker build -t framereader-backend:latest -f "$PROJECT_ROOT/docker/Dockerfile" "$PROJECT_ROOT"

echo "Docker image framereader-backend:latest built successfully."
echo "--- Build Complete ---"