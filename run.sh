#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"

echo "--- Running Stack ---"

if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r key value; do
        if [[ -n $key && $key != '#'* ]]; then
            key=$(echo "$key" | tr -d '\r')
            value=$(echo "$value" | tr -d '\r')
            export "$key"="$value"
        fi
    done < "$ENV_FILE"
else
    echo "WARNING: .env file not found, using defaults"
fi

cleanup() {
    echo "--- Stopping Stack ---"
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    echo "--- Stack Stopped ---"
}

trap cleanup EXIT INT TERM

if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: docker-compose не установлен"
    exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "ERROR: Файл docker-compose.yml не найден: $COMPOSE_FILE"
    exit 1
fi

echo "Start up services..."
docker-compose -f "$COMPOSE_FILE" up framereader-backend-app

echo "--- Backend Container Stopped ---"