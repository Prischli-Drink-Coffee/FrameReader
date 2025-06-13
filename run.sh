#!/bin/bash

set -e

echo "--- Running FrameReader Backend ---"

IMAGE_NAME="framereader-backend:latest"

# Define environment variables (replace with your actual values or load from .env)

# Example:
# DB_HOST="host.docker.internal" # Use this if running Docker on Linux/Windows and DB is on host
# DB_HOST="localhost" # Use this if DB is running directly on the same host as Docker
# DB_HOST="your_db_container_name" # Use this if DB is in another Docker container in the same network
# DB_PORT="3306"
# DB="your_database_name"
# DB_USER="your_db_user"
# DB_PASSWORD="your_db_password"
# HOST="0.0.0.0"
# SERVER_PORT="8000"
# DEBUG="TRUE" # or "FALSE"
# TRITON_API_URL="http://host.docker.internal:8000" # Or your Triton HTTP URL
# TRITON_WS_URL="ws://host.docker.internal:8000"   # Or your Triton WebSocket URL

# Example of how to pass environment variables (uncomment and set your values)
# ENV_VARS="-e DB_HOST=localhost -e DB_PORT=3306 -e DB=framereader_db -e DB_USER=user -e DB_PASSWORD=password"
# ENV_VARS+=" -e HOST=0.0.0.0 -e SERVER_PORT=8000 -e DEBUG=TRUE"
# ENV_VARS+=" -e TRITON_API_URL=http://localhost:8000 -e TRITON_WS_URL=ws://localhost:8000"

# You might need to adjust the network and port mapping based on your setup.
# -p 8000:8000 maps host port 8000 to container port 8000
# --network host might be needed if you want the container to use the host's network stack
# --add-host host.docker.internal:host-gateway is useful for Docker Desktop on Linux/Windows to access host services

echo "Running Docker container for backend..."
docker run --rm -it \
  -p 8005:8005 \
  --name framereader-backend-app \
  $ENV_VARS \
  $IMAGE_NAME

echo "--- Backend Container Stopped ---"