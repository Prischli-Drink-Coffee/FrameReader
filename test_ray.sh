#!/bin/bash

set -e

echo "Testing Triton Server with Ray Serve..."

TEST_IMAGE="/workspace/docs/test.jpg"

if [ ! -f "$TEST_IMAGE" ]; then
    echo "Creating test image..."
    mkdir -p /workspace/docs
    convert -size 512x512 canvas:white -font Arial -pointsize 40 -draw "text 50,250 'Hello Triton Server!'" $TEST_IMAGE
fi

echo "Testing YOLO endpoint..."
curl -X POST -F "image=@$TEST_IMAGE" http://localhost:8000/generate/yolo

echo -e "\n\nTesting Donut endpoint..."
curl -X POST -F "image=@$TEST_IMAGE" http://localhost:8000/generate/donut

echo -e "\n\nTests completed"