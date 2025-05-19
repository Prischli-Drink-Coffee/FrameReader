echo "Building models..."

MODEL_DIR="/workspace/models"

mkdir -p "${MODEL_DIR}/yolo/1/engine-batch-size-16"
mkdir -p "${MODEL_DIR}/donut/1/pytorch-batch-size-1"
mkdir -p "${MODEL_DIR}/donut/1/pytorch-batch-size-1/checkpoint"

# Здесь можно добавить логику для загрузки моделей с Hugging Face или других источников
# Например:
# python3 -c "
# from huggingface_hub import hf_hub_download
# import os
# 
# HF_TOKEN = os.environ.get('HF_TOKEN')
# 
# # Для YOLOv8
# yolo_path = hf_hub_download(repo_id='ultralytics/yolov8', filename='yolov8s.pt', token=HF_TOKEN)
# os.system(f'cp {yolo_path} /workspace/models/yolo/1/pytorch-batch-size-1/yolo_int8.engine')
# 
# # Для Donut
# donut_path = hf_hub_download(repo_id='naver-clova-ix/donut-base', filename='pytorch_model.bin', token=HF_TOKEN)
# os.system(f'cp {donut_path} /workspace/models/donut/1/pytorch-batch-size-1/checkpoint/')
# "

SOURCE_DIR=$(dirname "$(readlink -f "$0")")

find /opt/tritonserver/python -maxdepth 1 -type f -name \
     "tritonserver-*.whl" | xargs -I {} pip3 install --upgrade {}[all]

python3 $SOURCE_DIR/build_models.py "$@"

echo "Models built successfully"
exit 0
