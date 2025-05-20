import os
from pprint import pprint
from typing import Optional, Dict, List
import base64
import io
import sys
import numpy as np
import requests
import torch
import tritonclient.http as http_client
from tritonclient.utils import np_to_triton_dtype
from PIL import Image
from ray import serve
import logging

from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile, status, Form, Query
from fastapi.openapi.models import Tag as OpenApiTag
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


app = FastAPI(title="OCR API", version="1.3.2",
              description="This API triton server is intended for the FrameReader project. For rights, contact the service owner (dfvolkhin@edu.hse.ru).")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MainTag = OpenApiTag(name="Main", description="CRUD operations main")

app.openapi_tags = [
    MainTag.model_dump()
]

S3_BUCKET_URL = None

if "S3_BUCKET_URL" in os.environ:
    S3_BUCKET_URL = os.environ["S3_BUCKET_URL"]


def _print_heading(message):
    print("")
    print(message)
    print("-" * len(message))



@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "max_ongoing_requests": 1,
        "target_ongoing_requests": 1,
        "upscale_delay_s": 2,
        "downscale_delay_s": 120,
        "upscaling_factor": 1,
        "downscaling_factor": 1,
        "metrics_interval_s": 2,
        "look_back_period_s": 4,
    },
)
@serve.ingress(app)
class TritonDeployment:
    def __init__(self):
        self._triton_client = http_client.InferenceServerClient(url="localhost:8000")

        try:
            if not self._triton_client.is_server_live():
                logger.warning("ВНИМАНИЕ: Triton Server должен быть запущен отдельно")
            else:
                logger.warning("Triton Server доступен.")
        except Exception as e:
            logger.error(f"Ошибка при подключении к Triton Server: {e}")
            logger.warning("ВНИМАНИЕ: Убедитесь, что Triton Server запущен на localhost:8000")

        try:
            if not self._triton_client.is_model_ready("donut") or not self._triton_client.is_model_ready("yolo"):
                logger.error("YOLO or Donut models are not ready on Triton Server.")
        except Exception as error:
            logger.error(f"Error checking model readiness on Triton Server: {error}")
            return

        _print_heading("Triton Server Ready")
        _print_heading("Models Status (as reported by client)")
        try:
            pprint(self._triton_client.get_model_repository_index())
        except Exception as e:
            logger.error(f"Error getting model repository index: {e}")


    @app.post("/generate/yolo", tags=["Main"])
    async def generate_yolo(self, images: List[UploadFile] = File(...)) -> Dict:
        try:
            inputs = []
            image_arrays = []
            
            for image_file in images:
                contents = await image_file.read()
                img = Image.open(io.BytesIO(contents))
                image_arrays.append(np.array(img).astype(np.int8)) 
            
            if not image_arrays:
                raise ValueError("No images provided for processing.")

            batch_data = np.stack(image_arrays)
        
            input_image = http_client.InferInput("image", batch_data.shape, np_to_triton_dtype(batch_data.dtype))
            input_image.set_data_from_numpy(batch_data)
            inputs.append(input_image)

            output_result = http_client.InferRequestedOutput("result") 
            response = self._triton_client.infer(model_name="yolo", inputs=inputs, outputs=[output_result])
            
            result = {"status": "success", "model": "yolo"}

            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @app.post("/generate/donut", tags=["Main"])
    async def generate_donut(self, images: List[UploadFile] = File(...)) -> Dict:
        try:
            inputs = []
            image_arrays = []

            for image_file in images:
                contents = await image_file.read()
                img = Image.open(io.BytesIO(contents))
                image_arrays.append(np.array(img).astype(np.float16))

            if not image_arrays:
                raise ValueError("No images provided for processing.")
                
            batch_data = np.stack(image_arrays)

            input_image = http_client.InferInput("image", batch_data.shape, np_to_triton_dtype(batch_data.dtype))
            input_image.set_data_from_numpy(batch_data)
            inputs.append(input_image)

            output_text = http_client.InferRequestedOutput("text_sequence")
            response = self._triton_client.infer(model_name="donut", inputs=inputs, outputs=[output_text])
            
            result = {"status": "success", "model": "donut"}

            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}


def deployment(_args):
    return TritonDeployment.bind()


if __name__ == "__main__":
    serve.run(TritonDeployment.bind(), route_prefix="/")
    
    test_image_path = "/workspace/docs/test.jpg"

    test_image_paths = [test_image_path, test_image_path]

    valid_test_images = [path for path in test_image_paths if os.path.exists(path)]

    if valid_test_images:
        multipart_files = []
        for i, path in enumerate(valid_test_images):
            with open(path, "rb") as f:
                multipart_files.append(("images", (os.path.basename(path), f.read(), "image/jpeg"))) 
        
        logger.info(f"Testing YOLO model with {len(valid_test_images)} images:")

        yolo_response = requests.post(
            "http://localhost:8000/generate/yolo",
            files=multipart_files
        )
        logger.info(yolo_response.json())

        for file_tuple in multipart_files:
            file_tuple[1].seek(0)
        
        multipart_files_donut = []
        for i, path in enumerate(valid_test_images):
            with open(path, "rb") as f:
                multipart_files_donut.append(("images", (os.path.basename(path), f.read(), "image/jpeg")))

        logger.info(f"\nTesting Donut model with {len(valid_test_images)} images:")
        donut_response = requests.post(
            "http://localhost:8000/generate/donut",
            files=multipart_files_donut
        )
        logger.info(donut_response.json())
    else:
        logger.info(f"No test images found at specified paths: {test_image_paths}. Please provide valid images for testing.")

