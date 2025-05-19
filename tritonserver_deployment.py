import os
from pprint import pprint
from typing import Optional, Dict, List
import base64
import io

import numpy as np
import requests
import torch
import tritonclient.http as tritonserver
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from ray import serve

app = FastAPI()

S3_BUCKET_URL = None

if "S3_BUCKET_URL" in os.environ:
    S3_BUCKET_URL = os.environ["S3_BUCKET_URL"]


def _print_heading(message):
    print("")
    print(message)
    print("-" * len(message))


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=1,
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
class BaseDeployment:
    def __init__(self, use_torch_compile=False):
        self._image_size = 1024
        if use_torch_compile:
            print("compiling")
            print(torch._dynamo.list_backends())

    @app.post("/generate")
    async def generate(self, image: UploadFile = File(...), filename: Optional[str] = None) -> Dict:
        try:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            
            result = {
                "status": "success",
                "model": "base",
                "image_size": [img.width, img.height],
                "message": "BaseDeployment не реализует обработку изображений"
            }
            
            if filename:
                result["filename"] = filename
                
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=1,
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
        self._triton_server = tritonserver

        if S3_BUCKET_URL is not None:
            model_repository = S3_BUCKET_URL
        else:
            model_repository = [
                "/workspace/models"
            ]

        self._triton_server = tritonserver.Server(
            model_repository=model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
        )
        self._triton_server.start(wait_until_ready=True)

        _print_heading("Triton Server Started")
        _print_heading("Metadata")
        pprint(self._triton_server.metadata())
        self._yolo = None
        self._donut = None

        try:
            if not self._triton_server.model("donut").ready() or not self._triton_server.model("yolo").ready():
                self._yolo = self._triton_server.load("yolo")
                self._donut = self._triton_server.load("donut")

                if not self._yolo.ready() or not self._donut.ready():
                    raise Exception("Models not ready")
        except Exception as error:
            print("Error can't load yolo or donut models!")
            print(f"Please ensure dependencies are met {error}")
            return
        
        _print_heading("Models")
        pprint(self._triton_server.models())

    @app.post("/generate/yolo")
    async def generate_yolo(self, image: UploadFile = File(...), filename: Optional[str] = None) -> Dict:
        try:
            if not self._yolo or not self._yolo.ready():
                return {"status": "error", "message": "YOLO model not loaded or not ready"}
            
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            img_array = np.array(img)
            response = self._yolo.infer(inputs={"image": [img_array]})
            result = {"status": "success", "model": "yolo"}
            
            for output in response.outputs:
                result[output.name] = output.as_numpy().tolist() if output.as_numpy().size < 100 else "Large output"
            
            if filename:
                result["filename"] = filename
                
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @app.post("/generate/donut")
    async def generate_donut(self, image: UploadFile = File(...), filename: Optional[str] = None) -> Dict:
        try:
            if not self._donut or not self._donut.ready():
                return {"status": "error", "message": "Donut model not loaded or not ready"}
            
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            img_array = np.array(img)
            response = self._donut.infer(inputs={"image": [img_array]})
            result = {"status": "success", "model": "donut"}
            
            for output in response.outputs:
                if output.name == "text_sequence":
                    texts = []
                    for text_data in output.as_numpy():
                        texts.append(text_data.item().decode('utf-8'))
                    result[output.name] = texts
                else:
                    result[output.name] = output.as_numpy().tolist() if output.as_numpy().size < 100 else "Large output"
            
            if filename:
                result["filename"] = filename
                
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}


def deployment(_args):
    return TritonDeployment.bind()


def baseline(_args):
    if "use-torch-compile" in _args:
        return BaseDeployment.bind(use_torch_compile=True)
    else:
        return BaseDeployment.bind(use_torch_compile=False)


if __name__ == "__main__":
    serve.run(TritonDeployment.bind(), route_prefix="/")
    test_image_path = "/workspace/docs/test.jpg"
    
    if os.path.exists(test_image_path):
        with open(test_image_path, "rb") as f:
            files = {"image": ("test.jpg", f)}
            
            print("Testing YOLO model:")
            yolo_response = requests.post(
                "http://localhost:8000/generate/yolo",
                files=files
            )
            print(yolo_response.json())
            
            f.seek(0)
            print("\nTesting Donut model:")
            donut_response = requests.post(
                "http://localhost:8000/generate/donut",
                files=files
            )
            print(donut_response.json())
    else:
        print(f"Test image not found at {test_image_path}. Please provide a valid image for testing.")