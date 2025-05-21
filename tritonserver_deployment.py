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
        self._triton_client = http_client.InferenceServerClient(url="localhost:8080")

        try:
            if not self._triton_client.is_server_live():
                logger.error("ВНИМАНИЕ: Triton Server должен быть запущен отдельно")
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                    detail="Triton Server is not live.")
            else:
                logger.info("Triton Server доступен.")
        except Exception as e:
            logger.error(f"Ошибка при подключении к Triton Server: {e}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail=f"Error connecting to Triton Server: {e}")

        try:
            if not self._triton_client.is_model_ready("donut"):
                logger.error("Donut model is not ready on Triton Server.")
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                    detail="Donut model is not ready on Triton Server.")
            if not self._triton_client.is_model_ready("yolo"):
                logger.error("YOLO model is not ready on Triton Server.")
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                    detail="YOLO model is not ready on Triton Server.")
        except Exception as error:
            logger.error(f"Error checking model readiness on Triton Server: {error}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail=f"Error checking model readiness on Triton Server: {error}")

        _print_heading("Triton Server Ready")
        _print_heading("Models Status (as reported by client)")
        try:
            pprint(self._triton_client.get_model_repository_index())
        except Exception as e:
            logger.error(f"Error getting model repository index: {e}")


    @app.post("/generate/yolo", tags=["Main"])
    async def generate_yolo(self, images: List[UploadFile] = File(...)) -> Dict:
        try:
            if not images:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail="No images provided"
                )
            
            YOLO_HEIGHT = 640
            YOLO_WIDTH = 640
            BATCH_SIZE = 16
            
            image_arrays = []
            image_names = []
            
            for image_file in images:
                contents = await image_file.read()
                
                if not contents:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Empty image file: {image_file.filename}"
                    )
                    
                try:
                    img = Image.open(io.BytesIO(contents))
                    
                    if img.width == 0 or img.height == 0:
                        raise ValueError(f"Invalid image dimensions for {image_file.filename}")
                        
                    img_resized = img.resize((YOLO_WIDTH, YOLO_HEIGHT))
                    img_array = np.array(img_resized)
                    
                    logger.info(f"Resized image {image_file.filename} to YOLO format: {img_array.shape}")
                    
                    if img_array.size == 0:
                        raise ValueError(f"Empty image array for {image_file.filename}")
                        
                    image_arrays.append(img_array)
                    image_names.append(image_file.filename)
                    
                except Exception as img_error:
                    logger.error(f"Error processing image {image_file.filename}: {img_error}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to process image {image_file.filename}: {str(img_error)}"
                    )
            
            if not image_arrays:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid images provided for processing"
                )
                
            black_image = np.zeros((YOLO_HEIGHT, YOLO_WIDTH, 3), dtype=np.uint8)
            
            all_results = []
            num_images = len(image_arrays)
            num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, num_images)
                actual_batch_size = end_idx - start_idx
                
                logger.info(f"Processing YOLO batch {batch_idx+1}/{num_batches} with {actual_batch_size} actual images")
                
                batch_arrays = []
                
                for i in range(start_idx, end_idx):
                    batch_arrays.append(image_arrays[i])
                
                padding_needed = BATCH_SIZE - len(batch_arrays)
                if padding_needed > 0:
                    logger.info(f"Adding {padding_needed} black images to complete batch of {BATCH_SIZE}")
                    batch_arrays.extend([black_image.copy() for _ in range(padding_needed)])
                    
                batch_data = np.stack(batch_arrays)
                logger.info(f"YOLO batch shape: {batch_data.shape}, dtype: {batch_data.dtype}")

                if batch_data.shape != (BATCH_SIZE, YOLO_HEIGHT, YOLO_WIDTH, 3):
                    error_msg = f"Invalid batch shape: {batch_data.shape}, expected: ({BATCH_SIZE}, {YOLO_HEIGHT}, {YOLO_WIDTH}, 3)"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                    
                input_image = http_client.InferInput(
                    "image", 
                    batch_data.shape, 
                    np_to_triton_dtype(batch_data.dtype)
                )
                input_image.set_data_from_numpy(batch_data)
                
                model_metadata = self._triton_client.get_model_metadata(model_name="yolo")
                output_names = [output['name'] for output in model_metadata['outputs']]
                logger.info(f"Model outputs available: {output_names}")
                
                if not output_names:
                    logger.error("Model has no defined outputs")
                    all_results.extend([{"error": "Model has no outputs defined"} for _ in range(actual_batch_size)])
                    continue
                
                output_name = output_names[0]
                logger.info(f"Using output name: {output_name}")
                
                output_result = http_client.InferRequestedOutput(output_name)

                try:
                    response = self._triton_client.infer(
                        model_name="yolo", 
                        inputs=[input_image], 
                        outputs=[output_result]
                    )
                    logger.info(f"Output response: {response.get_response()}")
                    
                    output_metadata = next(
                        (out for out in response.get_response()['outputs'] 
                        if out['name'] == output_name),
                        None
                    )
                    logger.info(f"Output output_metadata: {output_metadata}")

                    if not output_metadata:
                        logger.warning(f"No output named '{output_name}' in response")
                        all_results.append({"error": "No model output"})
                        continue
                    
                    try:
                        raw_results = response.as_numpy(output_name)
                        
                        if isinstance(raw_results, np.ndarray) and raw_results.size == 0:
                            logger.warning(f"Empty numpy array received")
                            all_results.append({"error": "Empty numpy array received"})
                            continue
                        
                        if raw_results is not None:
                            try:
                                if raw_results.dtype == np.bytes_:
                                    decoded = raw_results[0].item().decode('utf-8')
                                    logger.warning(f"Decoded output: {decoded}")
                                    parsed_result = json.loads(decoded)
                                    all_results.extend(parsed_result)
                                else:
                                    all_results.append(raw_results.tolist())
                            except json.JSONDecodeError:
                                logger.error("Failed to decode JSON output")
                                all_results.append({"error": "Failed to decode JSON output"})
                            except Exception as e:
                                logger.error(f"Output processing error: {str(e)}")
                                all_results.append({"error": str(e)})
                        else:
                            logger.warning("Raw results is None")
                            all_results.append({"error": "Raw results is None"})
                    
                    except ValueError as e:
                        if "cannot reshape array of size 0" in str(e):
                            logger.warning("Received empty array, returning empty result")
                            all_results.append({"error": "Received empty array, returning empty result"})
                        else:
                            logger.error(f"ValueError: {str(e)}")
                            all_results.append({"error": str(e)})
                
                except Exception as e:
                    logger.error(f"Inference failed: {str(e)}")
                    all_results.append({"error": str(e)})

            return {
                "status": "success", 
                "model": "yolo", 
                "detections": all_results[:num_images]
            }
        
        except Exception as e:
            logger.error(f"Error in YOLO inference: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
     

    @app.post("/generate/donut", tags=["Main"])
    async def generate_donut(self, images: List[UploadFile] = File(...)) -> Dict:
        try:
            if not images:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No images provided"
                )

            DONUT_HEIGHT = 384
            DONUT_WIDTH = 384
            BATCH_SIZE = 1
            
            image_arrays = []
            image_names = []
            
            for image_file in images:
                contents = await image_file.read()
                
                if not contents:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Empty image file: {image_file.filename}"
                    )
                    
                try:
                    img = Image.open(io.BytesIO(contents))

                    if img.width == 0 or img.height == 0:
                        raise ValueError(f"Invalid image dimensions for {image_file.filename}")

                    img_resized = img.resize((DONUT_WIDTH, DONUT_HEIGHT))
                    img_array = np.array(img_resized)
                    
                    logger.info(f"Resized image {image_file.filename} to Donut format: {img_array.shape}")
                    
                    if img_array.size == 0:
                        raise ValueError(f"Empty image array for {image_file.filename}")
                    
                    image_arrays.append(img_array)
                    image_names.append(image_file.filename)
                    
                except Exception as img_error:
                    logger.error(f"Error processing image {image_file.filename}: {img_error}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to process image {image_file.filename}: {str(img_error)}"
                    )
            
            if not image_arrays:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid images provided for processing"
                )
                
            black_image = np.zeros((DONUT_HEIGHT, DONUT_WIDTH, 3), dtype=np.uint8)
            
            all_results = []
            num_images = len(image_arrays)
            num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, num_images)
                actual_batch_size = end_idx - start_idx
                
                logger.info(f"Processing Donut batch {batch_idx+1}/{num_batches} with {actual_batch_size} actual images")
                
                batch_arrays = []
                
                for i in range(start_idx, end_idx):
                    batch_arrays.append(image_arrays[i])
                
                padding_needed = BATCH_SIZE - len(batch_arrays)
                if padding_needed > 0:
                    logger.info(f"Adding {padding_needed} placeholder images to complete Donut batch of {BATCH_SIZE}")
                    batch_arrays.extend([black_image.copy() for _ in range(padding_needed)])
                
                batch_data = np.stack(batch_arrays)
                
                if batch_data.shape != (BATCH_SIZE, DONUT_HEIGHT, DONUT_WIDTH, 3):
                    error_msg = f"Invalid batch shape: {batch_data.shape}, expected: ({BATCH_SIZE}, {DONUT_HEIGHT}, {DONUT_WIDTH}, 3)"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                    
                logger.info(f"Donut batch shape: {batch_data.shape}, dtype: {batch_data.dtype}")
                
                input_image = http_client.InferInput(
                    "image", 
                    batch_data.shape, 
                    np_to_triton_dtype(batch_data.dtype)
                )
                input_image.set_data_from_numpy(batch_data)
                
                model_metadata = self._triton_client.get_model_metadata(model_name="donut")
                output_names = [output['name'] for output in model_metadata['outputs']]
                logger.info(f"Model outputs available: {output_names}")
                
                if not output_names:
                    logger.error("Model has no defined outputs")
                    all_results.extend([{"error": "Model has no outputs defined"} for _ in range(actual_batch_size)])
                    continue
                
                output_name = output_names[0]
                logger.info(f"Using output name: {output_name}")
                
                output_result = http_client.InferRequestedOutput(output_name)

                try:
                    response = self._triton_client.infer(
                        model_name="donut", 
                        inputs=[input_image], 
                        outputs=[output_result]
                    )
                    logger.info(f"Output response: {response.get_response()}")
                    
                    output_metadata = next(
                        (out for out in response.get_response()['outputs'] 
                        if out['name'] == output_name),
                        None
                    )
                    logger.info(f"Output output_metadata: {output_metadata}")

                    if not output_metadata:
                        logger.warning(f"No output named '{output_name}' in response")
                        all_results.append({"error": "No model output"})
                        continue
                    
                    try:
                        raw_results = response.as_numpy(output_name)
                        
                        if isinstance(raw_results, np.ndarray) and raw_results.size == 0:
                            logger.warning(f"Empty numpy array received")
                            all_results.append({"error": "Empty numpy array received"})
                            continue
                        
                        if raw_results is not None:
                            try:
                                if raw_results.dtype == np.bytes_:
                                    decoded = raw_results[0].item().decode('utf-8')
                                    logger.warning(f"Decoded output: {decoded}")
                                    parsed_result = json.loads(decoded)
                                    all_results.extend(parsed_result)
                                else:
                                    all_results.append(raw_results.tolist())
                            except json.JSONDecodeError:
                                logger.error("Failed to decode JSON output")
                                all_results.append({"error": "Failed to decode JSON output"})
                            except Exception as e:
                                logger.error(f"Output processing error: {str(e)}")
                                all_results.append({"error": str(e)})
                        else:
                            logger.warning("Raw results is None")
                            all_results.append({"error": "Raw results is None"})
                    
                    except ValueError as e:
                        if "cannot reshape array of size 0" in str(e):
                            logger.warning("Received empty array, returning empty result")
                            all_results.append({"error": "Received empty array, returning empty result"})
                        else:
                            logger.error(f"ValueError: {str(e)}")
                            all_results.append({"error": str(e)})
                
                except Exception as e:
                    logger.error(f"Inference failed: {str(e)}")
                    all_results.append({"error": str(e)})
            
            return {
                "status": "success", 
                "model": "donut", 
                "ocr_results": all_results[:num_images]
            }
            
        except Exception as e:
            logger.error(f"Error in Donut inference: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=str(e)
            )


def deployment(_args):
    return TritonDeployment.bind()


def configure_logging(log_dir: str = None) -> logging.Logger:
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    script_name = Path(sys.argv[0]).stem
    log_file = os.path.join(log_dir, f"{script_name}.log")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)
    app_logger = logging.getLogger('triton_app')
    app_logger.info(f"Логирование настроено. Файл логов: {log_file}")
    return app_logger


def configure_ray_logging():
    try:
        from ray.serve._private.constants import SERVE_LOGGER_NAME
        serve_logger = logging.getLogger(SERVE_LOGGER_NAME)
        serve_logger.setLevel(logging.INFO)
        serve_logger.propagate = True
        ray_logger = logging.getLogger("ray")
        ray_logger.setLevel(logging.INFO)
        ray_logger.propagate = True
        logger.info("Ray Serve логирование настроено")
    except ImportError:
        logger.error("Не удалось импортировать Ray Serve для настройки логирования")


if __name__ == "__main__":
    
    configure_ray_logging()
    os.environ["RAY_BACKEND_LOG_LEVEL"] = "info"
    sys.stdout.flush()
    sys.stderr.flush()
    logger.info("Запуск Ray Serve deployment")
    serve.run(TritonDeployment.bind(), route_prefix="/")
    
    test_image_path = "/workspace/docs/test.jpg"
    test_image_paths = [test_image_path, test_image_path]
    valid_test_images = [path for path in test_image_paths if os.path.exists(path)]

    if valid_test_images:
        multipart_files_yolo = []
        for i, path in enumerate(valid_test_images):
            with open(path, "rb") as f:
                multipart_files_yolo.append(("images", (os.path.basename(path), f.read(), "image/jpeg"))) 
        
        logger.info(f"Testing YOLO model with {len(valid_test_images)} images:")
        try:
            yolo_response = requests.post(
                "http://localhost:8000/generate/yolo",
                files=multipart_files_yolo
            )
            logger.info(f"YOLO Response Status: {yolo_response.status_code}")
            logger.info(f"YOLO Response Body: {yolo_response.json()}")
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"Could not connect to FastAPI app for YOLO test: {ce}. Is Ray Serve running?")
        except Exception as e:
            logger.error(f"An error occurred during YOLO test: {e}")

        for file_tuple in multipart_files_yolo:
            file_tuple[1].seek(0)
         
        multipart_files_donut = []
        for i, path in enumerate(valid_test_images):
            with open(path, "rb") as f:
                multipart_files_donut.append(("images", (os.path.basename(path), f.read(), "image/jpeg")))

        logger.info(f"\nTesting Donut model with {len(valid_test_images)} images:")
        try:
            donut_response = requests.post(
                "http://localhost:8000/generate/donut",
                files=multipart_files_donut
            )
            logger.info(f"Donut Response Status: {donut_response.status_code}")
            logger.info(f"Donut Response Body: {donut_response.json()}")
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"Could not connect to FastAPI app for Donut test: {ce}. Is Ray Serve running?")
        except Exception as e:
            logger.error(f"An error occurred during Donut test: {e}")
    else:
        logger.info(f"No test images found at specified paths: {test_image_paths}. Please provide valid images for testing.")
