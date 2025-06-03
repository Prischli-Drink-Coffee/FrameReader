import os
from typing import Optional, Dict, List, Any
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
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException, File, UploadFile, status
from fastapi.openapi.models import Tag as OpenApiTag
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class AppConfig:
    TRITON_URL = "localhost:8080"
    YOLO_BATCH_SIZE = 16
    YOLO_IMAGE_SIZE = (640, 640)
    DONUT_BATCH_SIZE = 1
    DONUT_IMAGE_SIZE = (384, 384)


class ImageProcessor:
    @staticmethod
    def resize_image(image: Image.Image, target_size: tuple) -> np.ndarray:
        resized = image.resize(target_size)
        return np.array(resized, dtype=np.uint8)
    
    @staticmethod
    def create_padding_image(size: tuple) -> np.ndarray:
        return np.zeros((*size, 3), dtype=np.uint8)
    
    @classmethod
    def create_batches(cls, images: List[np.ndarray], batch_size: int, image_size: tuple) -> List[np.ndarray]:
        batches = []
        padding_image = cls.create_padding_image(image_size)
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            padding_needed = batch_size - len(batch)
            
            if padding_needed > 0:
                batch.extend([padding_image] * padding_needed)
            
            batches.append(np.stack(batch))
        
        return batches


class TritonResponseExtractor:
    @staticmethod
    def extract_string_result(response, output_name: str) -> str:
        try:

            raw_data = response.as_numpy(output_name)
            
            if raw_data.size == 0:
                raise ValueError("Empty response from Triton")
            
            result_item = raw_data.flatten()[0]
            
            if isinstance(result_item, (bytes, np.bytes_)):
                decoded = result_item.decode('utf-8')
            elif hasattr(result_item, 'decode'):
                decoded = result_item.decode('utf-8')
            else:
                decoded = str(result_item)
            
            decoded = decoded.strip()
            if not decoded:
                raise ValueError("Empty decoded response")
                
            return decoded
                
        except (UnicodeDecodeError, AttributeError) as e:
            raise ValueError(f"Cannot decode response from {output_name}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to extract result from {output_name}: {e}")


class BaseInferenceService:
    def __init__(self, triton_client: http_client.InferenceServerClient, model_name: str):
        self._client = triton_client
        self._model_name = model_name
        self._extractor = TritonResponseExtractor()

    def _get_model_outputs(self) -> List[str]:
        try:
            metadata = self._client.get_model_metadata(model_name=self._model_name)
            return [output['name'] for output in metadata['outputs']]
        except Exception as e:
            logger.error(f"Failed to get model metadata for {self._model_name}: {e}")
            return []

    def _create_input_tensor(self, batch_data: np.ndarray, input_name: str = "image"):
        input_tensor = http_client.InferInput(
            input_name,
            batch_data.shape,
            np_to_triton_dtype(batch_data.dtype)
        )
        input_tensor.set_data_from_numpy(batch_data)
        return input_tensor

    def _infer_single_batch(self, batch_data: np.ndarray) -> Dict[str, Any]:
        try:
            logger.info(f"Starting inference for batch shape: {batch_data.shape}")
            
            input_tensor = self._create_input_tensor(batch_data)
            output_names = self._get_model_outputs()
            
            if not output_names:
                return {"status": "error", "error": f"Model {self._model_name} has no outputs defined"}
            
            output_tensors = [http_client.InferRequestedOutput(name) for name in output_names]
            
            response = self._client.infer(
                model_name=self._model_name,
                inputs=[input_tensor],
                outputs=output_tensors
            )

            result = response.get_response()
            logger.info(f"response.get_response(): {result}")

            result_str = self._extractor.extract_string_result(response, output_names[0])
            logger.info(f"Raw response from {self._model_name}: {result_str[:200]}...")
            
            if not result_str.strip():
                return {"status": "error", "error": "Empty response from model"}
            
            try:
                parsed_result = json.loads(result_str)
                logger.info(f"Successfully parsed JSON with {len(parsed_result)} items")
                return {"status": "success", "result": parsed_result}
                
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing failed for {self._model_name}. Error: {json_err}")
                logger.error(f"Raw result (first 500 chars): {result_str[:500]}")
                return {"status": "error", "error": f"Invalid JSON response: {json_err}"}
            
        except Exception as e:
            logger.error(f"Inference error for {self._model_name}: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}


class YOLOInferenceService(BaseInferenceService):
    def __init__(self, triton_client: http_client.InferenceServerClient):
        super().__init__(triton_client, "yolo")

    def process_image_batches(self, batches: List[np.ndarray], actual_count: int) -> List[Dict[str, Any]]:
        all_results = []
        
        for batch_idx, batch_data in enumerate(batches):
            logger.info(f"Processing YOLO batch {batch_idx + 1}/{len(batches)}")
            
            result = self._infer_single_batch(batch_data)
            
            if result["status"] == "success":
                batch_start = batch_idx * AppConfig.YOLO_BATCH_SIZE
                batch_end = min(batch_start + AppConfig.YOLO_BATCH_SIZE, actual_count)
                batch_size = batch_end - batch_start
                
                batch_results = result["result"]
                if not isinstance(batch_results, list):
                    logger.warning(f"Expected list result, got {type(batch_results)}")
                    batch_results = [batch_results]
                
                valid_results = batch_results[:batch_size]
                all_results.extend(valid_results)
                
                logger.info(f"Successfully processed {len(valid_results)} images in batch {batch_idx + 1}")
            else:
                logger.error(f"Batch {batch_idx + 1} failed: {result}")
                batch_start = batch_idx * AppConfig.YOLO_BATCH_SIZE
                batch_end = min(batch_start + AppConfig.YOLO_BATCH_SIZE, actual_count)
                batch_size = batch_end - batch_start
                
                for _ in range(batch_size):
                    all_results.append({
                        "status": "error", 
                        "error": f"Batch processing failed: {result.get('error', 'Unknown error')}"
                    })
        
        return all_results


class DonutInferenceService(BaseInferenceService):
    def __init__(self, triton_client: http_client.InferenceServerClient):
        super().__init__(triton_client, "donut")

    def process_image_batches(self, batches: List[np.ndarray], actual_count: int) -> List[Dict[str, Any]]:
        all_results = []
        
        for batch_idx, batch_data in enumerate(batches):
            result = self._infer_single_batch(batch_data)
            
            if result["status"] == "success":
                batch_start = batch_idx * AppConfig.DONUT_BATCH_SIZE
                batch_end = min(batch_start + AppConfig.DONUT_BATCH_SIZE, actual_count)
                batch_size = batch_end - batch_start
                
                batch_results = result["result"] if isinstance(result["result"], list) else [result["result"]]
                all_results.extend(batch_results[:batch_size])
            else:
                all_results.append(result)
        
        return all_results


class ImageValidator:
    @staticmethod
    def validate_content_type(image_file: UploadFile) -> None:
        if not image_file.content_type or not image_file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {image_file.filename}"
            )

    @staticmethod
    def validate_image_content(contents: bytes, filename: str) -> Image.Image:
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Empty image file: {filename}"
            )
        
        try:
            img = Image.open(io.BytesIO(contents))
            
            if img.width == 0 or img.height == 0:
                raise ValueError(f"Invalid image dimensions for {filename}")
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return img
            
        except Exception as img_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process image {filename}: {str(img_error)}"
            )


class ResultValidator:
    @staticmethod
    def validate_yolo_result(result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        
        required_keys = ["boxes", "confidences", "classes"]
        for key in required_keys:
            if key not in result:
                return False
            if not isinstance(result[key], list):
                return False
        
        # Все массивы должны быть одинаковой длины
        if result["boxes"] and result["confidences"] and result["classes"]:
            boxes_len = len(result["boxes"])
            if len(result["confidences"]) != boxes_len or len(result["classes"]) != boxes_len:
                return False
        
        return True
    
    @staticmethod
    def sanitize_yolo_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized = []
        
        for i, result in enumerate(results):
            if ResultValidator.validate_yolo_result(result):
                sanitized.append(result)
            else:
                logger.warning(f"Invalid YOLO result at index {i}: {result}")
                sanitized.append({
                    "boxes": [],
                    "confidences": [],
                    "classes": [],
                    "status": "invalid_format"
                })
        
        return sanitized


app = FastAPI(
    title="OCR API",
    version="1.3.2",
    description="OCR API with Triton Server integration for YOLO and Donut inference"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MainTag = OpenApiTag(name="Main", description="CRUD operations main")
app.openapi_tags = [MainTag.model_dump()]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for {request.url}: {exc.errors()}")
    
    safe_errors = []
    for error in exc.errors():
        safe_error = {
            "loc": error.get("loc", []),
            "msg": error.get("msg", "Unknown error"),
            "type": error.get("type", "unknown")
        }
        if "input" in error and not isinstance(error["input"], bytes):
            safe_error["input"] = error["input"]
        safe_errors.append(safe_error)
    
    return JSONResponse(
        status_code=422,
        content={"detail": safe_errors}
    )


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
        self._triton_client = http_client.InferenceServerClient(url=AppConfig.TRITON_URL)
        self._validate_triton_connection()
        self._yolo_service = YOLOInferenceService(self._triton_client)
        self._donut_service = DonutInferenceService(self._triton_client)
        self._image_validator = ImageValidator()
        self._result_validator = ResultValidator()

    def _validate_triton_connection(self) -> None:
        try:
            if not self._triton_client.is_server_live():
                logger.error("Triton Server не доступен")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Triton Server недоступен"
                )
            
            for model_name in ["yolo", "donut"]:
                if not self._triton_client.is_model_ready(model_name):
                    logger.error(f"{model_name} модель не готова")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"{model_name} модель недоступна"
                    )
            
            logger.info("Triton Server готов к работе")
            
        except Exception as e:
            logger.error(f"Ошибка подключения к Triton: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ошибка Triton Server: {e}"
            )

    async def _process_uploaded_images(self, images: List[UploadFile], target_size: tuple) -> List[np.ndarray]:
        if not images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Не предоставлены изображения"
            )
        
        image_arrays = []
        
        for image_file in images:
            self._image_validator.validate_content_type(image_file)
            contents = await image_file.read()
            img = self._image_validator.validate_image_content(contents, image_file.filename)
            processed_image = ImageProcessor.resize_image(img, target_size)
            image_arrays.append(processed_image)
        
        return image_arrays

    @app.post("/generate/yolo", tags=["Main"])
    async def generate_yolo(self, images: List[UploadFile] = File(...)) -> Dict[str, Any]:
        try:
            logger.info(f"Processing {len(images)} images for YOLO inference")
            
            image_arrays = await self._process_uploaded_images(images, AppConfig.YOLO_IMAGE_SIZE)
            batches = ImageProcessor.create_batches(
                image_arrays, 
                AppConfig.YOLO_BATCH_SIZE, 
                AppConfig.YOLO_IMAGE_SIZE
            )
            
            logger.info(f"Created {len(batches)} batches for processing")
            
            detections = self._yolo_service.process_image_batches(batches, len(image_arrays))
            clean_detections = self._result_validator.sanitize_yolo_results(detections)
            successful_count = sum(1 for d in clean_detections if d.get("status") != "error")
            
            return {
                "status": "success",
                "model": "yolo",
                "processed_images": len(image_arrays),
                "successful_detections": successful_count,
                "results": clean_detections
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка в YOLO inference: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Внутренняя ошибка сервера: {e}"
            )

    @app.post("/generate/donut", tags=["Main"])
    async def generate_donut(self, images: List[UploadFile] = File(...)) -> Dict[str, Any]:
        try:
            image_arrays = await self._process_uploaded_images(images, AppConfig.DONUT_IMAGE_SIZE)
            batches = ImageProcessor.create_batches(
                image_arrays,
                AppConfig.DONUT_BATCH_SIZE,
                AppConfig.DONUT_IMAGE_SIZE
            )
            ocr_results = self._donut_service.process_image_batches(batches, len(image_arrays))
            
            return {
                "status": "success",
                "model": "donut",
                "processed_images": len(image_arrays),
                "results": ocr_results
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка в Donut inference: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Внутренняя ошибка сервера: {e}"
            )


def deployment(_args):
    return TritonDeployment.bind()


def configure_logging(log_dir: Optional[str] = None) -> logging.Logger:
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


def configure_ray_logging() -> None:
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


def test_models(test_image_path: str) -> None:
    if not os.path.exists(test_image_path):
        logger.info(f"Test image not found: {test_image_path}")
        return
    
    test_endpoints = [
        ("http://localhost:8000/generate/yolo", "YOLO"),
        ("http://localhost:8000/generate/donut", "Donut")
    ]
    
    for endpoint, model_name in test_endpoints:
        with open(test_image_path, "rb") as f:
            files = [("images", (os.path.basename(test_image_path), f.read(), "image/jpeg"))]
        
        logger.info(f"Testing {model_name} model:")
        try:
            response = requests.post(endpoint, files=files)
            logger.info(f"{model_name} Response Status: {response.status_code}")
            logger.info(f"{model_name} Response Body: {response.json()}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to FastAPI app for {model_name} test")
        except Exception as e:
            logger.error(f"Error during {model_name} test: {e}")


if __name__ == "__main__":
    configure_ray_logging()
    os.environ["RAY_BACKEND_LOG_LEVEL"] = "info"
    sys.stdout.flush()
    sys.stderr.flush()
    
    logger.info("Запуск Ray Serve deployment")
    serve.run(TritonDeployment.bind(), route_prefix="/")
    
    test_models("/workspace/docs/test.jpg")