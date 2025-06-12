import asyncio
import json
import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from logging.handlers import RotatingFileHandler

import numpy as np
import torch
import tritonclient.http as http_client
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.models import Tag as OpenApiTag
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from PIL import Image
from ray import serve
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from tritonclient.utils import np_to_triton_dtype


sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    TRITON_URL: str = "localhost:8080"
    YOLO_BATCH_SIZE: int = 16
    YOLO_IMAGE_SIZE: Tuple[int, int] = (640, 640)
    DONUT_BATCH_SIZE: int = 1
    DONUT_IMAGE_SIZE: Tuple[int, int] = (384, 384)
    MAX_CHUNK_SIZE: int = 10
    STREAM_DELAY: float = 0.01


@dataclass
class StreamEvent:
    event: str
    data: Dict[str, Any]


@dataclass
class ProcessedImage:
    filename: str
    data: np.ndarray
    original_size: Tuple[int, int]


class ImageProcessor:
    @staticmethod
    def resize_image(image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
        resized = image.resize(target_size)
        return np.array(resized, dtype=np.uint8)

    @staticmethod
    def create_padding_image(size: Tuple[int, int]) -> np.ndarray:
        return np.zeros((*size, 3), dtype=np.uint8)

    @classmethod
    def create_batches(
        cls,
        images: List[np.ndarray],
        batch_size: int,
        image_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        batches = []
        padding_image = cls.create_padding_image(image_size)

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            padding_needed = batch_size - len(batch)

            if padding_needed > 0:
                batch.extend([padding_image] * padding_needed)

            batches.append(np.stack(batch))

        return batches


class FileManager:
    @staticmethod
    async def read_and_cache_upload_files(
        upload_files: List[UploadFile]
    ) -> List[Tuple[str, bytes]]:
        cached_files = []

        for upload_file in upload_files:
            try:
                # Ensure the file is not closed before reading
                if hasattr(upload_file.file, 'closed') and upload_file.file.closed:
                    raise ValueError(f"File {upload_file.filename} is already closed")

                contents = await upload_file.read()

                if not contents:
                    raise ValueError(f"Empty file content for {upload_file.filename}")

                filename = upload_file.filename or f"image_{len(cached_files)}.jpg"
                cached_files.append((filename, contents))

                # Attempt to seek back to the beginning for potential re-reads (though not strictly needed here)
                try:
                    if hasattr(upload_file, 'seek'):
                        await upload_file.seek(0)
                    elif hasattr(upload_file.file, 'seek'):
                        upload_file.file.seek(0)
                except Exception as seek_err:
                    logger.warning(f"Could not seek file {upload_file.filename}: {seek_err}")

            except Exception as e:
                logger.error(f"Error reading file {upload_file.filename}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to read file {upload_file.filename}: {e}"
                )

        return cached_files

    # Removed create_temp_files and cleanup_temp_files as they are no longer needed for WebSocket processing
    # and are generally inefficient for in-memory operations.
    # If they are used elsewhere for specific file-based operations, they should be re-evaluated.
    @staticmethod
    def cleanup_temp_files(temp_files: List[str]) -> None:
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")


class TritonResponseExtractor:
    @staticmethod
    def extract_string_result(response, output_name: str) -> str:
        try:
            raw_data = response.as_numpy(output_name)

            if raw_data is None or raw_data.size == 0:
                raise ValueError("Empty or None response from Triton")

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
            input_tensor = self._create_input_tensor(batch_data)
            output_names = self._get_model_outputs()

            if not output_names:
                return {
                    "status": "error",
                    "error": f"Model {self._model_name} has no outputs defined"
                }

            output_tensors = [http_client.InferRequestedOutput(name) for name in output_names]

            response = self._client.infer(
                model_name=self._model_name,
                inputs=[input_tensor],
                outputs=output_tensors
            )

            result_str = self._extractor.extract_string_result(response, output_names[0])

            if not result_str.strip():
                return {"status": "error", "error": "Empty response from model"}

            try:
                parsed_result = json.loads(result_str)
                return {"status": "success", "result": parsed_result}

            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing failed for {self._model_name}: {json_err}")
                return {"status": "error", "error": f"Invalid JSON response: {json_err}"}

        except Exception as e:
            logger.error(f"Inference error for {self._model_name}: {e}")
            return {"status": "error", "error": str(e)}


class YOLOInferenceService(BaseInferenceService):
    def __init__(self, triton_client: http_client.InferenceServerClient):
        super().__init__(triton_client, "yolo")

    def process_image_batches(self, batches: List[np.ndarray], actual_count: int) -> List[Dict[str, Any]]:
        all_results = []

        for batch_idx, batch_data in enumerate(batches):
            result = self._infer_single_batch(batch_data)

            if result["status"] == "success":
                batch_start = batch_idx * AppConfig.YOLO_BATCH_SIZE
                batch_end = min(batch_start + AppConfig.YOLO_BATCH_SIZE, actual_count)
                batch_size = batch_end - batch_start

                batch_results = result["result"]
                if not isinstance(batch_results, list):
                    batch_results = [batch_results]

                valid_results = batch_results[:batch_size]
                all_results.extend(valid_results)
            else:
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
            img = Image.open(BytesIO(contents))

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
                sanitized.append({
                    "boxes": [],
                    "confidences": [],
                    "classes": [],
                    "status": "invalid_format"
                })

        return sanitized


class StreamingProcessor:
    def __init__(self, triton_deployment):
        self._deployment = triton_deployment
        self._config = AppConfig()
        self._file_manager = FileManager() # Keep for other uses if any, but not for WebSocket temp files

    async def process_cached_images_stream(
        self,
        cached_files: List[Tuple[str, bytes]], # This is the correct input type
        model_type: str,
        chunk_size: int = 1
    ) -> AsyncGenerator[StreamEvent, None]:
        try:
            if model_type == "yolo":
                target_size = self._config.YOLO_IMAGE_SIZE
                batch_size = min(chunk_size, self._config.YOLO_BATCH_SIZE)
                service = self._deployment._yolo_service
                validator = self._deployment._result_validator.sanitize_yolo_results
            elif model_type == "donut":
                target_size = self._config.DONUT_IMAGE_SIZE
                batch_size = min(chunk_size, self._config.DONUT_BATCH_SIZE)
                service = self._deployment._donut_service
                validator = lambda x: x
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            yield StreamEvent(
                event="start",
                data={
                    "total_images": len(cached_files),
                    "model": model_type,
                    "batch_size": batch_size,
                    "chunk_size": chunk_size
                }
            )

            image_arrays = []
            for filename, contents in cached_files:
                try:
                    # Validate and process image directly from bytes
                    img = self._deployment._image_validator.validate_image_content(contents, filename)
                    processed_image = ImageProcessor.resize_image(img, target_size)
                    image_arrays.append(processed_image)
                except Exception as e:
                    logger.error(f"Failed to process cached image {filename}: {e}")
                    # Append a padding image or a placeholder for failed images
                    padding_image = ImageProcessor.create_padding_image(target_size)
                    image_arrays.append(padding_image)

            for i in range(0, len(image_arrays), chunk_size):
                chunk = image_arrays[i:i + chunk_size]
                chunk_index = i // chunk_size

                yield StreamEvent(
                    event="processing",
                    data={
                        "chunk": chunk_index,
                        "images_in_chunk": len(chunk),
                        "progress": f"{i + len(chunk)}/{len(image_arrays)}",
                        "percentage": round((i + len(chunk)) / len(image_arrays) * 100, 2)
                    }
                )

                batches = ImageProcessor.create_batches(chunk, batch_size, target_size)
                results = service.process_image_batches(batches, len(chunk))
                clean_results = validator(results)

                yield StreamEvent(
                    event="result",
                    data={
                        "chunk": chunk_index,
                        "results": clean_results,
                        "images_processed": len(chunk),
                        "successful": sum(1 for r in clean_results if r.get("status") != "error")
                    }
                )

                await asyncio.sleep(self._config.STREAM_DELAY)

            yield StreamEvent(
                event="complete",
                data={
                    "status": "success",
                    "total_processed": len(image_arrays),
                    "model": model_type
                }
            )

        except Exception as e:
            logger.error(f"Streaming error for {model_type}: {e}")
            yield StreamEvent(
                event="error",
                data={
                    "error": str(e),
                    "status": "failed",
                    "model": model_type
                }
            )


class WebSocketManager:
    def __init__(self):
        self._active_connections: Dict[str, WebSocket] = {}
        self._connection_counter = 0

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        await websocket.accept()

        if client_id is None:
            client_id = f"client_{self._connection_counter}"
            self._connection_counter += 1

        self._active_connections[client_id] = websocket
        logger.info(f"WebSocket client {client_id} connected")
        return client_id

    def disconnect(self, client_id: str):
        if client_id in self._active_connections:
            del self._active_connections[client_id]
            logger.info(f"WebSocket client {client_id} disconnected")

    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        if client_id in self._active_connections:
            try:
                await self._active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: Dict[str, Any]):
        disconnected_clients = []
        for client_id, connection in self._active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to {client_id}: {e}")
                disconnected_clients.append(client_id)

        for client_id in disconnected_clients:
            self.disconnect(client_id)


class WebSocketProcessor:
    def __init__(self, streaming_processor: StreamingProcessor, ws_manager: WebSocketManager):
        self._streaming_processor = streaming_processor
        self._ws_manager = ws_manager
        # self._file_manager = FileManager() # No longer needed for temp file creation

    async def handle_inference_request(
        self,
        websocket: WebSocket,
        client_id: str,
        message: Dict[str, Any]
    ):
        model_type = message.get("model", "yolo")
        chunk_size = min(message.get("chunk_size", 1), AppConfig.MAX_CHUNK_SIZE)

        await self._ws_manager.send_personal_message({
            "type": "status",
            "message": f"Processing request for {model_type}",
            "client_id": client_id
        }, client_id)

        try:
            images_data = message.get("images", [])
            if not images_data:
                await self._ws_manager.send_personal_message({
                    "type": "error",
                    "message": "No images provided",
                    "client_id": client_id
                }, client_id)
                return

            # Directly prepare cached_files (filename, bytes) from incoming data
            cached_files: List[Tuple[str, bytes]] = []
            for i, img_data in enumerate(images_data):
                try:
                    if isinstance(img_data, str):
                        import base64
                        img_bytes = base64.b64decode(img_data)
                    elif isinstance(img_data, bytes):
                        img_bytes = img_data
                    else:
                        raise ValueError("Image data must be base64 string or bytes")

                    filename = f"image_{i}.jpg" # Or use original filename if available in message
                    cached_files.append((filename, img_bytes))
                except Exception as e:
                    logger.error(f"Error decoding image data {i}: {e}")
                    await self._ws_manager.send_personal_message({
                        "type": "error",
                        "message": f"Failed to decode image {i}: {e}",
                        "client_id": client_id
                    }, client_id)
                    return # Stop processing if an image is invalid

            # Process images directly from cached_files (bytes in memory)
            async for event in self._streaming_processor.process_cached_images_stream(
                cached_files, model_type, chunk_size
            ):
                await self._ws_manager.send_personal_message({
                    "type": "stream_event",
                    "event": event.event,
                    "data": event.data,
                    "client_id": client_id
                }, client_id)

        except Exception as e:
            logger.error(f"WebSocket inference request error for {client_id}: {e}")
            await self._ws_manager.send_personal_message({
                "type": "error",
                "message": str(e),
                "client_id": client_id
            }, client_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting OCR Streaming API")
    yield
    logger.info("Shutting down OCR Streaming API")


app = FastAPI(
    title="OCR Streaming API",
    version="2.0.1",
    description="OCR API with streaming support for YOLO and Donut inference - Optimized WebSocket file handling",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MainTag = OpenApiTag(name="Main", description="Standard inference endpoints")
StreamingTag = OpenApiTag(name="Streaming", description="Streaming inference endpoints")
WebSocketTag = OpenApiTag(name="WebSocket", description="WebSocket endpoints")

app.openapi_tags = [
    MainTag.model_dump(),
    StreamingTag.model_dump(),
    WebSocketTag.model_dump()
]


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
        "max_ongoing_requests": 10,
        "target_ongoing_requests": 2,
        "upscale_delay_s": 2,
        "downscale_delay_s": 120,
        "upscaling_factor": 2,
        "downscaling_factor": 0.5,
        "metrics_interval_s": 2,
        "look_back_period_s": 4,
    },
)
@serve.ingress(app)
class TritonStreamingDeployment:
    def __init__(self):
        self._config = AppConfig()
        self._triton_client = http_client.InferenceServerClient(url=self._config.TRITON_URL)
        self._validate_triton_connection()

        self._yolo_service = YOLOInferenceService(self._triton_client)
        self._donut_service = DonutInferenceService(self._triton_client)
        self._image_validator = ImageValidator()
        self._result_validator = ResultValidator()
        self._file_manager = FileManager() # Keep for other uses if any

        self._streaming_processor = StreamingProcessor(self)
        self._ws_manager = WebSocketManager()
        self._ws_processor = WebSocketProcessor(self._streaming_processor, self._ws_manager)

    def _validate_triton_connection(self) -> None:
        try:
            if not self._triton_client.is_server_live():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Triton Server недоступен"
                )

            for model_name in ["yolo", "donut"]:
                if not self._triton_client.is_model_ready(model_name):
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

    async def _process_uploaded_images(
        self,
        images: List[UploadFile],
        target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        if not images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Не предоставлены изображения"
            )

        image_arrays = []

        for image_file in images:
            try:
                self._image_validator.validate_content_type(image_file)

                contents = await image_file.read()

                if not contents:
                    raise ValueError(f"Empty file: {image_file.filename}")

                img = self._image_validator.validate_image_content(contents, image_file.filename)
                processed_image = ImageProcessor.resize_image(img, target_size)
                image_arrays.append(processed_image)

            except Exception as e:
                logger.error(f"Failed to process image {image_file.filename}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to process image {image_file.filename}: {e}"
                )

        return image_arrays

    @app.post("/generate/yolo", tags=["Main"])
    async def generate_yolo(self, images: List[UploadFile] = File(...)) -> Dict[str, Any]:
        try:
            logger.info(f"Processing {len(images)} images for YOLO inference")

            image_arrays = await self._process_uploaded_images(images, self._config.YOLO_IMAGE_SIZE)
            batches = ImageProcessor.create_batches(
                image_arrays,
                self._config.YOLO_BATCH_SIZE,
                self._config.YOLO_IMAGE_SIZE
            )

            detections = self._yolo_service.process_image_batches(batches, len(image_arrays))
            clean_detections = self._result_validator.sanitize_yolo_results(detections)

            return {
                "status": "success",
                "model": "yolo",
                "processed_images": len(image_arrays),
                "results": clean_detections
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка в YOLO inference: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Внутренняя ошибка сервера: {e}"
            )

    @app.post("/generate/donut", tags=["Main"])
    async def generate_donut(self, images: List[UploadFile] = File(...)) -> Dict[str, Any]:
        try:
            image_arrays = await self._process_uploaded_images(images, self._config.DONUT_IMAGE_SIZE)
            batches = ImageProcessor.create_batches(
                image_arrays,
                self._config.DONUT_BATCH_SIZE,
                self._config.DONUT_IMAGE_SIZE
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
            logger.error(f"Ошибка в Donut inference: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Внутренняя ошибка сервера: {e}"
            )

    @app.post("/stream/yolo", tags=["Streaming"])
    async def stream_yolo(
        self,
        images: List[UploadFile] = File(...),
        chunk_size: int = 1
    ) -> StreamingResponse:
        if chunk_size > self._config.MAX_CHUNK_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chunk size too large. Maximum allowed: {self._config.MAX_CHUNK_SIZE}"
            )

        cached_files = []
        for upload_file in images:
            try:
                contents = await upload_file.read()
                filename = upload_file.filename or f"image_{len(cached_files)}.jpg"
                cached_files.append((filename, contents))
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to read file {upload_file.filename}: {e}"
                )

        async def event_generator():
            async for event in self._streaming_processor.process_cached_images_stream(
                cached_files, "yolo", chunk_size
            ):
                yield {
                    "event": event.event,
                    "data": json.dumps(event.data, ensure_ascii=False)
                }

        return EventSourceResponse(event_generator())

    @app.post("/stream/donut", tags=["Streaming"])
    async def stream_donut(
        self,
        images: List[UploadFile] = File(...),
        chunk_size: int = 1
    ) -> StreamingResponse:
        if chunk_size > self._config.MAX_CHUNK_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chunk size too large. Maximum allowed: {self._config.MAX_CHUNK_SIZE}"
            )

        cached_files = []
        for upload_file in images:
            try:
                contents = await upload_file.read()
                filename = upload_file.filename or f"image_{len(cached_files)}.jpg"
                cached_files.append((filename, contents))
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to read file {upload_file.filename}: {e}"
                )

        async def event_generator():
            async for event in self._streaming_processor.process_cached_images_stream(
                cached_files, "donut", chunk_size
            ):
                yield {
                    "event": event.event,
                    "data": json.dumps(event.data, ensure_ascii=False)
                }

        return EventSourceResponse(event_generator())

    @app.get("/stream/status/{model_name}", tags=["Streaming"])
    async def get_stream_status(self, model_name: str) -> Dict[str, Any]:
        if model_name not in ["yolo", "donut"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model name must be 'yolo' or 'donut'"
            )

        try:
            is_ready = self._triton_client.is_model_ready(model_name)
            metadata = self._triton_client.get_model_metadata(model_name)

            return {
                "model": model_name,
                "ready": is_ready,
                "version": metadata.get("versions", ["unknown"])[0] if metadata else "unknown",
                "streaming_available": True,
                "max_chunk_size": self._config.MAX_CHUNK_SIZE,
                "batch_size": (
                    self._config.YOLO_BATCH_SIZE if model_name == "yolo"
                    else self._config.DONUT_BATCH_SIZE
                )
            }

        except Exception as e:
            return {
                "model": model_name,
                "ready": False,
                "error": str(e),
                "streaming_available": False
            }

    @app.websocket("/ws/inference/{model_name}")
    async def websocket_inference_endpoint(self, websocket: WebSocket, model_name: str):
        if model_name not in ["yolo", "donut"]:
            await websocket.close(code=1008, reason="Invalid model name")
            return

        client_id = await self._ws_manager.connect(websocket)

        try:
            await self._ws_manager.send_personal_message({
                "type": "connected",
                "client_id": client_id,
                "model": model_name
            }, client_id)

            while True:
                message = await websocket.receive_json()

                if message.get("type") == "inference":
                    message["model"] = model_name
                    await self._ws_processor.handle_inference_request(
                        websocket, client_id, message
                    )
                elif message.get("type") == "ping":
                    await self._ws_manager.send_personal_message({
                        "type": "pong",
                        "client_id": client_id
                    }, client_id)
                else:
                    await self._ws_manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message.get('type')}",
                        "client_id": client_id
                    }, client_id)

        except WebSocketDisconnect:
            self._ws_manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
            self._ws_manager.disconnect(client_id)

    @app.get("/health", tags=["Main"])
    async def health_check(self) -> Dict[str, Any]:
        try:
            triton_status = self._triton_client.is_server_live()
            yolo_ready = self._triton_client.is_model_ready("yolo")
            donut_ready = self._triton_client.is_model_ready("donut")

            return {
                "status": "healthy" if all([triton_status, yolo_ready, donut_ready]) else "degraded",
                "triton_server": triton_status,
                "models": {
                    "yolo": yolo_ready,
                    "donut": donut_ready
                },
                "features": {
                    "streaming": True,
                    "websockets": True,
                    "batch_processing": True,
                    "file_caching": True
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


def deployment(_args):
    return TritonStreamingDeployment.bind()


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

    app_logger = logging.getLogger('triton_streaming_app')
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


if __name__ == "__main__":
    configure_ray_logging()
    os.environ["RAY_BACKEND_LOG_LEVEL"] = "info"
    sys.stdout.flush()
    sys.stderr.flush()

    logger.info("Запуск Ray Serve streaming deployment")
    serve.run(TritonStreamingDeployment.bind(), route_prefix="/")