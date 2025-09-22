import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import aiofiles
import httpx
import numpy as np
from PIL import Image
import asyncio

from src.utils.custom_logging import get_logger
from load_dotenv import load_dotenv

load_dotenv()
log = get_logger(__name__)


class MainEndpointClient:
    def __init__(
        self,
        base_url: str = None,
        timeout: float = 300.0,
        max_retries: int = 3
    ):
        self.base_url = base_url or os.getenv("TRITON_API_URL")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    def _validate_image_file(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if not path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        return path

    def _validate_numpy_array(self, array: np.ndarray) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if array.ndim not in [2, 3]:
            raise ValueError("Array must be 2D (grayscale) or 3D (color)")
        if array.ndim == 3 and array.shape[2] not in [1, 3, 4]:
            raise ValueError("Color images must have 1, 3, or 4 channels")
        if array.dtype not in [np.uint8, np.float32, np.float64]:
            array = array.astype(np.uint8)
        if array.max() <= 1.0 and array.dtype in [np.float32, np.float64]:
            array = (array * 255).astype(np.uint8)
        return array

    def _numpy_to_bytes(self, array: np.ndarray, format: str = 'JPEG') -> bytes:
        validated_array = self._validate_numpy_array(array)
        if validated_array.ndim == 2:
            img = Image.fromarray(validated_array, mode='L')
        elif validated_array.shape[2] == 1:
            img = Image.fromarray(validated_array.squeeze(), mode='L')
        elif validated_array.shape[2] == 3:
            img = Image.fromarray(validated_array, mode='RGB')
        elif validated_array.shape[2] == 4:
            img = Image.fromarray(validated_array, mode='RGBA')
        else:
            raise ValueError(f"Unsupported array shape: {validated_array.shape}")
        buffer = BytesIO()
        img.save(buffer, format=format)
        return buffer.getvalue()

    def _validate_image_content(self, content: bytes) -> bytes:
        if not content:
            raise ValueError("Empty image content")
        try:
            img = Image.open(BytesIO(content))
            img.verify()
        except Exception as e:
            raise ValueError(f"Invalid image content: {e}")
        return content

    async def _prepare_image_files_from_paths(
        self, 
        image_paths: List[Union[str, Path]]
    ) -> List[tuple]:
        files = []
        
        for idx, image_path in enumerate(image_paths):
            try:
                validated_path = self._validate_image_file(image_path)
                async with aiofiles.open(validated_path, 'rb') as f:
                    content = await f.read()
                self._validate_image_content(content)
                files.append((
                    'images',
                    (validated_path.name, content, 'image/jpeg')
                ))
                log.debug(f"Prepared image file {idx + 1}/{len(image_paths)}: {validated_path.name}")
            except Exception as e:
                log.error(f"Failed to prepare image {image_path}: {e}")
                raise
        return files

    async def _prepare_image_files_from_arrays(
        self, 
        image_arrays: List[np.ndarray],
        filenames: Optional[List[str]] = None
    ) -> List[tuple]:
        files = []
    
        if filenames and len(filenames) != len(image_arrays):
            raise ValueError("Number of filenames must match number of arrays")
        for idx, array in enumerate(image_arrays):
            try:
                content = self._numpy_to_bytes(array)
                self._validate_image_content(content)
                filename = (filenames[idx] if filenames 
                           else f"numpy_image_{idx + 1}.jpg")
                files.append((
                    'images',
                    (filename, content, 'image/jpeg')
                ))
                log.debug(f"Prepared numpy array {idx + 1}/{len(image_arrays)}: {filename}")
            except Exception as e:
                log.error(f"Failed to prepare numpy array {idx}: {e}")
                raise
        return files

    async def _make_request(
        self,
        endpoint: str,
        files: List[tuple],
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        for attempt in range(self.max_retries):
            try:
                log.info(f"_make_request: attempt {attempt+1}")
                response = await self._client.post(
                    endpoint,
                    files=files,
                    data=data or {}
                )
                log.info(f"_make_request: response.status_code={response.status_code}")
                try:
                    log.info(f"_make_request: response.text={response.text}")
                except Exception as e:
                    log.error(f"_make_request: failed to read response.text: {e}")
                log.info("_make_request: response received")
                response.raise_for_status()
                result = response.json()
                log.info(f"_make_request: response status {response.status_code}")
                return result
            except Exception as e:
                log.error(f"_make_request error: {e}")
                if attempt == self.max_retries - 1:
                    raise

    async def yolo_inference_from_paths(
        self, 
        image_paths: List[Union[str, Path]]
    ) -> Dict[str, Any]:
        if not image_paths:
            raise ValueError("No image paths provided")
        log.info(f"Starting YOLO inference for {len(image_paths)} images from paths")
        files = await self._prepare_image_files_from_paths(image_paths)
        result = await self._make_request('/generate/yolo', files)
        log.info(f"YOLO inference completed. Processed {result.get('processed_images', 0)} images")
        return result

    async def yolo_inference_from_arrays(
        self, 
        image_arrays: List[np.ndarray],
        filenames: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not image_arrays:
            raise ValueError("No image arrays provided")
        
        log.info(f"Starting YOLO inference for {len(image_arrays)} numpy arrays")
        
        try:
            files = await self._prepare_image_files_from_arrays(image_arrays, filenames)
        except Exception as e:
            log.error(f"Exception in _prepare_image_files_from_arrays: {e}")
            raise
        result = await self._make_request('/generate/yolo', files)
        
        log.info(f"YOLO inference completed. Processed {result.get('processed_images', 0)} arrays")
        return result

    async def donut_inference_from_paths(
        self, 
        image_paths: List[Union[str, Path]]
    ) -> Dict[str, Any]:
        if not image_paths:
            raise ValueError("No image paths provided")
        
        log.info(f"Starting Donut inference for {len(image_paths)} images from paths")
        
        files = await self._prepare_image_files_from_paths(image_paths)
        result = await self._make_request('/generate/donut', files)
        
        log.info(f"Donut inference completed. Processed {result.get('processed_images', 0)} images")
        return result

    async def donut_inference_from_arrays(self, image_arrays: List[np.ndarray], filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        log.info(f"donut_inference_from_arrays: processing {len(image_arrays)} arrays")
        files = await self._prepare_image_files_from_arrays(image_arrays, filenames)
        log.info(f"donut_inference_from_arrays: files prepared, sending request")
        result = await self._make_request('/generate/donut', files)
        log.info(f"donut_inference_from_arrays: request done, got result")
        return result

    async def health_check(self) -> Dict[str, Any]:
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        try:
            response = await self._client.get('/health')
            response.raise_for_status()
            result = response.json()
            log.info(f"Health check completed. Status: {result.get('status', 'unknown')}")
            return result
        except Exception as e:
            log.error(f"Health check failed: {e}")
            raise

    async def batch_inference_from_arrays(
        self,
        image_arrays: List[np.ndarray],
        models: List[str] = None,
        filenames: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        if models is None:
            models = ['yolo', 'donut']
        
        results = {}
        if 'yolo' in models:
            try:
                results['yolo'] = await self.yolo_inference_from_arrays(image_arrays, filenames)
            except Exception as e:
                log.error(f"YOLO batch inference failed: {e}")
                results['yolo'] = {'status': 'error', 'error': str(e)}
        if 'donut' in models:
            try:
                results['donut'] = await self.donut_inference_from_arrays(image_arrays, filenames)
            except Exception as e:
                log.error(f"Donut batch inference failed: {e}")
                results['donut'] = {'status': 'error', 'error': str(e)}
        log.info(f"Batch inference completed for models: {models}")
        return results

    async def yolo_inference(
        self, 
        images: Union[List[Union[str, Path]], List[np.ndarray]],
        filenames: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not images:
            raise ValueError("No images provided")
        if isinstance(images[0], np.ndarray):
            return await self.yolo_inference_from_arrays(images, filenames)
        else:
            return await self.yolo_inference_from_paths(images)

    async def donut_inference(
        self, 
        images: Union[List[Union[str, Path]], List[np.ndarray]],
        filenames: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not images:
            raise ValueError("No images provided")
        
        if isinstance(images[0], np.ndarray):
            return await self.donut_inference_from_arrays(images, filenames)
        else:
            return await self.donut_inference_from_paths(images)


async def example_usage():

    import numpy as np
    from pathlib import Path
    
    test_array_1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_array_2 = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    test_arrays = [test_array_1, test_array_2]
    
    image_paths = [Path("test1.jpg"), Path("test2.png")]
    
    async with MainEndpointClient() as client:
        health = await client.health_check()
        print(f"Server status: {health['status']}")
        
        yolo_results = await client.yolo_inference_from_arrays(
            test_arrays, 
            filenames=["array_1.jpg", "array_2.jpg"]
        )
        print(f"YOLO results: {len(yolo_results['results'])} detections")
        
        donut_results = await client.donut_inference_from_arrays(test_arrays)
        print(f"Donut results: {len(donut_results['results'])} OCR results")
        
        mixed_results = await client.yolo_inference(test_arrays)
        print(f"Mixed YOLO results: {mixed_results['status']}")
        
        batch_results = await client.batch_inference_from_arrays(
            test_arrays, 
            models=['yolo', 'donut']
        )
        print(f"Batch results keys: {list(batch_results.keys())}")

        if all(path.exists() for path in image_paths):
            file_results = await client.yolo_inference_from_paths(image_paths)
            print(f"File-based results: {file_results['status']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
