import asyncio
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Union, Any

import aiofiles
import httpx
import numpy as np
from PIL import Image

from src.utils.custom_logging import setup_logging
from src.utils.env import Env

env = Env()
log = setup_logging()


class StreamEndpointClient:
    def __init__(
        self,
        base_url: str = None,
        timeout: float = 600.0,
        chunk_size: int = 1,
        max_chunk_size: int = 10
    ):
        self.base_url = base_url or env.TRITON_API_URL
        self.timeout = timeout
        self.chunk_size = min(chunk_size, max_chunk_size)
        self.max_chunk_size = max_chunk_size
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
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

    def _validate_chunk_size(self, chunk_size: int) -> int:
        if chunk_size < 1:
            raise ValueError("Chunk size must be at least 1")
        if chunk_size > self.max_chunk_size:
            raise ValueError(f"Chunk size cannot exceed {self.max_chunk_size}")
        return chunk_size

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
                if not content:
                    raise ValueError(f"Empty image file: {validated_path}")
                try:
                    img = Image.open(BytesIO(content))
                    img.verify()
                except Exception as e:
                    raise ValueError(f"Invalid image content in {validated_path}: {e}")
                files.append((
                    'images',
                    (validated_path.name, content, 'image/jpeg')
                ))
                log.debug(f"Prepared streaming image {idx + 1}/{len(image_paths)}: {validated_path.name}")
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
                if not content:
                    raise ValueError(f"Empty content from numpy array {idx}")
                try:
                    img = Image.open(BytesIO(content))
                    img.verify()
                except Exception as e:
                    raise ValueError(f"Invalid converted image content from array {idx}: {e}")
                filename = (filenames[idx] if filenames 
                           else f"stream_array_{idx + 1}.jpg")
                files.append((
                    'images',
                    (filename, content, 'image/jpeg')
                ))
                log.debug(f"Prepared streaming array {idx + 1}/{len(image_arrays)}: {filename}")
            except Exception as e:
                log.error(f"Failed to prepare numpy array {idx}: {e}")
                raise
        return files

    async def _parse_sse_event(self, line: str) -> Optional[Dict[str, Any]]:
        line = line.strip()
        if line.startswith('event:'):
            return {'type': 'event', 'value': line[6:].strip()}
        elif line.startswith('data:'):
            data_str = line[5:].strip()
            try:
                return {'type': 'data', 'value': json.loads(data_str)}
            except json.JSONDecodeError as e:
                log.warning(f"Failed to parse SSE data: {e}")
                return {'type': 'data', 'value': data_str}
        elif line == '':
            return {'type': 'separator'}
        return None

    async def _stream_request(
        self,
        endpoint: str,
        files: List[tuple],
        chunk_size: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        validated_chunk_size = self._validate_chunk_size(chunk_size)
        data = {'chunk_size': str(validated_chunk_size)}
        log.info(f"Starting streaming request to {endpoint} with chunk_size={validated_chunk_size}")
        try:
            async with self._client.stream(
                'POST',
                endpoint,
                files=files,
                data=data,
                headers={'Accept': 'text/event-stream'}
            ) as response:
                response.raise_for_status()
                current_event = None
                current_data = None
                async for line in response.aiter_lines():
                    parsed = await self._parse_sse_event(line)
                    if not parsed:
                        continue
                    if parsed['type'] == 'event':
                        current_event = parsed['value']
                    elif parsed['type'] == 'data':
                        current_data = parsed['value']
                    elif parsed['type'] == 'separator' and current_event and current_data is not None:
                        yield {
                            'event': current_event,
                            'data': current_data
                        }
                        current_event = None
                        current_data = None
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error {e.response.status_code} for {endpoint}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            log.error(f"Request error for {endpoint}: {e}")
            raise

    async def yolo_stream_from_paths(
        self, 
        image_paths: List[Union[str, Path]], 
        chunk_size: int = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not image_paths:
            raise ValueError("No image paths provided")
        
        chunk_size = chunk_size or self.chunk_size
        log.info(f"Starting YOLO streaming for {len(image_paths)} paths with chunk_size={chunk_size}")
        files = await self._prepare_image_files_from_paths(image_paths)
        async for event in self._stream_request('/stream/yolo', files, chunk_size):
            log.debug(f"YOLO stream event: {event['event']}")
            yield event

    async def yolo_stream_from_arrays(
        self, 
        image_arrays: List[np.ndarray], 
        chunk_size: int = None,
        filenames: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not image_arrays:
            raise ValueError("No image arrays provided")
        
        chunk_size = chunk_size or self.chunk_size
        log.info(f"Starting YOLO streaming for {len(image_arrays)} arrays with chunk_size={chunk_size}")
        files = await self._prepare_image_files_from_arrays(image_arrays, filenames)
        async for event in self._stream_request('/stream/yolo', files, chunk_size):
            log.debug(f"YOLO stream event: {event['event']}")
            yield event

    async def donut_stream_from_paths(
        self, 
        image_paths: List[Union[str, Path]], 
        chunk_size: int = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not image_paths:
            raise ValueError("No image paths provided")
        
        chunk_size = chunk_size or self.chunk_size
        log.info(f"Starting Donut streaming for {len(image_paths)} paths with chunk_size={chunk_size}")
        files = await self._prepare_image_files_from_paths(image_paths)
        async for event in self._stream_request('/stream/donut', files, chunk_size):
            log.debug(f"Donut stream event: {event['event']}")
            yield event

    async def donut_stream_from_arrays(
        self, 
        image_arrays: List[np.ndarray], 
        chunk_size: int = None,
        filenames: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not image_arrays:
            raise ValueError("No image arrays provided")
        
        chunk_size = chunk_size or self.chunk_size
        log.info(f"Starting Donut streaming for {len(image_arrays)} arrays with chunk_size={chunk_size}")
        files = await self._prepare_image_files_from_arrays(image_arrays, filenames)
        async for event in self._stream_request('/stream/donut', files, chunk_size):
            log.debug(f"Donut stream event: {event['event']}")
            yield event

    async def get_stream_status(self, model_name: str) -> Dict[str, Any]:
        if model_name not in ['yolo', 'donut']:
            raise ValueError("Model name must be 'yolo' or 'donut'")
        
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        try:
            response = await self._client.get(f'/stream/status/{model_name}')
            response.raise_for_status()
            result = response.json()
            log.info(f"Stream status for {model_name}: {result.get('ready', False)}")
            return result
        except Exception as e:
            log.error(f"Failed to get stream status for {model_name}: {e}")
            raise

    async def collect_stream_results(
        self,
        stream_generator: AsyncGenerator[Dict[str, Any], None]
    ) -> Dict[str, Any]:
        collected_results = []
        final_status = None
        total_processed = 0
        
        async for event in stream_generator:
            event_type = event.get('event')
            event_data = event.get('data', {})
            if event_type == 'start':
                log.info(f"Stream started: {event_data}")
                total_processed = event_data.get('total_images', 0)
            elif event_type == 'processing':
                log.debug(f"Processing chunk: {event_data}")
            elif event_type == 'result':
                results = event_data.get('results', [])
                collected_results.extend(results)
                log.debug(f"Collected {len(results)} results from chunk")
            elif event_type == 'complete':
                final_status = event_data
                log.info(f"Stream completed: {event_data}")
            elif event_type == 'error':
                log.error(f"Stream error: {event_data}")
                raise RuntimeError(f"Stream failed: {event_data.get('error', 'Unknown error')}")
        return {
            'status': 'success',
            'results': collected_results,
            'total_processed': total_processed,
            'final_status': final_status
        }

    async def yolo_stream(
        self, 
        images: Union[List[Union[str, Path]], List[np.ndarray]], 
        chunk_size: int = None,
        filenames: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not images:
            raise ValueError("No images provided")
        
        if isinstance(images[0], np.ndarray):
            async for event in self.yolo_stream_from_arrays(images, chunk_size, filenames):
                yield event
        else:
            async for event in self.yolo_stream_from_paths(images, chunk_size):
                yield event

    async def donut_stream(
        self, 
        images: Union[List[Union[str, Path]], List[np.ndarray]], 
        chunk_size: int = None,
        filenames: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not images:
            raise ValueError("No images provided")
        
        if isinstance(images[0], np.ndarray):
            async for event in self.donut_stream_from_arrays(images, chunk_size, filenames):
                yield event
        else:
            async for event in self.donut_stream_from_paths(images, chunk_size):
                yield event

    async def stream_collect_from_arrays(
        self,
        image_arrays: List[np.ndarray],
        model_name: str,
        chunk_size: int = None,
        filenames: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if model_name == 'yolo':
            stream = self.yolo_stream_from_arrays(image_arrays, chunk_size, filenames)
        elif model_name == 'donut':
            stream = self.donut_stream_from_arrays(image_arrays, chunk_size, filenames)
        else:
            raise ValueError("Model name must be 'yolo' or 'donut'")
        
        return await self.collect_stream_results(stream)


async def example_usage():

    from stream_endpoint import StreamEndpointClient
    import numpy as np
    
    test_array_1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_array_2 = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    test_array_3 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_arrays = [test_array_1, test_array_2, test_array_3]
    
    async with StreamEndpointClient(chunk_size=2) as client:
        yolo_status = await client.get_stream_status('yolo')
        print(f"YOLO stream status: {yolo_status['ready']}")
        
        print("Starting YOLO streaming with arrays...")
        async for event in client.yolo_stream_from_arrays(
            test_arrays, 
            chunk_size=2,
            filenames=["arr1.jpg", "arr2.jpg", "arr3.jpg"]
        ):
            print(f"Event: {event['event']}")
            if event['event'] == 'result':
                print(f"  Results count: {len(event['data'].get('results', []))}")
        
        print("Collecting Donut stream results...")
        donut_results = await client.stream_collect_from_arrays(
            test_arrays, 
            'donut', 
            chunk_size=1
        )
        print(f"Donut results: {len(donut_results['results'])} items")

        print("Universal streaming method...")
        async for event in client.yolo_stream(test_arrays, chunk_size=3):
            if event['event'] == 'complete':
                print(f"Completed: {event['data']}")
        
        float_arrays = [arr.astype(np.float32) / 255.0 for arr in test_arrays]
        print("Streaming with normalized float arrays...")
        float_results = await client.stream_collect_from_arrays(
            float_arrays, 
            'yolo', 
            chunk_size=2
        )
        print(f"Float array results: {float_results['status']}")


if __name__ == "__main__":
    asyncio.run(example_usage())
