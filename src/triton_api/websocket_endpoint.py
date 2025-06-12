import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union

import aiofiles
import numpy as np
import websockets 
from websockets.exceptions import ConnectionClosed, WebSocketException

try:
    from websockets import ConnectionState
except ImportError:
    ConnectionState = None 

from PIL import Image

from src.utils.custom_logging import setup_logging
from src.utils.env import Env

env = Env()
log = setup_logging()


class WebSocketEndpointClient:
    def __init__(
        self,
        base_url: str = None,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0,
        max_reconnect_attempts: int = 5,
        reconnect_delay: float = 1.0
    ):
        self.base_url = base_url or env.TRITON_WS_URL
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._client_id: Optional[str] = None
        self._connected_event = asyncio.Event() # Событие для сигнализации о подключении
        self._message_handlers: Dict[str, Callable] = {}
        self._reconnect_count = 0
        self._listener_task: Optional[asyncio.Task] = None
        
        self._setup_default_handlers()

    def _is_websocket_potentially_open(self) -> bool:
        """
        Проверяет, существует ли веб-сокет и есть ли основания полагать, что он открыт.
        Это менее надежная проверка, используемая, когда .state или .open недоступны.
        """
        if not self._websocket:
            return False
        
        if ConnectionState: # Предпочтительный способ
            return self._websocket.state == ConnectionState.OPEN
        
        # Фолбэк: если нет ConnectionState, и мы не можем проверить .open,
        # то просто проверяем, что объект _websocket существует.
        # Операции send/recv должны будут обрабатывать ConnectionClosed.
        # log.debug("Cannot determine WebSocket state via .state or .open. Assuming 'potentially open' if object exists.")
        return True # Если _websocket не None, предполагаем, что он может быть открыт.

    def _setup_default_handlers(self):
        self._message_handlers.update({
            'connected': self._handle_connected,
            'pong': self._handle_pong,
            'status': self._handle_status,
            'stream_event': self._handle_stream_event,
            'error': self._handle_error
        })

    async def _handle_connected(self, message: Dict[str, Any]):
        self._client_id = message.get('client_id')
        self._reconnect_count = 0 
        log.info(f"WebSocket connected with client_id: {self._client_id}. Server model: {message.get('model')}")
        self._connected_event.set() # Сигнализируем, что событие "connected" получено

    async def _handle_pong(self, message: Dict[str, Any]): # ... (без изменений)
        # log.debug(f"Received pong from client: {self._client_id}")
        pass

    async def _handle_status(self, message: Dict[str, Any]): # ... (без изменений)
        # log.info(f"Status update: {message.get('message', 'Unknown status')}")
        pass

    async def _handle_stream_event(self, message: Dict[str, Any]): # ... (без изменений)
        event_type = message.get('event')
        event_data = message.get('data', {})
        # log.debug(f"Stream event: {event_type}, Data: {event_data}")

    async def _handle_error(self, message: Dict[str, Any]): # ... (без изменений)
        error_msg = message.get('message', 'Unknown error')
        log.error(f"WebSocket error: {error_msg}")
        # Возможно, здесь также стоит установить событие ошибки, если оно используется глобально

    def register_handler(self, message_type: str, handler: Callable): # ... (без изменений)
        self._message_handlers[message_type] = handler
        # log.debug(f"Registered handler for message type: {message_type}")

    def _validate_image_file(self, file_path: Union[str, Path]) -> Path: # ... (без изменений)
        path = Path(file_path)
        if not path.exists(): raise FileNotFoundError(f"Image file not found: {path}")
        if not path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        return path

    def _validate_numpy_array(self, array: np.ndarray) -> np.ndarray: # ... (без изменений)
        if not isinstance(array, np.ndarray): raise TypeError("Input must be a numpy.ndarray")
        if array.ndim not in [2, 3]: raise ValueError("Array must be 2D (grayscale) or 3D (color)")
        if array.ndim == 3 and array.shape[2] not in [1, 3, 4]:
            raise ValueError("Color images must have 1, 3, or 4 channels")
        if array.dtype not in [np.uint8, np.float32, np.float64]: array = array.astype(np.uint8)
        if array.dtype in [np.float32, np.float64] and array.max()<=1.0 and array.min()>=0.0:
            array = (array * 255).astype(np.uint8)
        elif array.dtype != np.uint8: array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _numpy_to_base64(self, array: np.ndarray, format: str = 'JPEG') -> str: # ... (без изменений)
        validated_array = self._validate_numpy_array(array)
        if validated_array.ndim == 2: img = Image.fromarray(validated_array, mode='L')
        elif validated_array.shape[2] == 1: img = Image.fromarray(validated_array.squeeze(), mode='L')
        elif validated_array.shape[2] == 3: img = Image.fromarray(validated_array, mode='RGB')
        elif validated_array.shape[2] == 4: img = Image.fromarray(validated_array, mode='RGBA')
        else: raise ValueError(f"Unsupported array shape: {validated_array.shape}")
        buffer = BytesIO(); img.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    async def _encode_images_from_paths(self, image_paths: List[Union[str, Path]]) -> List[str]: # ... (без изменений)
        encoded_images = []
        for idx, image_path in enumerate(image_paths):
            try:
                validated_path = self._validate_image_file(image_path)
                async with aiofiles.open(validated_path, 'rb') as f: content = await f.read()
                if not content: raise ValueError(f"Empty image file: {validated_path}")
                try: Image.open(BytesIO(content)).verify()
                except Exception as e: raise ValueError(f"Invalid image content in {validated_path}: {e}")
                encoded_images.append(base64.b64encode(content).decode('utf-8'))
                # log.debug(f"Encoded image {idx+1}/{len(image_paths)}: {validated_path.name}")
            except Exception as e: log.error(f"Failed to encode image {image_path}: {e}"); raise
        return encoded_images

    async def _encode_images_from_arrays(self, image_arrays: List[np.ndarray]) -> List[str]: # ... (без изменений)
        encoded_images = []
        for idx, array in enumerate(image_arrays):
            try:
                encoded_images.append(self._numpy_to_base64(array))
                # log.debug(f"Encoded numpy array {idx+1}/{len(image_arrays)}")
            except Exception as e: log.error(f"Failed to encode numpy array {idx}: {e}"); raise
        return encoded_images

    async def connect(self, model_name: str) -> None:
        if model_name not in ['yolo', 'donut']:
            raise ValueError(f"Unsupported model name for WebSocket: {model_name}")
        ws_url = f"{self.base_url}/ws/inference/{model_name}"
        
        if self._websocket and self._is_websocket_potentially_open() and self._connected_event.is_set():
            log.info(f"Already connected and confirmed to {ws_url}. Skipping new connection.")
            return
        
        # Если есть старый listener, остановим его перед новым подключением
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try: await self._listener_task
            except asyncio.CancelledError: log.debug("Previous listener task cancelled.")
        
        self._connected_event.clear() # Сбрасываем событие перед новым подключением

        try:
            log.info(f"Connecting to WebSocket: {ws_url}")
            self._websocket = await websockets.connect(
                ws_url, ping_interval=self.ping_interval, ping_timeout=self.ping_timeout
            )
            log.info(f"WebSocket connection process initiated for model: {model_name}. Starting listener.")
            # Запускаем listener СРАЗУ после websockets.connect, чтобы он мог поймать событие "connected"
            self._listener_task = asyncio.create_task(self._handle_incoming_messages())
            
            # Теперь ждем подтверждения от сервера через _connected_event
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=10.0) # Таймаут ожидания события "connected"
                log.info("Successfully connected and 'connected' event received from server.")
            except asyncio.TimeoutError:
                log.error("Timeout waiting for 'connected' event from server after establishing WebSocket.")
                await self.disconnect() # Закрываем соединение, если не дождались
                raise RuntimeError("Timeout waiting for 'connected' event from server.")

        except Exception as e:
            log.error(f"Failed to connect to WebSocket {ws_url}: {e}")
            self._websocket = None
            if self._listener_task and not self._listener_task.done(): # Убедимся, что listener остановлен, если был запущен
                self._listener_task.cancel()
            raise

    async def disconnect(self) -> None:
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try: await self._listener_task
            except asyncio.CancelledError: log.debug("Listener task cancelled during disconnect.")
        self._listener_task = None

        if self._websocket:
            try: 
                # Используем ConnectionState если доступно, иначе просто пытаемся закрыть
                should_close = True
                if ConnectionState:
                    if self._websocket.state == ConnectionState.CLOSED:
                        should_close = False
                
                if should_close and hasattr(self._websocket, 'close'): # hasattr для безопасности
                     await self._websocket.close() # type: ignore
                log.info("WebSocket connection closed")
            except Exception as e: log.error(f"Error closing WebSocket: {e}")
            finally: self._websocket = None; self._connected_event.clear() ; self._client_id = None

    async def _send_message(self, message: Dict[str, Any]) -> None:
        if not self._websocket or not self._is_websocket_potentially_open() or not self._connected_event.is_set():
            log.error("WebSocket not confirmed connected. Cannot send message.")
            raise RuntimeError("WebSocket not confirmed connected. Please connect first.")
        try:
            await self._websocket.send(json.dumps(message)) # type: ignore
            # log.debug(f"Sent message: {message.get('type', 'unknown')}")
        except ConnectionClosed as e:
            log.error(f"Failed to send message, connection closed: {e}"); await self.disconnect(); raise
        except Exception as e: log.error(f"Failed to send message: {e}"); raise

    async def _receive_message(self) -> Optional[Dict[str, Any]]:
        if not self._websocket or not self._is_websocket_potentially_open(): # _connected_event.is_set() здесь не так критично, как для send
            log.error("WebSocket not available to receive message.")
            raise RuntimeError("WebSocket not available to receive message.")
        try:
            message_str = await self._websocket.recv() # type: ignore
            message = json.loads(message_str)
            # log.debug(f"Received message: {message.get('type', 'unknown')}")
            return message
        except json.JSONDecodeError as e: log.error(f"Failed to decode JSON message: {e}"); return None
        except ConnectionClosed as e:
            log.info(f"WebSocket connection closed while receiving: {e}"); await self.disconnect(); raise
        except WebSocketException as e: # Более специфичные ошибки websockets
            log.error(f"WebSocket error during receive: {e}"); await self.disconnect(); raise
        except Exception as e: log.error(f"Failed to receive message: {e}"); raise

    async def _handle_incoming_messages(self) -> None:
        if not self._websocket:
            log.warning("Attempted to handle incoming messages with no WebSocket object.")
            return
        # log.debug("Listener task started for incoming WebSocket messages.")
        try:
            while self._is_websocket_potentially_open(): # Цикл, пока сокет "потенциально" открыт
                message = await self._receive_message()
                if not message: 
                    if not self._is_websocket_potentially_open(): # Перепроверка после recv
                        # log.debug("WebSocket seems closed after trying to receive a message.")
                        break
                    continue # Если recv вернул None (например, ошибка JSON), но сокет еще может быть жив
                
                message_type = message.get('type')
                if message_type in self._message_handlers:
                    await self._message_handlers[message_type](message)
                else: log.warning(f"No handler for message type: {message_type}")
        except ConnectionClosed: log.info("Connection closed during message handling loop.")
        except WebSocketException as e: log.error(f"WebSocketException in message handling loop: {e}")
        except RuntimeError as e: # Если _receive_message вызывает RuntimeError (сокет закрыт)
            log.error(f"RuntimeError in message handling loop (likely connection closed): {e}")
        except Exception as e:
            log.error(f"Unexpected error in _handle_incoming_messages: {type(e).__name__}: {e}", exc_info=True)
        finally:
            log.debug("Listener task for incoming WebSocket messages ended.")


    async def ping_server(self) -> None: # ... (без изменений)
        await self._send_message({'type': 'ping', 'timestamp': asyncio.get_event_loop().time()})

    async def send_inference_request_from_paths( # ... (без изменений)
        self, image_paths: List[Union[str, Path]], chunk_size: int = 1, request_id: Optional[str] = None
    ) -> None:
        if not image_paths: raise ValueError("No image paths provided")
        log.info(f"Preparing inference request for {len(image_paths)} image paths")
        encoded_images = await self._encode_images_from_paths(image_paths)
        message = {'type': 'inference', 'images': encoded_images, 'chunk_size': chunk_size}
        if request_id: message['request_id'] = request_id
        await self._send_message(message)
        log.info(f"Sent inference request with {len(encoded_images)} images from paths (ID: {request_id or 'N/A'})")

    async def send_inference_request_from_arrays( # ... (без изменений)
        self, image_arrays: List[np.ndarray], chunk_size: int = 1, request_id: Optional[str] = None
    ) -> None:
        if not image_arrays: raise ValueError("No image arrays provided")
        log.info(f"Preparing inference request for {len(image_arrays)} numpy arrays")
        encoded_images = await self._encode_images_from_arrays(image_arrays)
        message = {'type': 'inference', 'images': encoded_images, 'chunk_size': chunk_size}
        if request_id: message['request_id'] = request_id
        await self._send_message(message)
        log.info(f"Sent inference request with {len(encoded_images)} arrays (ID: {request_id or 'N/A'})")

    async def send_inference_request( # ... (без изменений)
        self, images: Union[List[Union[str, Path]], List[np.ndarray]], chunk_size: int = 1, request_id: Optional[str] = None
    ) -> None:
        if not images: raise ValueError("No images provided")
        if isinstance(images[0], np.ndarray):
            await self.send_inference_request_from_arrays(images, chunk_size, request_id) # type: ignore
        else:
            await self.send_inference_request_from_paths(images, chunk_size, request_id) # type: ignore
    
    # Метод _wait_for_connection_event удален, так как connect теперь сам ждет _connected_event

    async def run_inference_session_from_arrays(
        self, model_name: str, image_arrays: List[np.ndarray], chunk_size: int = 1, timeout: float = 300.0
    ) -> List[Dict[str, Any]]:
        # Убедимся, что мы подключены и listener запущен
        if not self._websocket or not self._is_websocket_potentially_open() or not self._connected_event.is_set():
            await self.connect(model_name) # connect теперь сам ждет _connected_event

        session_results_data: List[Dict[str, Any]] = []
        completion_event = asyncio.Event()
        error_occurred = asyncio.Event()
        error_message_holder = {"msg": ""}
        active_request_id = f"req_arr_{asyncio.get_event_loop().time()}"

        # Временный обработчик для этого сеанса
        async def session_stream_event_handler(message: Dict[str, Any]):
            event_type, event_data = message.get('event'), message.get('data', {})
            msg_request_id = event_data.get('request_id')
            if msg_request_id and msg_request_id != active_request_id:
                # log.debug(f"Session handler ignoring event for different request_id: {msg_request_id} (expected {active_request_id})")
                return
            if event_type == 'result':
                session_results_data.append(event_data)
                log.info(f"Session handler: Received result chunk (ReqID: {active_request_id}). Total chunks: {len(session_results_data)}")
            elif event_type == 'complete':
                log.info(f"Session handler: Inference session (ReqID: {active_request_id}) completed by server: {event_data}")
                completion_event.set()
            elif event_type == 'error':
                error_message_holder["msg"] = event_data.get('error', 'Unknown error during inference')
                log.error(f"Session handler: Inference error (ReqID: {active_request_id}): {error_message_holder['msg']}")
                error_occurred.set(); completion_event.set()

        original_handler = self._message_handlers.get('stream_event')
        self.register_handler('stream_event', session_stream_event_handler)
        
        try:
            # Listener уже должен быть запущен из connect()
            if not self._listener_task or self._listener_task.done():
                 log.warning("Listener task not running at start of inference session. This is unexpected.")
                 # Попытка перезапуска, если connect не справился или был вызван некорректно
                 if self._websocket and self._is_websocket_potentially_open():
                    self._listener_task = asyncio.create_task(self._handle_incoming_messages())
                 else:
                    raise RuntimeError("Cannot start listener, WebSocket not ready.")


            await self.send_inference_request_from_arrays(image_arrays, chunk_size, request_id=active_request_id)
            await asyncio.wait_for(completion_event.wait(), timeout=timeout)
            if error_occurred.is_set():
                raise RuntimeError(f"Inference failed for request {active_request_id}: {error_message_holder['msg']}")
            log.info(f"Inference session (ReqID: {active_request_id}) finished. Collected {len(session_results_data)} result data blocks.")
            return session_results_data
        except asyncio.TimeoutError:
            log.error(f"Inference session (ReqID: {active_request_id}) timed out after {timeout} seconds"); raise
        finally:
            # Восстанавливаем оригинальный обработчик
            if original_handler: self.register_handler('stream_event', original_handler)
            else: self._message_handlers.pop('stream_event', None)
            # Listener task НЕ останавливаем здесь, он должен жить пока жив клиент или до disconnect()

    async def run_inference_session_from_paths( # Аналогично from_arrays
        self, model_name: str, image_paths: List[Union[str, Path]], chunk_size: int = 1, timeout: float = 300.0
    ) -> List[Dict[str, Any]]:
        if not self._websocket or not self._is_websocket_potentially_open() or not self._connected_event.is_set():
            await self.connect(model_name)

        session_results_data: List[Dict[str, Any]] = []
        completion_event = asyncio.Event()
        error_occurred = asyncio.Event()
        error_message_holder = {"msg": ""}
        active_request_id = f"req_path_{asyncio.get_event_loop().time()}"

        async def session_stream_event_handler(message: Dict[str, Any]): # ... (такой же, как в from_arrays)
            event_type, event_data = message.get('event'), message.get('data', {})
            msg_request_id = event_data.get('request_id')
            if msg_request_id and msg_request_id != active_request_id:
                # log.debug(f"Session handler ignoring event for different request_id: {msg_request_id} (expected {active_request_id})")
                return
            if event_type == 'result':
                session_results_data.append(event_data)
                log.info(f"Session handler: Received result chunk (ReqID: {active_request_id}). Total chunks: {len(session_results_data)}")
            elif event_type == 'complete':
                log.info(f"Session handler: Inference session (ReqID: {active_request_id}) completed by server: {event_data}")
                completion_event.set()
            elif event_type == 'error':
                error_message_holder["msg"] = event_data.get('error', 'Unknown error during inference')
                log.error(f"Session handler: Inference error (ReqID: {active_request_id}): {error_message_holder['msg']}")
                error_occurred.set(); completion_event.set()

        original_handler = self._message_handlers.get('stream_event')
        self.register_handler('stream_event', session_stream_event_handler)
        
        try:
            if not self._listener_task or self._listener_task.done():
                 log.warning("Listener task not running at start of inference session (paths). This is unexpected.")
                 if self._websocket and self._is_websocket_potentially_open():
                    self._listener_task = asyncio.create_task(self._handle_incoming_messages())
                 else:
                    raise RuntimeError("Cannot start listener (paths), WebSocket not ready.")

            await self.send_inference_request_from_paths(image_paths, chunk_size, request_id=active_request_id)
            await asyncio.wait_for(completion_event.wait(), timeout=timeout)
            if error_occurred.is_set():
                raise RuntimeError(f"Inference failed for request {active_request_id}: {error_message_holder['msg']}")
            log.info(f"Inference session (ReqID: {active_request_id}) finished. Collected {len(session_results_data)} result data blocks.")
            return session_results_data
        except asyncio.TimeoutError:
            log.error(f"Inference session (ReqID: {active_request_id}) timed out after {timeout} seconds"); raise
        finally:
            if original_handler: self.register_handler('stream_event', original_handler)
            else: self._message_handlers.pop('stream_event', None)

    async def run_inference_session( # ... (без изменений)
        self, model_name: str, images: Union[List[Union[str, Path]], List[np.ndarray]], chunk_size: int = 1, timeout: float = 300.0
    ) -> List[Dict[str, Any]]:
        if not images: raise ValueError("No images provided")
        # connect() будет вызван внутри run_inference_session_from_arrays/paths если необходимо
        if isinstance(images[0], np.ndarray):
            return await self.run_inference_session_from_arrays(model_name, images, chunk_size, timeout) # type: ignore
        else:
            return await self.run_inference_session_from_paths(model_name, images, chunk_size, timeout) # type: ignore

    async def __aenter__(self):
        # connect() не вызывается здесь, т.к. model_name неизвестен.
        # Пользователь должен вызвать connect() или run_inference_session() сам.
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect() # Гарантирует остановку listener'а и закрытие сокета


async def example_usage(): # ... (без изменений, как в предыдущем ответе)
    import numpy as np
    rgb_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_arrays = [rgb_array, rgb_array] 
    if not env.TRITON_WS_URL:
        print("TRITON_WS_URL is not set. Skipping example.")
        return
    print(f"Attempting to connect to WebSocket server at: {env.TRITON_WS_URL}")
    try:
        async with WebSocketEndpointClient() as client: # __aenter__
            print("Running YOLO inference with numpy arrays via WebSocket...")
            # Первый вызов run_inference_session вызовет client.connect() внутри себя
            yolo_results_data = await client.run_inference_session(
                model_name='yolo', images=test_arrays, chunk_size=1, timeout=60.0 
            )
            total_yolo_detections = sum(len(data_item.get('detections', [])) for data_item in yolo_results_data)
            print(f"YOLO WebSocket session processed {len(yolo_results_data)} data blocks, {total_yolo_detections} total detections.")

            # Пример второго вызова с тем же клиентом, соединение должно быть уже установлено
            print("Running a second YOLO inference with the same client...")
            more_yolo_results = await client.run_inference_session(
                model_name='yolo', images=[rgb_array], chunk_size=1, timeout=30.0
            )
            total_more_detections = sum(len(data_item.get('detections', [])) for data_item in more_yolo_results)
            print(f"Second YOLO session processed {len(more_yolo_results)} data blocks, {total_more_detections} total detections.")

        # __aexit__ будет вызван здесь, вызывая client.disconnect()
    except ConnectionRefusedError:
        print(f"Connection refused. Ensure WebSocket server is running at {env.TRITON_WS_URL} and accessible.")
    except websockets.exceptions.InvalidURI:
        print(f"Invalid WebSocket URI: {env.TRITON_WS_URL}. Check configuration.")
    except asyncio.TimeoutError: print("Operation timed out. Server might be slow or unresponsive.")
    except RuntimeError as e: print(f"Runtime error during example: {e}")
    except Exception as e: print(f"An unexpected error occurred in example_usage: {type(e).__name__}: {e}", exc_info=True)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    # log.setLevel(logging.DEBUG) # Для более детального логирования от WebSocketEndpointClient
    asyncio.run(example_usage())