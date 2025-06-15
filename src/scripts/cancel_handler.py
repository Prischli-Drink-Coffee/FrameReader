import asyncio
import signal
import sys
from typing import Optional
import threading
from src.utils.custom_logging import get_logger

log = get_logger(__name__)


class CancellationHandler:
    _instance: Optional['CancellationHandler'] = None
    _shutdown_event: asyncio.Event
    _signal_handlers_set: bool = False
    _lock = threading.Lock()
    
    def __new__(cls) -> 'CancellationHandler':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._shutdown_event = asyncio.Event()
        return cls._instance
    
    def setup_signal_handlers(self) -> None:
        if self._signal_handlers_set:
            return
            
        def signal_handler(signum: int, frame) -> None:
            log.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.signal_shutdown()
            
            if signum == signal.SIGINT:
                sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self._signal_handlers_set = True
    
    def signal_shutdown(self) -> None:
        if not self._shutdown_event.is_set():
            self._shutdown_event.set()
    
    def is_shutting_down(self) -> bool:
        return self._shutdown_event.is_set()
    
    def check_cancellation(self) -> None:
        if self.is_shutting_down():
            raise asyncio.CancelledError("Processing was cancelled by user")
    
    def reset(self) -> None:
        """Reset shutdown state for testing purposes"""
        self._shutdown_event.clear()