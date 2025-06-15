import logging
import sys
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from src.utils.env import Env

env = Env()


def get_logger(name: str) -> logging.Logger:
    """
    Возвращает настроенный логгер с RichHandler.
    Уровень логирования устанавливается в зависимости от settings.debug.
    """
    console = Console(
        log_time=True,
        log_time_format='%H:%M:%S-%f',
        theme=Theme({
            "traceback.border": "black",
            "traceback.border.syntax_error": "black",
            "inspect.value.border": "black",
        })
    )

    debug = env.__getattr__('DEBUG')

    log_level = logging.DEBUG if debug else logging.INFO

    handler = RichHandler(
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format='%H:%M:%S-%f',
        level=log_level,
        console=console
    )

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(log_level)

    return logger