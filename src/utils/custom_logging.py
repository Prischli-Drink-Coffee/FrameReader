import os
import logging
import time
import sys
from typing import Optional

from rich.theme import Theme
from rich.logging import RichHandler
from rich.console import Console
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install

log: Optional[logging.Logger] = None


def setup_logging(debug: bool = False) -> logging.Logger:
    global log

    if log is not None:
        return log

    if sys.version_info >= (3, 9):
        logging.basicConfig(
            level=logging.DEBUG, 
            format='%(asctime)s | %(levelname)s | %(pathname)s | %(message)s',
            encoding='utf-8', 
            force=True
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG, 
            format='%(asctime)s | %(levelname)s | %(pathname)s | %(message)s',
            force=True
        )

    console = Console(
        log_time=True, 
        log_time_format='%H:%M:%S-%f', 
        theme=Theme({
            "traceback.border": "black",
            "traceback.border.syntax_error": "black",
            "inspect.value.border": "black",
        })
    )
    
    pretty_install(console=console)
    traceback_install(
        console=console, 
        extra_lines=1, 
        width=console.width, 
        word_wrap=False, 
        indent_guides=False,
        suppress=[]
    )
    
    rh = RichHandler(
        show_time=True, 
        omit_repeated_times=False, 
        show_level=True, 
        show_path=False, 
        markup=False,
        rich_tracebacks=True, 
        log_time_format='%H:%M:%S-%f',
        level=logging.DEBUG if debug else logging.INFO, 
        console=console
    )
    rh.set_name(str(logging.DEBUG if debug else logging.INFO))
    
    log = logging.getLogger("sd")
    log.addHandler(rh)

    return log
