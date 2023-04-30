import os
from typing import Optional
import logging
from datetime import datetime

def _get_logger(debug: bool=True, filename: str="output.log"):
    logger = logging.getLogger("logger")
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    handlers = (
        logging.StreamHandler(), 
        logging.FileHandler(filename))
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        handler.setLevel(level)
        logger.addHandler(handler)
    logger.propagate = False
    return logger

