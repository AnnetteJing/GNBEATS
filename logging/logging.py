import os
from typing import Optional
import logging
from datetime import datetime

def get_logger(debug: bool=True, filename: str="output.log"):
    logger = logging.getLogger("logger")
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    handlers = (
        logging.StreamHandler(), 
        logging.FileHandler(filename))
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        handler.setLevel(level)
        logger.addHandler(handler)
    return logger

