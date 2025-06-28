import logging
import sys
from pathlib import Path

def setup_logger():
    """Config for logger in project"""

    logger = logging.getLogger('RealEstateProject')
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / 'project.log')
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = setup_logger()