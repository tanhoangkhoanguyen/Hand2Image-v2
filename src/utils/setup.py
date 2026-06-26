from typing import List

import json

from src.utils.logger import get_logger

CONFIG_PATH = "src/utils/config.json"
CONFIG_DATA = None
LOGGER = get_logger("setup")

def get_config() -> None:
    try:
        global CONFIG_DATA
        with open(CONFIG_PATH, encoding="utf-8") as f:
            CONFIG_DATA = json.load(f)
    except Exception as e:
        LOGGER.error(f"No config data found\n\t{str(e)}")
        raise

def get_camera_id() -> int:
    try:
        if CONFIG_DATA is None:
            get_config()
        return CONFIG_DATA["camera_id"]
    except Exception as e:
        LOGGER.error(f"No config data found\n\t{str(e)}")
        raise

def get_classes() -> List[str]:
    try:
        if CONFIG_DATA is None:
            get_config()
        return CONFIG_DATA["classes"]
    except Exception as e:
        LOGGER.error(f"No config data found\n\t{str(e)}")
        raise

def get_colors() -> List[List[int]]:
    try:
        if CONFIG_DATA is None:
            get_config()
        return CONFIG_DATA["colors"]
    except Exception as e:
        LOGGER.error(f"No config data found\n\t{str(e)}")
        raise