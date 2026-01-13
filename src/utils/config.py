import yaml
from src.utils.logger import logger

def load_config(path):
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {path}: {e}")
        raise
