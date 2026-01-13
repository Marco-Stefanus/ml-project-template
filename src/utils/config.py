import yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_config(path: str) -> dict:
    logger.info(f"Loading config from {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
