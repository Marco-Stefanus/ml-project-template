import pandas as pd
from src.utils.logger import logger

def load_data(path):
    try:
        data = pd.read_csv(path)
        logger.info(f"Loaded data from {path} with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise
