import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.data.loader import load_data
from src.data.preprocessor import preprocess

def run(config_path):
    config = load_config(config_path)

    df = load_data(config["data"]["path"])
    df = preprocess(df)

    if config["mode"] == "ml":
        from src.models.ml.train import train
        train(df, config)

    elif config["mode"] == "dl":
        from src.models.dl.train import train
        train(df, config)

    else:
        raise ValueError("Invalid mode: choose ml or dl")
