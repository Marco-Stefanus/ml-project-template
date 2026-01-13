from src.utils.config import load_config
from src.data.loader import load_data
from src.data.preprocessor import preprocess

def run(config_path):
    config = load_config(config_path)

    data = load_data(config["data"]["path"])
    data = preprocess(data)

    if config["mode"] == "ml":
        from src.models.ML.train import train
        train(data, config)

    elif config["mode"] == "dl":
        from src.models.DL.train import train
        train(data, config)

    else:
        raise ValueError("Invalid mode (ml | dl)")
