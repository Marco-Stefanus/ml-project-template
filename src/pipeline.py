from src.utils.config import load_config
from src.data.loader import load_data
from src.data.preprocessor import preprocess
from src.features.engineering import build_features
from src.models.train import train_model
from src.models.evaluate import evaluate

def run(config_path="configs/base.yaml"):
    config = load_config(config_path)

    df = load_data("data/raw/data.csv")
    df = preprocess(df)

    X, y = build_features(df, config["data"]["target_column"])
    model = train_model(X, y, "models/artifacts/model.pkl")

    metrics = evaluate(model, X, y)
    print(metrics)
