from sklearn.ensemble import RandomForestClassifier
import joblib
from src.utils.logger import logger

def train(X, y, config):
    try:
        model = RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"]
        )
        model.fit(X, y)
        joblib.dump(model, "models/artifacts/model.pkl")
        logger.info("Model trained and saved to models/artifacts/model.pkl")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
