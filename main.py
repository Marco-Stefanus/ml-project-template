data = load_data("data/processed/data.csv")

from src.utils.config import load_config
from src.data.loader import load_data
from src.models.train import train
from src.utils.logger import logger

def main():
	try:
		logger.info("Loading config...")
		config = load_config("configs/base.yaml")
	except Exception as e:
		logger.error(f"Failed to load config: {e}")
		return

	try:
		logger.info("Loading data...")
		data = load_data("data/processed/data.csv")
	except Exception as e:
		logger.error(f"Failed to load data: {e}")
		return

	try:
		logger.info("Splitting data...")
		X = data.drop("target", axis=1)
		y = data["target"]
	except Exception as e:
		logger.error(f"Failed to split data: {e}")
		return

	try:
		logger.info("Training model...")
		train(X, y, config)
		logger.success("DONE! Model saved successfully.")
	except Exception as e:
		logger.error(f"Model training failed: {e}")

if __name__ == "__main__":
	main()
