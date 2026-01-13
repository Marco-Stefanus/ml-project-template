from src.utils.config import load_config
from src.data.loader import load_data
from src.models.train import train

config = load_config("configs/base.yaml")

data = load_data("data/processed/data.csv")

X = data.drop("target", axis=1)
y = data["target"]

train(X, y, config)
print("Loading config...")
config = load_config("configs/base.yaml")

print("Loading data...")
data = load_data("data/processed/data.csv")

print("Splitting data...")
X = data.drop("target", axis=1)
y = data["target"]

print("Training model...")
train(X, y, config)

print("DONE! Model saved successfully.")
