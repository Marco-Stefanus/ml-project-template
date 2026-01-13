import torch
from torch.utils.data import DataLoader, TensorDataset
from .model import MLP
from src.utils.device import get_device
from src.utils.logger import logger
import os

def train(df, config):
    try:
        device = get_device(config)

        X = torch.tensor(
            df.drop(columns=[config["data"]["target"]]).values,
            dtype=torch.float32
        )
        # Ensure target is long for CrossEntropyLoss
        y = torch.tensor(df[config["data"]["target"]].values, dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True
        )

        model = MLP(
            config["model"]["input_dim"],
            config["model"]["hidden_dim"],
            config["model"]["output_dim"],
            config["model"]["dropout"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["lr"]
        )
        criterion = torch.nn.CrossEntropyLoss()

        logger.info(f"Starting training for {config['training']['epochs']} epochs.")
        for epoch in range(config["training"]["epochs"]):
            epoch_loss = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {epoch_loss/len(loader):.4f}")

        # Ensure output directory exists
        os.makedirs("models/artifacts", exist_ok=True)
        torch.save(model.state_dict(), "models/artifacts/model.pt")
        logger.success("DL model trained and saved to models/artifacts/model.pt")
    except Exception as e:
        logger.error(f"Error in DL training: {e}")
        raise
