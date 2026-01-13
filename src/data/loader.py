import pandas as pd
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported file format")
