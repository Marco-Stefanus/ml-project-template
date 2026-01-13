import pandas as pd

def build_features(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
