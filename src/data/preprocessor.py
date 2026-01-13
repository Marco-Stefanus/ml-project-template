import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(inplace=True)
    return df
