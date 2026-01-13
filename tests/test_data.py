import pandas as pd
from src.data.preprocessor import preprocess

def test_preprocess():
    df = pd.DataFrame({"a": [1, None]})
    out = preprocess(df)
    assert out.isnull().sum().sum() == 0
