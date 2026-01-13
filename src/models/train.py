import joblib
from sklearn.linear_model import LogisticRegression

def train_model(X, y, model_path: str):
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model
