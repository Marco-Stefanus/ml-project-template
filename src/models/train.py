from sklearn.ensemble import RandomForestClassifier
import joblib

def train(X, y, config):
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"]
    )
    model.fit(X, y)
    joblib.dump(model, "models/artifacts/model.pkl")
