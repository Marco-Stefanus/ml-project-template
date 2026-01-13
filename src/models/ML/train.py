import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train(df, config):
    X = df.drop(columns=[config["data"]["target"]])
    y = df[config["data"]["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["training"]["test_size"]
    )

    model = LogisticRegression(**config["model"]["params"])
    model.fit(X_train, y_train)

    joblib.dump(model, "models/artifacts/model.pkl")
