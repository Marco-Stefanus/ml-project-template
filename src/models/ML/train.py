import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train(df, config):

    print("🚀 Training started...")

    # Split feature & target
    X = df.drop(columns=[config["data"]["target"]])
    y = df[config["data"]["target"]]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=42
    )

    print("📊 Data split:")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    # Model
    model = LogisticRegression(**config["model"]["params"])
    print("🧠 Training model...")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Training finished | Accuracy: {acc:.4f}")

    # Save model
    path = "models/artifacts/model.pkl"
    joblib.dump(model, path)
    print(f"💾 Model saved to: {path}")

    return model

import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train(df, config):
    print("🚀 Training started...")

    # Split feature & target
    X = df.drop(columns=[config["data"]["target"]])
    y = df[config["data"]["target"]]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=42
    )

    print("📊 Data split:")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    # Dataset untuk LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)


    # Config LightGBM (check GPU availability)
    import subprocess
    lgb_params = config["model"]["params"].copy()  # avoid mutating config
    lgb_params["verbose"] = -1  # suppress logs
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        lgb_params["device"] = "gpu"
        print("✅ GPU detected, training on GPU")
    except Exception:
        lgb_params["device"] = "cpu"
        print("⚠ GPU not available, fallback to CPU")

    print(f"🧠 Training LightGBM model on {lgb_params['device'].upper()}...")
    model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "eval"],
        num_boost_round=config["training"]["num_boost_round"],
        early_stopping_rounds=config["training"].get("early_stopping_rounds", 50),
        verbose_eval=50
    )

    # Evaluation
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_labels = [1 if p > 0.5 else 0 for p in y_pred]  # binary classification
    acc = accuracy_score(y_test, y_pred_labels)

    print(f"✅ Training finished | Accuracy: {acc:.4f}")

    # Save model
    path = "models/artifacts/lgbm_model.pkl"
    joblib.dump(model, path)
    print(f"💾 Model saved to: {path}")

    return model
# Cek GPU & fallback
try:
    import subprocess
    # cek nvidia-smi (Windows)
    subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    lgb_params["device"] = "gpu"
    print("✅ GPU detected, training on GPU")
except Exception:
    lgb_params["device"] = "cpu"
    print("⚠ GPU not available, fallback to CPU")
