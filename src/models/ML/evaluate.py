from sklearn.metrics import accuracy_score

def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds)
    }
