import numpy as np
from sklearn.linear_model import LogisticRegression

def test_model_fit():
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model = LogisticRegression().fit(X, y)
    assert model.predict([[1]])[0] == 1
