import pytest
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from ml.model import train_model, inference, compute_model_metrics


def test_train_model():
    """
    Test that train_model returns a GradientBoostingClassifier model.
    """
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)

    model = train_model(X_train, y_train)

    assert model is not None
    assert isinstance(model, GradientBoostingClassifier)


def test_inference():
    """
    Test that inference returns predictions with the correct shape.
    """
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert preds is not None
    assert len(preds) == 20
    assert all(p in [0, 1] for p in preds)


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns precision, recall, and F1 scores.
    """
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)