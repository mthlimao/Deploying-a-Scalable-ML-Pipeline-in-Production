import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
from source.ml.model import train_model, compute_model_metrics, inference

def test_train_model():
    # Generate mock training data
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Check if the returned model is an instance of RandomForestClassifier
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"
    
    # Check if the model has been fitted by making a prediction
    predictions = model.predict(X_train)
    assert len(predictions) == len(y_train), "Predictions length does not match training labels"
    assert set(predictions) <= {0, 1}, "Predictions contain unexpected classes"


def test_compute_model_metrics():
    # Generate mock true labels and predictions
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_preds = np.array([1, 0, 1, 0, 0, 1])
    
    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    
    # Compute expected values using sklearn
    expected_precision = precision_score(y_true, y_preds, zero_division=1)
    expected_recall = recall_score(y_true, y_preds, zero_division=1)
    expected_fbeta = fbeta_score(y_true, y_preds, beta=1, zero_division=1)
    
    # Validate results
    assert precision == expected_precision, "Precision score mismatch"
    assert recall == expected_recall, "Recall score mismatch"
    assert fbeta == expected_fbeta, "F-beta score mismatch"


def test_inference():
    # Create a simple trained model
    model = RandomForestClassifier()
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model.fit(X_train, y_train)
    
    # Create test input
    X_test = np.array([[2, 3], [6, 7]])
    
    # Run inference
    preds = inference(model, X_test)
    
    # Check output
    assert len(preds) == len(X_test), "Inference output length mismatch"
    assert set(preds) <= {0, 1}, "Predictions contain unexpected classes"