
import pytest
import pandas as pd
import numpy as np
from src.train import evaluate_model, train_and_evaluate
from unittest.mock import patch, MagicMock

def test_evaluate_model():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.6, 0.8])
    
    metrics = evaluate_model(y_true, y_pred, y_proba)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] == 0.75

@patch("mlflow.sklearn.log_model")
@patch("mlflow.log_metrics")
@patch("mlflow.log_params")
@patch("mlflow.start_run")
@patch("mlflow.set_experiment")
@patch("mlflow.register_model")
@patch("mlflow.tracking.MlflowClient")
def test_train_and_evaluate_executes(
    mock_client, mock_register, mock_set_exp, mock_start_run, 
    mock_log_params, mock_log_metrics, mock_log_model
):
    # Mocking MLflow context manager
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"
    mock_start_run.return_value.__enter__.return_value = mock_run
    
    # Create dummy data
    data = {
        'CustomerId': range(50),
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'cat_feature': ['A', 'B'] * 25,
        'is_high_risk': [0, 1] * 25
    }
    df = pd.DataFrame(data)
    
    # This should run without error even if MLflow is mocked
    train_and_evaluate(df, target_col="is_high_risk")
    
    assert mock_set_exp.called
    assert mock_start_run.called
    assert mock_log_metrics.called
    assert mock_log_model.called
    assert mock_register.called
