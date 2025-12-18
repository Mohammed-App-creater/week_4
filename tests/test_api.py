import pytest
from fastapi.testclient import TestClient
from src.api.main import app, transform_probability_to_score

client = TestClient(app)

def test_transform_probability_to_score():
    assert transform_probability_to_score(0.0) == 850
    assert transform_probability_to_score(1.0) == 300
    assert transform_probability_to_score(0.5) == 575
    assert 300 <= transform_probability_to_score(0.73) <= 850

def test_health_check_unhealthy():
    # Model is None by default in tests unless mocked/loaded
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "unhealthy"]

def test_predict_no_model():
    response = client.post("/predict", json={
        "total_transaction_amount": 5000.0,
        "avg_transaction_amount": 500.0,
        "transaction_count": 10,
        "std_transaction_amount": 50.0
    })
    # If model is not loaded, it returns 503
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"

def test_predict_invalid_input():
    response = client.post("/predict", json={
        "total_transaction_amount": "invalid",
        "avg_transaction_amount": 500.0,
        "transaction_count": 10
    })
    assert response.status_code == 422 # Validation error
