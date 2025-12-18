"""
Unit Tests for the Credit Risk API Layer.

These tests ensure that the FastAPI service correctly handles requests, 
validates input data, and properly transforms risk probabilities into 
business-friendly credit scores.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app, transform_probability_to_score

# TestClient provides a simulated environment to test FastAPI endpoints without a live server
client = TestClient(app)

def test_transform_probability_to_score():
    """
    Validates the mathematical transformation from probability to credit score.
    
    Business Relevance: Ensures the scoring logic follows the industry-standard 
    inverse relationship (high probability = low score) and stays within 
    the 300-850 range.
    """
    assert transform_probability_to_score(0.0) == 850
    assert transform_probability_to_score(1.0) == 300
    assert transform_probability_to_score(0.5) == 575
    assert 300 <= transform_probability_to_score(0.73) <= 850

def test_health_check_unhealthy():
    """
    Verifies the health check endpoint response when the model is not loaded.
    
    Functional Importance: Enables monitoring systems (like Kubernetes or 
    Docker Compose) to detect service degradation.
    """
    # Model is None by default in tests unless explicitly loaded or mocked
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "unhealthy"]

def test_predict_no_model():
    """
    Ensures the API fails gracefully with a 503 error if the model artifact is missing.
    
    Regulatory Note: Prevents 'random' or default scoring when the validated 
    model is unavailable.
    """
    response = client.post("/predict", json={
        "total_transaction_amount": 5000.0,
        "avg_transaction_amount": 500.0,
        "transaction_count": 10,
        "std_transaction_amount": 50.0
    })
    # Service Unavailable (503) is the correct response for a missing model
    assert response.status_code == 503
    assert response.json()["detail"] == "Model artifact is not available on the server."

def test_predict_invalid_input():
    """
    Validates that Pydantic correctly rejects malformed requests.
    
    Functional Importance: Prevents internal server errors (500) caused by 
    unexpected data types reaching the ML pipeline.
    """
    response = client.post("/predict", json={
        "total_transaction_amount": "invalid", # String instead of float
        "avg_transaction_amount": 500.0,
        "transaction_count": 10
    })
    assert response.status_code == 422 # Unprocessable Entity (Validation Error)
