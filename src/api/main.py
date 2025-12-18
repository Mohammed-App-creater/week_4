"""
FastAPI Service for Real-Time Credit Scoring.

This module implements a containerized REST API that serves the registered 
CreditRiskModel. It provides endpoints for health monitoring and risk 
prediction, converting model probabilities into business-friendly credit scores.
"""

import logging
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
import mlflow.sklearn
from src.api.pydantic_models import PredictionRequest, PredictionResponse, HealthResponse

# Configure logging to track API initialization and prediction events
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="A production-ready API for predicting credit risk likelihood and generating behavioral credit scores for BNPL customers.",
    version="1.0.0"
)

# Global variable to hold the model in memory for fast inference
model = None
MODEL_NAME = "CreditRiskModel"

def transform_probability_to_score(probability: float) -> int:
    """
    Maps a risk probability (0-1) to a standard Credit Score scale (300-850).
    
    The transformation logic follows an inverse relationship: 
    Higher probability of default leads to a lower credit score.
    Formula: score = 850 - (probability * 550)

    Args:
        probability (float): Model output (0.0 to 1.0).

    Returns:
        int: Scaled credit score between 300 and 850.
    """
    score = 850 - (probability * 550)
    # Clamping ensures the score stays within the standard industry range
    return int(max(300, min(850, score)))

@app.on_event("startup")
def load_model():
    """
    Initializes the API by loading the latest production model from MLflow.
    
    This occurs at startup to ensure that the first prediction request 
    does not suffer from 'cold start' latency.
    """
    global model
    try:
        # Configuration via environment variables for cloud/docker flexibility
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        logger.info(f"Loading latest version of model '{MODEL_NAME}' from {tracking_uri}...")
        
        # Pull the latest model version registered under the designated name
        model_uri = f"models:/{MODEL_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        
        logger.info("Successfully loaded model artifact.")
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        # In production, this might trigger an alert to the SRE team
        model = None

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Performs a system health check.
    
    Returns:
        HealthResponse: Status ('healthy' or 'unhealthy') and model availability.
    """
    if model is None:
        return HealthResponse(status="unhealthy", model_version=None)
    return HealthResponse(status="healthy", model_version="latest")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predicts credit risk and generates a score for a given customer profile.
    
    The endpoint orchestrates:
    1. Data validation via Pydantic.
    2. Feature formatting for the ML pipeline.
    3. Probability estimation.
    4. Score transformation and risk labeling.

    Args:
        request (PredictionRequest): Aggregated customer transaction features.

    Returns:
        PredictionResponse: Risk probability, credit score, and categorical label.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not available on the server.")

    try:
        # Convert Pydantic model to DataFrame - the ML pipeline expects structured data
        input_data = pd.DataFrame([request.model_dump()])
        
        # Execute the full ML pipeline (imputation -> scaling -> encoding -> prediction)
        probabilities = model.predict_proba(input_data)
        risk_probability = float(probabilities[0][1])
        
        # Business logic transformation
        credit_score = transform_probability_to_score(risk_probability)
        
        # Risk thresholds can be adjusted based on the bank's risk appetite
        risk_label = "HIGH_RISK" if risk_probability >= 0.5 else "LOW_RISK"
        
        return PredictionResponse(
            risk_probability=round(risk_probability, 4),
            credit_score=credit_score,
            risk_label=risk_label
        )
    except Exception as e:
        logger.error(f"Prediction logic failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during inference: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Local development entry point
    uvicorn.run(app, host="0.0.0.0", port=8000)
