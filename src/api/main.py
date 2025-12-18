import logging
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
import mlflow.sklearn
from src.api.pydantic_models import PredictionRequest, PredictionResponse, HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="API for predicting credit risk and generating credit scores.",
    version="1.0.0"
)

# Global variable for the model
model = None
MODEL_NAME = "CreditRiskModel"

def transform_probability_to_score(probability: float) -> int:
    """
    Maps probability (0-1) to credit score (300-850).
    score = 850 - (probability * 550)
    """
    score = 850 - (probability * 550)
    # Clamp between 300 and 850
    return int(max(300, min(850, score)))

@app.on_event("startup")
def load_model():
    """
    Load the latest production model from MLflow.
    """
    global model
    try:
        # MLflow tracking URI can be set via MLFLOW_TRACKING_URI env var
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        logger.info(f"Loading latest version of model '{MODEL_NAME}' from {tracking_uri}...")
        
        # Load the model from model registry
        # We assume the model is registered as 'CreditRiskModel'
        model_uri = f"models:/{MODEL_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        
        logger.info("Successfully loaded model.")
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        # In a real production scenario, you might want to retry or fail hard
        # For now, we'll log the error and requests will fail with 503
        model = None

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    """
    if model is None:
        return HealthResponse(status="unhealthy", model_version=None)
    return HealthResponse(status="healthy", model_version="latest")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk for a given customer.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to DataFrame (the model expects a DataFrame if it was trained on one)
        input_data = pd.DataFrame([request.dict()])
        
        # Predict probability
        # Assuming the model has predict_proba
        probabilities = model.predict_proba(input_data)
        risk_probability = float(probabilities[0][1])
        
        # Transform to score
        credit_score = transform_probability_to_score(risk_probability)
        
        # Assign label
        risk_label = "HIGH_RISK" if risk_probability >= 0.5 else "LOW_RISK"
        
        return PredictionResponse(
            risk_probability=round(risk_probability, 4),
            credit_score=credit_score,
            risk_label=risk_label
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
