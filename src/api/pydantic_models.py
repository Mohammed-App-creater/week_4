"""
Data Validation Models for the Credit Risk API.

This module defines the Pydantic schemas for request and response validation, 
ensuring type safety and providing documentation for the API's data contracts.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class PredictionRequest(BaseModel):
    """
    Schema for customer risk prediction requests.
    
    Requires customer-level aggregated transaction features which serve 
    as the primary predictors of creditworthiness.
    """
    total_transaction_amount: float = Field(..., description="Total cumulative transaction volume for the customer.")
    avg_transaction_amount: float = Field(..., description="The mathematical mean of all customer transactions.")
    transaction_count: int = Field(..., gt=0, description="Total number of completed transactions (must be > 0).")
    std_transaction_amount: float = Field(0.0, description="Measure of spending volatility (standard deviation of amounts).")

    # Metadata for API documentation and automated testing
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_transaction_amount": 5000.0,
                "avg_transaction_amount": 500.0,
                "transaction_count": 10,
                "std_transaction_amount": 50.0
            }
        }
    )

class PredictionResponse(BaseModel):
    """
    Schema for risk prediction results.
    
    Returns the raw probability and its derived business metrics.
    """
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of default (0.0 to 1.0).")
    credit_score: int = Field(..., ge=300, le=850, description="Scaled credit score following standard industry ranges.")
    risk_label: str = Field(..., pattern="^(HIGH_RISK|LOW_RISK)$", description="Categorical risk classification.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "risk_probability": 0.73,
                "credit_score": 527,
                "risk_label": "HIGH_RISK"
            }
        }
    )

class HealthResponse(BaseModel):
    """
    Schema for API health check status.
    """
    status: str = Field(..., description="Current server health status.")
    model_version: Optional[str] = Field(None, description="The version of the ML model currently loaded in memory.")
    
    model_config = ConfigDict()