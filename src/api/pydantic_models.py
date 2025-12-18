from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class PredictionRequest(BaseModel):
    total_transaction_amount: float = Field(..., description="Sum of all transaction amounts for the customer")
    avg_transaction_amount: float = Field(..., description="Mean transaction amount")
    transaction_count: int = Field(..., gt=0, description="Total number of transactions")
    std_transaction_amount: float = Field(0.0, description="Standard deviation of transaction amounts")

    # Pydantic v2 style configuration
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
    risk_probability: float = Field(..., ge=0, le=1)
    credit_score: int = Field(..., ge=300, le=850)
    risk_label: str = Field(..., pattern="^(HIGH_RISK|LOW_RISK)$")

    # Pydantic v2 style configuration
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
    status: str
    model_version: Optional[str] = None
    
    # Optional: Add ConfigDict if you need any configuration
    model_config = ConfigDict()