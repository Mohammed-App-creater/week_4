import os
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Initialize app
app = FastAPI(title="Credit Risk API", description="API for predicting credit risk")

# Load model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model.pkl")

# Define request body
class InputData(BaseModel):
    features: list[float]

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        model = None
        print(f"Warning: Model not found at {MODEL_PATH}. Run training script first.")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}
    
    # Convert input to numpy array and reshape for prediction
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features).tolist()
    
    return {
        "prediction": int(prediction[0]),
        "probability": probability[0]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
