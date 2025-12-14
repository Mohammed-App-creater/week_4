import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Define paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def train():
    print("Training model...")
    # Generate dummy data for demonstration purposes
    # In a real scenario, you would load data from data/processed/
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    
    # Train a simple model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
