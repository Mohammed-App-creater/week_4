
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Relative imports from src
try:
    from src.data_processing import build_feature_pipeline
except ImportError:
    # Handle cases where src is not in the path (e.g., running from project root)
    from data_processing import build_feature_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Computes key classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }
    return metrics

def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str = "is_high_risk"
) -> None:
    """
    Trains multiple models, logs experiments to MLflow,
    and registers the best model based on ROC-AUC.
    """
    logger.info("Starting model training pipeline...")

    # 1. Data Preparation
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Dynamically identify categorical and numerical features for the pipeline
    # Excluding IDs or non-predictive columns if present
    cols_to_exclude = ['CustomerId', 'rfm_cluster']
    X_train_full = X.drop(columns=[col for col in cols_to_exclude if col in X.columns])
    
    categorical_features = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X_train_full.select_dtypes(include=['number']).columns.tolist()

    logger.info(f"Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    # 2. Experiment Setup
    mlflow.set_experiment("credit-risk-model")
    
    models_to_train = [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42),
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0]
            }
        },
        {
            "name": "RandomForest",
            "estimator": RandomForestClassifier(class_weight='balanced', random_state=42),
            "params": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [None, 10, 20]
            }
        }
    ]

    best_roc_auc = -1.0
    best_run_id = None
    best_model_name = ""

    # 3. Training Loop
    for model_info in models_to_train:
        model_name = model_info["name"]
        logger.info(f"Training {model_name}...")

        with mlflow.start_run(run_name=model_name) as run:
            # Create Pipeline
            feature_pipeline = build_feature_pipeline(categorical_features, numerical_features)
            full_pipeline = Pipeline(steps=[
                ('preprocessor', feature_pipeline.named_steps['preprocessor']),
                ('classifier', model_info["estimator"])
            ])

            # Hyperparameter Tuning
            grid_search = GridSearchCV(
                full_pipeline, 
                param_grid=model_info["params"], 
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Evaluation
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_proba)

            # Log Params and Metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            
            # Log Model
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            # Explainability: Log coefficients or importance
            if model_name == "LogisticRegression":
                # Get feature names after preprocessing
                # Note: This requires accessing transformer names from preprocessor
                try:
                    preprocessor = best_model.named_steps['preprocessor']
                    ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
                    all_feature_names = numerical_features + list(ohe_feature_names)
                    
                    coeffs = best_model.named_steps['classifier'].coef_[0]
                    importance_df = pd.DataFrame({
                        'feature': all_feature_names,
                        'coefficient': coeffs
                    }).sort_values(by='coefficient', ascending=False)
                    
                    importance_path = "coefficients.csv"
                    importance_df.to_csv(importance_path, index=False)
                    mlflow.log_artifact(importance_path)
                    os.remove(importance_path)
                    logger.info(f"Logged coefficients for {model_name}")
                except Exception as e:
                    logger.warning(f"Could not log coefficients: {e}")

            logger.info(f"{model_name} ROC-AUC: {metrics['roc_auc']:.4f}")

            # Keep track of the best model
            if metrics['roc_auc'] > best_roc_auc:
                best_roc_auc = metrics['roc_auc']
                best_run_id = run.info.run_id
                best_model_name = model_name

    # 4. Model Registration
    if best_run_id:
        logger.info(f"Registering best model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, "CreditRiskModel")
        
        # Add description to version 1 (assuming it's the first)
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(
            name="CreditRiskModel",
            version=1,
            description="Proxy target: is_high_risk (RFM+KMeans). Features: aggregated transaction data. Use: Credit scoring and risk assessment."
        )

if __name__ == "__main__":
    # Example usage / testing entry point
    # In a real scenario, you'd load the proxy-labeled data here
    pass
