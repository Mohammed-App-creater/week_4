"""
Unit Tests for Data Processing and Feature Engineering.

These tests validate that raw transaction data is correctly aggregated into 
customer profiles and that secondary features (temporal, WoE) are extracted 
accurately for the ML pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    generate_customer_features, 
    TemporalFeatureExtractor, 
    build_feature_pipeline,
    apply_woe_transformation
)
from sklearn.pipeline import Pipeline

@pytest.fixture
def sample_transaction_data():
    """
    Provides a standardized transaction-level dataset for testing.
    """
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
        'Amount': [100.0, 200.0, 50.0, 150.0, 300.0, 120.0, 80.0, 250.0, 50.0, 400.0],
        'TransactionStartTime': [
            '2023-01-01 10:00:00', '2023-01-01 12:00:00', 
            '2023-01-02 09:00:00', '2023-01-02 15:00:00', 
            '2023-01-03 11:00:00', '2023-01-04 10:00:00',
            '2023-01-05 08:00:00', '2023-01-06 14:00:00',
            '2023-01-07 16:00:00', '2023-01-08 09:00:00'
        ],
        'ProductCategory': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A', 'B'],
        'ChannelId': [1, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        'is_high_risk': [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

def test_generate_customer_features(sample_transaction_data):
    """
    Verifies the aggregation logic from transactions to customer profiles.
    
    Business Relevance: Accurate aggregation is the foundation of behavioral 
    credit scoring.
    """
    customer_features = generate_customer_features(sample_transaction_data)
    
    assert len(customer_features) == 8 # Unique count of C1 to C8
    assert 'total_transaction_amount' in customer_features.columns
    assert 'avg_transaction_amount' in customer_features.columns
    assert 'transaction_count' in customer_features.columns
    assert 'std_transaction_amount' in customer_features.columns
    
    # Check specific values for CustomerId C1 to ensure sum/count/mean are correct
    c1 = customer_features[customer_features['CustomerId'] == 'C1'].iloc[0]
    assert c1['total_transaction_amount'] == 300.0
    assert c1['transaction_count'] == 2
    assert c1['avg_transaction_amount'] == 150.0

def test_temporal_feature_extractor(sample_transaction_data):
    """
    Ensures datetime strings are correctly decomposed into temporal components.
    """
    extractor = TemporalFeatureExtractor()
    transformed_df = extractor.transform(sample_transaction_data)
    
    assert 'transaction_hour' in transformed_df.columns
    assert 'transaction_day' in transformed_df.columns
    assert 'transaction_month' in transformed_df.columns
    assert 'transaction_year' in transformed_df.columns
    assert 'TransactionStartTime' not in transformed_df.columns
    
    # Check hour for first row (10:00:00)
    assert transformed_df.iloc[0]['transaction_hour'] == 10

def test_build_feature_pipeline(sample_transaction_data):
    """
    Tests the integration of OHE and Scaling within the unified pipeline.
    """
    categorical = ['ProductCategory', 'ChannelId']
    numerical = ['Amount']
    
    pipeline = build_feature_pipeline(categorical, numerical)
    assert isinstance(pipeline, Pipeline)
    
    # Fit and transform to check output dimensionality
    transformed = pipeline.fit_transform(sample_transaction_data)
    
    # Calculation: 1 numeric + OHE(3 categories for Prod + 2 for Channel) = 6
    assert transformed.shape[1] == 6

def test_apply_woe_transformation(sample_transaction_data):
    """
    Validates the Weight of Evidence (WoE) and Information Value (IV) calculation.
    
    Regulatory Note: WoE transformations must be deterministic and statistically 
    valid for banking model approval.
    """
    df_woe, iv_table = apply_woe_transformation(sample_transaction_data, 'is_high_risk')
    assert isinstance(df_woe, pd.DataFrame)
    assert isinstance(iv_table, pd.DataFrame)
    # The target should be excluded from the feature set
    assert 'is_high_risk' not in df_woe.columns
