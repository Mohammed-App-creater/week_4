
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
    customer_features = generate_customer_features(sample_transaction_data)
    
    assert len(customer_features) == 8 # C1 to C8
    assert 'total_transaction_amount' in customer_features.columns
    assert 'avg_transaction_amount' in customer_features.columns
    assert 'transaction_count' in customer_features.columns
    assert 'std_transaction_amount' in customer_features.columns
    
    # Check specific values for CustomerId C1
    c1 = customer_features[customer_features['CustomerId'] == 'C1'].iloc[0]
    assert c1['total_transaction_amount'] == 300.0
    assert c1['transaction_count'] == 2
    assert c1['avg_transaction_amount'] == 150.0

def test_temporal_feature_extractor(sample_transaction_data):
    extractor = TemporalFeatureExtractor()
    transformed_df = extractor.transform(sample_transaction_data)
    
    assert 'transaction_hour' in transformed_df.columns
    assert 'transaction_day' in transformed_df.columns
    assert 'transaction_month' in transformed_df.columns
    assert 'transaction_year' in transformed_df.columns
    assert 'TransactionStartTime' not in transformed_df.columns
    
    # Check hour for first row
    assert transformed_df.iloc[0]['transaction_hour'] == 10

def test_build_feature_pipeline(sample_transaction_data):
    categorical = ['ProductCategory', 'ChannelId']
    numerical = ['Amount']
    
    pipeline = build_feature_pipeline(categorical, numerical)
    assert isinstance(pipeline, Pipeline)
    
    # Fit and transform
    transformed = pipeline.fit_transform(sample_transaction_data)
    
    # 1 numerical + 3 unique categories for ProductCategory (A, B, C) + 2 for ChannelId (1, 2)
    # Total cols = 1 + 3 + 2 = 6
    assert transformed.shape[1] == 6

def test_apply_woe_transformation(sample_transaction_data):
    # This test might fail if xverse is not installed, but let's assume it is or will be.
    try:
        df_woe, iv_table = apply_woe_transformation(sample_transaction_data, 'is_high_risk')
        assert isinstance(df_woe, pd.DataFrame)
        assert isinstance(iv_table, pd.DataFrame)
        assert 'is_high_risk' not in df_woe.columns
    except ImportError:
        pytest.skip("xverse not installed")
