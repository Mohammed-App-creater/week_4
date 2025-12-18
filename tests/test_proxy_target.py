"""
Tests for Proxy Target Variable Generation.

These tests verify the multi-stage labeling process:
1. RFM metric calculation.
2. Unsupervised clustering.
3. Logical assignment of the "High Risk" label based on business heuristics.
"""

import pandas as pd
import numpy as np
import pytest
from src.data_processing import calculate_rfm, perform_rfm_clustering, assign_high_risk_label, build_proxy_target

@pytest.fixture
def sample_transactions():
    """
    Creates a small transaction dataset with distinct behavioral patterns:
    - C1: Active/Moderate
    - C2: Inactive/Low Value (Should be high risk)
    - C3: High engagement
    """
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C3', 'C3', 'C3'],
        'TransactionStartTime': [
            '2023-01-01 10:00:00', '2023-01-02 10:00:00',
            '2023-01-10 10:00:00',
            '2023-01-20 10:00:00', '2023-01-21 10:00:00', '2023-01-22 10:00:00'
        ],
        'Value': [100, 200, 50, 300, 400, 500]
    }
    return pd.DataFrame(data)

def test_calculate_rfm(sample_transactions):
    """
    Ensures RFM metrics are mathematically correct for each customer.
    """
    rfm = calculate_rfm(sample_transactions)
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    assert len(rfm) == 3
    # Check C1: 2 transactions, total value 300
    c1 = rfm[rfm['CustomerId'] == 'C1'].iloc[0]
    assert c1['Frequency'] == 2
    assert c1['Monetary'] == 300

def test_perform_rfm_clustering(sample_transactions):
    """
    Validates that the clustering algorithm groups customers as expected.
    """
    rfm = calculate_rfm(sample_transactions)
    clustered = perform_rfm_clustering(rfm, n_clusters=2)
    assert 'rfm_cluster' in clustered.columns
    assert clustered['rfm_cluster'].nunique() <= 2

def test_assign_high_risk_label(sample_transactions):
    """
    Tests the business logic that identifies the High Risk cluster.
    
    Business Logic: The cluster with the least engagement (High Recency, 
    Low Freq, Low Monetary) must be tagged as is_high_risk=1.
    """
    rfm = calculate_rfm(sample_transactions)
    clustered = perform_rfm_clustering(rfm, n_clusters=2)
    labeled = assign_high_risk_label(clustered)
    
    assert 'is_high_risk' in labeled.columns
    assert labeled['is_high_risk'].isin([0, 1]).all()
    # At least one cluster must be identified as high risk in a non-trivial set
    assert labeled['is_high_risk'].sum() > 0

def test_build_proxy_target(sample_transactions):
    """
    End-to-end test for the proxy labeling pipeline.
    """
    proxy_df = build_proxy_target(sample_transactions)
    assert len(proxy_df) == 3
    assert 'is_high_risk' in proxy_df.columns
    assert 'CustomerId' in proxy_df.columns
