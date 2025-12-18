import pandas as pd
import numpy as np
import pytest
from src.data_processing import calculate_rfm, perform_rfm_clustering, assign_high_risk_label, build_proxy_target

@pytest.fixture
def sample_transactions():
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
    rfm = calculate_rfm(sample_transactions)
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    assert len(rfm) == 3
    # Check C1: 2 transactions, total 300
    c1 = rfm[rfm['CustomerId'] == 'C1'].iloc[0]
    assert c1['Frequency'] == 2
    assert c1['Monetary'] == 300

def test_perform_rfm_clustering(sample_transactions):
    rfm = calculate_rfm(sample_transactions)
    clustered = perform_rfm_clustering(rfm, n_clusters=2)
    assert 'rfm_cluster' in clustered.columns
    assert clustered['rfm_cluster'].nunique() <= 2

def test_assign_high_risk_label(sample_transactions):
    rfm = calculate_rfm(sample_transactions)
    # Mocking high risk identification: C2 is least engaged (1 tx, lowest value, high recency)
    # However, KMeans with n=2 on 3 data points is a bit stochastic but we check the logic flow.
    clustered = perform_rfm_clustering(rfm, n_clusters=2)
    labeled = assign_high_risk_label(clustered)
    assert 'is_high_risk' in labeled.columns
    assert labeled['is_high_risk'].isin([0, 1]).all()
    assert labeled['is_high_risk'].sum() > 0

def test_build_proxy_target(sample_transactions):
    proxy_df = build_proxy_target(sample_transactions)
    assert len(proxy_df) == 3
    assert 'is_high_risk' in proxy_df.columns
    assert 'CustomerId' in proxy_df.columns
