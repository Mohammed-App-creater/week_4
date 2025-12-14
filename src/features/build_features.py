
import pandas as pd
import numpy as np
import os

def calculate_rfm(df):
    """
    Calculates Recency, Frequency, and Monetary (RFM) metrics for each customer.
    """
    # Ensure datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Reference date (day after the last transaction in the dataset)
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    # Use 'Value' for Monetary (absolute amount)
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Value': 'sum',
        'FraudResult': 'max' # Capture if customer ever committed fraud
    })
    
    # Rename columns
    rfm.rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Value': 'Monetary',
        'FraudResult': 'IsFraud'
    }, inplace=True)
    
    return rfm

def assign_risk_label(rfm_df):
    """
    Assigns a 'Risk_Label' (0=Good, 1=Bad) based on RFM scores and Fraud history.
    """
    # 1. Scoring (Simple Quantile-based)
    # Recency: Lower is better (higher score). 
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
    
    # Frequency: Higher is better.
    # Note: Duplicates in quantiles can happen if many users have 1 transaction. using rank method 'first' or checking unique counts
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    
    # Monetary: Higher is better.
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    
    # Combined RFM Score
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(int) + rfm_df['F_Score'].astype(int) + rfm_df['M_Score'].astype(int)
    
    # 2. Risk Definition Strategy
    # Threshold: Users with very low RFM scores are "High Risk" (Bad).
    # Fraudsters are automatically "High Risk" (Bad).
    
    # Define "Bad" (1) as Bottom Quartile of RFM Score or Fraud
    rfm_threshold = rfm_df['RFM_Score'].quantile(0.25)
    
    rfm_df['Risk_Label'] = np.where(
        (rfm_df['RFM_Score'] <= rfm_threshold) | (rfm_df['IsFraud'] == 1), 
        1, # Bad / High Risk
        0  # Good / Low Risk
    )
    
    print(f"RFM Threshold for High Risk: <= {rfm_threshold}")
    print("\nClass Distribution:")
    print(rfm_df['Risk_Label'].value_counts(normalize=True))
    
    return rfm_df

def main():
    # Paths
    input_path = os.path.join(os.path.dirname(__file__), '../../data/data.csv')
    output_path = os.path.join(os.path.dirname(__file__), '../../data/rfm_features.csv')
    
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    print("Calculating RFM...")
    rfm = calculate_rfm(df)
    
    print("Assigning Risk Labels...")
    rfm_labeled = assign_risk_label(rfm)
    
    print(f"Saving features to {output_path}...")
    rfm_labeled.to_csv(output_path)
    print("Done.")

if __name__ == '__main__':
    main()
