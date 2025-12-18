"""
Data Processing and Feature Engineering Module for Credit Risk Modeling.

This module provides tools for transforming raw transaction-level eCommerce data 
into customer-level features suitable for credit scoring. It includes custom 
transformers for temporal features, Weight of Evidence (WoE) encoding, and 
RFM-based (Recency, Frequency, Monetary) proxy target labeling.

Adherence to Basel II principles is maintained by prioritizing interpretable 
feature transformations like WoE and RFM analysis.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract datetime features from TransactionStartTime.
    
    Extracting temporal components allows the model to capture cyclical spending 
    behavior and potential risk patterns associated with specific times (e.g., 
    night-time transactions or end-of-month spikes).
    """
    def __init__(self, time_col: str = 'TransactionStartTime'):
        """
        Initializes the extractor.
        
        Args:
            time_col (str): Name of the column containing timestamp strings.
        """
        self.time_col = time_col

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op fit method for pipeline compatibility.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts hour, day, month, and year from the timestamp column.
        
        Args:
            X (pd.DataFrame): Input dataframe with defined time_col.
            
        Returns:
            pd.DataFrame: Dataframe with new temporal features and dropped time_col.
        """
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col])
        X['transaction_hour'] = X[self.time_col].dt.hour
        X['transaction_day'] = X[self.time_col].dt.day
        X['transaction_month'] = X[self.time_col].dt.month
        X['transaction_year'] = X[self.time_col].dt.year
        return X.drop(columns=[self.time_col])

def generate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction-level data to customer-level profiles.
    
    Credit risk is typically assessed at the customer level. This function 
    summarizes raw spending behavior into predictive aggregates.

    Args:
        df (pd.DataFrame): Transaction-level dataset containing CustomerId and Amount.

    Returns:
        pd.DataFrame: Customer-level dataset with columns:
            - total_transaction_amount (Total volume)
            - avg_transaction_amount (Mean value - spending power proxy)
            - transaction_count (Frequency - engagement proxy)
            - std_transaction_amount (Volatility - risk proxy)
    """
    agg_funcs = {
        'Amount': ['sum', 'mean', 'count', 'std']
    }
    
    customer_features = df.groupby('CustomerId').agg(agg_funcs).reset_index()
    
    # Flatten multi-index columns for cleaner downstream processing
    customer_features.columns = [
        'CustomerId', 
        'total_transaction_amount', 
        'avg_transaction_amount', 
        'transaction_count', 
        'std_transaction_amount'
    ]
    
    # Fill NaN for std where count is 1 (standard deviation is undefined)
    customer_features['std_transaction_amount'] = customer_features['std_transaction_amount'].fillna(0)
    
    return customer_features

def build_feature_pipeline(categorical_features: List[str], numerical_features: List[str]) -> Pipeline:
    """
    Returns a unified sklearn Pipeline for robust feature engineering.
    
    This centralized pipeline ensures that the same transformations are applied 
    identically during training and production inference (FastAPI), preventing 
    training-serving skew.

    Args:
        categorical_features (List[str]): List of column names for OHE.
        numerical_features (List[str]): List of column names for Scaling.

    Returns:
        Pipeline: Sklearn pipeline object containing preprocessor.
    """
    
    # Handle missing numerical values with median to reduce outlier influence
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Handle missing categorical values with mode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return Pipeline(steps=[('preprocessor', preprocessor)])

class WeightOfEvidenceTransformer(BaseEstimator, TransformerMixin):
    """
    Custom WoE Transformer that calculates Weight of Evidence and Information Value.
    
    WoE is a standard technique in banking (Basel II compliant) used to linearize 
    categorical features and handle continuous variables via binning. It provides 
    direct explainability for feature contribution to risk.
    """
    def __init__(self, bins: int = 10):
        """
        Initializes the WoE transformer.
        
        Args:
            bins (int): Number of bins for continuous numerical features.
        """
        self.bins = bins
        self.woe_maps_ = {}
        self.iv_table_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Calculates WoE and IV for each feature in X based on target y.
        
        WoE = ln(Distribution of Good / Distribution of Bad)
        IV = sum((Distr Good - Distr Bad) * WoE)

        Args:
            X (pd.DataFrame): Features to transform.
            y (pd.Series): Binary target (1 = High Risk/Bad, 0 = Low Risk/Good).
        """
        X = X.copy()
        iv_data = []
        
        for col in X.columns:
            # Pre-bin numerical features to treat them as categories
            if X[col].dtype.kind in 'iuf': # Numerical
                X_binned = pd.qcut(X[col], q=self.bins, duplicates='drop').astype(str)
            else: # Categorical
                X_binned = X[col].astype(str)
            
            temp_df = pd.DataFrame({'feature': X_binned, 'target': y})
            
            # Aggregate stats for PD calculation
            stats = temp_df.groupby('feature')['target'].agg(['count', 'sum'])
            stats.columns = ['Total', 'Bad']
            stats['Good'] = stats['Total'] - stats['Bad']
            
            # Add small epsilon to avoid log(0) and division by zero
            # Regulatory Note: Using an epsilon is common to ensure model stability
            eps = 1e-6
            global_good = stats['Good'].sum() + eps
            global_bad = stats['Bad'].sum() + eps
            
            stats['Distr_Good'] = (stats['Good'] + eps) / global_good
            stats['Distr_Bad'] = (stats['Bad'] + eps) / global_bad
            
            stats['WoE'] = np.log(stats['Distr_Good'] / stats['Distr_Bad'])
            stats['IV'] = (stats['Distr_Good'] - stats['Distr_Bad']) * stats['WoE']
            
            self.woe_maps_[col] = stats['WoE'].to_dict()
            iv_data.append({'Variable': col, 'Information Value': stats['IV'].sum()})
            
        self.iv_table_ = pd.DataFrame(iv_data).sort_values(by='Information Value', ascending=False)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces raw feature values with their calculated WoE.
        
        Args:
            X (pd.DataFrame): Data to transform.
            
        Returns:
            pd.DataFrame: WoE encoded dataframe.
        """
        X_tf = X.copy()
        for col, woe_map in self.woe_maps_.items():
            if X_tf[col].dtype.kind in 'iuf':
                # Simplified re-binning for transformation consistency
                X_binned = pd.qcut(X_tf[col], q=self.bins, duplicates='drop').astype(str)
                X_tf[col] = X_binned.map(woe_map).fillna(0)
            else:
                X_tf[col] = X_tf[col].astype(str).map(woe_map).fillna(0)
        return X_tf

def apply_woe_transformation(
    df: pd.DataFrame, 
    target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies Weight of Evidence (WoE) transformation to the entire dataset.
    
    Used to prepare features for Logistic Regression, ensuring each feature 
    linearly contributes to the log-odds of risk.

    Args:
        df (pd.DataFrame): Customer-level dataset.
        target_col (str): Name of the binary target column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - df_woe: WoE transformed features.
            - iv_table: Summary of feature predictive power (Information Value).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Filter for valid feature types
    X = X.select_dtypes(include=['number', 'object', 'category'])
    
    transformer = WeightOfEvidenceTransformer()
    transformer.fit(X, y)
    df_woe = transformer.transform(X)
    
    return df_woe, transformer.iv_table_

def calculate_rfm(
    df: pd.DataFrame,
    customer_col: str = "CustomerId",
    date_col: str = "TransactionStartTime",
    monetary_col: str = "Value"
) -> pd.DataFrame:
    """
    Calculates Recency, Frequency, and Monetary (RFM) metrics per customer.
    
    RFM is a proven behavioral segmentation framework. In credit risk without 
    hard labels, dormancy (high recency) and low spend (low monetary) are often 
    early indicators of customer distress or churn.

    Args:
        df (pd.DataFrame): Transaction-level dataset.

    Returns:
        pd.DataFrame: RFM features per CustomerId.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Snapshot date is 1 day after the latest transaction to avoid 0-day recency
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        customer_col: 'count',
        monetary_col: 'sum'
    })
    
    rfm.rename(columns={
        date_col: 'Recency',
        customer_col: 'Frequency',
        monetary_col: 'Monetary'
    }, inplace=True)
    
    return rfm.reset_index()

def perform_rfm_clustering(
    rfm_df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Segments customers into behavioral groups using K-Means clustering.
    
    Clustering allows us to discover natural groupings in spending patterns. 
    One of these groups will be designated as "High Risk" based on being outliers 
    in low engagement.

    Args:
        rfm_df (pd.DataFrame): Customer RFM metrics.
        n_clusters (int): Number of clusters (default 3: High/Med/Low risk proxy).

    Returns:
        pd.DataFrame: RFM dataframe with a new 'rfm_cluster' column.
    """
    rfm_df = rfm_df.copy()
    
    # Preprocessing: RFM features are often skewed; log transform stabilizes variance
    eps = 1e-6
    features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
    features['Frequency'] = np.log1p(features['Frequency'])
    features['Monetary'] = np.log1p(features['Monetary'] + eps)
    
    # Scaling is CRITICAL for distance-based algorithms like K-Means
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    rfm_df['rfm_cluster'] = kmeans.fit_predict(scaled_features)
    
    return rfm_df

def assign_high_risk_label(
    rfm_clustered_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Identifies the "High Risk" cluster and assigns binary labels.
    
    Business Logic:
    High Risk is behaviorally defined as High Recency (inactive longest) 
    AND Low Frequency (least number of transactions) 
    AND Low Monetary (lowest financial volume). This represents customers 
    who have effectively disengaged from the BNPL service.

    Args:
        rfm_clustered_df (pd.DataFrame): RFM data with cluster IDs.

    Returns:
        pd.DataFrame: Dataset with 'is_high_risk' (binary proxy target).
    """
    rfm_clustered_df = rfm_clustered_df.copy()
    
    cluster_stats = rfm_clustered_df.groupby('rfm_cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    # Rank clusters: 
    # High Recency = High Risk (ascending rank for Recency mean)
    # Low Freq/Monetary = High Risk (ascending rank for Freq/Monetary means)
    
    recency_rank = cluster_stats['Recency'].rank(ascending=False)
    freq_rank = cluster_stats['Frequency'].rank(ascending=True)
    monetary_rank = cluster_stats['Monetary'].rank(ascending=True)
    
    # The cluster with the minimum sum of ranks is the most "extreme" on the risk end
    total_rank = recency_rank + freq_rank + monetary_rank
    high_risk_cluster = total_rank.idxmin()
    
    rfm_clustered_df['is_high_risk'] = (rfm_clustered_df['rfm_cluster'] == high_risk_cluster).astype(int)
    
    return rfm_clustered_df

def build_proxy_target(
    transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Executes the full pipeline to generate a financial risk proxy target.
    
    This function bridges the gap between raw data and a model-ready target, 
    enabling supervised learning in an unsupervised labeling environment.

    Args:
        transactions_df (pd.DataFrame): Raw transaction-level data.

    Returns:
        pd.DataFrame: Customer-level data with proxy labels.
    """
    rfm = calculate_rfm(transactions_df)
    rfm_clustered = perform_rfm_clustering(rfm)
    proxy_df = assign_high_risk_label(rfm_clustered)
    
    return proxy_df
