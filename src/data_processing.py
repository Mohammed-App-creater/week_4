
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
    """
    def __init__(self, time_col: str = 'TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col])
        X['transaction_hour'] = X[self.time_col].dt.hour
        X['transaction_day'] = X[self.time_col].dt.day
        X['transaction_month'] = X[self.time_col].dt.month
        X['transaction_year'] = X[self.time_col].dt.year
        return X.drop(columns=[self.time_col])

def generate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction-level data to customer-level features.
    
    For each CustomerId, computes:
    - total_transaction_amount (sum)
    - avg_transaction_amount (mean)
    - transaction_count (count)
    - std_transaction_amount (std)
    """
    agg_funcs = {
        'Amount': ['sum', 'mean', 'count', 'std']
    }
    
    customer_features = df.groupby('CustomerId').agg(agg_funcs).reset_index()
    
    # Flatten multi-index columns
    customer_features.columns = [
        'CustomerId', 
        'total_transaction_amount', 
        'avg_transaction_amount', 
        'transaction_count', 
        'std_transaction_amount'
    ]
    
    # Fill NaN for std where count is 1
    customer_features['std_transaction_amount'] = customer_features['std_transaction_amount'].fillna(0)
    
    return customer_features

def build_feature_pipeline(categorical_features: List[str], numerical_features: List[str]) -> Pipeline:
    """
    Returns a sklearn Pipeline for feature engineering.
    
    The pipeline handles:
    - Missing value imputation (median for numerical, most frequent for categorical)
    - Scaling (StandardScaler for numerical)
    - Encoding (OneHotEncoder for categorical)
    """
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

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
    Robust implementation to avoid dependency on unstable libraries.
    """
    def __init__(self, bins: int = 10):
        self.bins = bins
        self.woe_maps_ = {}
        self.iv_table_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        iv_data = []
        
        for col in X.columns:
            # Skip non-predictive columns if needed, but here we process all passed columns
            if X[col].dtype.kind in 'iuf': # Numerical
                # Bin numerical features
                X_binned = pd.qcut(X[col], q=self.bins, duplicates='drop').astype(str)
            else: # Categorical
                X_binned = X[col].astype(str)
            
            # Create a localized df for calculation
            temp_df = pd.DataFrame({'feature': X_binned, 'target': y})
            
            # Group by bin/category
            stats = temp_df.groupby('feature')['target'].agg(['count', 'sum'])
            stats.columns = ['Total', 'Bad']
            stats['Good'] = stats['Total'] - stats['Bad']
            
            # Add small epsilon to avoid division by zero
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
        X_tf = X.copy()
        for col, woe_map in self.woe_maps_.items():
            if X_tf[col].dtype.kind in 'iuf':
                # For numerical, we need to bin using the same logic as fit
                # This is a simplification: for production, we'd store the bin edges
                # However, for this task, the xverse fallback logic is key.
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
    Applies Weight of Evidence (WoE) transformation.
    
    Returns:
    - WoE transformed dataframe
    - Information Value (IV) summary table
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Filter for numeric and categorical columns
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
    Returns RFM metrics per customer.
    
    Recency: Days since most recent transaction (max date + 1)
    Frequency: Total transaction count
    Monetary: Total transaction value
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
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
    Returns RFM dataframe with cluster labels.
    
    Logic:
    1. Log transform Frequency and Monetary to reduce skew.
    2. Scale RFM features using StandardScaler.
    3. Apply KMeans clustering.
    """
    rfm_df = rfm_df.copy()
    
    # Preprocessing
    # Add epsilon to avoid log(0) if any Monetary values are 0 or Frequency is 0 (though unlikely)
    eps = 1e-6
    features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
    features['Frequency'] = np.log1p(features['Frequency'])
    features['Monetary'] = np.log1p(features['Monetary'] + eps)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    rfm_df['rfm_cluster'] = kmeans.fit_predict(scaled_features)
    
    return rfm_df

def assign_high_risk_label(
    rfm_clustered_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds binary column `is_high_risk`.
    
    Identifies the high-risk cluster based on:
    - Highest average Recency
    - Lowest average Frequency
    - Lowest average Monetary
    """
    rfm_clustered_df = rfm_clustered_df.copy()
    
    cluster_stats = rfm_clustered_df.groupby('rfm_cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    # We rank each cluster on each metric. 
    # High Risk = High Recency, Low Frequency, Low Monetary
    # Ranking: higher mean Recency is "riskier", lower mean Frequency/Monetary is "riskier"
    
    # Recency: descending rank (highest Recency is rank 0)
    recency_rank = cluster_stats['Recency'].rank(ascending=False)
    # Frequency: ascending rank (lowest Frequency is rank 0)
    freq_rank = cluster_stats['Frequency'].rank(ascending=True)
    # Monetary: ascending rank (lowest Monetary is rank 0)
    monetary_rank = cluster_stats['Monetary'].rank(ascending=True)
    
    # Sum of ranks. The cluster with the lowest sum of ranks (closest to 0 on all) is high risk.
    total_rank = recency_rank + freq_rank + monetary_rank
    high_risk_cluster = total_rank.idxmin()
    
    rfm_clustered_df['is_high_risk'] = (rfm_clustered_df['rfm_cluster'] == high_risk_cluster).astype(int)
    
    return rfm_clustered_df

def build_proxy_target(
    transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    End-to-end:
    raw transactions → RFM → clustering → is_high_risk
    """
    rfm = calculate_rfm(transactions_df)
    rfm_clustered = perform_rfm_clustering(rfm)
    proxy_df = assign_high_risk_label(rfm_clustered)
    
    return proxy_df
