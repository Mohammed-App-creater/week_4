import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

# --- Configuration ---
DATA_PATH = 'data/data.csv'
OUTPUT_DIR = 'report_assets'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
SUMMARY_FILE = os.path.join(OUTPUT_DIR, 'summary.md')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def prepare_output_dirs():
    """Creates the output directory structure, clearing it if it exists."""
    if os.path.exists(OUTPUT_DIR):
        print(f"Directory {OUTPUT_DIR} exists, cleaning up...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(PLOTS_DIR)
    print(f"Created output directories: {OUTPUT_DIR}, {PLOTS_DIR}")

def load_data(filepath):
    """Loads dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded data with shape: {df.shape}")
    return df

def generate_summary(df):
    """Generates a summary string of the dataset."""
    summary = []
    summary.append("# Interim Report - Data Analysis Summary\n")
    
    summary.append("## 1. Dataset Overview")
    summary.append(f"- **Total Rows:** {df.shape[0]}")
    summary.append(f"- **Total Columns:** {df.shape[1]}")
    summary.append(f"- **Missing Values:** {df.isnull().sum().sum()} total missing cells.\n")
    
    summary.append("## 2. Key Observations")
    
    # Class Imbalance
    if 'FraudResult' in df.columns:
        fraud_counts = df['FraudResult'].value_counts()
        fraud_rate = (fraud_counts[1] / len(df)) * 100
        summary.append(f"### Class Imbalance (FraudResult)")
        summary.append(f"- **Non-Fraud (0):** {fraud_counts.get(0, 0)}")
        summary.append(f"- **Fraud (1):** {fraud_counts.get(1, 0)}")
        summary.append(f"- **Fraud Rate:** {fraud_rate:.2f}%\n")
        
    # Numerical Stats
    summary.append("### Numerical Statistics")
    num_desc = df.describe().to_markdown()
    summary.append(num_desc + "\n")
    
    # Skewness
    summary.append("### Skewness")
    numeric_df = df.select_dtypes(include=[np.number])
    skew_vals = numeric_df.skew()
    summary.append("High skewness observed in:")
    for col, val in skew_vals.items():
        if abs(val) > 1:
            summary.append(f"- **{col}:** {val:.2f}")
    summary.append("\n")
    
    return "\n".join(summary)

def save_plot(filename):
    """Helper to save plot and close figure."""
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filename}")
    return filepath

def generate_visualizations(df):
    """Generates and saves all required plots."""
    plot_md_links = []
    
    # 1. Transaction Amount Distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Amount'], bins=50, kde=True)
    plt.title('Distribution of Transaction Amount')
    
    plt.subplot(1, 2, 2)
    # Log transform for Value avoids negative issues if Value > 0, otherwise shift
    # Using Value for log plot as Amount can be negative
    sns.histplot(np.log1p(df['Value']), bins=50, kde=True, color='orange')
    plt.title('Log-Distribution of Transaction Value')
    save_plot('dist_amount_value.png')
    plot_md_links.append("## 3. Visualizations\n")
    plot_md_links.append("### Transaction Amount Distribution")
    plot_md_links.append("![Amount Distribution](plots/dist_amount_value.png)\n")
    
    # 2. Fraud Distribution
    if 'FraudResult' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='FraudResult', data=df)
        plt.title('Fraud Count Distribution')
        plt.yscale('log') # Log scale because of high imbalance
        plt.ylabel('Count (Log Scale)')
        save_plot('fraud_count_log.png')
        plot_md_links.append("### Fraud Distribution (Log Scale)")
        plot_md_links.append("![Fraud Distribution](plots/fraud_count_log.png)\n")
    
    # 3. Correlation Matrix
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        save_plot('correlation_matrix.png')
        plot_md_links.append("### Correlation Matrix")
        plot_md_links.append("![Correlation Matrix](plots/correlation_matrix.png)\n")

    # 4. Boxplot for Value vs Fraud
    if 'FraudResult' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='FraudResult', y='Value', data=df)
        plt.title('Transaction Value by Fraud Result')
        plt.yscale('log')
        save_plot('boxplot_value_fraud.png')
        plot_md_links.append("### Outlier Detection: Value vs Fraud")
        plot_md_links.append("![Boxplot](plots/boxplot_value_fraud.png)\n")

    return "\n".join(plot_md_links)

def main():
    try:
        print("Starting Interim Report Asset Generation...")
        
        # 1. Setup
        prepare_output_dirs()
        
        # 2. Load
        df = load_data(DATA_PATH)
        
        # 3. Analysis & Summary Text
        summary_text = generate_summary(df)
        
        # 4. Plots
        plots_markdown = generate_visualizations(df)
        
        # 5. Write Full Report
        full_report = summary_text + plots_markdown
        
        with open(SUMMARY_FILE, 'w') as f:
            f.write(full_report)
            
        print(f"\nSuccess! Summary report saved to: {SUMMARY_FILE}")
        
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()
