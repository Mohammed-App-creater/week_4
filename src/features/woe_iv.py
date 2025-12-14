
import pandas as pd
import numpy as np
import os

def calculate_woe_iv(df, feature, target):
    """
    Calculates WoE and IV for a specific feature.
    Assumes feature is categorical or binned.
    """
    lst = []
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': df[df[feature] == val].count()[feature],
            'Good': df[(df[feature] == val) & (df[target] == 0)].count()[feature],
            'Bad': df[(df[feature] == val) & (df[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    
    # Avoid div by zero
    dset['Distr_Good'] = dset['Distr_Good'].replace(0, 0.0001)
    dset['Distr_Bad'] = dset['Distr_Bad'].replace(0, 0.0001)

    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    
    iv = dset['IV'].sum()
    
    # Sort for display
    dset = dset.sort_values(by='WoE')
    
    return dset, iv

def main():
    input_path = os.path.join(os.path.dirname(__file__), '../../data/rfm_features.csv')
    df = pd.read_csv(input_path)
    
    # Analyze IV for the Score columns (R_Score, F_Score, M_Score)
    # These are already binned (1-5)
    
    features = ['R_Score', 'F_Score', 'M_Score']
    target = 'Risk_Label'
    
    print("Information Value (IV) Analysis:")
    print("-" * 30)
    
    iv_dict = {}
    
    for feat in features:
        # Ensure it's treated as categorical/int
        df[feat] = df[feat].astype(int)
        
        woe_df, iv = calculate_woe_iv(df, feat, target)
        iv_dict[feat] = iv
        
        print(f"\nFeature: {feat}")
        print(f"IV: {iv:.4f}")
        print(woe_df[['Value', 'All', 'Good', 'Bad', 'WoE', 'IV']])

    print("\n" + "-" * 30)
    print("IV Summary:")
    for k, v in iv_dict.items():
        print(f"{k}: {v:.4f}")
        
    # Standard IV Interpretation:
    # < 0.02: Useless
    # 0.02 - 0.1: Weak
    # 0.1 - 0.3: Medium
    # 0.3 - 0.5: Strong
    # > 0.5: Suspiciously Good

if __name__ == '__main__':
    main()
