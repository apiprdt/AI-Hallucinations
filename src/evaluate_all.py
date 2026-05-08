import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error, r2_score
import os

def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def evaluate_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for k, v in data["results"].items():
        results.append(v)
    df = pd.DataFrame(results)
    
    # Identify if regression or classification
    labels = df['label'].unique()
    is_classification = len(labels) <= 2 and all(l in [0, 1] for l in labels)
    
    metrics = []
    # Use conditions found in columns, ensure they are prediction columns (numeric)
    conditions = [c for c in df.columns if re.match(r'C\d', c) and not c.endswith('_text')]
    
    for cond in conditions:
        temp_df = df[['label', cond]].dropna()
        if temp_df.empty: continue
        
        y_true = temp_df['label'].values
        y_pred = pd.to_numeric(temp_df[cond], errors='coerce').values
        
        # Filter out NaNs from coercion
        valid_idx = ~np.isnan(y_pred)
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]
        
        if len(y_true) < 10: continue

        if is_classification:
            auc = roc_auc_score(y_true, y_pred)
            brier = brier_score_loss(y_true, y_pred)
            ece = calculate_ece(y_true, y_pred)
            metrics.append({
                "Condition": cond,
                "ROC-AUC": auc,
                "Brier": brier,
                "ECE": ece
            })
        else:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            metrics.append({
                "Condition": cond,
                "RMSE": rmse,
                "R2": r2
            })
            
    return pd.DataFrame(metrics), is_classification

import re
if __name__ == "__main__":
    processed_dir = "data/processed/"
    files = [f for f in os.listdir(processed_dir) if f.endswith("_checkpoint.json")]
    
    all_metrics = {}
    for f in files:
        name = f.replace("results_", "").replace("_checkpoint.json", "")
        print(f"\n--- Evaluating {name.upper()} ---")
        try:
            m_df, is_clf = evaluate_dataset(os.path.join(processed_dir, f))
            print(m_df.to_string(index=False))
            all_metrics[name] = m_df
        except Exception as e:
            print(f"Error evaluating {f}: {e}")
