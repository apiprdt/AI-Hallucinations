import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine if points fall in this bin
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def analyze_calibration(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for i in range(data["metadata"]["total_molecules"]):
        results.append(data["results"][str(i)])
    
    df = pd.DataFrame(results)
    
    # Conditions to evaluate (mapped to paper codes)
    condition_map = {
        "C0_pure": "C0",
        "C1_rdkit": "C1",
        "C2_chempriming": "C2",
        "C2b_pure_gibberish": "C2b",
        "C3_hallu_free": "C3",
        "C4a_SP_intent": "C4a",
        "C4b_PI_intent": "C4b",
        "C4c_MF_intent": "C4c"
    }
    
    report = []
    for col, label in condition_map.items():
        if col in df.columns:
            y_true = df['label'].values
            y_prob = df[col].values
            
            brier = brier_score_loss(y_true, y_prob)
            ece = calculate_ece(y_true, y_prob)
            
            report.append({
                "Condition": label,
                "Brier Score": brier,
                "ECE": ece
            })
    
    return pd.DataFrame(report)

if __name__ == "__main__":
    print("Evaluating Calibration: BBBP")
    bbbp_cal = analyze_calibration("data/processed/results_bbbp_checkpoint.json")
    print(bbbp_cal.to_string(index=False))
    
    print("\nEvaluating Calibration: BACE")
    bace_cal = analyze_calibration("data/processed/results_bace_checkpoint.json")
    print(bace_cal.to_string(index=False))
    
    # Save to CSV for LaTeX integration
    bbbp_cal.to_csv("bbbp_calibration.csv", index=False)
    bace_cal.to_csv("bace_calibration.csv", index=False)
