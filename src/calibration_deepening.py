"""
Calibration Deepening Analysis.
Computes Brier Score decomposition (Reliability, Resolution, Uncertainty)
and prediction entropy per class for C0 vs C3 to show the distributional compression.
"""
import json
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from pathlib import Path

def load_data(path):
    with open(path) as f:
        ckpt = json.load(f)
    n = ckpt['metadata']['total_molecules']
    return pd.DataFrame([ckpt['results'][str(i)] for i in range(n)])

def brier_decomposition(y_true, y_prob, n_bins=10):
    """
    Decompose Brier Score into Reliability, Resolution, and Uncertainty.
    Brier = Reliability - Resolution + Uncertainty
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    # Need counts per bin to compute proper weighted sums
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.searchsorted(bins[1:-1], y_prob)
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total > 0
    prob_true_bin = np.zeros(len(bins))
    prob_pred_bin = np.zeros(len(bins))
    
    prob_true_bin[nonzero] = bin_true[nonzero] / bin_total[nonzero]
    prob_pred_bin[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
    
    base_rate = np.mean(y_true)
    
    reliability = np.sum(bin_total[nonzero] * (prob_true_bin[nonzero] - prob_pred_bin[nonzero])**2) / len(y_true)
    resolution = np.sum(bin_total[nonzero] * (prob_true_bin[nonzero] - base_rate)**2) / len(y_true)
    uncertainty = base_rate * (1.0 - base_rate)
    
    brier = reliability - resolution + uncertainty
    return brier, reliability, resolution, uncertainty

def analyze_calibration(df, dataset_name):
    print(f"\n{'='*60}")
    print(f"CALIBRATION DECOMPOSITION: {dataset_name}")
    print(f"{'='*60}")
    
    y_true = np.array(df['label'])
    base_rate = np.mean(y_true)
    print(f"Base rate (Positive class ratio): {base_rate:.3f}")
    
    conditions = ['C0_pure', 'C3_hallu_free', 'C4a_hallu_sp', 'C5_shuffled']
    results = []
    
    for cond in conditions:
        y_prob = np.array(df[cond])
        brier, rel, res, unc = brier_decomposition(y_true, y_prob, n_bins=10)
        
        # Entropy per class
        pos_probs = y_prob[y_true == 1]
        neg_probs = y_prob[y_true == 0]
        
        pos_clipped = np.clip(pos_probs, 1e-8, 1 - 1e-8)
        neg_clipped = np.clip(neg_probs, 1e-8, 1 - 1e-8)
        
        pos_ent = -np.mean(pos_clipped * np.log2(pos_clipped) + (1 - pos_clipped) * np.log2(1 - pos_clipped))
        neg_ent = -np.mean(neg_clipped * np.log2(neg_clipped) + (1 - neg_clipped) * np.log2(1 - neg_clipped))
        
        results.append({
            'Condition': cond.replace('_pure', '').replace('_hallu_free', '').replace('_hallu_sp', '').replace('_shuffled', ''),
            'Brier': f"{brier:.3f}",
            'Reliability': f"{rel:.3f}",
            'Resolution': f"{res:.3f}",
            'Pos Entropy': f"{pos_ent:.3f}",
            'Neg Entropy': f"{neg_ent:.3f}"
        })
        
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    return df_res

def main():
    bbbp_path = 'data/processed/results_bbbp_checkpoint.json'
    bace_path = 'data/processed/results_bace_checkpoint.json'
    
    df_bbbp = load_data(bbbp_path)
    df_bace = load_data(bace_path)
    
    res_bbbp = analyze_calibration(df_bbbp, 'BBBP')
    res_bace = analyze_calibration(df_bace, 'BACE')

if __name__ == '__main__':
    main()
