import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score
from scipy import stats
from statsmodels.stats.multitest import multipletests

def load_data(checkpoint_file: str):
    with open(checkpoint_file, 'r') as f:
        ckpt = json.load(f)
    # Filter only completed results to allow analysis of partial checkpoints
    completed_indices = [int(k) for k in ckpt["results"].keys()]
    results_list = [ckpt["results"][str(i)] for i in sorted(completed_indices)]
    return pd.DataFrame(results_list)

def cohen_d_paired(x1, x2):
    diff = x1 - x2
    return np.mean(diff) / np.std(diff, ddof=1)

def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, ci=95, seed=42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.sort(bootstrapped_scores)
    lower = (100 - ci) / 2
    upper = 100 - lower
    return np.percentile(sorted_scores, lower), np.percentile(sorted_scores, upper)

def generate_final_report(df: pd.DataFrame):
    labels = df['label']
    conditions = ['C0_pure', 'C1_factual', 'C2_chempriming', 'C2b_pure_gibberish', 'C3_hallu_free', 'C4a_hallu_sp', 'C4b_hallu_pi', 'C4c_hallu_mf', 'C5_shuffled']
    
    # BAGIAN 1 — ROC-AUC Table
    auc_data = []
    c0_preds = df['C0_pure']
    
    p_values_raw = [1.0] # C0 vs C0 is 1.0
    cohen_ds = [0.0]
    for cond in conditions[1:]:
        _, p = stats.ttest_rel(df[cond], c0_preds)
        p_values_raw.append(p)
        cohen_ds.append(cohen_d_paired(df[cond], c0_preds))
    
    # BH Correction (excluding C0)
    _, p_corrected_others, _, _ = multipletests(p_values_raw[1:], method='fdr_bh')
    p_corrected = [1.0] + p_corrected_others.tolist()
    
    print("Calculating Confidence Intervals (Bootstrapping 1000x)...")
    for i, cond in enumerate(conditions):
        auc = roc_auc_score(labels, df[cond])
        ci_low, ci_high = bootstrap_auc_ci(labels, df[cond])
        
        auc_data.append({
            "Kondisi": cond,
            "ROC-AUC": f"{auc:.4f}",
            "95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
            "Cohen's d": f"{cohen_ds[i]:.3f}",
            "p-val(BH)": f"{p_corrected[i]:.2e}",
            "Sig?": "YES" if p_corrected[i] < 0.05 else "no"
        })
    
    report_df = pd.DataFrame(auc_data)
    
    print("\n" + "="*80)
    print("BAGIAN 1 — STATISTICAL RIGOR TABLE")
    print("="*80)
    print(report_df.to_string(index=False))
    
    # BAGIAN 2 — Taxonomy Distribution
    print("\n" + "="*80)
    print("BAGIAN 2 — Taxonomy Distribution:")
    print("="*80)
    df['tax_type'] = df['taxonomy_free'].apply(lambda x: json.loads(x)['dominant_type'])
    tax_counts = df['tax_type'].value_counts()
    print(tax_counts)
    
    # BAGIAN 3 — Skenario B Analysis (C5 vs C3)
    print("\n" + "="*80)
    print("BAGIAN 3 — Skenario B Analysis (The Tone vs Semantic Debate):")
    print("="*80)
    auc_c3 = roc_auc_score(labels, df['C3_hallu_free'])
    auc_c5 = roc_auc_score(labels, df['C5_shuffled'])
    auc_c0 = roc_auc_score(labels, df['C0_pure'])
    
    print(f"C0 (Baseline)  : {auc_c0:.4f}")
    print(f"C3 (Semantic)  : {auc_c3:.4f}")
    print(f"C5 (Pure Tone) : {auc_c5:.4f}")
    
    if abs(auc_c5 - auc_c3) < abs(auc_c5 - auc_c0):
        print("\nKESIMPULAN: Skenario B Terkonfirmasi.")
        print("ROC-AUC C5 naik mengikuti C3. Ini membuktikan bahwa LLM ")
        print("mendapatkan 'confidence boost' BUKAN dari relevansi kimia (semantic),")
        print("tetapi dari framing gaya bahasa ilmiah yang kompleks (scientific tone).")
    else:
        print("\nKESIMPULAN: C5 tetap di level baseline (C0). Semantik konten dominan.")
        print("Ini membantah hipotesis bahwa 'Scientific Tone' adalah faktor tunggal.")

if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "results_bbbp_checkpoint.json"
    if os.path.exists(checkpoint):
        df = load_data(checkpoint)
        generate_final_report(df)
    else:
        print(f"Checkpoint {checkpoint} not found.")
