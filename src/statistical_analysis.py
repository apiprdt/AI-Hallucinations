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
    """
    Cohen's d for paired samples.
    NOTE: This measures the effect size of prediction SCORE shifts,
    not the effect size of AUC differences.
    """
    diff = x1 - x2
    return np.mean(diff) / np.std(diff, ddof=1)

def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, ci=95, seed=42):
    """Bootstrap confidence interval for a single AUC estimate."""
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

def bootstrap_paired_auc_test(y_true, y_pred_cond, y_pred_baseline, n_bootstraps=1000, seed=42):
    """
    Bootstrap paired AUC difference test.
    
    Tests whether AUC(condition) - AUC(baseline) is significantly different from 0.
    Returns the observed delta, the bootstrap distribution, and a two-sided p-value.
    
    This is the CORRECT test for comparing AUC values, unlike paired t-test on
    raw predictions which only measures whether prediction scores shift (not whether
    discrimination improves).
    """
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred_cond = np.array(y_pred_cond)
    y_pred_baseline = np.array(y_pred_baseline)
    
    # Observed delta
    auc_cond = roc_auc_score(y_true, y_pred_cond)
    auc_base = roc_auc_score(y_true, y_pred_baseline)
    observed_delta = auc_cond - auc_base
    
    # Bootstrap
    bootstrap_deltas = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            auc_c = roc_auc_score(y_true[indices], y_pred_cond[indices])
            auc_b = roc_auc_score(y_true[indices], y_pred_baseline[indices])
            bootstrap_deltas.append(auc_c - auc_b)
        except ValueError:
            continue
    
    bootstrap_deltas = np.array(bootstrap_deltas)
    
    # Two-sided p-value: proportion of bootstrap samples where delta has opposite sign
    if observed_delta >= 0:
        p_value = 2 * np.mean(bootstrap_deltas <= 0)
    else:
        p_value = 2 * np.mean(bootstrap_deltas >= 0)
    
    p_value = min(p_value, 1.0)
    
    return observed_delta, bootstrap_deltas, p_value

def generate_final_report(df: pd.DataFrame):
    labels = df['label']
    conditions = ['C0_pure', 'C1_factual', 'C2_chempriming', 'C2b_pure_gibberish', 'C3_hallu_free', 'C4a_hallu_sp', 'C4b_hallu_pi', 'C4c_hallu_mf', 'C5_shuffled']
    
    # PART 1 — ROC-AUC Table with Bootstrap Paired AUC Test
    auc_data = []
    c0_preds = df['C0_pure']
    
    print("Calculating Bootstrap Paired AUC Tests (1000 iterations)...")
    
    p_values_raw = []
    cohen_ds = []
    
    for cond in conditions:
        if cond == 'C0_pure':
            p_values_raw.append(1.0)
            cohen_ds.append(0.0)
        else:
            # Bootstrap paired AUC difference test (CORRECT method)
            delta, _, p_val = bootstrap_paired_auc_test(labels, df[cond], c0_preds)
            p_values_raw.append(p_val)
            # Cohen's d on prediction scores (documented as score shift, not AUC effect)
            cohen_ds.append(cohen_d_paired(df[cond], c0_preds))
    
    # BH Correction (excluding C0)
    _, p_corrected_others, _, _ = multipletests(p_values_raw[1:], method='fdr_bh')
    p_corrected = [1.0] + p_corrected_others.tolist()
    
    print("Calculating Confidence Intervals (Bootstrapping 1000x)...")
    for i, cond in enumerate(conditions):
        auc = roc_auc_score(labels, df[cond])
        ci_low, ci_high = bootstrap_auc_ci(labels, df[cond])
        
        auc_data.append({
            "Condition": cond,
            "ROC-AUC": f"{auc:.4f}",
            "95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
            "Cohen's d (score shift)": f"{cohen_ds[i]:.3f}",
            "p-val(BH)": f"{p_corrected[i]:.2e}",
            "Sig?": "YES" if p_corrected[i] < 0.05 else "no"
        })
    
    report_df = pd.DataFrame(auc_data)
    
    print("\n" + "="*80)
    print("PART 1 — STATISTICAL RIGOR TABLE (Bootstrap Paired AUC Test)")
    print("="*80)
    print("NOTE: p-values test whether AUC(condition) differs from AUC(C0)")
    print("      Cohen's d measures prediction SCORE shift magnitude (not AUC effect)")
    print(report_df.to_string(index=False))
    
    # PART 2 — Taxonomy Distribution
    print("\n" + "="*80)
    print("PART 2 — Taxonomy Distribution:")
    print("="*80)
    df['tax_type'] = df['taxonomy_free'].apply(lambda x: json.loads(x)['dominant_type'])
    tax_counts = df['tax_type'].value_counts()
    print(tax_counts)
    
    # PART 3 — Semantic vs Stylistic Analysis (C5 vs C3)
    print("\n" + "="*80)
    print("PART 3 — Semantic Content vs Stylistic Priming (C5 vs C3):")
    print("="*80)
    auc_c3 = roc_auc_score(labels, df['C3_hallu_free'])
    auc_c5 = roc_auc_score(labels, df['C5_shuffled'])
    auc_c0 = roc_auc_score(labels, df['C0_pure'])
    
    print(f"C0 (Baseline)           : {auc_c0:.4f}")
    print(f"C3 (Semantic + Style)   : {auc_c3:.4f}")
    print(f"C5 (Style only, shifted): {auc_c5:.4f}")
    
    if abs(auc_c5 - auc_c0) < abs(auc_c3 - auc_c0) * 0.3:
        print("\nCONCLUSION: C5 aligns with baseline (C0). Semantic content appears")
        print("to drive discriminative gains, not scientific register alone.")
    elif abs(auc_c5 - auc_c3) < abs(auc_c5 - auc_c0):
        print("\nCONCLUSION: C5 tracks C3. Scientific register may contribute to gains.")
    else:
        print("\nCONCLUSION: C5 aligns closer to baseline. Semantic alignment matters.")

if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "results_bbbp_checkpoint.json"
    if os.path.exists(checkpoint):
        df = load_data(checkpoint)
        generate_final_report(df)
    else:
        print(f"Checkpoint {checkpoint} not found.")
