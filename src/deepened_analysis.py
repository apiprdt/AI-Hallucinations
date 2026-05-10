"""
Deepened statistical analysis for manuscript hardening.
Computes effect sizes, entropy, KL divergence, and permutation tests
from existing checkpoint data. Zero API cost.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from scipy.special import kl_div
from sklearn.metrics import roc_auc_score
from pathlib import Path
from collections import OrderedDict

# Professional styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

OUT_DIR = Path('hallucination-paper-overleaf/figures')
TABLE_DIR = Path('hallucination-paper-overleaf/tables')

CONDITIONS = ['C0_pure', 'C1_factual', 'C2_chempriming', 'C2b_pure_gibberish',
              'C3_hallu_free', 'C4a_hallu_sp', 'C4b_hallu_pi', 'C4c_hallu_mf', 'C5_shuffled']

SHORT = {
    'C0_pure': 'C0', 'C1_factual': 'C1', 'C2_chempriming': 'C2',
    'C2b_pure_gibberish': 'C2b', 'C3_hallu_free': 'C3',
    'C4a_hallu_sp': 'C4a', 'C4b_hallu_pi': 'C4b',
    'C4c_hallu_mf': 'C4c', 'C5_shuffled': 'C5',
}

COLORS = {
    'C0': '#8c8c8c', 'C1': '#b0b0b0', 'C2': '#4e79a7', 'C2b': '#76b7b2',
    'C3': '#f28e2b', 'C4a': '#e15759', 'C4b': '#ff9d9a',
    'C4c': '#d4723c', 'C5': '#59a14f',
}

def load_data(path):
    with open(path) as f:
        ckpt = json.load(f)
    n = ckpt['metadata']['total_molecules']
    return pd.DataFrame([ckpt['results'][str(i)] for i in range(n)])


# ============================================================
# 1. EFFECT SIZE ANALYSIS (Cohen's d)
# ============================================================
def compute_effect_sizes(df, dataset_name):
    """Compute Cohen's d for each condition vs C0 baseline."""
    print(f"\n{'='*60}")
    print(f"EFFECT SIZE ANALYSIS: {dataset_name}")
    print(f"{'='*60}")
    
    baseline = np.array(df['C0_pure'])
    labels = np.array(df['label'])
    
    results = []
    for cond in CONDITIONS:
        preds = np.array(df[cond])
        
        # Cohen's d (paired)
        diff = preds - baseline
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
        
        # Mean prediction shift
        mean_shift = np.mean(diff)
        
        # Classification
        if abs(d) < 0.2:
            magnitude = "negligible"
        elif abs(d) < 0.5:
            magnitude = "small"
        elif abs(d) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        results.append({
            'Condition': SHORT[cond],
            'Mean Pred': f"{np.mean(preds):.3f}",
            'Mean Shift': f"{mean_shift:+.3f}",
            "Cohen's d": f"{d:+.3f}",
            'Magnitude': magnitude,
        })
        
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    return df_results


# ============================================================
# 2. ENTROPY & KL DIVERGENCE
# ============================================================
def compute_entropy_kl(df, dataset_name):
    """Compute Shannon entropy and KL divergence for prediction distributions."""
    print(f"\n{'='*60}")
    print(f"ENTROPY & KL DIVERGENCE: {dataset_name}")
    print(f"{'='*60}")
    
    results = []
    baseline_preds = np.array(df['C0_pure'])
    
    for cond in CONDITIONS:
        preds = np.array(df[cond])
        
        # Clip to avoid log(0)
        preds_clipped = np.clip(preds, 1e-8, 1 - 1e-8)
        baseline_clipped = np.clip(baseline_preds, 1e-8, 1 - 1e-8)
        
        # Shannon entropy of predictions (binary entropy per sample, then mean)
        entropy = -np.mean(preds_clipped * np.log2(preds_clipped) + 
                          (1 - preds_clipped) * np.log2(1 - preds_clipped))
        
        # Prediction spread (std)
        spread = np.std(preds)
        
        # KL divergence: approximate via histogram
        bins = np.linspace(0, 1, 21)
        hist_cond, _ = np.histogram(preds, bins=bins, density=True)
        hist_base, _ = np.histogram(baseline_preds, bins=bins, density=True)
        
        # Smooth to avoid division by zero
        hist_cond = hist_cond + 1e-8
        hist_base = hist_base + 1e-8
        hist_cond = hist_cond / hist_cond.sum()
        hist_base = hist_base / hist_base.sum()
        
        kl = np.sum(hist_cond * np.log(hist_cond / hist_base))
        
        results.append({
            'Condition': SHORT[cond],
            'Mean Entropy': round(entropy, 3),
            'Pred Spread': round(spread, 3),
            'KL vs C0': round(kl, 3),
        })
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    return df_results


# ============================================================
# 3. PERMUTATION SIGNIFICANCE TEST
# ============================================================
def permutation_test(df, cond1, cond2, n_perms=5000, seed=42):
    """Two-sided permutation test for AUC difference."""
    rng = np.random.RandomState(seed)
    labels = np.array(df['label'])
    preds1 = np.array(df[cond1])
    preds2 = np.array(df[cond2])
    
    observed_diff = roc_auc_score(labels, preds1) - roc_auc_score(labels, preds2)
    
    count = 0
    for _ in range(n_perms):
        # Randomly swap condition assignments per sample
        swap = rng.rand(len(labels)) > 0.5
        p1 = np.where(swap, preds2, preds1)
        p2 = np.where(swap, preds1, preds2)
        
        if len(np.unique(labels)) < 2:
            continue
        
        perm_diff = roc_auc_score(labels, p1) - roc_auc_score(labels, p2)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    
    p_value = count / n_perms
    return observed_diff, p_value


def run_permutation_tests(df, dataset_name):
    """Run permutation tests for key comparisons."""
    print(f"\n{'='*60}")
    print(f"PERMUTATION TESTS: {dataset_name}")
    print(f"{'='*60}")
    
    comparisons = [
        ('C4a_hallu_sp', 'C0_pure', 'C4a vs C0'),
        ('C3_hallu_free', 'C0_pure', 'C3 vs C0'),
        ('C4a_hallu_sp', 'C3_hallu_free', 'C4a vs C3'),
        ('C2_chempriming', 'C0_pure', 'C2 vs C0'),
        ('C5_shuffled', 'C0_pure', 'C5 vs C0'),
    ]
    
    results = []
    for cond1, cond2, name in comparisons:
        diff, p = permutation_test(df, cond1, cond2)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        results.append({
            'Comparison': name,
            'AUC Diff': f"{diff:+.3f}",
            'p-value': f"{p:.4f}",
            'Sig': sig,
        })
        print(f"  {name}: delta={diff:+.3f}, p={p:.4f} {sig}")
    
    return pd.DataFrame(results)


# ============================================================
# 4. VISUALIZATION: Combined Entropy + KL Figure
# ============================================================
def plot_entropy_kl(entropy_bbbp, entropy_bace):
    """Generate Figure 7: Entropy and KL divergence comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    
    for ax, df_ent, title in zip(axes, [entropy_bbbp, entropy_bace], ['BBBP', 'BACE']):
        conditions = df_ent['Condition'].tolist()
        kl_vals = df_ent['KL vs C0'].tolist()
        entropy_vals = df_ent['Mean Entropy'].tolist()
        colors = [COLORS.get(c, '#999') for c in conditions]
        
        x = np.arange(len(conditions))
        bars = ax.bar(x, kl_vals, color=colors, edgecolor='white', linewidth=0.5, 
                     width=0.7, zorder=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylabel('KL Divergence from C0')
        ax.set_title(f'{title}', fontweight='bold')
        ax.yaxis.grid(True, linestyle=':', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        # Annotate values
        for bar, val in zip(bars, kl_vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='#333')
    
    fig.suptitle('Distributional Divergence from Baseline (KL)', 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'figure7_kl_divergence.png')
    print(f"[OK] Saved figure7_kl_divergence.png")
    plt.close(fig)


# ============================================================
# 5. EFFECT SIZE LATEX TABLE
# ============================================================
def generate_effect_size_table(es_bbbp, es_bace):
    """Generate LaTeX table for effect sizes."""
    tex = r"""\begin{table}[ht]
\centering
\caption{Standardized effect sizes (Cohen's $d$, paired) for prediction probability shifts relative to the C0 baseline. Effect magnitudes are classified following standard conventions: negligible ($|d| < 0.2$), small ($0.2 \leq |d| < 0.5$), medium ($0.5 \leq |d| < 0.8$), and large ($|d| \geq 0.8$).}
\label{tab:effect_sizes}
\begin{tabular}{lcccc}
\toprule
\textbf{Condition} & \multicolumn{2}{c}{\textbf{BBBP}} & \multicolumn{2}{c}{\textbf{BACE}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & Cohen's $d$ & Magnitude & Cohen's $d$ & Magnitude \\
\midrule
"""
    for i in range(len(es_bbbp)):
        cond = es_bbbp.iloc[i]['Condition']
        d_bbbp = es_bbbp.iloc[i]["Cohen's d"]
        mag_bbbp = es_bbbp.iloc[i]['Magnitude']
        d_bace = es_bace.iloc[i]["Cohen's d"]
        mag_bace = es_bace.iloc[i]['Magnitude']
        tex += f"{cond} & {d_bbbp} & {mag_bbbp} & {d_bace} & {mag_bace} \\\\\n"
    
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    table_path = TABLE_DIR / 'effect_size_table.tex'
    with open(table_path, 'w') as f:
        f.write(tex)
    print(f"[OK] Saved {table_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    bbbp_path = 'data/processed/results_bbbp_checkpoint.json'
    bace_path = 'data/processed/results_bace_checkpoint.json'
    
    df_bbbp = load_data(bbbp_path)
    df_bace = load_data(bace_path)
    
    # 1. Effect sizes
    es_bbbp = compute_effect_sizes(df_bbbp, 'BBBP')
    es_bace = compute_effect_sizes(df_bace, 'BACE')
    
    # 2. Entropy & KL
    ent_bbbp = compute_entropy_kl(df_bbbp, 'BBBP')
    ent_bace = compute_entropy_kl(df_bace, 'BACE')
    
    # 3. Permutation tests
    perm_bbbp = run_permutation_tests(df_bbbp, 'BBBP')
    perm_bace = run_permutation_tests(df_bace, 'BACE')
    
    # 4. Generate figures
    plot_entropy_kl(ent_bbbp, ent_bace)
    
    # 5. Generate LaTeX table
    generate_effect_size_table(es_bbbp, es_bace)
    
    # Save all results to JSON for reference
    report = {
        'effect_sizes': {
            'BBBP': es_bbbp.to_dict('records'),
            'BACE': es_bace.to_dict('records'),
        },
        'entropy_kl': {
            'BBBP': ent_bbbp.to_dict('records'),
            'BACE': ent_bace.to_dict('records'),
        },
        'permutation_tests': {
            'BBBP': perm_bbbp.to_dict('records'),
            'BACE': perm_bace.to_dict('records'),
        },
    }
    
    report_path = Path('data/processed/deepened_analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved full report to {report_path}")

if __name__ == '__main__':
    main()
