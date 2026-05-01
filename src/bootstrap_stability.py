import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy import stats

# Premium Styling for Q1
plt.style.use('default')
sns.set_theme(style="ticks", font_scale=1.1)

# Custom palette for Q1 look
PALETTE = sns.color_palette("muted")
COLORS = {
    'C0_pure': '#7f8c8d',          # Gray
    'C1_factual': '#95a5a6',       # Light Gray
    'C2_chempriming': '#3498db',   # Blue
    'C2b_pure_gibberish': '#2980b9',# Dark Blue
    'C3_hallu_free': '#e67e22',    # Orange
    'C4a_hallu_sp': '#c0392b',     # Red (Highest)
    'C4b_hallu_pi': '#d35400',     # Dark Orange
    'C4c_hallu_mf': '#e74c3c',     # Light Red
    'C5_shuffled': '#2c3e50'       # Dark Slate
}

def load_data(checkpoint_file: str):
    with open(checkpoint_file, 'r') as f:
        ckpt = json.load(f)
    results_list = [ckpt["results"][str(i)] for i in range(ckpt["metadata"]["total_molecules"])]
    return pd.DataFrame(results_list)

def bootstrap_stability(checkpoint_file: str, n_bootstraps=1000, seed=42):
    if not os.path.exists(checkpoint_file):
        print(f"File {checkpoint_file} not found.")
        return

    df = load_data(checkpoint_file)
    conditions = ['C0_pure', 'C1_factual', 'C2_chempriming', 'C2b_pure_gibberish', 'C3_hallu_free', 'C4a_hallu_sp', 'C4b_hallu_pi', 'C4c_hallu_mf', 'C5_shuffled']
    labels = np.array(df['label'])
    
    rng = np.random.RandomState(seed)
    
    bootstrap_results = {cond: [] for cond in conditions}
    
    print(f"Running {n_bootstraps} bootstrap iterations...")
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
            
        for cond in conditions:
            preds = np.array(df[cond])[indices]
            auc = roc_auc_score(labels[indices], preds)
            bootstrap_results[cond].append(auc)
            
    # Calculate stability metrics
    print("\n" + "="*80)
    print("BOOTSTRAP STABILITY REPORT")
    print("="*80)
    
    rows = []
    for cond in conditions:
        arr = np.array(bootstrap_results[cond])
        mean_auc = np.mean(arr)
        std_auc = np.std(arr)
        ci_lower = np.percentile(arr, 2.5)
        ci_upper = np.percentile(arr, 97.5)
        
        rows.append({
            'Condition': cond,
            'Mean AUC': f"{mean_auc:.3f}",
            'Std AUC': f"{std_auc:.3f}",
            '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]"
        })
        
    res_df = pd.DataFrame(rows)
    print(res_df.to_string(index=False))
    
    # Calculate ranking stability (P(C4a > C3), P(C4a > C0))
    c4a_arr = np.array(bootstrap_results['C4a_hallu_sp'])
    c3_arr = np.array(bootstrap_results['C3_hallu_free'])
    c0_arr = np.array(bootstrap_results['C0_pure'])
    
    p_c4a_gt_c0 = np.mean(c4a_arr > c0_arr)
    p_c4a_gt_c3 = np.mean(c4a_arr > c3_arr)
    p_c3_gt_c0 = np.mean(c3_arr > c0_arr)
    
    print("\nRanking Stability:")
    print(f"P(C4a > C0 baseline) = {p_c4a_gt_c0:.1%}")
    print(f"P(C4a > C3 free)     = {p_c4a_gt_c3:.1%}")
    print(f"P(C3 > C0 baseline)  = {p_c3_gt_c0:.1%}")
    
    # Generate Figure 4: Bootstrap Violin Plot
    plot_data = []
    for cond in conditions:
        for auc in bootstrap_results[cond]:
            plot_data.append({'Condition': cond.replace('_hallu', '').replace('_pure', '').replace('_chempriming', ' (chem)').replace('_gibberish', ' (gibberish)'), 'AUC': auc})
    
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 6))
    
    # Q1 formatting: thin lines, clean look
    sns.violinplot(
        data=plot_df, x='Condition', y='AUC', 
        palette=[COLORS.get(c, '#bdc3c7') for c in conditions], 
        linewidth=1.0, 
        inner='quartile' # shows median and quartiles clearly
    )
    
    # Add baseline reference line based on median of C0
    c0_median = np.median(c0_arr)
    plt.axhline(c0_median, color='#34495e', linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)
    
    plt.title('Bootstrap Distribution of ROC-AUC Scores (BBBP)', pad=20, fontweight='bold')
    plt.ylabel('ROC-AUC Score (1000 Bootstraps)')
    plt.xlabel('')
    plt.xticks(rotation=30, ha='right')
    
    # Despine top and right borders (Q1 standard)
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    plt.savefig('hallucination-paper-overleaf/figures/figure4_bootstrap_stability.png', dpi=300, bbox_inches='tight')
    print("\nSaved hallucination-paper-overleaf/figures/figure4_bootstrap_stability.png")

if __name__ == "__main__":
    bootstrap_stability("data/processed/results_bbbp_checkpoint.json")
