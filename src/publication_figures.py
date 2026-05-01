"""
Publication-grade figure generator for Q1 journal submission.
Design principles:
  - Consistent color palette with semantic grouping
  - Human-readable labels (no code names)
  - Clean typography (no default matplotlib fonts)
  - Despined axes, minimal chartjunk
  - 300 DPI, tight bounding boxes
  - Horizontal bar chart for Figure 1 (easier to read condition names)
  - Bar chart replaces pie chart for Figure 2 (pie charts are unprofessional)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import json
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ============================================================
# DESIGN SYSTEM
# ============================================================

# Use a professional serif-compatible font
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
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Semantic color palette — grouped by condition type
COLORS = {
    # Baselines (cool grays)
    'C0': '#8c8c8c',
    'C1': '#b0b0b0',
    # Stylistic controls (blues)
    'C2': '#4e79a7',
    'C2b': '#76b7b2',
    # Hallucinations (warm reds/oranges — the "interesting" conditions)
    'C3': '#f28e2b',
    'C4a': '#e15759',
    'C4b': '#ff9d9a',
    'C4c': '#d4723c',
    # Semantic control (dark)
    'C5': '#59a14f',
}

# Human-readable labels
LABELS = {
    'C0_pure': 'C0: Baseline\n(SMILES only)',
    'C1_factual': 'C1: Factual\n(RDKit)',
    'C2_chempriming': 'C2: Chem.\nPriming',
    'C2b_pure_gibberish': 'C2b: Pure\nGibberish',
    'C3_hallu_free': 'C3: Free\nHallucination',
    'C4a_hallu_sp': 'C4a: Structural\nPhantom',
    'C4b_hallu_pi': 'C4b: Property\nInversion',
    'C4c_hallu_mf': 'C4c: Mechanism\nFabrication',
    'C5_shuffled': 'C5: Shuffled\n(Semantic Ctrl)',
}

SHORT_LABELS = {
    'C0_pure': 'C0',
    'C1_factual': 'C1',
    'C2_chempriming': 'C2',
    'C2b_pure_gibberish': 'C2b',
    'C3_hallu_free': 'C3',
    'C4a_hallu_sp': 'C4a',
    'C4b_hallu_pi': 'C4b',
    'C4c_hallu_mf': 'C4c',
    'C5_shuffled': 'C5',
}

COLOR_LIST = [COLORS['C0'], COLORS['C1'], COLORS['C2'], COLORS['C2b'],
              COLORS['C3'], COLORS['C4a'], COLORS['C4b'], COLORS['C4c'], COLORS['C5']]

CONDITIONS = ['C0_pure', 'C1_factual', 'C2_chempriming', 'C2b_pure_gibberish',
              'C3_hallu_free', 'C4a_hallu_sp', 'C4b_hallu_pi', 'C4c_hallu_mf', 'C5_shuffled']

OUT_DIR = Path('hallucination-paper-overleaf/figures')

def load_data(checkpoint_file: str):
    with open(checkpoint_file, 'r') as f:
        ckpt = json.load(f)
    results_list = [ckpt["results"][str(i)] for i in range(ckpt["metadata"]["total_molecules"])]
    return pd.DataFrame(results_list)


# ============================================================
# FIGURE 1: Performance Comparison (Horizontal Bar + CI)
# ============================================================
def figure1_performance(df):
    """Horizontal bar chart with bootstrap 95% CI error bars."""
    labels_arr = np.array(df['label'])
    
    aucs = []
    ci_lowers = []
    ci_uppers = []
    rng = np.random.RandomState(42)
    
    for cond in CONDITIONS:
        preds = np.array(df[cond])
        auc = roc_auc_score(labels_arr, preds)
        
        # Bootstrap CI
        boot_aucs = []
        for _ in range(1000):
            idx = rng.randint(0, len(labels_arr), len(labels_arr))
            if len(np.unique(labels_arr[idx])) < 2:
                continue
            boot_aucs.append(roc_auc_score(labels_arr[idx], preds[idx]))
        boot_aucs = np.array(boot_aucs)
        
        aucs.append(auc)
        ci_lowers.append(auc - np.percentile(boot_aucs, 2.5))
        ci_uppers.append(np.percentile(boot_aucs, 97.5) - auc)
    
    # Sort by AUC for visual clarity
    order = np.argsort(aucs)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    y_pos = np.arange(len(CONDITIONS))
    sorted_aucs = [aucs[i] for i in order]
    sorted_colors = [COLOR_LIST[i] for i in order]
    sorted_labels = [LABELS[CONDITIONS[i]].replace('\n', ' ') for i in order]
    sorted_ci = np.array([[ci_lowers[i] for i in order], [ci_uppers[i] for i in order]])
    
    bars = ax.barh(y_pos, sorted_aucs, xerr=sorted_ci, 
                   color=sorted_colors, edgecolor='white', linewidth=0.5,
                   error_kw={'linewidth': 1.0, 'capsize': 3, 'capthick': 1.0, 'color': '#333333'},
                   height=0.7, zorder=3)
    
    # Baseline reference line
    baseline_auc = aucs[0]  # C0
    ax.axvline(x=baseline_auc, color='#555555', linestyle='--', linewidth=1.0, 
               zorder=2, alpha=0.7, label=f'Baseline C0 ({baseline_auc:.3f})')
    
    # Chance level
    ax.axvline(x=0.5, color='#cccccc', linestyle=':', linewidth=0.8, zorder=1)
    # Place 'chance' text below the bottom bar to avoid overlap
    ax.text(0.502, -0.6, 'chance', fontsize=8, color='#999999', va='center', style='italic')
    
    # Value annotations
    for i, (bar, val, ci_upper) in enumerate(zip(bars, sorted_aucs, sorted_ci[1])):
        # Place the text slightly after the end of the error bar so it's not crossed out
        text_x = val + ci_upper + 0.005
        ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', ha='left', fontsize=9, color='#333333')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.set_xlabel('ROC-AUC Score')
    ax.set_xlim(0.45, 0.72)
    ax.set_title('Performance Across Hallucination Conditions (BBBP)', pad=15, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='#cccccc')
    
    # Light grid on x-axis only
    ax.xaxis.grid(True, linestyle=':', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'figure1_performance_comparison.png')
    print("✓ Saved figure1_performance_comparison.png")
    plt.close(fig)


# ============================================================
# FIGURE 2: Taxonomy Distribution (Horizontal Bar, NOT pie)
# ============================================================
def figure2_taxonomy(df):
    """Horizontal bar chart showing taxonomy distribution."""
    # Extract taxonomy from checkpoint data
    tax_col = 'taxonomy_free'
    if tax_col not in df.columns:
        print("⚠ taxonomy_free column not found, skipping Figure 2")
        return
    
    tax_types = df[tax_col].apply(lambda x: json.loads(x)['dominant_type'] if pd.notna(x) else 'Unknown')
    tax_counts = tax_types.value_counts()
    
    # Define consistent ordering and colors for taxonomy
    TAX_ORDER = ['CC', 'PI', 'MF', 'SP']
    TAX_COLORS = {
        'CC': '#4e79a7',   # Blue - most common
        'PI': '#e15759',   # Red
        'MF': '#f28e2b',   # Orange
        'SP': '#edc948',   # Gold - rarest but most interesting
    }
    TAX_FULL = {
        'CC': 'Contextual\nConfabulation',
        'PI': 'Property\nInversion', 
        'MF': 'Mechanism\nFabrication',
        'SP': 'Structural\nPhantom',
    }
    
    counts = [tax_counts.get(t, 0) for t in TAX_ORDER]
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]
    colors = [TAX_COLORS[t] for t in TAX_ORDER]
    labels = [f'{TAX_FULL[t]}' for t in TAX_ORDER]
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    y_pos = np.arange(len(TAX_ORDER))
    bars = ax.barh(y_pos, pcts, color=colors, edgecolor='white', linewidth=0.5, height=0.6, zorder=3)
    
    for i, (bar, pct, count) in enumerate(zip(bars, pcts, counts)):
        ax.text(pct + 0.8, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}% (n={count})', va='center', ha='left', fontsize=10, color='#333333')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Proportion of C3 Hallucinations (%)')
    ax.set_xlim(0, 100)
    ax.set_title('Distribution of Hallucination Types\n(Naturally Generated, C3)', pad=15, fontweight='bold')
    
    ax.xaxis.grid(True, linestyle=':', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'figure2_taxonomy_distribution.png')
    print("✓ Saved figure2_taxonomy_distribution.png")
    plt.close(fig)


# ============================================================
# FIGURE 3: Prediction Collapse (BACE) — Dual KDE
# ============================================================
def figure3_collapse(bace_checkpoint: str):
    """KDE plot showing prediction distribution collapse on BACE."""
    if not os.path.exists(bace_checkpoint):
        print(f"⚠ {bace_checkpoint} not found, skipping Figure 3")
        return
    
    df = load_data(bace_checkpoint)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    # Left panel: Baseline (C0) — bimodal
    ax0 = axes[0]
    c0_neg = df.loc[df['label'] == 0, 'C0_pure']
    c0_pos = df.loc[df['label'] == 1, 'C0_pure']
    
    sns.kdeplot(c0_neg, ax=ax0, fill=True, color='#4e79a7', alpha=0.4, linewidth=1.5, label='True Negative')
    sns.kdeplot(c0_pos, ax=ax0, fill=True, color='#e15759', alpha=0.4, linewidth=1.5, label='True Positive')
    ax0.set_title('Baseline (C0)', fontweight='bold')
    ax0.set_xlabel('Predicted Probability')
    ax0.set_ylabel('Density')
    ax0.legend(framealpha=0.9, edgecolor='#cccccc', fontsize=9)
    ax0.set_xlim(-0.05, 1.0)
    
    # Right panel: Hallucination (C3) — collapsed
    ax1 = axes[1]
    c3_neg = df.loc[df['label'] == 0, 'C3_hallu_free']
    c3_pos = df.loc[df['label'] == 1, 'C3_hallu_free']
    
    sns.kdeplot(c3_neg, ax=ax1, fill=True, color='#4e79a7', alpha=0.4, linewidth=1.5, label='True Negative')
    sns.kdeplot(c3_pos, ax=ax1, fill=True, color='#e15759', alpha=0.4, linewidth=1.5, label='True Positive')
    ax1.axvline(x=0.17, color='#333333', linestyle='--', linewidth=1.0, alpha=0.7)
    ax1.text(0.20, ax1.get_ylim()[1] * 0.85 if ax1.get_ylim()[1] > 0 else 3.0, 
             'collapse\ncenter', fontsize=8, color='#555555', style='italic')
    ax1.set_title('Free Hallucination (C3)', fontweight='bold')
    ax1.set_xlabel('Predicted Probability')
    ax1.legend(framealpha=0.9, edgecolor='#cccccc', fontsize=9)
    ax1.set_xlim(-0.05, 1.0)
    
    fig.suptitle('Prediction Distribution Collapse on BACE Dataset', 
                 fontweight='bold', fontsize=14, y=1.02)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'figure3_negativity_collapse.png')
    print("✓ Saved figure3_negativity_collapse.png")
    plt.close(fig)


# ============================================================
# FIGURE 4: Bootstrap Stability (Violin + Strip)
# ============================================================
def figure4_bootstrap(df):
    """Violin plot with strip overlay showing bootstrap AUC distributions."""
    labels_arr = np.array(df['label'])
    rng = np.random.RandomState(42)
    
    bootstrap_data = []
    for _ in range(1000):
        idx = rng.randint(0, len(labels_arr), len(labels_arr))
        if len(np.unique(labels_arr[idx])) < 2:
            continue
        for cond in CONDITIONS:
            preds = np.array(df[cond])[idx]
            auc = roc_auc_score(labels_arr[idx], preds)
            bootstrap_data.append({
                'Condition': SHORT_LABELS[cond],
                'AUC': auc,
            })
    
    boot_df = pd.DataFrame(bootstrap_data)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Order by median AUC
    medians = boot_df.groupby('Condition')['AUC'].median()
    order = medians.sort_values().index.tolist()
    
    palette = {SHORT_LABELS[c]: COLOR_LIST[i] for i, c in enumerate(CONDITIONS)}
    
    sns.violinplot(data=boot_df, x='Condition', y='AUC', order=order,
                   palette=palette, inner='quartile', linewidth=0.8,
                   saturation=0.85, zorder=3, cut=1)
    
    # Baseline reference line
    c0_median = medians['C0']
    ax.axhline(y=c0_median, color='#555555', linestyle='--', linewidth=1.0,
               alpha=0.7, zorder=2, label=f'Baseline C0 median ({c0_median:.3f})')
    
    # Chance level
    ax.axhline(y=0.5, color='#cccccc', linestyle=':', linewidth=0.8, zorder=1)
    
    ax.set_xlabel('')
    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('Bootstrap Distribution of ROC-AUC Scores (BBBP, n=1000)', 
                 pad=15, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='#cccccc')
    
    ax.yaxis.grid(True, linestyle=':', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'figure4_bootstrap_stability.png')
    print("✓ Saved figure4_bootstrap_stability.png")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    bbbp_ckpt = "data/processed/results_bbbp_checkpoint.json"
    bace_ckpt = "data/processed/results_bace_checkpoint.json"
    
    print("Generating publication-grade figures...")
    print(f"Output directory: {OUT_DIR.resolve()}\n")
    
    df_bbbp = load_data(bbbp_ckpt)
    
    figure1_performance(df_bbbp)
    figure2_taxonomy(df_bbbp)
    figure3_collapse(bace_ckpt)
    figure4_bootstrap(df_bbbp)
    
    print("\n✓ All figures generated successfully.")
