import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.calibration import calibration_curve
from pathlib import Path

# Use professional design system
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT_DIR = Path('hallucination-paper-overleaf/figures')

def load_data(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        ckpt = json.load(f)
    results_list = [ckpt["results"][str(i)] for i in range(ckpt["metadata"]["total_molecules"])]
    return pd.DataFrame(results_list)

def plot_reliability_diagram(df, conditions, title, filename):
    # Setup professional full-framed style
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.right'] = True
    
    fig, ax = plt.subplots(figsize=(6, 5.5))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.5, zorder=1, label="Perfectly calibrated")
    
    # Elegant color palette
    colors = {'C0_pure': '#4a4a4a', 'C3_hallu_free': '#d95f02', 'C4a_hallu_sp': '#1b9e77'}
    labels = {'C0_pure': 'C0 (Baseline)', 'C3_hallu_free': 'C3 (Free Hallu)', 'C4a_hallu_sp': 'C4a (Struct. Aug.)'}
    
    for cond in conditions:
        if cond not in df.columns: continue
        y_true = df['label']
        y_prob = df[cond]
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
        
        ax.plot(prob_pred, prob_true, marker="s", color=colors.get(cond, '#333333'), 
                label=labels.get(cond, cond), markersize=7, linewidth=2.5, 
                markerfacecolor="white", markeredgewidth=2, alpha=0.9, zorder=3)
        
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontweight="bold", fontsize=11)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    
    # Clear, visible grid for easy reading
    ax.grid(True, linestyle='-', color='#e0e0e0', alpha=0.8, zorder=0)
    ax.set_axisbelow(True) # Put grid behind lines
    
    # Move legend to upper left to avoid blocking data at the bottom right
    legend = ax.legend(loc="upper left", fontsize=10, framealpha=1.0, edgecolor='#999999', shadow=False)
    legend.set_zorder(5)
    
    ax.set_title(title, fontweight='bold', fontsize=13, pad=15)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / filename, bbox_inches='tight', dpi=300)
    print(f"[OK] Saved {filename}")
    plt.close(fig)

if __name__ == "__main__":
    bbbp_ckpt = "data/processed/results_bbbp_checkpoint.json"
    bace_ckpt = "data/processed/results_bace_checkpoint.json"
    
    if os.path.exists(bbbp_ckpt):
        df_bbbp = load_data(bbbp_ckpt)
        plot_reliability_diagram(df_bbbp, ['C0_pure', 'C3_hallu_free', 'C4a_hallu_sp'], 
                                 "Reliability Diagram (BBBP)", "figure6_reliability_bbbp.png")
                                 
    if os.path.exists(bace_ckpt):
        df_bace = load_data(bace_ckpt)
        plot_reliability_diagram(df_bace, ['C0_pure', 'C3_hallu_free'], 
                                 "Reliability Diagram (BACE)", "figure6_reliability_bace.png")
