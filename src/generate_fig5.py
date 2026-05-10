import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Apply the same professional styling used across all figures
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

# Define a curated color palette matching other figures
COLORS = {
    "Baseline (C0)": "#95a5a6",
    "Hallucination (C3)": "#e74c3c",
}

def generate_scaling_figure():
    """Figure 5: Scaling divergence across model sizes.
    Uses the same styling as other publication figures for visual consistency.
    """
    data = [
        {"Task": "BBBP", "Model": "Llama-8B (Mean)", "ROC-AUC": 0.601, "Condition": "Hallucination (C3)"},
        {"Task": "BBBP", "Model": "Llama-8B (Mean)", "ROC-AUC": 0.502, "Condition": "Baseline (C0)"},
        {"Task": "BBBP", "Model": "Llama-70B", "ROC-AUC": 0.549, "Condition": "Hallucination (C3)"},
        {"Task": "BBBP", "Model": "Llama-70B", "ROC-AUC": 0.644, "Condition": "Baseline (C0)"},
        {"Task": "BACE", "Model": "Llama-8B (Mean)", "ROC-AUC": 0.512, "Condition": "Hallucination (C3)"},
        {"Task": "BACE", "Model": "Llama-8B (Mean)", "ROC-AUC": 0.644, "Condition": "Baseline (C0)"},
        {"Task": "BACE", "Model": "Llama-70B", "ROC-AUC": 0.506, "Condition": "Hallucination (C3)"},
        {"Task": "BACE", "Model": "Llama-70B", "ROC-AUC": 0.564, "Condition": "Baseline (C0)"},
    ]
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")

    g = sns.catplot(
        data=df,
        kind="bar",
        x="Model",
        y="ROC-AUC",
        hue="Condition",
        col="Task",
        palette=COLORS,
        alpha=0.9,
        height=4.5,
        aspect=1,
    )
    g.set_axis_labels("", "ROC-AUC")
    g.set_titles("{col_name} Dataset")

    # Add reference line at chance level (0.5)
    for ax in g.axes.flat:
        ax.axhline(0.5, ls='--', color='black', alpha=0.6)
        ax.set_ylim(0.4, 0.75)

    out_path = os.path.join('hallucination-paper-overleaf', 'figures', 'figure5_scaling_divergence.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved {out_path}")

if __name__ == "__main__":
    generate_scaling_figure()
