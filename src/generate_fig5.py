import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def generate_scaling_figure():
    # Data from final robustness analysis
    data = [
        {"Task": "BBBP", "Model": "Llama-8B (Mean)", "ROC-AUC": 0.601, "Condition": "Hallucination (C3)"},
        {"Task": "BBBP", "Model": "Llama-8B (Mean)", "ROC-AUC": 0.502, "Condition": "Baseline (C0)"},
        {"Task": "BBBP", "Model": "Llama-70B", "ROC-AUC": 0.549, "Condition": "Hallucination (C3)"},
        {"Task": "BBBP", "Model": "Llama-70B", "ROC-AUC": 0.644, "Condition": "Baseline (C0)"},
        
        {"Task": "BACE", "Model": "Llama-8B (Mean)", "ROC-AUC": 0.512, "Condition": "Hallucination (C3)"},
        {"Task": "BACE", "Model": "Llama-8B (Mean)", "ROC-AUC": 0.644, "Condition": "Baseline (C0)"},
        {"Task": "BACE", "Model": "Llama-70B", "ROC-AUC": 0.506, "Condition": "Hallucination (C3)"},
        {"Task": "BACE", "Model": "Llama-70B", "ROC-AUC": 0.564, "Condition": "Baseline (C0)"}
    ]
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    g = sns.catplot(
        data=df, kind="bar",
        x="Model", y="ROC-AUC", hue="Condition", col="Task",
        palette={"Baseline (C0)": "#95a5a6", "Hallucination (C3)": "#e74c3c"},
        alpha=.8, height=5, aspect=1
    )
    
    g.set_axis_labels("", "ROC-AUC")
    g.set_titles("{col_name} Dataset")
    # Legend title is now derived from the column name 'Condition'
    
    # Add horizontal line at 0.5
    for ax in g.axes.flat:
        ax.axhline(0.5, ls='--', color='black', alpha=0.5)
        ax.set_ylim(0.4, 0.75)
        
    output_path = "hallucination-paper-overleaf/figures/figure5_scaling_divergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved {output_path}")

if __name__ == "__main__":
    generate_scaling_figure()
