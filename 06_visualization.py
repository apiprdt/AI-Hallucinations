import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import roc_auc_score

# Set style for premium look
plt.style.use('dark_background')
sns.set_palette("viridis")

def load_data(checkpoint_file: str):
    with open(checkpoint_file, 'r') as f:
        ckpt = json.load(f)
    results_list = [ckpt["results"][str(i)] for i in range(ckpt["metadata"]["total_molecules"])]
    return pd.DataFrame(results_list)

def create_visualizations(checkpoint_file: str):
    if not os.path.exists(checkpoint_file):
        print(f"File {checkpoint_file} not found.")
        return

    df = load_data(checkpoint_file)
    conditions = ['C0_pure', 'C1_factual', 'C2_gibberish', 'C3_hallu_free', 'C4a_hallu_sp', 'C4b_hallu_pi', 'C4c_hallu_mf', 'C5_shuffled']
    labels = df['label']

    # 1. Calculate AUCs
    aucs = []
    for cond in conditions:
        auc = roc_auc_score(labels, df[cond])
        aucs.append(auc)

    auc_df = pd.DataFrame({
        'Condition': conditions,
        'ROC-AUC': aucs
    })

    # 2. Plot ROC-AUC Bar Chart
    plt.figure(figsize=(14, 7))
    
    # Custom palette highlighting C5
    colors = sns.color_palette("magma", len(conditions)-1)
    colors.append((0.9, 0.4, 0.6)) # Highlight C5 with a distinct color
    
    ax = sns.barplot(x='Condition', y='ROC-AUC', data=auc_df, palette=colors)
    plt.title('Performance Across Hallucination Conditions (Including Skenario B)', fontsize=18, pad=20, weight='bold')
    
    # Add a horizontal line for Baseline
    baseline_val = auc_df.iloc[0]['ROC-AUC']
    plt.axhline(y=baseline_val, color='grey', linestyle='--', label=f'Baseline C0 ({baseline_val:.3f})')
    
    # Adjust limits
    plt.ylim(0.4, 0.95)
    plt.ylabel('ROC-AUC Score', fontsize=14)
    plt.xlabel('Condition', fontsize=14)
    plt.xticks(rotation=20, fontsize=12)
    plt.legend()
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        font_weight = 'bold' if i == len(conditions)-1 or i == 3 else 'normal' # Highlight C3 and C5 texts
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='white', weight=font_weight, xytext=(0, 10),
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    print("Saved performance_comparison.png")

    # 3. Taxonomy Distribution Pie Chart
    plt.figure(figsize=(10, 8))
    df['tax_type'] = df['taxonomy_free'].apply(lambda x: json.loads(x)['dominant_type'])
    tax_counts = df['tax_type'].value_counts()
    
    plt.pie(tax_counts, labels=tax_counts.index, autopct='%1.1f%%', startangle=140, 
            colors=sns.color_palette("plasma", len(tax_counts)), wedgeprops={'edgecolor': 'black'})
    plt.title('Distribution of Hallucination Types (C3 Free)', fontsize=16)
    plt.tight_layout()
    plt.savefig('taxonomy_distribution.png', dpi=300)
    print("Saved taxonomy_distribution.png")

    # 4. Confidence/Probability Heatmap
    # Sort by label and then C0 prediction for a cleaner heatmap
    sorted_df = df.sort_values(by=['label', 'C0_pure'])
    heatmap_data = sorted_df[conditions].T
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Predicted Probability'}, yticklabels=True, xticklabels=False)
    plt.title('Prediction Probabilities for All 204 Molecules', fontsize=16, pad=15)
    plt.ylabel('Condition')
    plt.xlabel('Molecules (Sorted by True Label)')
    
    # Add a line separating negative and positive samples
    num_neg = (sorted_df['label'] == 0).sum()
    plt.axvline(x=num_neg, color='red', linestyle='--', linewidth=2, label='Label Boundary (0 vs 1)')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('probability_heatmap.png', dpi=300)
    print("Saved probability_heatmap.png")

if __name__ == "__main__":
    checkpoint_file = "results_bbbp_checkpoint.json"
    create_visualizations(checkpoint_file)
