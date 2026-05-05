"""
Graphical Abstract Generator for JMGM Submission
Creates a clean schematic showing the perturbation framework pipeline
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis('off')

# Color palette
C_BG = '#f8f9fa'
C_SMILES = '#4e79a7'
C_LLM = '#f28e2b'
C_BBBP = '#59a14f'
C_BACE = '#e15759'
C_ARROW = '#555555'
C_FRAME = '#333333'

fig.patch.set_facecolor('white')

# === LEFT: Input ===
# SMILES box
smiles_box = FancyBboxPatch((0.3, 2.0), 1.8, 1.0, 
    boxstyle="round,pad=0.1", facecolor=C_SMILES, edgecolor=C_FRAME, 
    linewidth=1.2, alpha=0.9)
ax.add_patch(smiles_box)
ax.text(1.2, 2.5, 'SMILES\nInput', ha='center', va='center', 
    fontsize=10, fontweight='bold', color='white')

# Arrow 1
ax.annotate('', xy=(2.5, 2.5), xytext=(2.1, 2.5),
    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))

# === CENTER: LLM + Perturbations ===
# LLM box
llm_box = FancyBboxPatch((2.6, 1.5), 2.4, 2.0,
    boxstyle="round,pad=0.15", facecolor=C_LLM, edgecolor=C_FRAME,
    linewidth=1.2, alpha=0.15)
ax.add_patch(llm_box)
ax.text(3.8, 3.25, 'LLM Augmentation', ha='center', va='center',
    fontsize=9, fontweight='bold', color=C_FRAME)

# Perturbation conditions (stacked)
conditions = [
    ('C0: Baseline', '#8c8c8c'),
    ('C1: Factual', '#b0b0b0'),
    ('C3: Free Hallu.', '#f28e2b'),
    ('C4a: Struct. Aug.', '#e15759'),
    ('C5: Perm. Ctrl.', '#59a14f'),
]
for i, (label, color) in enumerate(conditions):
    y = 2.9 - i * 0.32
    ax.add_patch(FancyBboxPatch((2.8, y-0.12), 2.0, 0.24,
        boxstyle="round,pad=0.03", facecolor=color, alpha=0.7,
        edgecolor='white', linewidth=0.5))
    ax.text(3.8, y, label, ha='center', va='center', fontsize=6.5,
        color='white', fontweight='bold')

# Arrow 2
ax.annotate('', xy=(5.5, 2.5), xytext=(5.0, 2.5),
    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))

# === CENTER-RIGHT: Framework label ===
ax.text(5.75, 2.5, '9-Condition\nAblation', ha='center', va='center',
    fontsize=9, fontweight='bold', color=C_FRAME,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8e8e8', 
              edgecolor=C_FRAME, linewidth=1))

# Arrow 3 splits into two
ax.annotate('', xy=(7.0, 3.3), xytext=(6.5, 2.7),
    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
ax.annotate('', xy=(7.0, 1.7), xytext=(6.5, 2.3),
    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))

# === RIGHT: Results ===
# BBBP result box (green = improvement)
bbbp_box = FancyBboxPatch((7.0, 2.8), 2.2, 1.4,
    boxstyle="round,pad=0.1", facecolor=C_BBBP, edgecolor=C_FRAME,
    linewidth=1.2, alpha=0.2)
ax.add_patch(bbbp_box)
ax.text(8.1, 3.9, 'BBBP (Physicochemical)', ha='center', va='center',
    fontsize=8, fontweight='bold', color=C_BBBP)
ax.text(8.1, 3.5, 'C4a: 0.626 vs 0.533 baseline', ha='center', va='center',
    fontsize=7, color=C_FRAME)
ax.text(8.1, 3.15, '▲ Directional trend under\n   structural topic focus', ha='center', va='center',
    fontsize=6.5, color='#2d6a2e', style='italic')

# BACE result box (red = collapse)
bace_box = FancyBboxPatch((7.0, 0.8), 2.2, 1.4,
    boxstyle="round,pad=0.1", facecolor=C_BACE, edgecolor=C_FRAME,
    linewidth=1.2, alpha=0.2)
ax.add_patch(bace_box)
ax.text(8.1, 1.9, 'BACE (Enzymatic)', ha='center', va='center',
    fontsize=8, fontweight='bold', color=C_BACE)
ax.text(8.1, 1.5, 'C3: 0.504 vs 0.674 baseline', ha='center', va='center',
    fontsize=7, color=C_FRAME)
ax.text(8.1, 1.15, '▼ Distributional compression\n   toward non-discriminative peak', ha='center', va='center',
    fontsize=6.5, color='#a03030', style='italic')

# === FAR RIGHT: Key insight ===
# Arrow from both results to insight
ax.annotate('', xy=(9.8, 2.5), xytext=(9.2, 3.3),
    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2))
ax.annotate('', xy=(9.8, 2.5), xytext=(9.2, 1.7),
    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2))

insight_box = FancyBboxPatch((9.7, 1.7), 2.1, 1.6,
    boxstyle="round,pad=0.1", facecolor='#fff3cd', edgecolor=C_FRAME,
    linewidth=1.5)
ax.add_patch(insight_box)
ax.text(10.75, 2.85, 'Key Finding', ha='center', va='center',
    fontsize=8, fontweight='bold', color=C_FRAME)
ax.text(10.75, 2.5, 'Semantic perturbation\nsensitivity is', ha='center', va='center',
    fontsize=7, color=C_FRAME)
ax.text(10.75, 2.0, 'task-dependent\n& scale-dependent', ha='center', va='center',
    fontsize=8, fontweight='bold', color='#8b6914')

# Title bar at top
ax.text(6.0, 4.7, 'Differential Sensitivity to Semantic Perturbations in LLM-Based Molecular Property Prediction',
    ha='center', va='center', fontsize=10, fontweight='bold', color=C_FRAME,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=C_FRAME, linewidth=1))

plt.tight_layout()
plt.savefig('hallucination-paper-overleaf/figures/graphical_abstract.png', 
    dpi=300, bbox_inches='tight', facecolor='white')
print("[OK] Saved graphical_abstract.png")
plt.close()
