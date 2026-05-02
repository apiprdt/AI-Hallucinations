# Architecture-Dependent Sensitivity to Structured Semantic Perturbations

**A Controlled Perturbation Study on LLM-Based Molecular Property Prediction**

This repository contains the code, data, and manuscript sources for the study investigating how different types of LLM-generated hallucinations—and the scale of the model—influence molecular property prediction performance.

## 🧪 Overview

We introduce a controlled semantic perturbation framework to evaluate how structured hallucination types (Structural Phantom, Property Inversion, Mechanism Fabrication, and Contextual Confabulation) interact with model scale (Llama-8B vs. Llama-70B) and task domain (BBBP and BACE).

### Key Observations:
- **Architecture-Dependent Sensitivity**: Smaller models (8B) can exhibit performance shifts under hallucination augmentation on certain tasks, whereas larger models (70B) exhibit reversed sensitivity or performance degradation.
- **Task-Dependent Divergence**: Hallucination augmentation consistently degrades performance on enzymatic tasks (BACE), while physicochemical tasks (BBBP) show directional sensitivity to structured perturbations.
- **Semantic Control**: A random-permutation control (C5) demonstrates that the observed effects are semantically grounded rather than driven by stylistic priming (scientific register).

## 📁 Repository Structure

```
AI-Hallucinations/
├── src/                          # Core source code
│   ├── publication_figures.py    # Main figure generation (Figure 1-4)
│   ├── generate_fig5.py          # Scaling divergence figure
│   ├── statistical_analysis.py   # Statistical engine (Bootstrap AUC tests)
│   ├── taxonomy_classifier.py    # Hallucination taxonomy engine
│   └── data_prep.py              # Dataset loading & descriptors
├── hallucination-paper-overleaf/ # LaTeX manuscript
├── data/processed/               # Final evaluation datasets (Reproducibility)
├── prompts/                      # Prompt templates for all conditions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Reproducibility

To reproduce the analysis and figures reported in the paper:

### 1. Installation
```bash
git clone https://github.com/apiprdt/AI-Hallucinations.git
cd AI-Hallucinations
pip install -r requirements.txt
```

### 2. Run Statistical Analysis
This will generate the ROC-AUC values, confidence intervals, and p-values reported in Table 1.
```bash
python src/statistical_analysis.py data/processed/results_bbbp_checkpoint.json
python src/statistical_analysis.py data/processed/results_bace_checkpoint.json
```

### 3. Generate Publication Figures
This will regenerate the figures used in the manuscript.
```bash
python src/publication_figures.py
python src/generate_fig5.py
```

## 📊 Experimental Conditions
| Code | Condition | Description |
|------|-----------|-------------|
| C0   | Baseline  | SMILES only |
| C1   | Factual   | RDKit descriptors |
| C2   | Chem Priming | Scientific gibberish |
| C3   | Free Hallu | Unconstrained hallucination |
| C4a  | SP        | Structural Phantom |
| C5   | Random-Perm | Random-permutation control |

## 📄 License
This project is licensed under the MIT License.

## 🤝 Citation
Please cite the F1000Research version (TBA).
