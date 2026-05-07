# Differential Sensitivity to Semantic Perturbations in LLM-Based Molecular Property Prediction

**Investigating the Impact of Structured Hallucinations on LLM Molecular Reasoning**

This repository contains the source code, processed datasets, and manuscript LaTeX files for our study on how structured semantic perturbations (hallucinations) influence the performance of Large Language Models (LLMs) in molecular property prediction tasks.

## 🧪 Overview

We introduce a controlled semantic perturbation framework to evaluate how structured hallucination types—including Structural Phantom, Property Inversion, and Mechanism Fabrication—interact with model scale (Llama-8B vs. Llama-70B) and task domain (BBBP and BACE).

### Key Findings:
- **Orientation Semantics**: Experimental evidence suggests that the structural topic focus of a perturbation, rather than its factual accuracy, is a primary driver of performance shifts in LLM-based molecular property prediction.
- **Task-Specific Sensitivity**: We demonstrate a robust divergence between task domains; enzymatic tasks (BACE) exhibit significant performance degradation and "distributional compression" toward a low-confidence state, while physicochemical tasks (BBBP) show directional sensitivity to structural prompts.
- **Informative Hallucination**: Our framework characterizes hallucinations not as mere errors, but as activations of latent semantic priors that can be leveraged for probing model calibration.
- **Semantic Grounding**: Multi-seed random-permutation controls (C5) confirm that observed effects are driven by semantic alignment rather than scientific register or stylistic priming alone.

## 📁 Repository Structure

```
AI-Hallucinations/
├── src/                          # Core source code
│   ├── publication_figures.py    # Main figure generation
│   ├── statistical_analysis.py   # Bootstrap ROC-AUC testing engine
│   └── taxonomy_classifier.py    # Heuristic hallucination classifier
├── hallucination-paper-overleaf/ # LaTeX manuscript (Springer Nature Template)
├── data/processed/               # Final evaluation datasets for reproducibility
├── prompts/                      # Experimental prompt templates
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Reproducibility

### 1. Installation
```bash
git clone https://github.com/apiprdt/AI-Hallucinations.git
cd AI-Hallucinations
pip install -r requirements.txt
```

### 2. Run Statistical Analysis
Generate ROC-AUC values and confidence intervals reported in the manuscript:
```bash
python src/statistical_analysis.py data/processed/results_bbbp_checkpoint.json
python src/statistical_analysis.py data/processed/results_bace_checkpoint.json
```

### 3. Generate Figures
Regenerate publication-quality figures:
```bash
python src/publication_figures.py
python src/generate_fig5.py
```

## 📊 Experimental Conditions

| Code | Condition | Description |
|------|-----------|-------------|
| C0   | Baseline  | SMILES only |
| C1   | Factual   | RDKit descriptors |
| C2   | Chem Priming | Scientific gibberish control |
| C3   | Free Hallu | Unconstrained hallucination |
| C4a  | SP        | Structural Phantom (Prompt-constrained) |
| C5   | Random-Perm | Semantic permutation control |

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Citation
*Manuscript under review at the Journal of Molecular Modeling (Springer).*
If you use this work in your research, please cite:
> Erdita, M. A. (2026). Differential Sensitivity to Semantic Perturbations in LLM-Based Molecular Property Prediction. *Journal of Molecular Modeling* (In Submission).
