# Semantic Orientation Effects in Zero-Shot LLM Molecular Inference

**Task-Dependent Sensitivity and Distributional Compression**

This repository contains the complete source code, evaluation datasets, and analysis pipelines for our study investigating how semantic orientation—specifically, the structural topic focus vs. semantic fabrication of prompted textual augmentations—modulates zero-shot molecular property predictions in Large Language Models (LLMs).

## 🧪 Overview

We introduce a controlled semantic perturbation framework to evaluate how factually incorrect or structured "hallucinations" interact with prediction tasks. We evaluated the `Llama-3.1-8B-Instant` model across four benchmarks: BBBP, BACE, Tox21, and ESOL.

### Key Findings:
- **Distributional Compression**: On enzymatic (BACE) and solubility (ESOL) tasks, semantic fabrication systematically destroys the model's discriminative capacity. The prediction distribution collapses into a narrow, low-confidence state, marked by a massive drop in Shannon Entropy and high Kullback-Leibler divergence ($D_{KL} > 14$).
- **Task-Dependent Sensitivity**: On general physicochemical tasks (BBBP, Tox21), prompt-constrained structural topic focus produces directional positive shifts in predictive discriminability (Cohen's $d = +0.51$) even without factual taxonomic correctness.
- **Semantic Conditioning**: We demonstrate that the semantic orientation of a prompt can drive significant behavioral shifts regardless of its factual accuracy, functioning as a form of semantic conditioning that modulates the LLM's inference state.
- **AI Taxonomy Validation**: LLM-as-judge evaluation demonstrates that strictly categorizing generative chemical text (e.g., Structural Phantoms vs. Mechanism Fabrication) is highly ambiguous, supporting the treatment of such categories as *prompt design intents* rather than definitive semantic truths.

## 📁 Repository Structure

```
AI-Hallucinations/
├── src/                          # Analysis scripts
│   ├── ai_taxonomy_validator.py  # LLM-as-judge taxonomy classification
│   ├── deepened_analysis.py      # Entropy, KL divergence, Cohen's d
│   ├── calibration_deepening.py  # Brier score decomposition
│   └── publication_figures.py    # Main figure generation
├── hallucination-paper-overleaf/ # LaTeX manuscript files
├── data/processed/               # Final inference results (JSON/CSV)
├── requirements.txt              # Python dependencies
├── run_all.py                    # Master reproduction script
└── README.md                     # This file
```

## 🚀 Reproducibility

This project was built to be fully reproducible without requiring expensive GPU resources. All analytical pipelines run on the pre-computed API checkpoints.

### 1. Installation
```bash
git clone https://github.com/apiprdt/AI-Hallucinations.git
cd AI-Hallucinations
pip install -r requirements.txt
```

### 2. Run All Analysis Pipelines
Execute the master reproduction script to run the deepened statistical analyses, taxonomy classification, and figure generation:
```bash
python run_all.py
```

## 📊 Evaluation Framework

| Code | Condition | Description |
|------|-----------|-------------|
| C0   | Baseline  | SMILES only |
| C1   | Factual   | RDKit descriptors |
| C2   | Chem Priming | Scientific gibberish control |
| C3   | Free Hallucination | Unconstrained semantic fabrication |
| C4a  | Structural Topic | Prompt-constrained structural focus |
| C5   | Random-Perm | Semantic permutation control |

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Citation
*Manuscript under review at Molecular Informatics (Wiley).*
If you use this work in your research, please cite:
> Erdita, M. A. (2026). Semantic Orientation Effects in Zero-Shot LLM Molecular Inference: Task-Dependent Sensitivity and Distributional Compression. *Molecular Informatics* (In Submission).
