# Hallucination Taxonomy Matters

**A Taxonomy-Controlled Ablation Study on LLM-Augmented Molecular Property Prediction**

This repository contains the code and datasets for a pilot study investigating how different types of LLM-generated hallucinations affect molecular property prediction performance.

## 🧪 Overview

Recent studies reported that hallucinated molecular descriptions can improve prediction performance. This project deconstructs this effect using a formal **Hallucination Taxonomy** and a 9-condition controlled ablation study across two MoleculeNet benchmarks (BBBP and BACE).

### Key Observations:
- **Distributional Bias**: LLMs naturally favor generic Contextual Confabulation (~80%), but rare Structural Phantom hallucinations exhibit the strongest predictive association on BBBP.
- **Prediction Collapse**: On BACE (enzyme inhibition), hallucination augmentation degrades performance, collapsing prediction distributions.
- **Semantic vs Style**: A circular-shift control demonstrates that scientific writing style alone does not account for observed gains.

## 📁 Repository Structure

```
HallucinationProject/
├── src/                          # Core source code
│   ├── taxonomy_classifier.py    # Hallucination taxonomy engine (SP/PI/MF/CC)
│   ├── data_prep.py              # Dataset loading & factual description generation
│   ├── run_bbbp.py               # Main experiment runner (all 9 conditions)
│   ├── 01_baseline.py            # Non-LLM baseline (Random Forest)
│   ├── statistics.py             # Statistical analysis (bootstrap AUC tests)
│   ├── 06_visualization.py       # Figure generation
│   ├── print_report.py           # Quick report printer
│   ├── fase1_runner.py           # C2b pure gibberish runner
│   ├── audit_c2.py               # C2 vocabulary audit
│   ├── verify_c5.py              # C5 shuffle verification
│   └── verify_spearman.py        # Spearman correlation check
├── hallucination-paper-overleaf/ # LaTeX paper (canonical)
│   ├── main.tex                  # Main document
│   ├── sections/                 # Paper sections
│   ├── tables/                   # LaTeX tables
│   ├── figures/                  # Publication figures
│   └── references.bib            # Bibliography (20 references)
├── scratch/                      # Exploratory scripts
├── data/                         # Raw and processed data
├── figures/                      # Generated figures
├── examples/                     # Output examples per condition
├── prompts/                      # Prompt documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Groq API Key (for LLM inference)

### 2. Installation
```bash
git clone https://github.com/apiprdt/AI-Hallucinations.git
cd AI-Hallucinations
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_api_key_here
```

### 4. Reproducing the Experiment

**Step 1: Prepare data**
```bash
python src/data_prep.py
```

**Step 2: Run the experiment (BBBP)**
```bash
python src/run_bbbp.py
```

**Step 3: Analyze results**
```bash
python src/statistics.py results_bbbp_checkpoint.json
```

**Step 4: Generate figures**
```bash
python src/06_visualization.py
```

## 📊 Experimental Conditions
| Code | Condition | Description |
|------|-----------|-------------|
| C0   | Baseline  | SMILES only |
| C1   | Factual   | RDKit-derived descriptors |
| C2   | Chem Priming | Gibberish with chemical vocabulary |
| C2b  | Pure Gibberish | Gibberish without chemical vocabulary |
| C3   | Free Hallucination | Unconstrained LLM hallucination |
| C4a  | Structural Phantom | Structure-focused hallucination |
| C4b  | Property Inversion | Property-focused hallucination |
| C4c  | Mechanism Fabrication | Mechanism-focused hallucination |
| C5   | Shuffled | Circular-shift semantic control |

## ⚠️ Limitations
This is an **exploratory pilot study** with important constraints:
- Single model (Llama-3.1-8B-Instant)
- Single scaffold-split evaluation (no cross-validation)
- Limited sample sizes (BBBP N=204, BACE N=152)

See the paper's Limitations section for full details.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Citation
```bibtex
@article{hallucination2026taxonomy,
  title={Hallucination Type Matters: A Taxonomy-Controlled Ablation Study 
         on LLM-Augmented Molecular Property Prediction},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```
