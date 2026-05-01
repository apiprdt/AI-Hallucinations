# Hallucination Taxonomy Matters

**A Taxonomy-Controlled Ablation Study on LLM-Augmented Molecular Property Prediction**

This repository contains the code and datasets for the research project investigating the impact of different hallucination types on molecular property prediction using Large Language Models (LLMs).

## 🧪 Overview

Recent studies reported that hallucinated molecular descriptions can improve prediction performance. This project deconstructs this effect using a formal **Hallucination Taxonomy** and a 9-condition ablation study.

### Key Findings:
- **Lazy Architect Effect**: LLMs naturally favor generic hallucinations, but specific "Structural Phantom" hallucinations provide the highest utility.
- **Negativity Collapse**: Hallucinations can severely degrade performance in specific tasks (e.g., enzymatic inhibition) through signal destruction.
- **Semantic Alignment**: Improvements are driven by semantic relevance, not just scientific writing style.

## 📁 Repository Structure
- `hallucination.py`: Core engine for taxonomy-based hallucination generation.
- `04_experiment_runner.py`: Main execution script for the ablation study.
- `05_analysis.py`: Statistical analysis and visualization pipeline.
- `prompts/`: Example prompts used in various conditions.
- `examples/`: Actual output examples for each condition.
- `final_figures/`: High-resolution research figures.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Groq API Key (for LLM inference)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/apiprdt/AI-Hallucinations.git
cd AI-Hallucinations

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_api_key_here
MODEL_NAME=llama-3.1-8b-instant
```

### 4. Running the Experiment
To run the full prediction pipeline for a dataset (e.g., BBBP):
```bash
python 04_experiment_runner.py
```

To analyze results and generate figures:
```bash
python 05_analysis.py
```

## 📊 Experimental Conditions
We evaluate 9 distinct conditions:
- **C0**: Baseline (SMILES only)
- **C1**: Factual (RDKit-derived)
- **C2**: Style Control (Structured Gibberish)
- **C3**: Free Hallucination (Dresden Replication)
- **C4a-c**: Taxonomy-specific hallucinations (Structural, Property, Mechanism)
- **C5**: Semantic Control (Shuffled Descriptions)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Citation
If you use this work in your research, please cite:
*[Your Name et al.], "Hallucination Type Matters: A Taxonomy-Controlled Ablation Study", 2026.*
