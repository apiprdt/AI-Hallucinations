# Taxonomy-Controlled Hallucination in Molecular Property Prediction

This repository contains the replication and extension of the research on **LLM-Augmented Molecular Property Prediction**. We investigate the impact of different hallucination types on prediction accuracy across various chemical datasets (BBBP, BACE).

## 🚀 Key Findings

- **The Booster Effect (BBBP)**: In physicochemical tasks, strategic hallucination acts as a performance booster, improving ROC-AUC from 0.53 to 0.62.
- **The Negativity Collapse (BACE)**: In highly specific enzymatic tasks, hallucination can induce a systematic bias, compressing prediction scores toward zero (Negativity Collapse).
- **The Lazy Architect Bias**: LLMs exhibit a systematic preference for *Contextual Confabulation* (80%) over more predictively useful *Structural Phantoms* (2%).
- **Scientific Register Sensitivity**: LLMs respond to a "Scientific Tone" with increased confidence, but this boost is decoupled from discriminative accuracy (as proven by our Circular Shift control).

## 🧬 Taxonomy of Hallucinations

We classify LLM-generated descriptions into four distinct types:
1. **Structural Phantom (SP)**: Inventing non-existent functional groups or atoms.
2. **Property Inversion (PI)**: Attributing properties of the opposite class to the molecule.
3. **Mechanism Fabrication (MF)**: Inventing complex but plausible biological pathways.
4. **Contextual Confabulation (CC)**: Narrative descriptions that are plausible but generic.

## 🛠 Project Structure

- `hallucination.py`: Core engine for generating taxonomy-controlled hallucinations.
- `experiment_runner.py`: Parallelized execution engine using multiple Groq API keys.
- `05_analysis.py`: Statistical rigor analysis (Cohen's d, BH Correction, CI).
- `data_prep.py`: Dataset loading and factual description generation.

## 📈 Methodology

Our study uses a 6-condition ablation design:
- **C0**: Pure SMILES (Baseline)
- **C1**: Factual Description
- **C2**: Random Chemical Priming
- **C3**: Natural Hallucination (Free)
- **C4**: Forced Taxonomy Hallucination (SP, PI, MF)
- **C5**: Circular Shift Control (Scientific Tone Check)

## ⚖️ License
MIT License
