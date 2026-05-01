### C0: Pure SMILES Baseline
Task: Predict if the following molecule is a Blood-Brain Barrier Penetrant (1 for yes, 0 for no).
SMILES: CC(=O)Oc1ccccc1C(=O)O
Output only a single number representing the probability (0.0 to 1.0).

---

### C3: Free Hallucination
Task: Predict if the following molecule is a Blood-Brain Barrier Penetrant (1 for yes, 0 for no).
SMILES: CC(=O)Oc1ccccc1C(=O)O
Description: This compound exhibits a unique planar geometry with multiple heterocyclic rings that facilitate interaction with dopamine receptors. Its high lipophilicity suggests efficient passive diffusion across the blood-brain barrier.
Output only a single number representing the probability (0.0 to 1.0).

---

### C4a: Structural Phantom (SP)
Task: Predict if the following molecule is a Blood-Brain Barrier Penetrant (1 for yes, 0 for no).
SMILES: CC(=O)Oc1ccccc1C(=O)O
Description: Contains a reactive fluorine substituent at the para position which significantly enhances its metabolic stability and CNS penetration profile.
Output only a single number representing the probability (0.0 to 1.0).

---

### C5: Shuffled Hallucination (Semantic Control)
Task: Predict if the following molecule is a Blood-Brain Barrier Penetrant (1 for yes, 0 for no).
SMILES: CC(=O)Oc1ccccc1C(=O)O
Description: [Description taken from a completely different molecule, e.g., a large polar peptide]
Output only a single number representing the probability (0.0 to 1.0).
