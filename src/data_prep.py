import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd
import numpy as np
import random
import string
import os
from tqdm import tqdm

def generate_factual_description(smiles: str) -> str:
    """
    Generate factual description from SMILES using RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid molecule"
    
    # Calculate verifiable physicochemical properties
    mw      = Descriptors.MolWt(mol)
    logp    = Descriptors.MolLogP(mol)
    hbd     = rdMolDescriptors.CalcNumHBD(mol)   # H-bond donors
    hba     = rdMolDescriptors.CalcNumHBA(mol)   # H-bond acceptors
    tpsa    = Descriptors.TPSA(mol)
    n_atoms = mol.GetNumAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    
    # Extract atom symbols
    atoms = set([atom.GetSymbol() for atom in mol.GetAtoms()])
    
    # Format Deskriptif/Natural
    size = "small" if mw < 300 else "moderately sized" if mw < 500 else "large"
    solubility = "water-preferring" if logp < 2 else "lipophilic"
    hb = "limited" if hbd < 2 else "moderate"
    
    description = (
        f"This molecule is {size} (MW ~{int(mw)} g/mol), "
        f"{solubility} (LogP ~{logp:.1f}), has {hb} hydrogen bonding capacity, "
        f"and contains {', '.join(sorted(atoms))} atoms arranged in {n_rings} ring(s)."
    )
    
    return description

def generate_gibberish(target_length: int, seed: int = 42) -> str:
    """
    Generate random text with length matched to a hypothetical hallucination.
    We'll use a placeholder length or match it dynamically later.
    For this script, we'll generate gibberish based on the factual description length 
    as a proxy for now, but in Phase 4 we will match the actual LLM output length.
    """
    random.seed(seed)
    words = []
    current_length = 0
    while current_length < target_length:
        word_len = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
        current_length += word_len + 1
    
    return ' '.join(words)[:target_length]

if __name__ == "__main__":
    # --- BBBP ---
    print("Loading BBBP dataset for Phase 2...")
    tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer='Raw', splitter='scaffold')
    train, valid, test = datasets
    smiles_list = test.ids.tolist()
    labels = test.y.flatten().tolist()
    
    data = []
    print(f"Processing {len(smiles_list)} molecules from BBBP test set...")
    for smiles, label in tqdm(zip(smiles_list, labels), total=len(smiles_list)):
        factual = generate_factual_description(smiles)
        data.append({
            "smiles": smiles,
            "label": label,
            "factual_desc": factual
        })
    pd.DataFrame(data).to_csv("bbbp_prepared.csv", index=False)

    # --- BACE ---
    print("\nLoading BACE dataset for replication...")
    # Since BACE.csv is already in the dir, we can load it directly or via DeepChem
    # Direct load is safer for column mapping
    raw_bace = pd.read_csv("BACE.csv")
    
    # We should use a split. Let's use the same scaffold split logic via DeepChem to be consistent
    tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer='Raw', splitter='scaffold')
    train, valid, test = datasets
    
    smiles_bace = test.ids.tolist()
    labels_bace = test.y.flatten().tolist()
    
    data_bace = []
    print(f"Processing {len(smiles_bace)} molecules from BACE test set...")
    for smiles, label in tqdm(zip(smiles_bace, labels_bace), total=len(smiles_bace)):
        factual = generate_factual_description(smiles)
        data_bace.append({
            "smiles": smiles,
            "label": label,
            "factual_desc": factual
        })
    
    pd.DataFrame(data_bace).to_csv("bace_prepared.csv", index=False)
    print(f"Saved prepared data to bace_prepared.csv. Total: {len(data_bace)}")
