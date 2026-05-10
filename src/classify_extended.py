"""
Run heuristic taxonomy classifier on Tox21 and ESOL datasets.
These datasets were generated previously but the taxonomy was not computed
at generation time. This script computes it and updates the JSON.
"""
import json
import os
from collections import Counter
from pathlib import Path

def load_data(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# Hardcode the keyword logic from HallucinationEngine so we don't need to load models
ELEMENT_NAMES = {
    'F': ['fluorine', 'fluoro', 'fluorinated'],
    'Cl': ['chlorine', 'chloro', 'chlorinated'],
    'Br': ['bromine', 'bromo', 'brominated'],
    'I': ['iodine', 'iodo'],
    'P': ['phosphorus', 'phosphate'],
    'S': ['sulfur', 'sulfate', 'thiol'],
    'N': ['nitrogen', 'amine', 'amide', 'nitro']
}

def classify_text(llm_output, atoms, logp):
    llm_lower = llm_output.lower()
    
    results = {
        "structural_phantom": False,
        "property_inversion": False,
        "mechanism_fabrication": False,
        "dominant_type": "CC"
    }

    # 1. Structural Phantom (SP)
    for element, keywords in ELEMENT_NAMES.items():
        if element not in atoms:
            if any(k in llm_lower for k in keywords):
                results["structural_phantom"] = True
                results["dominant_type"] = "SP"
                break

    # 2. Property Inversion (PI)
    if logp > 3 and any(k in llm_lower for k in ["water soluble", "hydrophilic", "highly soluble"]):
        results["property_inversion"] = True
        results["dominant_type"] = "PI"
    elif logp < 1 and any(k in llm_lower for k in ["lipophilic", "hydrophobic", "membrane permeable"]):
        results["property_inversion"] = True
        results["dominant_type"] = "PI"

    # 3. Mechanism Fabrication (MF)
    common_targets = ['ace2', 'her2', 'egfr', 'vegf', 'p53', 'bcl-2', 'cox-2', 'dopamine', 'serotonin']
    if any(t in llm_lower for t in common_targets):
        results["mechanism_fabrication"] = True
        if results["dominant_type"] == "CC":
            results["dominant_type"] = "MF"

    return results

def process_dataset(path, name):
    print(f"\nProcessing {name}...")
    data = load_data(path)
    if not data:
        print(f"File not found: {path}")
        return

    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    n_total = data['metadata'].get('total_molecules', len(data['results']))
    
    counts = Counter()
    
    for i in range(n_total):
        r = data['results'][str(i)]
        
        # Check if taxonomy already exists
        if 'taxonomy_free' in r:
            counts[json.loads(r['taxonomy_free'])['dominant_type']] += 1
            continue
            
        # Get C3 text
        text = r.get('C3_text', r.get('hallu_free_text', ''))
        if not text:
            continue
            
        # Compute properties
        mol = Chem.MolFromSmiles(r['smiles'])
        if not mol:
            continue
            
        atoms = set([atom.GetSymbol() for atom in mol.GetAtoms()])
        logp = Descriptors.MolLogP(mol)
        
        # Classify
        tax = classify_text(text, atoms, logp)
        r['taxonomy_free'] = json.dumps(tax)
        counts[tax['dominant_type']] += 1
        
    # Save back
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Taxonomy distribution for {name}:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

def main():
    process_dataset('data/processed/results_tox21_checkpoint.json', 'Tox21')
    process_dataset('data/processed/results_esol_checkpoint.json', 'ESOL')

if __name__ == '__main__':
    main()
