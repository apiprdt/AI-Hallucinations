import os
import re
import time
import json
import pandas as pd
from groq import Groq
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import cohen_kappa_score
from dotenv import load_dotenv

# Load from .env if available
load_dotenv()

class HallucinationEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("WARNING: GROQ_API_KEY not found. API calls will fail.")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        
        # Atom names for taxonomy detection
        self.ELEMENT_NAMES = {
            'F': ['fluorine', 'fluoro', 'fluorinated'],
            'Cl': ['chlorine', 'chloro', 'chlorinated'],
            'Br': ['bromine', 'bromo', 'brominated'],
            'I': ['iodine', 'iodo'],
            'P': ['phosphorus', 'phosphate'],
            'S': ['sulfur', 'sulfate', 'thiol'],
            'N': ['nitrogen', 'amine', 'amide', 'nitro']
        }

    def generate_hallucination(self, smiles: str, h_type: str = "free") -> str:
        """
        Generate hallucination using Llama-3.1 via Groq.
        """
        if not self.client:
            return "MOCK: This is a hallucinated description of the molecule."

        prompts = {
            "free": f"You are a speculative chemist. Given SMILES: {smiles}. Generate a creative, scientifically-sounding description of this molecule's potential properties and biological mechanisms. Be creative and speculative. Write 2-3 sentences only.",
            "structural": f"Given SMILES: {smiles}. Describe the structural features, functional groups, and atomic composition. Be specific and speculative.",
            "property": f"Given SMILES: {smiles}. Describe its solubility, lipophilicity, and membrane permeability with specific values.",
            "mechanism": f"Given SMILES: {smiles}. Hypothesize biological targets and cellular mechanisms. Be specific about protein targets."
        }

        max_retries = 3
        retry_delay = 5 # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompts.get(h_type, prompts["free"])}],
                    max_tokens=150,
                    temperature=0.9 # High temperature for hallucination
                )
                return response.choices[0].message.content
            except Exception as e:
                if "rate_limit_exceeded" in str(e).lower() and attempt < max_retries - 1:
                    print(f"Rate limit hit in hallucination, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Error generating hallucination: {e}")
                    return ""
        return ""

    def classify_hallucination(self, llm_output: str, smiles: str) -> dict:
        """
        Rule-based taxonomy classification.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"dominant_type": "INVALID"}

        actual_atoms = set([atom.GetSymbol() for atom in mol.GetAtoms()])
        llm_lower = llm_output.lower()
        
        results = {
            "structural_phantom": False,
            "property_inversion": False,
            "mechanism_fabrication": False,
            "dominant_type": "CC" # Contextual Confabulation (default)
        }

        # 1. Structural Phantom (SP)
        for element, keywords in self.ELEMENT_NAMES.items():
            if element not in actual_atoms:
                if any(k in llm_lower for k in keywords):
                    results["structural_phantom"] = True
                    results["dominant_type"] = "SP"
                    break

        # 2. Property Inversion (PI)
        logp = Descriptors.MolLogP(mol)
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
            if results["dominant_type"] == "CC": # Only override if not already SP or PI
                results["dominant_type"] = "MF"

        return results

def validate_classifier(manual_labels_path, predictions):
    """
    Phase 3.5: Cohen's Kappa validation.
    """
    if not os.path.exists(manual_labels_path):
        print(f"Manual labels file {manual_labels_path} not found. Skipping validation.")
        return
    
    manual_df = pd.read_csv(manual_labels_path)
    # Assumes manual_df has 'manual_type' column
    kappa = cohen_kappa_score(manual_df['manual_type'], predictions)
    print(f"Cohen's Kappa Score: {kappa:.4f}")
    return kappa

if __name__ == "__main__":
    # Example usage
    engine = HallucinationEngine()
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O" # Aspirin
    print(f"SMILES: {test_smiles}")
    
    hallu = engine.generate_hallucination(test_smiles)
    print(f"Hallucination: {hallu}")
    
    classification = engine.classify_hallucination(hallu, test_smiles)
    print(f"Classification: {classification}")
