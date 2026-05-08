import json
import os
import random
import time
import re
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

# --- HELPER FUNCTIONS ---

def get_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return "Invalid SMILES"
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    return f"Molecular Weight: {mw:.2f}, LogP: {logp:.2f}, H-Bond Donors: {hbd}, H-Bond Acceptors: {hba}, TPSA: {tpsa:.2f}"

def call_groq(prompt, max_tokens=500, temperature=0.0):
    max_retries = 5
    retry_delay = 10
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                seed=42
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower() and attempt < max_retries - 1:
                print(f"Rate limit hit, sleeping {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Groq Error: {e}")
                time.sleep(5)
    return None

# --- GENERATION OPERATORS ---

def generate_hallucination(smiles, condition):
    if condition == "C3":
        prompt = f"Provide a detailed scientific description of the molecular structure and properties for the molecule with SMILES: {smiles}. Focus on its potential biological interactions and physicochemical characteristics."
    elif condition == "C4a":
        prompt = f"Describe the scaffold and structural features of the molecule with SMILES: {smiles}. Mention specific rings, functional groups, and their spatial arrangement."
    elif condition == "C4b":
        prompt = f"Describe the physicochemical properties (solubility, LogP, polar surface area) of the molecule with SMILES: {smiles}."
    elif condition == "C4c":
        prompt = f"Describe the mechanism of action and potential protein targets for the molecule with SMILES: {smiles} in the context of aqueous solubility."
    else:
        return ""
    
    return call_groq(prompt)

# --- PREDICTION ENGINE ---

def predict_solubility(smiles, context, condition):
    if context:
        prompt = f"""Given this molecule (SMILES: {smiles})
Additional information: {context}

What is the measured log solubility in mols per litre for this molecule? 
Respond with ONLY a numerical value (e.g., -2.5). Nothing else."""
    else:
        prompt = f"""Given this molecule (SMILES: {smiles})

What is the measured log solubility in mols per litre for this molecule? 
Respond with ONLY a numerical value (e.g., -2.5). Nothing else."""

    output = call_groq(prompt, max_tokens=10)
    if output:
        match = re.search(r"-?\d+\.?\d*", output)
        if match:
            try:
                return float(match.group())
            except:
                return 0.0
    return 0.0

# --- MAIN RUNNER ---

def run_esol_experiment(n_samples=200):
    df = pd.read_csv("data/raw/ESOL.csv")
    # Sample 200 compounds
    df_sampled = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    checkpoint_path = "data/processed/results_esol_checkpoint.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            ckpt = json.load(f)
    else:
        ckpt = {"metadata": {"dataset": "ESOL", "total": n_samples}, "results": {}}

    for i, row in tqdm(df_sampled.iterrows(), total=n_samples):
        idx_str = str(i)
        if idx_str not in ckpt["results"]:
            ckpt["results"][idx_str] = {
                "smiles": row["smiles"],
                "label": row["measured log solubility in mols per litre"]
            }
        
        res = ckpt["results"][idx_str]
        smiles = res["smiles"]
        
        # C0: Baseline
        if "C0" not in res:
            res["C0"] = predict_solubility(smiles, "", "C0")
            
        # C1: RDKit
        if "C1" not in res:
            desc = get_rdkit_descriptors(smiles)
            res["C1_text"] = desc
            res["C1"] = predict_solubility(smiles, desc, "C1")
            
        # C3: Free Hallucination
        if "C3_text" not in res:
            res["C3_text"] = generate_hallucination(smiles, "C3")
        if "C3" not in res and res["C3_text"]:
            res["C3"] = predict_solubility(smiles, res["C3_text"], "C3")
            
        # C4a: Structural
        if "C4a_text" not in res:
            res["C4a_text"] = generate_hallucination(smiles, "C4a")
        if "C4a" not in res and res["C4a_text"]:
            res["C4a"] = predict_solubility(smiles, res["C4a_text"], "C4a")

        # Save checkpoint every 10 molecules
        if (i+1) % 10 == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump(ckpt, f, indent=2)

    with open(checkpoint_path, 'w') as f:
        json.dump(ckpt, f, indent=2)

if __name__ == "__main__":
    run_esol_experiment(200)
