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
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                time.sleep(5)
    return None

def generate_hallucination(smiles, condition):
    if condition == "C3":
        prompt = f"Provide a detailed scientific description of the molecular structure and properties for the molecule with SMILES: {smiles}. Focus on its potential biological interactions and physicochemical characteristics."
    elif condition == "C4a":
        prompt = f"Describe the scaffold and structural features of the molecule with SMILES: {smiles}. Mention specific rings, functional groups, and their spatial arrangement."
    else:
        return ""
    return call_groq(prompt)

def predict_toxicity(smiles, context, task_name="NR-AR"):
    if context:
        prompt = f"""Given this molecule (SMILES: {smiles})
Additional information: {context}

Is this molecule toxic for the {task_name} pathway? Respond with ONLY a number representing the probability of toxicity (between 0 and 1). Nothing else."""
    else:
        prompt = f"""Given this molecule (SMILES: {smiles})

Is this molecule toxic for the {task_name} pathway? Respond with ONLY a number representing the probability of toxicity (between 0 and 1). Nothing else."""

    output = call_groq(prompt, max_tokens=10)
    if output:
        match = re.search(r"0?\.\d+|\d+\.?\d*", output)
        if match:
            try:
                return float(match.group())
            except:
                return 0.5
    return 0.5

def run_tox21_experiment(n_samples=200):
    # Try to download Tox21 if not exists
    tox_path = "data/raw/tox21.csv"
    if not os.path.exists(tox_path):
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
        print("Downloading Tox21...")
        import requests, gzip
        r = requests.get(url)
        with open("data/raw/tox21.csv.gz", "wb") as f: f.write(r.content)
        with gzip.open("data/raw/tox21.csv.gz", 'rb') as f_in:
            with open(tox_path, 'wb') as f_out: f_out.write(f_in.read())
            
    df = pd.read_csv(tox_path)
    # Use 'NR-AR' as the primary task for simplicity
    df = df.dropna(subset=['NR-AR'])
    df_sampled = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    checkpoint_path = "data/processed/results_tox21_checkpoint.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f: ckpt = json.load(f)
    else:
        ckpt = {"metadata": {"dataset": "Tox21", "total": n_samples}, "results": {}}

    for i, row in tqdm(df_sampled.iterrows(), total=n_samples):
        idx_str = str(i)
        if idx_str not in ckpt["results"]:
            ckpt["results"][idx_str] = {"smiles": row["smiles"], "label": int(row["NR-AR"])}
        
        res = ckpt["results"][idx_str]
        smiles = res["smiles"]
        
        if "C0" not in res: res["C0"] = predict_toxicity(smiles, "")
        if "C1" not in res:
            desc = get_rdkit_descriptors(smiles)
            res["C1"] = predict_toxicity(smiles, desc)
        if "C3_text" not in res: res["C3_text"] = generate_hallucination(smiles, "C3")
        if "C3" not in res and res["C3_text"]: res["C3"] = predict_toxicity(smiles, res["C3_text"])
        if "C4a_text" not in res: res["C4a_text"] = generate_hallucination(smiles, "C4a")
        if "C4a" not in res and res["C4a_text"]: res["C4a"] = predict_toxicity(smiles, res["C4a_text"])

        if (i+1) % 10 == 0:
            with open(checkpoint_path, 'w') as f: json.dump(ckpt, f, indent=2)

    with open(checkpoint_path, 'w') as f: json.dump(ckpt, f, indent=2)

if __name__ == "__main__":
    run_tox21_experiment(200)
