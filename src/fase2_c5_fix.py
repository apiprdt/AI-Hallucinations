import json
import random
import os
import time
import re
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

load_dotenv()

def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    return Groq(api_key=api_key)

client = get_client()

def predict_property(smiles, augmentation, task="bbbp"):
    task_prompts = {
        "bbbp": "penetrate the blood-brain barrier",
        "bace": "inhibit the BACE-1 enzyme"
    }
    
    context = f"Additional information: {augmentation}" if augmentation else ""
    prompt = f"""Given this molecule (SMILES: {smiles})
{context}

On a scale of 0 to 1, what is the probability that this molecule can {task_prompts[task]}?
Respond with ONLY a number between 0 and 1. Nothing else."""

    max_retries = 3
    retry_delay = 5 
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
                seed=42
            )
            output = response.choices[0].message.content.strip()
            match = re.search(r"0?\.\d+|\d+\.?\d*", output)
            if match:
                prob = float(match.group())
                return min(max(prob, 0.0), 1.0)
            return 0.5
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return 0.5
    return 0.5

def generate_derangement(n, seed=42):
    """Generate a random permutation where no element maps to itself."""
    random.seed(seed)
    indices = list(range(n))
    while True:
        p = indices[:]
        random.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p

def fix_checkpoint(file_path, task_name):
    print(f"\nProcessing {file_path} ({task_name})...")
    with open(file_path, 'r') as f:
        ckpt = json.load(f)
    
    total = ckpt["metadata"]["total_molecules"]
    # Generate true random derangement
    p = generate_derangement(total, seed=42)
    
    # Map index to its source hallucination
    # Molecule i will get hallucination from source_idx p[i]
    
    for i in tqdm(range(total)):
        idx_str = str(i)
        source_idx = p[i]
        source_idx_str = str(source_idx)
        
        # Get target molecule SMILES
        smiles = ckpt["results"][idx_str]["smiles"]
        # Get source hallucination text (from molecule source_idx)
        hallu_text = ckpt["c3_cache"][source_idx_str]
        
        # Check if we already have this prediction (very unlikely but for safety)
        # We are overwriting C5_shuffled
        old_val = ckpt["results"][idx_str].get("C5_shuffled", "N/A")
        old_src = ckpt["results"][idx_str].get("c5_source_index", "N/A")
        
        # Call API
        new_val = predict_property(smiles, hallu_text, task=task_name)
        
        # Update results
        ckpt["results"][idx_str]["C5_shuffled"] = new_val
        ckpt["results"][idx_str]["c5_source_index"] = source_idx
        ckpt["results"][idx_str]["c5_mechanism"] = "random_derangement_seed42"
        
        # Periodic save
        if (i+1) % 25 == 0:
            with open(file_path, 'w') as f:
                json.dump(ckpt, f, indent=2)
                
    # Final save
    with open(file_path, 'w') as f:
        json.dump(ckpt, f, indent=2)
    print(f"Finished {task_name}.")

if __name__ == "__main__":
    bbbp_path = "data/processed/results_bbbp_checkpoint.json"
    bace_path = "data/processed/results_bace_checkpoint.json"
    
    fix_checkpoint(bbbp_path, "bbbp")
    fix_checkpoint(bace_path, "bace")
