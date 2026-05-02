import json
import os
import time
import re
import pandas as pd
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    return Groq(api_key=api_key)

client = get_client()

def predict_property(smiles, augmentation, task="bbbp", model="llama-3.1-8b-instant", seed=42):
    task_prompts = {
        "bbbp": "penetrate the blood-brain barrier",
        "bace": "inhibit the BACE-1 enzyme"
    }
    
    context = f"Additional information: {augmentation}" if augmentation else ""
    prompt = f"""Given this molecule (SMILES: {smiles})
{context}

On a scale of 0 to 1, what is the probability that this molecule can {task_prompts[task]}?
Respond with ONLY a number between 0 and 1. Nothing else."""

    max_retries = 5
    retry_delay = 10 
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
                seed=seed
            )
            output = response.choices[0].message.content.strip()
            match = re.search(r"0?\.\d+|\d+\.?\d*", output)
            if match:
                prob = float(match.group())
                return min(max(prob, 0.0), 1.0)
            return 0.5
        except Exception as e:
            err_msg = str(e).lower()
            if "rate_limit_exceeded" in err_msg or "too many requests" in err_msg:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return 0.5
    return 0.5

def run_robustness(task_name, checkpoint_path, output_path):
    print(f"\n[INFO] Running Robustness for {task_name.upper()}...")
    with open(checkpoint_path, 'r') as f:
        ckpt = json.load(f)
        
    # Load or init new checkpoint
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            rob_ckpt = json.load(f)
    else:
        rob_ckpt = {"results": {}}

    configurations = [
        {"model": "llama-3.1-8b-instant", "seed": 123, "tag": "8b_s123"},
        {"model": "llama-3.1-8b-instant", "seed": 777, "tag": "8b_s777"},
        {"model": "llama-3.3-70b-versatile", "seed": 42, "tag": "70b_s42"}
    ]
    
    total = ckpt["metadata"]["total_molecules"]
    
    for config in configurations:
        tag = config["tag"]
        model = config["model"]
        seed = config["seed"]
        print(f"--- Config: {tag} ({model}, seed={seed}) ---")
        
        for i in tqdm(range(total)):
            idx_str = str(i)
            if idx_str not in rob_ckpt["results"]:
                rob_ckpt["results"][idx_str] = {}
            
            # Check if this config is already done for this molecule
            if f"C0_{tag}" in rob_ckpt["results"][idx_str] and f"C3_{tag}" in rob_ckpt["results"][idx_str]:
                continue
            
            smiles = ckpt["results"][idx_str]["smiles"]
            hallu_text = ckpt["results"][idx_str]["hallu_free_text"]
            
            # C0 (Baseline)
            c0_val = predict_property(smiles, None, task=task_name, model=model, seed=seed)
            rob_ckpt["results"][idx_str][f"C0_{tag}"] = c0_val
            
            # C3 (Free Hallu)
            c3_val = predict_property(smiles, hallu_text, task=task_name, model=model, seed=seed)
            rob_ckpt["results"][idx_str][f"C3_{tag}"] = c3_val
            
            # C4c (Mechanism Fabrication) - Only for 8b seeds to keep it light
            if "8b" in tag:
                hallu_mf = ckpt["results"][idx_str]["hallu_mf_text"]
                c4c_val = predict_property(smiles, hallu_mf, task=task_name, model=model, seed=seed)
                rob_ckpt["results"][idx_str][f"C4c_{tag}"] = c4c_val
                
            if (i+1) % 20 == 0:
                with open(output_path, 'w') as f:
                    json.dump(rob_ckpt, f, indent=2)
                    
    with open(output_path, 'w') as f:
        json.dump(rob_ckpt, f, indent=2)
    print(f"[OK] Finished {task_name}.")

if __name__ == "__main__":
    bbbp_ckpt = "data/processed/results_bbbp_checkpoint.json"
    bace_ckpt = "data/processed/results_bace_checkpoint.json"
    
    run_robustness("bbbp", bbbp_ckpt, "data/processed/results_bbbp_robustness.json")
    run_robustness("bace", bace_ckpt, "data/processed/results_bace_robustness.json")
