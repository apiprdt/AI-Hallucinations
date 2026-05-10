import sys
import pandas as pd
import numpy as np
import time
import os
import json
import re
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

class MixtralRunner:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)

    def predict_property(self, smiles: str, augmentation: str = None, task: str = "bbbp") -> float:
        task_prompts = {
            "bbbp": "penetrate the blood-brain barrier",
            "bace": "inhibit the BACE-1 enzyme"
        }

        context = f"Additional information: {augmentation}\n\n" if augmentation else ""
        prompt = f"""Given this molecule (SMILES: {smiles})
{context}On a scale of 0 to 1, what is the probability that this molecule can {task_prompts[task]}?
Respond with ONLY a number between 0 and 1. Nothing else."""

        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="mixtral-8x7b-32768",
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

    def run_eval(self, task: str):
        # We don't need the CSV, we can extract SMILES and labels from the checkpoint
        ckpt_path = f"data/processed/results_{task}_checkpoint.json"
        with open(ckpt_path, 'r') as f:
            llama_ckpt = json.load(f)
            
        results_file = f"data/processed/results_Mixtral_{task}.json"
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            results = {"completed": [], "data": {}}
            
        total_mols = llama_ckpt["metadata"]["total_molecules"]
        print(f"Running Mixtral-2-9b-it on {task.upper()} (N={total_mols})...")
        
        for i in tqdm(range(total_mols)):
            idx_str = str(i)
            if i in results["completed"]:
                continue
                
            item = llama_ckpt["results"][idx_str]
            smiles = item['smiles']
            label = item['label']
            
            # Get the exact C3 hallucination used for Llama
            hallu_free = item["hallu_free_text"]
            
            c0_pred = self.predict_property(smiles, None, task=task)
            c3_pred = self.predict_property(smiles, hallu_free, task=task)
            
            results["data"][idx_str] = {
                "smiles": smiles,
                "label": int(label),
                "C0_pure": c0_pred,
                "C3_hallu_free": c3_pred
            }
            results["completed"].append(i)
            
            if len(results["completed"]) % 20 == 0:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Calculate ROC-AUC
        y_true = [results["data"][str(i)]["label"] for i in range(total_mols)]
        c0_preds = [results["data"][str(i)]["C0_pure"] for i in range(total_mols)]
        c3_preds = [results["data"][str(i)]["C3_hallu_free"] for i in range(total_mols)]
        
        try:
            auc_c0 = roc_auc_score(y_true, c0_preds)
            auc_c3 = roc_auc_score(y_true, c3_preds)
            print(f"[{task.upper()}] Mixtral-2-9B AUC | C0: {auc_c0:.3f} | C3: {auc_c3:.3f}")
        except ValueError:
            print(f"Could not calculate AUC for {task}")

if __name__ == "__main__":
    runner = MixtralRunner()
    runner.run_eval("bbbp")
    runner.run_eval("bace")
