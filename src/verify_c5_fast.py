import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from groq import Groq
from tqdm import tqdm
import re
import time
from dotenv import load_dotenv

load_dotenv()

class Predictor:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def predict_property(self, smiles: str, augmentation: str) -> float:
        if not self.client: return 0.5
        prompt = f"""Given this molecule (SMILES: {smiles})
Additional information: {augmentation}

On a scale of 0 to 1, what is the probability that this molecule can penetrate the blood-brain barrier?
Respond with ONLY a number between 0 and 1. Nothing else."""
        
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
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
            except Exception:
                time.sleep(2)
        return 0.5

def main():
    with open("data/processed/results_bbbp_checkpoint.json", "r") as f:
        ckpt = json.load(f)
    
    data = []
    for k, v in ckpt["results"].items():
        data.append({
            "smiles": v["smiles"],
            "label": v["label"],
            "c3_text": v["hallu_free_text"]
        })
    df = pd.DataFrame(data)
    
    # We take 50 random samples to make AUC valid but fast
    df = df.sample(n=50, random_state=42).reset_index(drop=True)
    
    predictor = Predictor()
    seeds = [100, 200]
    results = {}
    
    for seed in seeds:
        np.random.seed(seed)
        shuffled_idx = np.random.permutation(50)
        c5_texts = df['c3_text'].iloc[shuffled_idx].values
        
        preds = []
        for i, row in tqdm(df.iterrows(), total=50):
            preds.append(predictor.predict_property(row['smiles'], c5_texts[i]))
            
        auc = roc_auc_score(df['label'], preds)
        results[seed] = auc
        print(f"Seed {seed}: {auc:.4f}")
        
    print("\n--- C5 FAST RESULTS ---")
    aucs = list(results.values())
    print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    with open("c5_fast_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
