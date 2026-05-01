import sys
import pandas as pd
import numpy as np
import time
import os
import json
import re
import random
from datetime import datetime
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Import our taxonomy logic
from hallucination import HallucinationEngine

load_dotenv()

ADJECTIVES = ["moderate", "intermediate", "variable", 
              "standard", "conventional", "typical"]
NOUNS = ["hypothetical", "theoretical", "computational",
         "structural", "molecular", "chemical"]

def generate_structured_gibberish(target_length: int = 0) -> str:
    random.seed(42 + target_length) # just to make it pseudo-random but deterministic
    template = (
        f"This compound exhibits {random.choice(ADJECTIVES)} "
        f"structural coherence with {random.choice(NOUNS)} "
        f"interaction domains, demonstrating "
        f"{random.choice(ADJECTIVES)} physicochemical "
        f"characteristics under {random.choice(NOUNS)} conditions."
    )
    return template

ADJ_BUREAUCRATIC = ["variable", "standard", "conventional", "typical", "moderate", "intermediate"]

NOUN_BUREAUCRATIC = ["portfolio", "transactional", "governance", "bureaucratic", "administrative", "regulatory"]

def generate_c2b_pure_gibberish(target_length: int = 0) -> str:
    random.seed(42 + target_length)
    template = (
        f"This entity exhibits {random.choice(ADJ_BUREAUCRATIC)} "
        f"portfolio coherence with {random.choice(NOUN_BUREAUCRATIC)} "
        f"governance domains, demonstrating "
        f"{random.choice(ADJ_BUREAUCRATIC)} bureaucratic "
        f"characteristics under {random.choice(NOUN_BUREAUCRATIC)} conditions."
    )
    return template

class ExperimentRunner:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.engine = HallucinationEngine(api_key=self.api_key)

    def predict_property(self, smiles: str, augmentation: str = None, task: str = "bbbp") -> float:
        """
        Ask LLM to predict property with temperature=0.0.
        """
        if not self.client:
            return 0.5 # Mock uncertainty

        task_prompts = {
            "bbbp": "penetrate the blood-brain barrier",
            "bace": "inhibit the BACE-1 enzyme",
            "hiv": "inhibit HIV replication"
        }

        context = f"Additional information: {augmentation}" if augmentation else ""
        prompt = f"""Given this molecule (SMILES: {smiles})
{context}

On a scale of 0 to 1, what is the probability that this molecule can {task_prompts[task]}?
Respond with ONLY a number between 0 and 1. Nothing else."""

        max_retries = 3
        retry_delay = 5 # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0, # Deterministic prediction
                    seed=42
                )
                output = response.choices[0].message.content.strip()
                # Extract number
                match = re.search(r"0?\.\d+|\d+\.?\d*", output)
                if match:
                    prob = float(match.group())
                    return min(max(prob, 0.0), 1.0)
                return 0.5
            except Exception as e:
                if "rate_limit_exceeded" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Exponential backoff
                else:
                    return 0.5
        return 0.5

    def run(self, input_csv: str, checkpoint_file: str, task: str = "bbbp", limit: int = None, test_mode: bool = False):
        df = pd.read_csv(input_csv)
        if limit:
            df = df.head(limit)
            
        total_molecules = len(df)
        
        # 1. Load or Initialize Checkpoint
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                ckpt = json.load(f)
        else:
            ckpt = {
                "completed": [],
                "c3_cache": {}, # index (str) -> c3 text
                "results": {},  # index (str) -> result dict
                "metadata": {
                    "dataset": task,
                    "total_molecules": total_molecules,
                    "conditions": ["C0","C1","C2_chempriming","C2b_pure_gibberish","C3","C4a","C4b","C4c","C5"],
                    "model": "llama-3.1-8b-instant",
                    "temperature_predict": 0.0,
                    "temperature_generate": 0.9,
                    "seed": 42,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        # Helper to save
        def save_checkpoint():
            with open(checkpoint_file, 'w') as f:
                json.dump(ckpt, f, indent=2)

        # STEP 1: Generate SEMUA C3 hallucinations dulu
        print("STEP 1: Generating all C3 hallucinations...")
        c3_generated = False
        for i in tqdm(range(total_molecules)):
            idx_str = str(i)
            if idx_str not in ckpt["c3_cache"]:
                smiles = df.iloc[i]['smiles']
                c3_text = self.engine.generate_hallucination(smiles, "free")
                ckpt["c3_cache"][idx_str] = c3_text
                c3_generated = True
        
        if c3_generated:
            save_checkpoint()
            
        # STEP 2 & 3: Buat array C5 dengan circular shift (i+1 % len)
        print("STEP 2 & 3: Building C5 array via circular shift...")
        c5_array = []
        for i in range(total_molecules):
            source_idx = (i + 1) % total_molecules
            c5_array.append(ckpt["c3_cache"][str(source_idx)])
            
        # VALIDASI SEBELUM FULL RUN (Test Mode)
        if test_mode:
            print("\n" + "="*40)
            print("VALIDASI C5 (TEST MODE)")
            print("="*40)
            print(f"Molekul 0 (SMILES): {df.iloc[0]['smiles']}")
            print(f"Sumber C5-nya: {ckpt['c3_cache']['1'][:50]}... (Dari Molekul 1)")
            print(f"Sama dengan C5 array[0]? {c5_array[0] == ckpt['c3_cache']['1']}")
            
            print(f"\nMolekul 1 (SMILES): {df.iloc[1]['smiles']}")
            print(f"Sumber C5-nya: {ckpt['c3_cache']['2'][:50]}... (Dari Molekul 2)")
            
            last_idx = total_molecules - 1
            print(f"\nMolekul Terakhir [{last_idx}] (SMILES): {df.iloc[last_idx]['smiles']}")
            print(f"Sumber C5-nya: {ckpt['c3_cache']['0'][:50]}... (Dari Molekul 0)")
            return

        # STEP 4: Jalankan semua prediksi C0-C5
        print("STEP 4: Running predictions for all conditions...")
        
        for i in tqdm(range(total_molecules)):
            idx_str = str(i)
            if i in ckpt["completed"]:
                continue
                
            row = df.iloc[i]
            smiles = row['smiles']
            label = row['label']
            factual = row['factual_desc']
            
            hallu_free = ckpt["c3_cache"][idx_str]
            hallu_sp   = self.engine.generate_hallucination(smiles, "structural")
            hallu_pi   = self.engine.generate_hallucination(smiles, "property")
            hallu_mf   = self.engine.generate_hallucination(smiles, "mechanism")
            
            chem_gibberish = generate_structured_gibberish(len(hallu_free))
            pure_gibberish = generate_c2b_pure_gibberish(len(hallu_free))
            
            c5_hallu = c5_array[i]
            c5_source_index = (i + 1) % total_molecules
            
            res = {
                "smiles": smiles,
                "label": label,
                "C0_pure": self.predict_property(smiles, None, task=task),
                "C1_factual": self.predict_property(smiles, factual, task=task),
                "C2_chempriming": self.predict_property(smiles, chem_gibberish, task=task),
                "C2b_pure_gibberish": self.predict_property(smiles, pure_gibberish, task=task),
                "C3_hallu_free": self.predict_property(smiles, hallu_free, task=task),
                "C4a_hallu_sp": self.predict_property(smiles, hallu_sp, task=task),
                "C4b_hallu_pi": self.predict_property(smiles, hallu_pi, task=task),
                "C4c_hallu_mf": self.predict_property(smiles, hallu_mf, task=task),
                "C5_shuffled": self.predict_property(smiles, c5_hallu, task=task),
                "c5_source_index": c5_source_index,
                "hallu_free_text": hallu_free,
                "hallu_mf_text": hallu_mf,
                "taxonomy_free": json.dumps(self.engine.classify_hallucination(hallu_free, smiles))
            }
            
            ckpt["results"][idx_str] = res
            ckpt["completed"].append(i)
            
            # Checkpoint every 25 molecules
            if len(ckpt["completed"]) % 25 == 0:
                save_checkpoint()
                
        # Final save
        save_checkpoint()
        print(f"\nExperiment finished. {total_molecules}/{total_molecules} molecules processed.")
        
        # LAPORAN FINAL
        self.generate_report(ckpt)

    def generate_report(self, ckpt):
        results_list = [ckpt["results"][str(i)] for i in range(ckpt["metadata"]["total_molecules"])]
        df_res = pd.DataFrame(results_list)
        conditions = ["C0_pure","C1_factual","C2_chempriming","C2b_pure_gibberish","C3_hallu_free","C4a_hallu_sp","C4b_hallu_pi","C4c_hallu_mf","C5_shuffled"]
        
        print("\n" + "="*80)
        print("LAPORAN FINAL (204 MOLEKUL)")
        print("="*80)
        
        print("\nBAGIAN 1 — Distribusi Prediksi:")
        stats_df = df_res[conditions].agg(['mean', 'std', 'min', 'max']).T
        print(stats_df.to_string())
        
        print("\nBAGIAN 2 — Taxonomy Distribution C3:")
        df_res['tax_type'] = df_res['taxonomy_free'].apply(lambda x: json.loads(x)['dominant_type'])
        tax_counts = df_res['tax_type'].value_counts()
        for t, count in tax_counts.items():
            print(f"{t}: {count} molekul")
            
        print("\nBAGIAN 3 — Sanity Checks:")
        labels = df_res['label']
        print(f"- Positive rate aktual di {len(df_res)} sampel: {labels.mean():.2%}")
        
        total_preds = len(df_res) * len(conditions)
        count_05 = (df_res[conditions] == 0.5).sum().sum()
        print(f"- % prediksi default ke 0.5: {count_05/total_preds:.2%}")
        
        low_std_cols = [c for c in conditions if df_res[c].std() < 0.05]
        if low_std_cols:
            print(f"- Apakah ada kondisi std < 0.05? YA: {', '.join(low_std_cols)}")
        else:
            print("- Apakah ada kondisi std < 0.05? TIDAK")
            
        mean_c3 = df_res['C3_hallu_free'].mean()
        mean_c0 = df_res['C0_pure'].mean()
        mean_c5 = df_res['C5_shuffled'].mean()
        print(f"- Mean C5 ({mean_c5:.4f}) vs Mean C3 ({mean_c3:.4f}) vs Mean C0 ({mean_c0:.4f})")
        if abs(mean_c5 - mean_c3) < abs(mean_c5 - mean_c0):
            print("  -> Mean C5 lebih mirip C3")
        else:
            print("  -> Mean C5 lebih mirip C0")
            
        print("\nBAGIAN 4 — 2 Contoh Kasus:")
        # 1. C4c >> C5
        df_res['diff_c4c_c5'] = df_res['C4c_hallu_mf'] - df_res['C5_shuffled']
        case1 = df_res.sort_values('diff_c4c_c5', ascending=False).iloc[0]
        print(f"\nContoh 1 (C4c >> C5 - konten penting):")
        print(f"SMILES: {case1['smiles']}")
        print(f"Label Benar: {case1['label']}")
        for c in conditions:
            print(f"{c}: {case1[c]:.4f}")
            
        # 2. C5 ≈ C4c
        df_res['abs_diff_c4c_c5'] = abs(df_res['C4c_hallu_mf'] - df_res['C5_shuffled'])
        # Sort by smallest difference, but pick one where C4c is high
        high_c4c_df = df_res[df_res['C4c_hallu_mf'] > 0.7]
        if len(high_c4c_df) > 0:
            case2 = high_c4c_df.sort_values('abs_diff_c4c_c5').iloc[0]
        else:
            case2 = df_res.sort_values('abs_diff_c4c_c5').iloc[0]
            
        print(f"\nContoh 2 (C5 ≈ C4c - gaya bahasa dominan):")
        print(f"SMILES: {case2['smiles']}")
        print(f"Label Benar: {case2['label']}")
        for c in conditions:
            print(f"{c}: {case2[c]:.4f}")

if __name__ == "__main__":
    runner = ExperimentRunner()
    
    # 1. Jalankan Validasi untuk BACE
    runner.run("bace_prepared.csv", "results_bace_checkpoint.json", task="bace", test_mode=True)
    
    # 2. Jalankan Full Run jika validasi selesai
    print("\n[SYSTEM] Validasi BACE selesai. Memulai FULL RUN...\n")
    runner.run("bace_prepared.csv", "results_bace_checkpoint.json", task="bace", test_mode=False)
