import json
import random
import os
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr
import time
import re

load_dotenv()

# --- 1. RENAME C2_gibberish -> C2_chempriming ---
checkpoint_file = "results_bbbp_checkpoint.json"
with open(checkpoint_file, 'r') as f:
    ckpt = json.load(f)

# Update metadata
if "C2_gibberish" in ckpt["metadata"]["conditions"]:
    idx = ckpt["metadata"]["conditions"].index("C2_gibberish")
    ckpt["metadata"]["conditions"][idx] = "C2_chempriming"
if "C2b_pure_gibberish" not in ckpt["metadata"]["conditions"]:
    ckpt["metadata"]["conditions"].append("C2b_pure_gibberish")

# Update results keys
for k, v in ckpt["results"].items():
    if "C2_gibberish" in v:
        v["C2_chempriming"] = v.pop("C2_gibberish")

# --- 2. GENERATOR C2b PURE GIBBERISH ---
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

# --- 3. AUDIT VOCABULARY C2b ---
print("============================================================")
print("AUDIT C2b: Teks Aktual yang Dikirim ke LLM (5 Contoh Pertama)")
print("============================================================")
samples = [generate_c2b_pure_gibberish(l) for l in [100, 200, 300, 400, 500]]
for i, s in enumerate(samples):
    print(f"Contoh {i+1}: {s}")

dilarang = {
    "molecule", "compound", "atom", "bond", "chemical", "structural",
    "physicochemical", "interaction", "receptor", "pathway", "inhibit",
    "biological", "molecular", "synthesis", "substrate", "protein", "enzyme", "ligand"
}

overlap_count = 0
total_words = 0
for s in samples:
    words = [w.strip('.,').lower() for w in s.split()]
    total_words += len(words)
    overlap_count += sum(1 for w in words if w in dilarang)

percent_overlap = (overlap_count / total_words) * 100
print(f"\nTotal kata dilarang di C2b: {overlap_count}")
print(f"Persentase Kosakata Terlarang di C2b: {percent_overlap:.2f}%")

if percent_overlap > 2.0:
    print("AUDIT FAILED! Revisi template.")
    exit(1)
print("AUDIT PASSED! Melanjutkan run LLM untuk C2b...\n")

# --- 4. JALANKAN C2b KE LLM (204 MOLEKUL) ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def predict_c2b(smiles: str, gibberish_text: str) -> float:
    prompt = f"""Given this molecule (SMILES: {smiles})
Additional information: {gibberish_text}

On a scale of 0 to 1, what is the probability that this molecule can penetrate the blood-brain barrier?
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

print("Running LLM prediction for C2b_pure_gibberish...")
for i in tqdm(range(ckpt["metadata"]["total_molecules"])):
    idx_str = str(i)
    res = ckpt["results"][idx_str]
    if "C2b_pure_gibberish" not in res:
        hallu_free = res.get("hallu_free_text", "")
        c2b_text = generate_c2b_pure_gibberish(len(hallu_free))
        res["C2b_pure_gibberish"] = predict_c2b(res["smiles"], c2b_text)
        
        if (i+1) % 25 == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump(ckpt, f, indent=2)

with open(checkpoint_file, 'w') as f:
    json.dump(ckpt, f, indent=2)
print("Finished LLM predictions for C2b.")

# --- 5. REPORT SPEARMAN ---
results_list = [ckpt["results"][str(i)] for i in range(ckpt["metadata"]["total_molecules"])]
df = pd.DataFrame(results_list)

conditions_to_report = ['C0_pure', 'C2b_pure_gibberish', 'C2_chempriming', 'C3_hallu_free']
report_data = []

for cond in conditions_to_report:
    r, p = spearmanr(df['label'], df[cond])
    report_data.append({'Condition': cond, 'Spearman_r': r, 'p-value': p})

res_df = pd.DataFrame(report_data).sort_values('Spearman_r', ascending=False)

print("\n" + "="*60)
print("Spearman Correlation Analysis: Isolating Length vs Semantics")
print("="*60)
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
