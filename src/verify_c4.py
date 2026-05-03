import os
import json
import time
from tqdm import tqdm
from taxonomy_classifier import HallucinationEngine

def main():
    print("Loading test set SMILES from checkpoint...")
    checkpoint_file = "data/processed/results_bbbp_checkpoint.json"
    with open(checkpoint_file, 'r') as f:
        ckpt = json.load(f)
    
    smiles_list = []
    for k, v in ckpt["results"].items():
        smiles_list.append(v["smiles"])
    
    print(f"Loaded {len(smiles_list)} SMILES.")
    
    engine = HallucinationEngine()
    
    results = {
        "C4a": {"total": 0, "SP": 0, "PI": 0, "MF": 0, "CC": 0},
        "C4b": {"total": 0, "SP": 0, "PI": 0, "MF": 0, "CC": 0},
        "C4c": {"total": 0, "SP": 0, "PI": 0, "MF": 0, "CC": 0}
    }
    
    # We will just verify a subset if we hit rate limits, but let's try to do all
    # Actually, to save time, let's sample 50 random molecules or do all 204 if fast enough.
    import random
    random.seed(42)
    sample_smiles = random.sample(smiles_list, 50)
    
    for i, smiles in enumerate(tqdm(sample_smiles)):
        # C4a - Structural -> Target SP
        text_a = engine.generate_hallucination(smiles, "structural")
        if text_a:
            cls_a = engine.classify_hallucination(text_a, smiles)
            dom_a = cls_a["dominant_type"]
            results["C4a"]["total"] += 1
            if dom_a in results["C4a"]: results["C4a"][dom_a] += 1
            
        # C4b - Property -> Target PI
        text_b = engine.generate_hallucination(smiles, "property")
        if text_b:
            cls_b = engine.classify_hallucination(text_b, smiles)
            dom_b = cls_b["dominant_type"]
            results["C4b"]["total"] += 1
            if dom_b in results["C4b"]: results["C4b"][dom_b] += 1
            
        # C4c - Mechanism -> Target MF
        text_c = engine.generate_hallucination(smiles, "mechanism")
        if text_c:
            cls_c = engine.classify_hallucination(text_c, smiles)
            dom_c = cls_c["dominant_type"]
            results["C4c"]["total"] += 1
            if dom_c in results["C4c"]: results["C4c"][dom_c] += 1
            
        # Optional save intermediate
        if (i+1) % 20 == 0:
            with open("c4_verify_temp.json", "w") as f:
                json.dump(results, f, indent=2)

    print("\n--- FINAL COMPLIANCE RATES ---")
    for cond, data in results.items():
        total = data["total"]
        if total == 0: continue
        
        target = "SP" if cond == "C4a" else ("PI" if cond == "C4b" else "MF")
        target_count = data[target]
        compliance = (target_count / total) * 100
        
        print(f"{cond} (Target: {target}): {target_count}/{total} ({compliance:.1f}%)")
        print(f"   Breakdown: SP={data['SP']}, PI={data['PI']}, MF={data['MF']}, CC={data['CC']}")
        
    with open("c4_verify_final.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
