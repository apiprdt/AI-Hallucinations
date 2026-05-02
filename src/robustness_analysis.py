import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def analyze_robustness(base_ckpt_path, rob_ckpt_path, task_name):
    if not os.path.exists(base_ckpt_path) or not os.path.exists(rob_ckpt_path):
        print(f"Skipping {task_name}: files missing.")
        return

    with open(base_ckpt_path, 'r') as f:
        base_ckpt = json.load(f)
    with open(rob_ckpt_path, 'r') as f:
        rob_ckpt = json.load(f)

    # Get labels
    labels = [base_ckpt["results"][str(i)]["label"] for i in range(base_ckpt["metadata"]["total_molecules"])]
    
    # Seeds for 8B
    seeds_8b = ["42", "s123", "s777"]
    conditions = ["C0", "C3"]
    
    report_data = []
    
    for cond in conditions:
        aucs_8b = []
        # Seed 42 is in the base_ckpt
        base_key = f"{cond}_pure" if cond == "C0" else f"{cond}_hallu_free"
        y_pred_42 = [base_ckpt["results"][str(i)][base_key] for i in range(len(labels))]
        aucs_8b.append(roc_auc_score(labels, y_pred_42))
        
        # Seeds 123, 777 are in rob_ckpt
        for s in ["s123", "s777"]:
            rob_key = f"{cond}_8b_{s}"
            y_pred = [rob_ckpt["results"][str(i)][rob_key] for i in range(len(labels))]
            aucs_8b.append(roc_auc_score(labels, y_pred))
            
        mean_8b = np.mean(aucs_8b)
        std_8b = np.std(aucs_8b)
        
        # 70B Result (Seed 42)
        rob_key_70b = f"{cond}_70b_s42"
        y_pred_70b = []
        labels_70b = []
        for i in range(len(labels)):
            if str(i) in rob_ckpt["results"] and rob_key_70b in rob_ckpt["results"][str(i)]:
                y_pred_70b.append(rob_ckpt["results"][str(i)][rob_key_70b])
                labels_70b.append(labels[i])
        
        if len(y_pred_70b) > 20: # Minimum sample to report
            auc_70b = roc_auc_score(labels_70b, y_pred_70b)
            val_70b = f"{auc_70b:.3f} (N={len(y_pred_70b)})"
        else:
            val_70b = "Pending"
        
        report_data.append({
            "Condition": cond,
            "Llama-8B (Mean±Std)": f"{mean_8b:.3f} ± {std_8b:.3f}",
            "Llama-70B": val_70b
        })

    print(f"\n[ROBUSTNESS REPORT: {task_name.upper()}]")
    df = pd.DataFrame(report_data)
    print(df.to_string(index=False))

if __name__ == "__main__":
    analyze_robustness("data/processed/results_bbbp_checkpoint.json", "data/processed/results_bbbp_robustness.json", "bbbp")
    analyze_robustness("data/processed/results_bace_checkpoint.json", "data/processed/results_bace_robustness.json", "bace")
