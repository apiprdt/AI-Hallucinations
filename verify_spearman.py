import json
import pandas as pd
from scipy.stats import spearmanr

with open("results_bbbp_checkpoint.json", "r") as f:
    ckpt = json.load(f)

results_list = [ckpt["results"][str(i)] for i in range(ckpt["metadata"]["total_molecules"])]
df = pd.DataFrame(results_list)

conditions = ['C0_pure', 'C1_factual', 'C2_gibberish', 'C3_hallu_free', 'C4a_hallu_sp', 'C4b_hallu_pi', 'C4c_hallu_mf', 'C5_shuffled']

results = []
for cond in conditions:
    r, p = spearmanr(df['label'], df[cond])
    results.append({'Condition': cond, 'Spearman_r': r, 'p-value': p})

res_df = pd.DataFrame(results).sort_values('Spearman_r', ascending=False)

print("="*60)
print("Spearman Correlation Analysis: Label vs All Predictions")
print("="*60)
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
