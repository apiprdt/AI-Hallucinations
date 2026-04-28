import json
import pandas as pd
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

with open("results_bbbp_checkpoint.json", "r") as f:
    ckpt = json.load(f)

results_list = [ckpt["results"][str(i)] for i in range(ckpt["metadata"]["total_molecules"])]
df_res = pd.DataFrame(results_list)
conditions = ["C0_pure","C1_factual","C2_gibberish","C3_hallu_free","C4a_hallu_sp","C4b_hallu_pi","C4c_hallu_mf","C5_shuffled"]

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
