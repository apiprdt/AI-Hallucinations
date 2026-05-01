import json
import pandas as pd
from scipy.stats import ttest_rel

with open("results_bbbp_checkpoint.json", "r") as f:
    ckpt = json.load(f)

results_list = [ckpt["results"][str(i)] for i in range(ckpt["metadata"]["total_molecules"])]
df = pd.DataFrame(results_list)

print("="*60)
print("Verifikasi 1: Distribusi Label, C3 vs C5 (20 Pertama)")
print("="*60)
print(df[['label', 'C3_hallu_free', 'C5_shuffled']].head(20).to_string())

print("\n" + "="*60)
print("Verifikasi 2: Paired t-test C3 vs C5")
print("="*60)
t, p = ttest_rel(df['C3_hallu_free'], df['C5_shuffled'])
print(f"t-statistic = {t:.3f}, p-value = {p:.4e}")

print("\n" + "="*60)
print("Verifikasi 3: Mekanisme Shuffle C5")
print("="*60)
print("C5 dibuat dengan menukar SELURUH blok teks halusinasi C3 antar molekul.")
print("Rumus yang digunakan dalam script: c5_source_index = (i + 1) % total_molecules")
print("Tidak ada pengacakan per-kata atau per-kalimat yang merusak tata bahasa.")
