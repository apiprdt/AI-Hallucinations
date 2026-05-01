import random

ADJECTIVES = ["moderate", "intermediate", "variable", 
              "standard", "conventional", "typical"]
NOUNS = ["hypothetical", "theoretical", "computational",
         "structural", "molecular", "chemical"]

def generate_structured_gibberish(target_length: int = 0) -> str:
    random.seed(42 + target_length)
    template = (
        f"This compound exhibits {random.choice(ADJECTIVES)} "
        f"structural coherence with {random.choice(NOUNS)} "
        f"interaction domains, demonstrating "
        f"{random.choice(ADJECTIVES)} physicochemical "
        f"characteristics under {random.choice(NOUNS)} conditions."
    )
    return template

print("="*60)
print("AUDIT C2: Teks Aktual yang Dikirim ke LLM (5 Contoh Pertama)")
print("="*60)

lengths = [100, 200, 300, 400, 500] # simulate different hallu_free lengths
samples = []
for l in lengths:
    samples.append(generate_structured_gibberish(l))

for i, s in enumerate(samples):
    print(f"Contoh {i+1}: {s}")

print("\n" + "="*60)
print("AUDIT KOSAKATA KIMIA/BIOLOGI DI C2")
print("="*60)

# Define explicit chem/bio vocabulary
chem_bio_vocab = {
    "compound", "structural", "interaction", "domains", 
    "physicochemical", "characteristics", "molecular", "chemical"
}

sample = samples[0].lower()
words = [w.strip('.,') for w in sample.split()]
total_words = len(words)

overlap = [w for w in words if w in chem_bio_vocab]
percent_overlap = (len(overlap) / total_words) * 100

print(f"Kosakata yang diuji: {chem_bio_vocab}")
print(f"Kata-kata yang terdeteksi di C2: {overlap}")
print(f"Total kata di C2: {total_words}")
print(f"Total kata kimia/biologi di C2: {len(overlap)}")
print(f"Persentase Kosakata Kimia/Biologi di C2: {percent_overlap:.2f}%")
