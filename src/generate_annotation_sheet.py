"""
Extract stratified hallucination samples for taxonomy validation.
Produces a CSV with balanced representation across categories and conditions.
"""
import json
import csv
import random
import os
from collections import Counter
from pathlib import Path

random.seed(42)

def load_checkpoint(path):
    with open(path) as f:
        ckpt = json.load(f)
    n = ckpt['metadata']['total_molecules']
    results = []
    for i in range(n):
        r = ckpt['results'][str(i)]
        results.append(r)
    return results

def extract_samples(results, dataset_name, n_target=50):
    """Extract stratified samples ensuring representation of all taxonomy categories."""
    
    # Group by heuristic label
    by_label = {'SP': [], 'PI': [], 'MF': [], 'CC': []}
    
    for i, r in enumerate(results):
        if 'hallu_free_text' not in r or 'taxonomy_free' not in r:
            continue
        
        tax = json.loads(r['taxonomy_free'])
        label = tax['dominant_type']
        text = r['hallu_free_text']
        
        if not text or text.strip() == '':
            continue
            
        by_label[label].append({
            'index': i,
            'dataset': dataset_name,
            'smiles': r['smiles'],
            'condition': 'C3_free',
            'generated_text': text,
            'heuristic_label': label,
        })
    
    # Print distribution
    print(f"\n{dataset_name} heuristic distribution:")
    for label, items in sorted(by_label.items()):
        print(f"  {label}: {len(items)}")
    
    # Stratified sampling: take ALL minority classes, then fill from CC
    samples = []
    
    # Take all SP (rare)
    samples.extend(by_label.get('SP', []))
    
    # Take all PI
    pi_items = by_label.get('PI', [])
    if len(pi_items) > 15:
        samples.extend(random.sample(pi_items, 15))
    else:
        samples.extend(pi_items)
    
    # Take all MF
    mf_items = by_label.get('MF', [])
    if len(mf_items) > 15:
        samples.extend(random.sample(mf_items, 15))
    else:
        samples.extend(mf_items)
    
    # Fill remaining from CC
    remaining = n_target - len(samples)
    cc_items = by_label.get('CC', [])
    if remaining > 0 and len(cc_items) > 0:
        samples.extend(random.sample(cc_items, min(remaining, len(cc_items))))
    
    random.shuffle(samples)
    return samples

def main():
    out_dir = Path('data/annotation')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    # BBBP
    bbbp = load_checkpoint('data/processed/results_bbbp_checkpoint.json')
    all_samples.extend(extract_samples(bbbp, 'BBBP', n_target=40))
    
    # BACE
    bace = load_checkpoint('data/processed/results_bace_checkpoint.json')
    all_samples.extend(extract_samples(bace, 'BACE', n_target=30))
    
    print(f"\nTotal samples extracted: {len(all_samples)}")
    print(f"Distribution:")
    label_counts = Counter(s['heuristic_label'] for s in all_samples)
    for k, v in sorted(label_counts.items()):
        print(f"  {k}: {v}")
    
    # Write CSV
    csv_path = out_dir / 'taxonomy_annotation_sheet.csv'
    fieldnames = [
        'sample_id', 'dataset', 'index', 'smiles', 'condition',
        'generated_text', 'heuristic_label',
        'ai_label', 'ai_confidence', 'ai_reasoning',
        'manual_review_needed', 'manual_label'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, s in enumerate(all_samples):
            row = {
                'sample_id': i,
                'dataset': s['dataset'],
                'index': s['index'],
                'smiles': s['smiles'],
                'condition': s['condition'],
                'generated_text': s['generated_text'],
                'heuristic_label': s['heuristic_label'],
                'ai_label': '',
                'ai_confidence': '',
                'ai_reasoning': '',
                'manual_review_needed': '',
                'manual_label': '',
            }
            writer.writerow(row)
    
    print(f"\nSaved annotation sheet to: {csv_path}")
    print(f"Total rows: {len(all_samples)}")

if __name__ == '__main__':
    main()
