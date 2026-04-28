import argparse
import os
import pandas as pd
import json
import deepchem as dc
import importlib.util
import sys

# Dynamic import for file starting with a digit
spec = importlib.util.spec_from_file_location("runner", "04_experiment_runner.py")
runner_mod = importlib.util.module_from_spec(spec)
sys.modules["runner"] = runner_mod
spec.loader.exec_module(runner_mod)
ExperimentRunner = runner_mod.ExperimentRunner

import numpy as np # Added missing numpy import
from data_prep import generate_factual_description
from tqdm import tqdm

def run_pilot():
    print("\n" + "="*50)
    print("RUNNING BACE PILOT (Hold-out Test Set, ~152 molecules)")
    print("="*50)
    runner = ExperimentRunner()
    # Ini sudah dihandle oleh 04_experiment_runner.py jika dipanggil dengan parameter tertentu
    # Tapi kita buat wrapper di sini agar lebih terkontrol
    runner.run("bace_prepared.csv", "results_bace_pilot.json", task="bace")

def run_full_cv():
    print("\n" + "="*50)
    print("RUNNING BACE FULL (5-Fold CV, ~1,513 molecules)")
    print("="*50)
    
    # Load full dataset via DeepChem for scaffold splitting
    tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer='Raw', splitter='scaffold')
    # load_bace_classification returns (tasks, (train, valid, test), transformers)
    # Kita butuh gabungan semuanya untuk di-split ulang menjadi 5 fold
    # Atau kita gunakan KFoldSplitter
    
    raw_data = pd.read_csv("BACE.csv")
    # Note: BACE.csv needs 'mol' and 'Class' columns
    
    # Simple strategy: manually split the full dataset into 5 chunks of ~300 molecules
    # using scaffold split for each fold if possible, or just standard CV for now
    # to maintain consistency with drug discovery ML standards.
    
    full_dataset = dc.data.DiskDataset.from_numpy(
        X=np.zeros(len(raw_data)), # Placeholders since we only need SMILES/IDs
        y=raw_data['Class'].values,
        ids=raw_data['mol'].values
    )
    
    splitter = dc.splits.ScaffoldSplitter()
    folds = splitter.k_fold_split(full_dataset, k=5)
    
    runner = ExperimentRunner()
    
    for fold_idx, (train_ds, test_ds) in enumerate(folds):
        print(f"\n--- FOLD {fold_idx+1}/5 ---")
        smiles_list = test_ds.ids.tolist()
        labels = test_ds.y.flatten().tolist()
        
        # Prepare fold data
        fold_data = []
        for smiles, label in tqdm(zip(smiles_list, labels), total=len(smiles_list), desc="Prepping Fold"):
            factual = generate_factual_description(smiles)
            fold_data.append({
                "smiles": smiles,
                "label": label,
                "factual_desc": factual
            })
        
        fold_csv = f"bace_fold_{fold_idx}.csv"
        pd.DataFrame(fold_data).to_csv(fold_csv, index=False)
        
        # Run experiment for this fold
        runner.run(fold_csv, f"results_bace_fold_{fold_idx}.json", task="bace")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BACE Robustness Check (Pilot vs Full CV)")
    parser.add_argument("--phase", type=str, choices=["pilot", "full"], default="pilot",
                        help="Choose 'pilot' for test-set only or 'full' for 5-fold CV")
    
    args = parser.parse_args()
    
    if args.phase == "pilot":
        run_pilot()
    else:
        # Full CV takes a long time and many API calls
        run_full_cv()
