import deepchem as dc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

def filter_nan(X, y):
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]

def run_rf_baseline(dataset_name):
    print(f"\n--- Running RF Baseline for {dataset_name} ---")
    
    # We use Morgan fingerprints (ECFP4) as it's the standard strong baseline
    featurizer = dc.feat.CircularFingerprint(size=2048, radius=2)
    
    if dataset_name.upper() == 'BBBP':
        tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer=featurizer, splitter='scaffold')
    elif dataset_name.upper() == 'BACE':
        tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer=featurizer, splitter='scaffold')
    else:
        raise ValueError("Unknown dataset")
        
    train_dataset, valid_dataset, test_dataset = datasets
    
    X_train = np.vstack([train_dataset.X, valid_dataset.X])
    y_train = np.concatenate([train_dataset.y.flatten(), valid_dataset.y.flatten()])
    
    X_test  = test_dataset.X
    y_test  = test_dataset.y.flatten()

    X_train, y_train = filter_nan(X_train, y_train)
    X_test, y_test = filter_nan(X_test, y_test)
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)} (should match LLM test set)")
    
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Random Forest (Morgan FP) ROC-AUC: {auc:.4f}")
    return auc

if __name__ == "__main__":
    run_rf_baseline("BBBP")
    run_rf_baseline("BACE")
