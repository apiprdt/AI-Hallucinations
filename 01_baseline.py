import deepchem as dc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ============================================
# LOAD DATASET
# ============================================
print("Loading BBBP dataset...")
featurizer = dc.feat.RDKitDescriptors()
tasks, datasets, transformers = dc.molnet.load_bbbp(
    featurizer=featurizer,
    splitter='scaffold'
)

train_dataset, valid_dataset, test_dataset = datasets

X_train = train_dataset.X
y_train = train_dataset.y.flatten()
X_test  = test_dataset.X
y_test  = test_dataset.y.flatten()

# RDKitDescriptors might contain NaNs for some molecules
def filter_nan(X, y):
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]

X_train, y_train = filter_nan(X_train, y_train)
X_test, y_test = filter_nan(X_test, y_test)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
print(f"Positive rate (train): {y_train.mean():.3f}")
print(f"Positive rate (test): {y_test.mean():.3f}")

# ============================================
# BASELINE: Random Forest (non-LLM)
# ============================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

auc_scores = []
for fold, (idx_train, idx_val) in enumerate(skf.split(X_all, y_all)):
    X_f_train, X_f_val = X_all[idx_train], X_all[idx_val]
    y_f_train, y_f_val = y_all[idx_train], y_all[idx_val]
    
    # Handle infinite or excessively large values
    X_f_train = np.nan_to_num(X_f_train, posinf=1e10, neginf=-1e10)
    X_f_val = np.nan_to_num(X_f_val, posinf=1e10, neginf=-1e10)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_f_train, y_f_train)
    
    y_pred = rf.predict_proba(X_f_val)[:, 1]
    auc = roc_auc_score(y_f_val, y_pred)
    auc_scores.append(auc)
    print(f"Fold {fold+1} ROC-AUC: {auc:.4f}")

print(f"\nBaseline RF ROC-AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
