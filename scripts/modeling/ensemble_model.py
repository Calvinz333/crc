"""
Ensemble Modeling Pipeline
============================
Trains and evaluates Random Forest, XGBoost, LightGBM, and a Soft-Voting Ensemble.
Uses Stratified 5-Fold Cross-Validation for robust performance estimation.

Outputs:
  - models/trained/ (individual and ensemble models saved)
  - results/ml/model_comparison.csv
  - figures/final/roc_curves.png
  - figures/final/model_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             roc_curve, classification_report)

# ── Paths ──────────────────────────────────────────────────────────────────
ML_DIR   = "data/processed/ml_ready"
RES_DIR  = "results/ml"
FIG_DIR  = "figures/final"
MOD_DIR  = "models/trained"
for d in [RES_DIR, FIG_DIR, MOD_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 150})

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  Ensemble Modeling Pipeline: RF + XGB + LGBM")
print("=" * 65)

X = pd.read_csv(f"{ML_DIR}/X_species_combat.csv", index_col=0)
y = pd.read_csv(f"{ML_DIR}/y_labels.csv", index_col=0).squeeze()

# XGBoost requires sanitized feature names without brackets/spaces
X.columns = (X.columns
               .str.replace(r'[\[\]<>]', '', regex=True)
               .str.replace(r'\s+', '_', regex=True)
               .str.replace(r'[^A-Za-z0-9_/.-]', '', regex=True))

print(f"  Samples  : {X.shape[0]} (CRC={y.sum()}, CTR={(y==0).sum()})")
print(f"  Features : {X.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════
# 2. DEFINE MODELS
# ══════════════════════════════════════════════════════════════════════════
scale_pos = (y == 0).sum() / y.sum()

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight='balanced', n_jobs=-1, random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, eval_metric='logloss', verbosity=0, random_state=42
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', verbose=-1, random_state=42
    )
}

# ══════════════════════════════════════════════════════════════════════════
# 3. 5-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════
print("\n[1] Running 5-Fold Stratified Cross-Validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store Out-Of-Fold (OOF) predictions for each model to build the ensemble safely
oof_preds = {name: np.zeros(X.shape[0]) for name in models}
results   = []

for name, model in models.items():
    print(f"    Evaluating {name}...")
    # cross_val_predict gives the OOF probability for each sample
    oof_proba = cross_val_predict(model, X.values, y.values, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    oof_preds[name] = oof_proba

# Soft Voting Ensemble OOF Probabilities
oof_preds['Ensemble'] = np.mean([oof_preds[m] for m in models], axis=0)

# ══════════════════════════════════════════════════════════════════════════
# 4. CALCULATE METRICS
# ══════════════════════════════════════════════════════════════════════════
print("\n[2] Calculating Performance Metrics...")

for name, proba in oof_preds.items():
    preds = (proba >= 0.5).astype(int)
    results.append({
        'Model'         : name,
        'Accuracy'      : accuracy_score(y, preds),
        'Precision'     : precision_score(y, preds),
        'Recall'        : recall_score(y, preds),
        'F1-Score'      : f1_score(y, preds),
        'ROC-AUC'       : roc_auc_score(y, proba),
        'Avg Precision' : average_precision_score(y, proba) # PR-AUC
    })

df_res = pd.DataFrame(results).round(4)
print("\n" + "=" * 80)
print(df_res.to_string(index=False))
print("=" * 80)

df_res.to_csv(f"{RES_DIR}/model_comparison.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] Generating visual reports...")

# ── 5.1 Bar Chart ──
fig, ax = plt.subplots(figsize=(10, 6))
df_melt = df_res.melt(id_vars='Model', var_name='Metric', value_name='Score')
sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model', palette='Set2', ax=ax)
ax.set_ylim(0.4, 1.0)
ax.set_title("Model Performance Comparison (5-Fold CV OOF)", fontsize=13, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_model_comparison_bar.png", bbox_inches='tight')
plt.close()

# ── 5.2 ROC Curves ──
plt.figure(figsize=(7, 6))
colors = {'Random Forest': '#457B9D', 'XGBoost': '#E63946', 'LightGBM': '#2A9D8F', 'Ensemble': '#111111'}

for name, proba in oof_preds.items():
    fpr, tpr, _ = roc_curve(y, proba)
    auc_val = df_res.loc[df_res['Model']==name, 'ROC-AUC'].values[0]
    
    if name == 'Ensemble':
        plt.plot(fpr, tpr, color=colors[name], lw=2.5, linestyle='--',
                 label=f'{name} (AUC = {auc_val:.3f})')
    else:
        plt.plot(fpr, tpr, color=colors[name], lw=1.5, alpha=0.8,
                 label=f'{name} (AUC = {auc_val:.3f})')

plt.plot([0, 1], [0, 1], ':', color='gray')
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('Out-of-Fold ROC Curves', fontsize=13, fontweight='bold')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_roc_curves.png", bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# 6. TRAIN FINAL MODELS & SAVE
# ══════════════════════════════════════════════════════════════════════════
print("\n[4] Training final models on full dataset & saving...")

for name, model in models.items():
    model.fit(X.values, y.values)
    joblib.dump(model, f"{MOD_DIR}/{name.lower().replace(' ', '_')}.pkl")

class SoftEnsemble:
    def __init__(self, models):
        self.models = models
    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

final_ensemble = SoftEnsemble(list(models.values()))
joblib.dump(final_ensemble, f"{MOD_DIR}/ensemble.pkl")

print(f"    Saved models to {MOD_DIR}/")
print("\n" + "=" * 65)
print("  ENSEMBLE MODELING COMPLETE ✅")
print("=" * 65)
