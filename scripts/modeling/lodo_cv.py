"""
LODO Cross-Validation + SHAP Analysis
=======================================
Leave-One-Dataset-Out (LODO) CV across 14 studies.
Models: Random Forest, XGBoost, LightGBM
Metrics: ROC-AUC per fold + mean ± std summary
Saves:  results/ml/, figures/ml/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb
import shap
import warnings, os, time
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ML_DIR  = "data/processed/ml_ready"
RES_DIR = "results/ml"
FIG_DIR = "figures/ml"
for d in [RES_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.spines.top': False,
                     'axes.spines.right': False,
                     'figure.dpi': 150})

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD & ALIGN DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  LODO Cross-Validation — CRC Microbiome Classifier")
print("=" * 65)
print("\n[1] Loading data...")

X    = pd.read_csv(f"{ML_DIR}/X_species_combat.csv", index_col=0)
y    = pd.read_csv(f"{ML_DIR}/y_labels.csv",          index_col=0).squeeze()
meta = pd.read_csv(f"{ML_DIR}/metadata.csv",          index_col=0)

# Align indices
common = X.index.intersection(y.index).intersection(meta.index)
X, y, meta = X.loc[common], y.loc[common], meta.loc[common]

print(f"    Samples  : {X.shape[0]}  (CRC={y.sum()}, CTR={(y==0).sum()})")
print(f"    Features : {X.shape[1]}")
print(f"    Studies  : {meta['Study'].nunique()}")

# Sanitize feature names (XGBoost rejects [ ] < > characters)
X.columns = (X.columns
               .str.replace(r'[\[\]<>]', '', regex=True)
               .str.replace(r'\s+', '_', regex=True)
               .str.replace(r'[^A-Za-z0-9_/.-]', '', regex=True))
print(f"    Feature names sanitized ✓")

# ══════════════════════════════════════════════════════════════════════════
# 2. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight='balanced', n_jobs=-1, random_state=42),

    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y==0).sum() / y.sum(),   # handle class imbalance
        eval_metric='logloss', verbosity=0, random_state=42),

    "LightGBM": lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', verbose=-1, random_state=42),
}

# ══════════════════════════════════════════════════════════════════════════
# 3. LODO CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════
print("\n[2] Running LODO Cross-Validation...")
print("-" * 65)

studies   = sorted(meta['Study'].unique())
results   = {name: [] for name in models}      # AUC per fold
fold_data  = []                                  # per-fold records

for study in studies:
    test_idx  = meta[meta['Study'] == study].index
    train_idx = meta[meta['Study'] != study].index

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # Skip fold if test set is missing one class
    if y_test.nunique() < 2:
        print(f"  [SKIP] {study:12s} — only one class in test set")
        continue

    n_crc = y_test.sum()
    n_ctr = (y_test == 0).sum()
    fold_row = {'Study': study, 'n_test': len(y_test),
                'n_CRC': n_crc, 'n_CTR': n_ctr}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        elapsed = time.time() - t0

        results[name].append(auc)
        fold_row[name] = auc
        print(f"  {study:15s}  {name:15s}  AUC={auc:.3f}  ({elapsed:.1f}s)")

    fold_data.append(fold_row)

# ══════════════════════════════════════════════════════════════════════════
# 4. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
df_folds = pd.DataFrame(fold_data)
df_folds.to_csv(f"{RES_DIR}/lodo_fold_results.csv", index=False)

print("\n" + "=" * 65)
print("  LODO SUMMARY — Mean ROC-AUC ± Std (across studies)")
print("=" * 65)

best_model_name = None
best_mean_auc   = 0.0

for name, aucs in results.items():
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    print(f"  {name:20s}:  {mean_auc:.3f} ± {std_auc:.3f}   (n={len(aucs)} folds)")
    if mean_auc > best_mean_auc:
        best_mean_auc   = mean_auc
        best_model_name = name

print(f"\n  🏆 Best model: {best_model_name}  (AUC={best_mean_auc:.3f})")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════
# 5. AUC PER-STUDY BAR CHART
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] Generating per-study AUC plot...")

model_cols   = list(models.keys())
auc_cols     = df_folds[model_cols]
study_labels = df_folds['Study']

x   = np.arange(len(study_labels))
w   = 0.25
colors = ['#457B9D', '#E63946', '#2A9D8F']

fig, ax = plt.subplots(figsize=(14, 5))
for i, (col, color) in enumerate(zip(model_cols, colors)):
    ax.bar(x + i*w, df_folds[col], width=w, color=color,
           alpha=0.85, label=col, edgecolor='white')

ax.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.7, label='Chance (0.5)')
ax.axhline(0.7, color='#F4A261', linestyle=':', lw=1.2, alpha=0.8, label='Good (0.7)')
ax.set_xticks(x + w)
ax.set_xticklabels(study_labels, rotation=35, ha='right', fontsize=9)
ax.set_ylabel("ROC-AUC", fontsize=11)
ax.set_title("LODO Cross-Validation — Per-Study ROC-AUC", fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=9)

# Annotate means
for i, (col, color) in enumerate(zip(model_cols, colors)):
    mean = df_folds[col].mean()
    ax.axhline(mean, color=color, linestyle='--', lw=1.2, alpha=0.6)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_lodo_auc_per_study.png", bbox_inches='tight')
plt.close()
print(f"    Saved: {FIG_DIR}/01_lodo_auc_per_study.png")

# ══════════════════════════════════════════════════════════════════════════
# 6. ROC CURVE — BEST MODEL (retrain on all data)
# ══════════════════════════════════════════════════════════════════════════
print("\n[4] Fitting best model on full data for SHAP + ROC curve...")

best_model = models[best_model_name]
best_model.fit(X, y)

# Aggregate ROC from LODO folds
fig, ax = plt.subplots(figsize=(6, 6))
all_fpr = np.linspace(0, 1, 200)
tprs    = []

for study in studies:
    test_idx  = meta[meta['Study'] == study].index
    train_idx = meta[meta['Study'] != study].index
    if y.loc[test_idx].nunique() < 2:
        continue
    best_model.fit(X.loc[train_idx], y.loc[train_idx])
    proba = best_model.predict_proba(X.loc[test_idx])[:, 1]
    fpr, tpr, _ = roc_curve(y.loc[test_idx], proba)
    tpr_interp = np.interp(all_fpr, fpr, tpr)
    tprs.append(tpr_interp)
    ax.plot(all_fpr, tpr_interp, lw=0.8, alpha=0.3, color='#457B9D')

mean_tpr = np.mean(tprs, axis=0)
std_tpr  = np.std(tprs, axis=0)
mean_auc = np.mean([roc_auc_score(y.loc[meta[meta['Study']==s].index],
                    best_model.fit(X.loc[meta[meta['Study']!=s].index], y.loc[meta[meta['Study']!=s].index])
                    .predict_proba(X.loc[meta[meta['Study']==s].index])[:, 1])
                    for s in studies if y.loc[meta[meta['Study']==s].index].nunique() == 2])

ax.plot(all_fpr, mean_tpr, color='#E63946', lw=2.5,
        label=f'Mean ROC (AUC = {best_mean_auc:.3f})')
ax.fill_between(all_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                alpha=0.15, color='#E63946', label='± 1 std')
ax.plot([0,1],[0,1],'--', color='gray', lw=1, label='Chance')
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title(f"LODO ROC Curve — {best_model_name}", fontsize=12, fontweight='bold')
ax.legend(frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_roc_curve.png", bbox_inches='tight')
plt.close()
print(f"    Saved: {FIG_DIR}/02_roc_curve.png")

# ══════════════════════════════════════════════════════════════════════════
# 7. SHAP ANALYSIS — GLOBAL FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════
print("\n[5] Running SHAP analysis (global feature importance)...")

# Retrain on full data for SHAP
best_model.fit(X, y)

# Use a background sample of 100 rows for SHAP (fixes XGBoost version mismatch)
background = shap.sample(X, 100, random_state=42)
explainer   = shap.TreeExplainer(best_model, background,
                                  feature_perturbation='interventional')
shap_values = explainer.shap_values(X)

# For binary classification, shap_values may be list[2] or 2d array
if isinstance(shap_values, list):
    sv = shap_values[1]   # class 1 (CRC) SHAP values
else:
    sv = shap_values

# Mean absolute SHAP per feature
mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
top20      = mean_shap.nlargest(20)

# ── Plot: Top-20 SHAP importance ──
fig, ax = plt.subplots(figsize=(9, 7))
colors_shap = ['#E63946' if v > 0 else '#457B9D'
               for v in pd.Series(sv.mean(axis=0), index=X.columns).loc[top20.index]]
bars = ax.barh(range(20), top20.values[::-1], color=colors_shap[::-1],
               alpha=0.85, edgecolor='white')
ax.set_yticks(range(20))
short_names = [n.split('[')[0].strip()[:50] for n in top20.index[::-1]]
ax.set_yticklabels(short_names, fontsize=9)
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title(f"Top 20 Species by SHAP Importance\n({best_model_name} — full dataset)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_shap_importance.png", bbox_inches='tight')
plt.close()
print(f"    Saved: {FIG_DIR}/03_shap_importance.png")

# ── SHAP Beeswarm (top 20 features) ──
top20_idx = mean_shap.nlargest(20).index
X_top20   = X[top20_idx]
sv_top20  = sv[:, X.columns.isin(top20_idx)]

# Rename for readability
rename = {c: c.split('[')[0].strip()[:45] for c in top20_idx}
X_plot = X_top20.rename(columns=rename)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv_top20, X_plot, plot_type='dot',
                  show=False, max_display=20, color_bar_label='Feature value')
plt.title(f"SHAP Beeswarm — {best_model_name}", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_shap_beeswarm.png", bbox_inches='tight')
plt.close()
print(f"    Saved: {FIG_DIR}/04_shap_beeswarm.png")

# Save SHAP summary table
shap_df = pd.DataFrame({
    'species'        : mean_shap.index,
    'mean_abs_shap'  : mean_shap.values,
    'mean_shap_dir'  : pd.Series(sv.mean(axis=0), index=X.columns).values
}).sort_values('mean_abs_shap', ascending=False)
shap_df.to_csv(f"{RES_DIR}/shap_importance.csv", index=False)
print(f"    Saved: {RES_DIR}/shap_importance.csv")

# ══════════════════════════════════════════════════════════════════════════
# 8. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY")
print("=" * 65)
for name, aucs in results.items():
    tag = "🏆" if name == best_model_name else "  "
    print(f"  {tag} {name:20s}: AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print()
print("  Top 5 most important species (by SHAP):")
for i, row in shap_df.head(5).iterrows():
    direction = "↑ CRC" if row['mean_shap_dir'] > 0 else "↑ CTR"
    print(f"    {i+1}. {row['species'].split('[')[0].strip()[:45]}  [{direction}]")
print()
print(f"  Figures → {FIG_DIR}/")
print(f"  Results → {RES_DIR}/")
print("=" * 65)
