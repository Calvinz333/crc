"""
Preprocessing + ComBat Batch Correction Pipeline
==================================================
Steps:
  1. Load & filter → CRC + CTR only
  2. Remove low-prevalence species (< 10% of samples)
  3. CLR transformation  (microbiome-standard)
  4. ComBat batch correction (batch = Study)
  5. PCA before/after plots to verify correction
  6. Save ML-ready feature matrix + metadata
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from neuroCombat import neuroCombat
import warnings, os
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = "data/processed/wirbel_2019"
OUT_DIR    = "data/processed/ml_ready"
FIG_DIR    = "figures/exploratory"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS    = {"CRC": "#E63946", "CTR": "#457B9D"}
STUDY_CM  = plt.cm.tab20

plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.spines.top': False,
                     'axes.spines.right': False,
                     'figure.dpi': 150})

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD & FILTER
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Load & Filter")
print("=" * 60)

meta = pd.read_csv(f"{DATA_DIR}/meta_all.tsv", sep='\t')
sp   = pd.read_csv(f"{DATA_DIR}/species_profiles.tsv", sep='\t', index_col=0)

meta_crc = meta[meta['Group'].isin(['CRC', 'CTR'])].copy()
common   = [s for s in meta_crc['Sample_ID'] if s in sp.columns]
meta_crc = meta_crc[meta_crc['Sample_ID'].isin(common)].set_index('Sample_ID')
sp_filt  = sp[meta_crc.index]

# Remove unclassified row
sp_filt = sp_filt[sp_filt.index != '-1']

print(f"  Samples : {sp_filt.shape[1]}  (CRC={( meta_crc['Group']=='CRC').sum()}, CTR={(meta_crc['Group']=='CTR').sum()})")
print(f"  Species : {sp_filt.shape[0]}  (raw)")

# ══════════════════════════════════════════════════════════════════════════
# 2. PREVALENCE FILTER  (keep species present in ≥ 10% of samples)
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 2 — Prevalence Filtering (≥ 10% of samples)")
print("=" * 60)

prevalence   = (sp_filt > 0).sum(axis=1) / sp_filt.shape[1]
sp_prev      = sp_filt[prevalence >= 0.10]

print(f"  Species retained : {sp_prev.shape[0]}  (removed {sp_filt.shape[0] - sp_prev.shape[0]})")

# ══════════════════════════════════════════════════════════════════════════
# 3. CLR TRANSFORMATION
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 3 — CLR Transformation")
print("=" * 60)

def clr_transform(df):
    """Centered Log-Ratio transform. df: features × samples."""
    # Add pseudocount to avoid log(0)
    X     = df.values.astype(float) + 1e-6
    log_X = np.log(X)
    geom_mean = log_X.mean(axis=0)          # per-sample geometric mean
    clr   = log_X - geom_mean
    return pd.DataFrame(clr, index=df.index, columns=df.columns)

sp_clr = clr_transform(sp_prev)
print(f"  CLR matrix : {sp_clr.shape}  (features × samples)")
print(f"  Value range: [{sp_clr.values.min():.2f}, {sp_clr.values.max():.2f}]")

# ══════════════════════════════════════════════════════════════════════════
# 4. BATCH CORRECTION (ComBat)
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 4 — ComBat Batch Correction (batch = Study)")
print("=" * 60)

batch = meta_crc.loc[sp_clr.columns, 'Study']
print(f"  Batches: {batch.nunique()} studies")
print(f"  {batch.value_counts().to_dict()}")

# pyComBat expects: data = features × samples, batch = series/list
# neuroCombat expects: data = features × samples, batch = 1D array
dat      = sp_clr.values                          # numpy array features × samples
bat      = batch.values                           # numpy array of batch labels
combat_out = neuroCombat(dat=dat, covars=pd.DataFrame({'batch': bat}), batch_col='batch')
sp_combat  = pd.DataFrame(combat_out['data'], index=sp_clr.index, columns=sp_clr.columns)
print(f"\n  ComBat complete. Output shape: {sp_combat.shape}")

# ══════════════════════════════════════════════════════════════════════════
# 5. PCA BEFORE / AFTER COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 5 — PCA Before / After Comparison")
print("=" * 60)

studies = meta_crc['Study'].unique()
s_colors = {s: STUDY_CM(i / len(studies)) for i, s in enumerate(studies)}

def run_pca(mat, meta_df, title, ax_grp, ax_study):
    """Run PCA and plot on given axes."""
    X       = mat.T.values                            # samples × features
    X       = np.nan_to_num(X, nan=0.0)
    pca     = PCA(n_components=2)
    coords  = pca.fit_transform(X)
    ve      = pca.explained_variance_ratio_ * 100
    idx     = mat.columns

    for ax, col_key, cmap_dict in [
        (ax_grp,   'Group', COLORS),
        (ax_study, 'Study', s_colors),
    ]:
        for label, color in cmap_dict.items():
            mask = meta_df.loc[idx, col_key] == label
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, alpha=0.55, s=14, label=label, edgecolors='none')
        ax.set_xlabel(f"PC1 ({ve[0]:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({ve[1]:.1f}%)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')

fig = plt.figure(figsize=(16, 10))
fig.suptitle("PCA: Before vs After ComBat Batch Correction", fontsize=14, fontweight='bold', y=1.01)
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

ax_before_grp   = fig.add_subplot(gs[0, 0])
ax_before_study = fig.add_subplot(gs[0, 1])
ax_after_grp    = fig.add_subplot(gs[1, 0])
ax_after_study  = fig.add_subplot(gs[1, 1])

run_pca(sp_clr,    meta_crc, "BEFORE — CLR only",           ax_before_grp, ax_before_study)
run_pca(sp_combat, meta_crc, "AFTER  — CLR + ComBat",       ax_after_grp,  ax_after_study)

# Labels
for ax, title in [(ax_before_grp,   "Coloured by Group"),
                  (ax_before_study, "Coloured by Study"),
                  (ax_after_grp,    "Coloured by Group"),
                  (ax_after_study,  "Coloured by Study")]:
    ax.set_title(title, fontsize=9)

# Legends
ax_before_grp.legend(handles=[
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=8, label=l)
    for l, c in COLORS.items()], frameon=False, fontsize=8)

ax_before_study.legend(handles=[
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=s_colors[s], markersize=7, label=s)
    for s in studies], frameon=False, fontsize=6, ncol=2, loc='upper right')

# Row labels
for ax, lbl in [(ax_before_grp, "BEFORE"), (ax_after_grp, "AFTER")]:
    ax.annotate(lbl, xy=(-0.18, 0.5), xycoords='axes fraction',
                fontsize=12, fontweight='bold', rotation=90, va='center', color='#333')

plt.savefig(f"{FIG_DIR}/05_pca_batch_correction.png", bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR}/05_pca_batch_correction.png")

# ══════════════════════════════════════════════════════════════════════════
# 6. SAVE ML-READY DATA
# ══════════════════════════════════════════════════════════════════════════
print("\nSTEP 6 — Saving ML-Ready Data")
print("=" * 60)

# Transpose: samples × features
X_ml = sp_combat.T.copy()                             # samples × species
X_ml.index.name = 'Sample_ID'

# Binary label: CRC=1, CTR=0
y_ml = (meta_crc.loc[X_ml.index, 'Group'] == 'CRC').astype(int)
y_ml.name = 'label'

# Full metadata aligned
meta_ml = meta_crc.loc[X_ml.index, ['Group','Study','Age','Gender','BMI','Country']].copy()

# Save
X_ml.to_csv(f"{OUT_DIR}/X_species_combat.csv")
y_ml.to_csv(f"{OUT_DIR}/y_labels.csv")
meta_ml.to_csv(f"{OUT_DIR}/metadata.csv")

# Also save feature names
with open(f"{OUT_DIR}/feature_names.txt", 'w') as f:
    f.write('\n'.join(X_ml.columns.tolist()))

print(f"  X (features) : {X_ml.shape}  → {OUT_DIR}/X_species_combat.csv")
print(f"  y (labels)   : {y_ml.shape}  → {OUT_DIR}/y_labels.csv")
print(f"  metadata     : {meta_ml.shape} → {OUT_DIR}/metadata.csv")

print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)
print(f"  Raw species          : {sp_filt.shape[0]}")
print(f"  After prevalence (10%): {sp_prev.shape[0]}")
print(f"  Final features       : {X_ml.shape[1]}")
print(f"  Total samples        : {X_ml.shape[0]}")
print(f"  CRC                  : {y_ml.sum()}")
print(f"  CTR                  : {(y_ml==0).sum()}")
print(f"  Class ratio (CRC:CTR): 1 : {(y_ml==0).sum() / y_ml.sum():.1f}")
print(f"  Studies (batches)    : {meta_ml['Study'].nunique()}")
print("\n  ✅ Data ready for ML modeling!")
print("=" * 60)
