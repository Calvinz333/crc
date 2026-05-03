"""
Differential Abundance Analysis
=================================
Identifies microbial species significantly enriched/depleted in CRC vs CTR.

Methods:
  - Wilcoxon rank-sum test (non-parametric, per feature)
  - Benjamini-Hochberg FDR correction
  - Cohen's d effect size
  - Log2 fold-change (median relative abundance)
  - Per-study sign-consistency check

Outputs:
  - results/differential/da_results.csv          (full ranked table)
  - results/differential/da_significant.csv      (FDR < 0.05)
  - figures/ml/05_volcano_plot.png
  - figures/ml/06_top_taxa_boxplots.png
  - figures/ml/07_study_consistency_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings, os
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ML_DIR  = "data/processed/ml_ready"
RES_DIR = "results/differential"
FIG_DIR = "figures/ml"
for d in [RES_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'figure.dpi'        : 150,
    'font.size'         : 10,
})

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  Differential Abundance Analysis — CRC vs CTR")
print("=" * 65)
print("\n[1] Loading data...")

X    = pd.read_csv(f"{ML_DIR}/X_species_combat.csv", index_col=0)
y    = pd.read_csv(f"{ML_DIR}/y_labels.csv",          index_col=0).squeeze()
meta = pd.read_csv(f"{ML_DIR}/metadata.csv",          index_col=0)

# Align
common = X.index.intersection(y.index).intersection(meta.index)
X, y, meta = X.loc[common], y.loc[common], meta.loc[common]

print(f"    Samples  : {X.shape[0]}  (CRC={y.sum()}, CTR={(y==0).sum()})")
print(f"    Features : {X.shape[1]}")
print(f"    Studies  : {meta['Study'].nunique()}")

# ── Data is already CLR-transformed + ComBat corrected ──
# We use X directly for testing and effect sizes
X_clr = X.copy()

crc_idx = y[y == 1].index
ctr_idx = y[y == 0].index


# ══════════════════════════════════════════════════════════════════════════
# 2. GLOBAL WILCOXON TEST + FDR
# ══════════════════════════════════════════════════════════════════════════
print("\n[2] Running Wilcoxon rank-sum tests on all features...")

records = []
for feat in X_clr.columns:
    crc_vals = X_clr.loc[crc_idx, feat].values
    ctr_vals = X_clr.loc[ctr_idx, feat].values

    stat, pval = mannwhitneyu(crc_vals, ctr_vals, alternative='two-sided')

    # Cohen's d on CLR values
    pooled_std = np.sqrt((crc_vals.std()**2 + ctr_vals.std()**2) / 2)
    cohens_d   = (crc_vals.mean() - ctr_vals.mean()) / (pooled_std + 1e-9)

    # Log2 fold-change equivalent (difference in means of log-transformed data)
    # Since CLR data is log-scaled, the difference of means is proportional to log2 fold change.
    med_crc = np.mean(X_clr.loc[crc_idx, feat])
    med_ctr = np.mean(X_clr.loc[ctr_idx, feat])
    log2fc  = (med_crc - med_ctr) / np.log(2)  # converting natural log diff to log2 diff

    # Prevalence (cannot be computed meaningfully on CLR data, setting to NaN)
    prev_crc = np.nan
    prev_ctr = np.nan

    records.append({
        'species'   : feat,
        'stat'      : stat,
        'pval'      : pval,
        'cohens_d'  : cohens_d,
        'log2fc'    : log2fc,
        'prev_CRC'  : prev_crc,
        'prev_CTR'  : prev_ctr,
        'mean_clr_CRC': crc_vals.mean(),
        'mean_clr_CTR': ctr_vals.mean(),
    })

df_res = pd.DataFrame(records)

# FDR correction (Benjamini-Hochberg)
reject, qvals, _, _ = multipletests(df_res['pval'], method='fdr_bh')
df_res['qval']       = qvals
df_res['significant'] = reject

df_res = df_res.sort_values('qval')
df_res.to_csv(f"{RES_DIR}/da_results.csv", index=False)

df_sig = df_res[df_res['significant']]
df_sig.to_csv(f"{RES_DIR}/da_significant.csv", index=False)

n_up   = (df_sig['log2fc'] > 0).sum()
n_down = (df_sig['log2fc'] < 0).sum()

print(f"    Significant (FDR < 0.05) : {len(df_sig)} / {len(df_res)} features")
print(f"      ↑ CRC-enriched : {n_up}")
print(f"      ↓ CTR-enriched : {n_down}")

print("\n    Top 10 CRC-enriched species:")
top_up = df_sig[df_sig['log2fc'] > 0].head(10)
for _, r in top_up.iterrows():
    name = r['species'].split('[')[0].strip()[:55]
    print(f"      {name:<55}  q={r['qval']:.2e}  d={r['cohens_d']:+.3f}  log2fc={r['log2fc']:+.2f}")

print("\n    Top 10 CTR-enriched species:")
top_dn = df_sig[df_sig['log2fc'] < 0].tail(10)
for _, r in top_dn.sort_values('log2fc').iterrows():
    name = r['species'].split('[')[0].strip()[:55]
    print(f"      {name:<55}  q={r['qval']:.2e}  d={r['cohens_d']:+.3f}  log2fc={r['log2fc']:+.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. VOLCANO PLOT
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] Generating volcano plot...")

fig, ax = plt.subplots(figsize=(10, 7))

# Colour coding
neg_log_q = -np.log10(df_res['qval'].clip(lower=1e-300))
colors_v = np.where(
    (df_res['qval'] < 0.05) & (df_res['log2fc'] > 0.5),  '#E63946',   # CRC-enriched
    np.where(
    (df_res['qval'] < 0.05) & (df_res['log2fc'] < -0.5), '#457B9D',   # CTR-enriched
    '#CCCCCC'))                                                          # NS

sizes = np.where(df_res['significant'], 30, 10)

sc = ax.scatter(df_res['log2fc'], neg_log_q,
                c=colors_v, s=sizes, alpha=0.75, linewidths=0)

# Thresholds
ax.axhline(-np.log10(0.05), color='gray', linestyle='--', lw=1, alpha=0.7)
ax.axvline( 0.5, color='gray', linestyle=':', lw=1, alpha=0.7)
ax.axvline(-0.5, color='gray', linestyle=':', lw=1, alpha=0.7)

# Label top hits
label_df = df_res[(df_res['qval'] < 0.05) & (np.abs(df_res['log2fc']) > 0.5)]
label_df = pd.concat([
    label_df[label_df['log2fc'] > 0].nlargest(8, 'log2fc'),
    label_df[label_df['log2fc'] < 0].nsmallest(6, 'log2fc'),
])
for _, row in label_df.iterrows():
    name  = row['species'].split('[')[0].strip()
    short = (name[:35] + '…') if len(name) > 35 else name
    xpos  = row['log2fc']
    ypos  = -np.log10(max(row['qval'], 1e-300))
    ax.annotate(short, (xpos, ypos),
                fontsize=6.5, ha='left' if xpos > 0 else 'right',
                xytext=(4 if xpos > 0 else -4, 2), textcoords='offset points',
                color='#333333')

from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor='#E63946', label=f'CRC-enriched (n={n_up}, FDR<0.05, |log2fc|>0.5)'),
    Patch(facecolor='#457B9D', label=f'CTR-enriched (n={n_down}, FDR<0.05, |log2fc|>0.5)'),
    Patch(facecolor='#CCCCCC', label='Not significant'),
]
ax.legend(handles=legend_els, frameon=False, fontsize=8, loc='upper left')
ax.set_xlabel("Log₂ Fold-Change  (CRC / CTR)", fontsize=11)
ax.set_ylabel("-log₁₀(FDR q-value)", fontsize=11)
ax.set_title("Differential Abundance — Volcano Plot\n(Wilcoxon + Benjamini-Hochberg FDR)",
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_volcano_plot.png", bbox_inches='tight')
plt.close()
print(f"    Saved: {FIG_DIR}/05_volcano_plot.png")

# ══════════════════════════════════════════════════════════════════════════
# 4. TOP-TAXA BOXPLOTS
# ══════════════════════════════════════════════════════════════════════════
print("\n[4] Generating top-taxa boxplots...")

# Pick top 6 CRC-enriched + top 4 CTR-enriched by Cohen's d
top_crc = (df_sig[df_sig['log2fc'] > 0]
           .nlargest(6, 'cohens_d')['species'].tolist())
top_ctr = (df_sig[df_sig['log2fc'] < 0]
           .nsmallest(4, 'cohens_d')['species'].tolist())
top_features = top_crc + top_ctr

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

label_map = {0: 'CTR', 1: 'CRC'}
palette   = {'CTR': '#457B9D', 'CRC': '#E63946'}

for ax, feat in zip(axes, top_features):
    plot_df = pd.DataFrame({
        'CLR abundance': X_clr[feat].values,
        'Group': [label_map[v] for v in y.values]
    })
    sns.boxplot(data=plot_df, x='Group', y='CLR abundance',
                palette=palette, width=0.5, linewidth=1.2,
                flierprops=dict(marker='.', markersize=3, alpha=0.5), ax=ax)
    sns.stripplot(data=plot_df, x='Group', y='CLR abundance',
                  palette=palette, size=2.5, alpha=0.35, jitter=True, ax=ax)

    # Stat annotation
    row = df_res[df_res['species'] == feat].iloc[0]
    stars = ('****' if row['qval'] < 1e-4 else
             '***'  if row['qval'] < 1e-3 else
             '**'   if row['qval'] < 0.01  else
             '*'    if row['qval'] < 0.05  else 'ns')
    ax.set_title(f"{feat.split('[')[0].strip()[:35]}\n"
                 f"q={row['qval']:.1e}  d={row['cohens_d']:+.2f}  {stars}",
                 fontsize=7, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("CLR abundance" if ax in axes[::5] else "", fontsize=8)

plt.suptitle("Top Differentially Abundant Species — CRC vs CTR",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_top_taxa_boxplots.png", bbox_inches='tight')
plt.close()
print(f"    Saved: {FIG_DIR}/06_top_taxa_boxplots.png")

# ══════════════════════════════════════════════════════════════════════════
# 5. PER-STUDY SIGN-CONSISTENCY HEATMAP
# ══════════════════════════════════════════════════════════════════════════
print("\n[5] Computing per-study consistency for top significant taxa...")

top_sig = df_sig.nlargest(30, 'cohens_d')['species'].tolist() + \
          df_sig.nsmallest(15, 'cohens_d')['species'].tolist()
top_sig = list(dict.fromkeys(top_sig))[:40]   # deduplicate, keep ≤40

studies = sorted(meta['Study'].unique())
consist = pd.DataFrame(index=top_sig, columns=studies, dtype=float)

for study in studies:
    s_idx = meta[meta['Study'] == study].index
    s_crc = s_idx.intersection(crc_idx)
    s_ctr = s_idx.intersection(ctr_idx)
    if len(s_crc) < 3 or len(s_ctr) < 3:
        consist[study] = np.nan
        continue
    for feat in top_sig:
        _, pv = mannwhitneyu(X_clr.loc[s_crc, feat],
                             X_clr.loc[s_ctr, feat],
                             alternative='two-sided')
        fc_dir = X_clr.loc[s_crc, feat].mean() - X_clr.loc[s_ctr, feat].mean()
        # encode: +1 = CRC-up sig, -1 = CTR-up sig, 0 = NS
        if   pv < 0.05 and fc_dir > 0: consist.loc[feat, study] =  1.0
        elif pv < 0.05 and fc_dir < 0: consist.loc[feat, study] = -1.0
        else:                           consist.loc[feat, study] =  0.0

# Short names for y-axis
short_names = [s.split('[')[0].strip()[:50] for s in top_sig]

fig, ax = plt.subplots(figsize=(14, 12))
cmap = plt.cm.RdBu   # red = CRC-enriched, blue = CTR-enriched
sns.heatmap(consist.astype(float),
            cmap=cmap, center=0, vmin=-1, vmax=1,
            linewidths=0.3, linecolor='#e0e0e0',
            yticklabels=short_names, xticklabels=studies,
            cbar_kws={'label': '−1: CTR↑ (p<0.05)  |  0: NS  |  +1: CRC↑ (p<0.05)',
                      'shrink': 0.6},
            ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
ax.set_title("Per-Study Sign Consistency — Top Differentially Abundant Taxa\n"
             "(colour = direction of significant difference, grey = not significant)",
             fontsize=11, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_study_consistency_heatmap.png", bbox_inches='tight')
plt.close()
print(f"    Saved: {FIG_DIR}/07_study_consistency_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════
# 6. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  DIFFERENTIAL ABUNDANCE SUMMARY")
print("=" * 65)
print(f"  Total features tested : {len(df_res)}")
print(f"  FDR < 0.05            : {len(df_sig)}")
print(f"    ↑ CRC-enriched      : {n_up}")
print(f"    ↓ CTR-enriched (protected): {n_down}")
print()
print("  Top 5 CRC-enriched (by Cohen's d):")
for i, (_, r) in enumerate(df_sig[df_sig['log2fc']>0].nlargest(5,'cohens_d').iterrows(), 1):
    print(f"    {i}. {r['species'].split('[')[0].strip()[:55]}")
    print(f"       q={r['qval']:.2e}  d={r['cohens_d']:+.3f}  log2fc={r['log2fc']:+.2f}")
print()
print("  Top 5 CTR-enriched (protective) by Cohen's d:")
for i, (_, r) in enumerate(df_sig[df_sig['log2fc']<0].nsmallest(5,'cohens_d').iterrows(), 1):
    print(f"    {i}. {r['species'].split('[')[0].strip()[:55]}")
    print(f"       q={r['qval']:.2e}  d={r['cohens_d']:+.3f}  log2fc={r['log2fc']:+.2f}")
print()
print(f"  Results → {RES_DIR}/")
print(f"  Figures → {FIG_DIR}/")
print("=" * 65)
