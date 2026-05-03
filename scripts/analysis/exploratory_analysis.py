"""
Exploratory Analysis - Wirbel 2019 CRC Meta-analysis Data
Generates: Alpha diversity, Beta diversity (PCA), Heatmaps, Study distribution
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu, shapiro
from scipy.spatial.distance import braycurtis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = "data/processed/wirbel_2019"
FIG_DIR    = "figures/exploratory"
import os; os.makedirs(FIG_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────
COLORS = {"CRC": "#E63946", "CTR": "#457B9D"}
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("Loading data...")
meta = pd.read_csv(f"{DATA_DIR}/meta_all.tsv", sep='\t')
sp   = pd.read_csv(f"{DATA_DIR}/species_profiles.tsv", sep='\t', index_col=0)

# Filter CRC + CTR only
meta_crc = meta[meta['Group'].isin(['CRC', 'CTR'])].copy()
print(f"CRC samples : {(meta_crc['Group']=='CRC').sum()}")
print(f"CTR samples : {(meta_crc['Group']=='CTR').sum()}")

# Align species table to filtered samples
common_samples = [s for s in meta_crc['Sample_ID'] if s in sp.columns]
sp_filt = sp[common_samples].copy()
meta_crc = meta_crc[meta_crc['Sample_ID'].isin(common_samples)].set_index('Sample_ID')
sp_filt  = sp_filt[meta_crc.index]   # reorder columns to match meta rows

print(f"Matched samples : {sp_filt.shape[1]}")
print(f"Species features: {sp_filt.shape[0]}")

# Relative abundance (column-wise)
sp_rel = sp_filt.div(sp_filt.sum(axis=0), axis=1)

# Remove unclassified row (-1)
sp_rel = sp_rel[sp_rel.index != '-1']
sp_filt = sp_filt[sp_filt.index != '-1']

# ══════════════════════════════════════════════════════════════════════════
# 2. ALPHA DIVERSITY
# ══════════════════════════════════════════════════════════════════════════
print("\nCalculating alpha diversity...")

def shannon(col):
    p = col[col > 0] / col.sum()
    return -np.sum(p * np.log(p))

def richness(col):
    return (col > 0).sum()

alpha_df = pd.DataFrame({
    'Shannon' : sp_filt.apply(shannon),
    'Richness': sp_filt.apply(richness),
    'Group'   : meta_crc['Group'],
    'Study'   : meta_crc['Study'],
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Alpha Diversity: CRC vs Healthy Controls", fontsize=14, fontweight='bold', y=1.01)

for ax, metric in zip(axes, ['Shannon', 'Richness']):
    crc_vals = alpha_df.loc[alpha_df['Group']=='CRC', metric]
    ctr_vals = alpha_df.loc[alpha_df['Group']=='CTR', metric]
    stat, pval = mannwhitneyu(crc_vals, ctr_vals, alternative='two-sided')
    sig = "***" if pval<0.001 else ("**" if pval<0.01 else ("*" if pval<0.05 else "ns"))

    # Violin + strip
    for i, (grp, color) in enumerate([('CTR', COLORS['CTR']), ('CRC', COLORS['CRC'])]):
        vals = alpha_df.loc[alpha_df['Group']==grp, metric]
        parts = ax.violinplot(vals, positions=[i], widths=0.6, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color); pc.set_alpha(0.6)
        parts['cmedians'].set_color('white'); parts['cmedians'].set_linewidth(2)
        ax.scatter(np.random.normal(i, 0.06, len(vals)), vals,
                   color=color, alpha=0.4, s=8, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control\n(n={})'.format(len(ctr_vals)),
                        'CRC\n(n={})'.format(len(crc_vals))], fontsize=11)
    ax.set_ylabel(metric + ' Index', fontsize=11)
    ax.set_title(f"{metric}  |  p={pval:.2e}  {sig}", fontsize=11)

    # significance bar
    y_max = alpha_df[metric].max() * 1.05
    ax.plot([0, 0, 1, 1], [y_max*0.97, y_max, y_max, y_max*0.97], lw=1.2, c='k')
    ax.text(0.5, y_max*1.01, sig, ha='center', va='bottom', fontsize=13)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_alpha_diversity.png", bbox_inches='tight')
plt.close()
print("  Saved: 01_alpha_diversity.png")

# ══════════════════════════════════════════════════════════════════════════
# 3. PCA (Beta Diversity Proxy)
# ══════════════════════════════════════════════════════════════════════════
print("Running PCA...")

# Use log-transformed relative abundance; fill NaN with 0
sp_log = np.log1p(sp_rel * 1e6).T.fillna(0)   # samples x species
pca = PCA(n_components=3)
coords = pca.fit_transform(sp_log)
var_exp = pca.explained_variance_ratio_ * 100

pca_df = pd.DataFrame(coords, columns=['PC1','PC2','PC3'], index=meta_crc.index)
pca_df['Group'] = meta_crc['Group']
pca_df['Study'] = meta_crc['Study']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("PCA of Species Profiles (log-transformed relative abundance)",
             fontsize=13, fontweight='bold')

# Panel A: colour by Group
ax = axes[0]
for grp, color in COLORS.items():
    sub = pca_df[pca_df['Group']==grp]
    ax.scatter(sub['PC1'], sub['PC2'], c=color, alpha=0.55, s=20, label=grp, edgecolors='none')
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11)
ax.set_title("Coloured by Group", fontsize=11)
ax.legend(frameon=False, fontsize=10)

# Panel B: colour by Study
ax = axes[1]
studies = pca_df['Study'].unique()
study_colors = plt.cm.tab20(np.linspace(0, 1, len(studies)))
for study, color in zip(studies, study_colors):
    sub = pca_df[pca_df['Study']==study]
    ax.scatter(sub['PC1'], sub['PC2'], color=color, alpha=0.6, s=20,
               label=study, edgecolors='none')
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11)
ax.set_title("Coloured by Study (batch effect check)", fontsize=11)
ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper right')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_pca.png", bbox_inches='tight')
plt.close()
print("  Saved: 02_pca.png")

# ══════════════════════════════════════════════════════════════════════════
# 4. TOP SPECIES HEATMAP
# ══════════════════════════════════════════════════════════════════════════
print("Generating heatmap...")

# Select top 30 species by mean relative abundance
top30 = sp_rel.mean(axis=1).nlargest(30).index
sp_top = sp_rel.loc[top30]

# Shorten species names
def shorten(name):
    parts = name.split('[')[0].strip().split('/')
    return parts[0][:45]
sp_top.index = [shorten(n) for n in sp_top.index]

# Sort samples: CTR first, CRC second
order = meta_crc.sort_values('Group').index
sp_top = sp_top[order]
group_colors = meta_crc.loc[order, 'Group'].map(COLORS)

fig, ax = plt.subplots(figsize=(16, 9))
sns.heatmap(
    np.log1p(sp_top * 1e4),
    ax=ax,
    cmap='YlOrRd',
    xticklabels=False,
    yticklabels=True,
    cbar_kws={'label': 'log(rel. abund. × 1e4 + 1)', 'shrink': 0.6},
    linewidths=0,
)
ax.set_title("Top 30 Species — Relative Abundance Heatmap\n(samples sorted: Control | CRC)",
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Samples", fontsize=11)
ax.set_ylabel("")
ax.tick_params(axis='y', labelsize=8)

# Group colour bar on top
ax2 = ax.inset_axes([0, 1.01, 1, 0.025])
import matplotlib.colors as mcolors
color_rgb = np.array([mcolors.to_rgb(COLORS[g]) for g in meta_crc.loc[order, 'Group']])
ax2.imshow([color_rgb], aspect='auto', extent=[0, len(order), 0, 1])
ax2.set_axis_off()
for grp, color in COLORS.items():
    ax2.plot([], [], color=color, linewidth=6, label=grp)
ax2.legend(loc='upper right', bbox_to_anchor=(1.12, 2), frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_heatmap_top30.png", bbox_inches='tight')
plt.close()
print("  Saved: 03_heatmap_top30.png")

# ══════════════════════════════════════════════════════════════════════════
# 5. STUDY DISTRIBUTION BAR CHART
# ══════════════════════════════════════════════════════════════════════════
print("Generating study distribution...")

study_grp = meta_crc.groupby(['Study','Group']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 5))
study_grp.plot(kind='bar', ax=ax, color=[COLORS['CTR'], COLORS['CRC']],
               width=0.65, edgecolor='white')
ax.set_xlabel("Study", fontsize=11)
ax.set_ylabel("Number of Samples", fontsize=11)
ax.set_title("Sample Distribution Across Studies", fontsize=13, fontweight='bold')
ax.legend(title="Group", frameon=False)
ax.tick_params(axis='x', rotation=35)
for bar in ax.patches:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h+1, str(int(h)),
                ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_study_distribution.png", bbox_inches='tight')
plt.close()
print("  Saved: 04_study_distribution.png")

# ══════════════════════════════════════════════════════════════════════════
# 6. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXPLORATORY ANALYSIS SUMMARY")
print("="*55)
print(f"Samples analysed : {sp_filt.shape[1]} (CRC={( meta_crc['Group']=='CRC').sum()}, CTR={(meta_crc['Group']=='CTR').sum()})")
print(f"Species features : {sp_filt.shape[0]}")
print(f"Studies          : {meta_crc['Study'].nunique()}")
print(f"PCA var explained: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%, PC3={var_exp[2]:.1f}%")

crc_sh = alpha_df.loc[alpha_df['Group']=='CRC','Shannon']
ctr_sh = alpha_df.loc[alpha_df['Group']=='CTR','Shannon']
_, p_sh = mannwhitneyu(crc_sh, ctr_sh, alternative='two-sided')
print(f"Shannon (CRC mean): {crc_sh.mean():.2f} vs CTR: {ctr_sh.mean():.2f}  p={p_sh:.2e}")
print(f"\nFigures saved to: {FIG_DIR}/")
print("="*55)
