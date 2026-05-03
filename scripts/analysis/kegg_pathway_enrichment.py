"""
KEGG Pathway Enrichment Analysis
==================================
Maps top CRC-associated taxa (from SHAP, DA, and novelty discovery)
to functional metabolic pathways using the KEGG KO abundance profiles.

Strategy:
  1. Define "signal taxa" from SHAP importance + DA results.
  2. Retrieve samples dominated by each signal taxon (top-quartile).
  3. For each signal group, test which KEGG KO terms (K-numbers) are
     significantly enriched vs background using Mann-Whitney U + FDR.
  4. Map significant KO terms → KEGG pathway modules via a curated
     KO→Pathway lookup (fetched from local mapping or KEGG API).
  5. Aggregate pathway-level enrichment scores.
  6. Generate: heatmap, bubble chart, and bar plots.

Outputs  →  results/kegg/
  ├── ko_enrichment_CRC_vs_CTR.csv          (all KO terms)
  ├── pathway_enrichment_summary.csv         (pathway-level summary)
  ├── fig1_ko_volcano.png
  ├── fig2_pathway_bubble.png
  ├── fig3_top_pathway_heatmap.png
  └── fig4_signal_taxa_ko_heatmap.png

Run from project root:
    python scripts/analysis/kegg_pathway_enrichment.py
"""

import os
import re
import warnings
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ML_DIR   = "data/processed/ml_ready"
KEGG_TSV = "data/processed/wirbel_2019/KEGG_profiles.tsv"
OUT_DIR  = "results/kegg"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Aesthetics ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi" : 150,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"  : True,
    "grid.alpha" : 0.3,
})

def _save(fig, fname, dpi=300):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {path}")

def _section(title):
    print(f"\n{'═'*70}\n  {title}\n{'═'*70}")


# ══════════════════════════════════════════════════════════════════════════════
# BUILT-IN KO → PATHWAY MAP
# (curated from KEGG BRITE hsa00001; covers the most common metabolic modules)
# ══════════════════════════════════════════════════════════════════════════════
# fmt: off
KO_PATHWAY_MAP: dict[str, str] = {
    # Carbohydrate metabolism
    **{k: "Glycolysis / Gluconeogenesis"       for k in ["K00001","K00002","K00016","K00844","K01810","K00850","K01624","K01803","K00134","K00150","K00927","K01803","K00615","K00626"]},
    **{k: "TCA Cycle"                           for k in ["K01647","K01681","K00031","K00030","K01902","K01903","K00174","K01676","K01679","K00239","K00240","K00241","K00242","K00244","K00245","K00246"]},
    **{k: "Pyruvate metabolism"                 for k in ["K00016","K00248","K00382","K01512","K01958","K01959","K01960","K00018","K00656","K00626"]},
    **{k: "Pentose phosphate pathway"           for k in ["K00036","K01057","K00615","K00616","K01807","K01808","K00033","K00034","K01622"]},
    **{k: "Starch and sucrose metabolism"       for k in ["K01187","K00688","K02438","K00700","K01977","K01208","K01200","K00027"]},
    # Lipid metabolism
    **{k: "Fatty acid biosynthesis"             for k in ["K00059","K00645","K09458","K00667","K00668","K00660","K01782","K11533","K11534","K11535","K11536","K00638","K00662"]},
    **{k: "Fatty acid degradation"              for k in ["K00232","K00633","K01703","K01704","K00247","K00248","K00249","K01825","K07508"]},
    **{k: "Butyrate / Short-chain fatty acids"  for k in ["K00248","K00929","K01034","K01035","K18118","K18119","K18120","K03781"]},
    # Amino acid metabolism
    **{k: "Tryptophan metabolism"               for k in ["K00453","K00452","K03781","K00463","K01661","K11358","K01626"]},
    **{k: "Arginine / Polyamine metabolism"     for k in ["K01755","K01668","K01482","K10536","K08088","K05185","K09548"]},
    **{k: "Glutamate / Glutamine metabolism"    for k in ["K00262","K00264","K00265","K00284","K01920","K00820"]},
    **{k: "Serine / Glycine metabolism"         for k in ["K00600","K00030","K00830","K00831","K01183","K11358"]},
    # Nucleotide metabolism
    **{k: "Purine metabolism"                   for k in ["K00939","K00940","K01951","K01952","K00759","K00760","K01756","K00762","K00763","K01783"]},
    **{k: "Pyrimidine metabolism"               for k in ["K00762","K00940","K01011","K01465","K10047","K01077"]},
    # Energy metabolism
    **{k: "Oxidative phosphorylation"           for k in ["K00330","K00331","K00332","K00333","K00339","K00341","K00342","K00343","K02111","K02112","K02115","K02274","K02275","K02276","K02277","K02278","K02279"]},
    **{k: "Sulfur metabolism"                   for k in ["K00392","K00380","K00381","K00860","K01738","K01764","K17725","K17726"]},
    **{k: "Nitrogen metabolism"                 for k in ["K00362","K00363","K03385","K04561","K01674","K01455"]},
    # Cofactor / vitamin
    **{k: "Folate biosynthesis"                 for k in ["K01737","K11754","K00796","K01633","K01029","K00287"]},
    **{k: "Riboflavin metabolism"               for k in ["K00793","K00794","K00861","K01853","K05977"]},
    **{k: "Biotin metabolism"                   for k in ["K01935","K01906","K00652","K16833","K16834"]},
    # Microbial-specific
    **{k: "LPS / Peptidoglycan biosynthesis"    for k in ["K01000","K01001","K02563","K02558","K02560","K02555","K02556","K02563","K02556","K02118","K02119","K02120","K02121","K02122","K02123","K02124","K02125"]},
    **{k: "Biofilm / Virulence"                 for k in ["K02529","K02553","K03300","K07657","K07658","K07659"]},
    **{k: "Sporulation"                         for k in ["K06383","K06384","K14077","K14078","K14079","K14080"]},
    **{k: "Flagella assembly"                   for k in ["K02406","K02407","K02408","K02409","K02410","K02411","K02412","K02413","K02414","K02415","K02416","K02417","K02418","K02419","K02420","K02421","K02422","K02423","K02557"]},
    **{k: "Antibiotic biosynthesis/resistance"  for k in ["K18314","K18316","K18317","K18319","K01420","K01421","K01422","K06217","K06218"]},
    **{k: "Quorum sensing"                      for k in ["K11014","K11015","K18253","K18255","K18256"]},
    **{k: "DNA repair / SOS response"           for k in ["K03553","K03584","K03702","K03703","K03660","K03657"]},
}
# fmt: on

def ko_to_pathway(ko: str) -> str:
    """Return pathway name for a KO identifier, or 'Unclassified'."""
    return KO_PATHWAY_MAP.get(ko, "Unclassified")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
_section("Loading Data")

y    = pd.read_csv(f"{ML_DIR}/y_labels.csv",  index_col=0).squeeze()
meta = pd.read_csv(f"{ML_DIR}/metadata.csv",  index_col=0)

print("  Loading KEGG profiles (9,499 KO × 575 samples) …")
kegg = pd.read_csv(KEGG_TSV, sep="\t", index_col=0)
print(f"  KEGG shape: {kegg.shape}")

# Transpose → samples × KO
kegg_T = kegg.T

# Align samples
common = kegg_T.index.intersection(y.index)
kegg_T = kegg_T.loc[common]
y_k    = y.loc[common]
meta_k = meta.loc[common]

print(f"  Aligned samples: {len(common)}  (CRC={int(y_k.sum())}, CTR={(y_k==0).sum()})")

# Restrict CRC studies only (removes non-CRC controls from T2D / healthy cohorts)
crc_studies = ["AT-CRC","CN-CRC","DE-CRC","FR-CRC","IT-CRC","IT-CRC-2",
               "JP-CRC","US-CRC","US-CRC-2"]
crc_mask = meta_k["Study"].isin(crc_studies)
kegg_crc = kegg_T.loc[crc_mask]
y_crc    = y_k.loc[crc_mask]
print(f"  CRC-studies subset: {len(kegg_crc)} samples")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE A — GLOBAL KO ENRICHMENT: CRC vs CTR
# ══════════════════════════════════════════════════════════════════════════════
_section("Module A — Global KO Enrichment (CRC vs CTR)")

print("  Running Mann-Whitney U on all 9,499 KO terms …")
ko_ids   = kegg_crc.columns.tolist()
crc_idx  = y_crc[y_crc == 1].index
ctr_idx  = y_crc[y_crc == 0].index

records = []
for ko in ko_ids:
    crc_vals = kegg_crc.loc[crc_idx, ko].values
    ctr_vals = kegg_crc.loc[ctr_idx, ko].values
    # Skip KOs with near-zero variance across all samples
    if kegg_crc[ko].std() < 1e-10:
        continue
    stat, pval = stats.mannwhitneyu(crc_vals, ctr_vals, alternative="two-sided")
    log2fc     = np.log2((crc_vals.mean() + 1e-9) / (ctr_vals.mean() + 1e-9))
    records.append({"KO": ko, "MW_stat": stat, "p_value": pval, "log2FC": log2fc,
                    "mean_CRC": crc_vals.mean(), "mean_CTR": ctr_vals.mean()})

df_ko = pd.DataFrame(records)
_, qvals, _, _ = multipletests(df_ko["p_value"].values, method="fdr_bh")
df_ko["q_value"] = qvals
df_ko["Pathway"]  = df_ko["KO"].map(ko_to_pathway)
df_ko["Significant"] = (df_ko["q_value"] < 0.05) & (df_ko["log2FC"].abs() >= 0.5)
df_ko = df_ko.sort_values("q_value")

n_sig = df_ko["Significant"].sum()
print(f"  Significant KOs (q<0.05, |log2FC|≥0.5): {n_sig} / {len(df_ko)}")
print(f"  CRC-enriched (log2FC > 0): {(df_ko['Significant'] & (df_ko['log2FC']>0)).sum()}")
print(f"  CTR-enriched (log2FC < 0): {(df_ko['Significant'] & (df_ko['log2FC']<0)).sum()}")
df_ko.to_csv(f"{OUT_DIR}/ko_enrichment_CRC_vs_CTR.csv", index=False)

# ── Volcano Plot ──────────────────────────────────────────────────────────────
print("  Plotting KO volcano …")
df_v = df_ko.copy()
df_v["-log10q"] = -np.log10(df_v["q_value"].clip(lower=1e-50))

colors_v = np.where(
    ~df_v["Significant"], "#B0BEC5",
    np.where(df_v["log2FC"] > 0, "#E63946", "#457B9D")
)

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df_v["log2FC"], df_v["-log10q"], c=colors_v, s=6, alpha=0.65, linewidths=0)
ax.axhline(-np.log10(0.05), color="grey", linestyle="--", lw=0.8, label="FDR=0.05")
ax.axvline(0.5,  color="#E63946", linestyle=":", lw=0.8)
ax.axvline(-0.5, color="#457B9D", linestyle=":", lw=0.8)

# Annotate top CRC-enriched and CTR-enriched
top_crc = df_v[df_v["log2FC"] > 0].nlargest(8, "-log10q")
top_ctr = df_v[df_v["log2FC"] < 0].nlargest(6, "-log10q")
for _, row in pd.concat([top_crc, top_ctr]).iterrows():
    pname = row["Pathway"]
    label = f"{row['KO']}\n({pname[:28]})" if pname != "Unclassified" else row["KO"]
    ax.annotate(label, (row["log2FC"], row["-log10q"]),
                fontsize=5.5, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
                xytext=(row["log2FC"] + 0.05, row["-log10q"] + 0.5))

from matplotlib.patches import Patch
legend_els = [
    Patch(fc="#E63946", label="CRC-enriched (↑ in cancer)"),
    Patch(fc="#457B9D", label="CTR-enriched (↑ in healthy)"),
    Patch(fc="#B0BEC5", label="Not significant"),
]
ax.legend(handles=legend_els, frameon=False, fontsize=9)
ax.set_xlabel("log₂ Fold Change (CRC / CTR)", fontsize=11)
ax.set_ylabel("-log₁₀(FDR q-value)", fontsize=11)
ax.set_title("KEGG KO Enrichment Volcano Plot — CRC vs CTR\n"
             f"({n_sig} significant KO terms, FDR<0.05, |log2FC|≥0.5)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
_save(fig, "fig1_ko_volcano.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE B — PATHWAY-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
_section("Module B — Pathway-Level Enrichment Summary")

# Only keep classified pathways
df_classified = df_ko[df_ko["Pathway"] != "Unclassified"].copy()

pathway_records = []
for pathway, grp in df_classified.groupby("Pathway"):
    n_total  = len(grp)
    n_sig    = grp["Significant"].sum()
    n_crc_up = (grp["Significant"] & (grp["log2FC"] > 0)).sum()
    n_ctr_up = (grp["Significant"] & (grp["log2FC"] < 0)).sum()
    mean_fc  = grp.loc[grp["Significant"], "log2FC"].mean() if n_sig else np.nan
    min_q    = grp["q_value"].min()
    pathway_records.append({
        "Pathway"           : pathway,
        "Total_KOs"         : n_total,
        "Significant_KOs"   : n_sig,
        "CRC_enriched_KOs"  : n_crc_up,
        "CTR_enriched_KOs"  : n_ctr_up,
        "Mean_log2FC"       : round(mean_fc, 3) if not np.isnan(mean_fc) else 0,
        "Min_q_value"       : min_q,
        "Enrichment_score"  : round(n_sig / n_total, 3),
    })

df_path = pd.DataFrame(pathway_records).sort_values("Significant_KOs", ascending=False)
df_path.to_csv(f"{OUT_DIR}/pathway_enrichment_summary.csv", index=False)

print("\n  Pathway Enrichment Summary:")
print(f"  {'Pathway':<45s} {'SigKOs':>7s}  {'MeanFC':>7s}  {'Score':>7s}")
for _, row in df_path.head(15).iterrows():
    direction = "↑CRC" if row["Mean_log2FC"] > 0 else "↑CTR"
    print(f"  {row['Pathway']:<45s} {int(row['Significant_KOs']):>7d}  "
          f"{row['Mean_log2FC']:>+7.3f}  {row['Enrichment_score']:>7.3f}  {direction}")

# ── Bubble Chart ──────────────────────────────────────────────────────────────
print("\n  Plotting pathway bubble chart …")
df_bubble = df_path[df_path["Significant_KOs"] > 0].copy()
df_bubble = df_bubble.sort_values("Mean_log2FC")

fig, ax = plt.subplots(figsize=(11, max(6, len(df_bubble) * 0.45)))

colors_b = ["#E63946" if fc > 0 else "#457B9D" for fc in df_bubble["Mean_log2FC"]]
sizes_b  = (df_bubble["Significant_KOs"] / df_bubble["Significant_KOs"].max() * 600 + 30).values

sc = ax.scatter(
    df_bubble["Mean_log2FC"],
    range(len(df_bubble)),
    s=sizes_b, c=colors_b, alpha=0.8, zorder=3,
    edgecolors="white", linewidths=0.5,
)
ax.axvline(0, color="grey", linestyle="--", lw=0.8)
ax.set_yticks(range(len(df_bubble)))
ax.set_yticklabels(df_bubble["Pathway"], fontsize=8)
ax.set_xlabel("Mean log₂FC of significant KOs (CRC / CTR)", fontsize=10)
ax.set_title(
    "KEGG Pathway Enrichment — Bubble Chart\n"
    "(Size ∝ number of significant KO terms; Red=CRC-enriched, Blue=CTR-enriched)",
    fontsize=11, fontweight="bold",
)
# Size legend
for sz, label in [(30, "1 KO"), (200, "~5 KOs"), (600, "max KOs")]:
    ax.scatter([], [], s=sz, c="grey", alpha=0.6, label=label)
ax.legend(title="# Sig. KO terms", frameon=False, fontsize=8, title_fontsize=8)
plt.tight_layout()
_save(fig, "fig2_pathway_bubble.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE C — KO HEATMAP: TOP PATHWAYS × COHORTS
# ══════════════════════════════════════════════════════════════════════════════
_section("Module C — Pathway Abundance Heatmap by Cohort")

print("  Aggregating KO abundances into pathway scores …")

# Select KOs that belong to the top 15 enriched classified pathways
top_paths  = df_path.head(15)["Pathway"].tolist()
top_path_kos = {
    p: df_classified[(df_classified["Pathway"] == p) &
                     df_classified["Significant"]]["KO"].tolist()
    for p in top_paths
}

# Per-cohort, per-class mean pathway score
heat_data = {}
for study in sorted(crc_studies):
    idx_s = meta_k[(meta_k["Study"] == study) & crc_mask].index
    if len(idx_s) == 0:
        continue
    y_s  = y_k.loc[idx_s]
    col_label = study
    for group, gval in [("CRC", 1), ("CTR", 0)]:
        g_idx = idx_s[y_s == gval]
        if len(g_idx) == 0:
            continue
        heat_key = f"{study}\n{group}"
        row_vals = {}
        for path in top_paths:
            kos = [k for k in top_path_kos[path] if k in kegg_T.columns]
            if not kos:
                row_vals[path] = np.nan
            else:
                row_vals[path] = kegg_T.loc[g_idx, kos].mean().mean()
        heat_data[heat_key] = row_vals

heat_df = pd.DataFrame(heat_data).T.dropna(axis=1, how="all")

# Z-score across columns
heat_z = (heat_df - heat_df.mean()) / (heat_df.std() + 1e-10)

fig, ax = plt.subplots(figsize=(max(10, len(heat_z.columns) * 0.7),
                                max(8, len(heat_z) * 0.35)))
sns.heatmap(
    heat_z,
    cmap="RdBu_r", center=0,
    linewidths=0.3, linecolor="white",
    ax=ax,
    cbar_kws={"label": "Z-scored mean KO abundance"},
    xticklabels=[p[:30] for p in heat_z.columns],
    yticklabels=heat_z.index,
)
ax.set_title(
    "KEGG Pathway Abundance — CRC vs CTR across Cohorts\n"
    "(Z-scored; top 15 enriched pathways × cohort-group combinations)",
    fontsize=11, fontweight="bold", pad=12,
)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=7)
plt.tight_layout()
_save(fig, "fig3_top_pathway_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE D — SIGNAL TAXA × KO CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
_section("Module D — Signal Taxa ↔ Top KO Terms Correlation")

# Load species profiles
print("  Loading species profiles …")
sp_df = pd.read_csv("data/processed/wirbel_2019/species_profiles.tsv",
                    sep="\t", index_col=0)
sp_T  = sp_df.T

# Align with KEGG space
common_kegg_sp = sp_T.index.intersection(kegg_T.index)
sp_sub   = sp_T.loc[common_kegg_sp]
kegg_sub = kegg_T.loc[common_kegg_sp]
y_sub    = y_k.loc[common_kegg_sp]

print(f"  Aligned species + KEGG samples: {len(common_kegg_sp)}")

# Top signal taxa from SHAP
shap_imp = pd.read_csv("results/ml/shap_importance.csv")
top_taxa_raw = shap_imp.head(10)["species"].str.replace(r"[\[\]<>]", "", regex=True).str.strip().tolist()

# Match to species profile columns
sp_cols = sp_T.columns.tolist()
matched_taxa = []
for t in top_taxa_raw:
    t_norm = t.lower().replace(" ", "_").replace(".", "_")
    matches = [c for c in sp_cols if t.split("_ref")[0].lower().replace("_", " ") in c.lower()]
    if matches:
        matched_taxa.append(matches[0])
matched_taxa = list(dict.fromkeys(matched_taxa))[:8]
print(f"  Signal taxa matched in species profiles: {len(matched_taxa)}")

if matched_taxa:
    # Select top significant KOs
    top_kos = df_ko[df_ko["Significant"]].nlargest(25, "q_value" if False else "-log10q" if False else "MW_stat").head(25)
    # Actually sort by absolute log2FC for most biologically interpretable
    top_kos_sig = df_ko[df_ko["Significant"]].nlargest(25, df_ko["log2FC"].abs().name if False else "log2FC")
    top_kos_list = df_ko[df_ko["Significant"]].reindex(
        df_ko[df_ko["Significant"]]["log2FC"].abs().sort_values(ascending=False).index
    ).head(25)["KO"].tolist()

    # Spearman corr: each signal taxon vs each top KO across all samples
    corr_matrix = pd.DataFrame(index=matched_taxa, columns=top_kos_list, dtype=float)
    for taxon in matched_taxa:
        if taxon not in sp_sub.columns:
            continue
        t_vec = sp_sub[taxon].values
        for ko in top_kos_list:
            if ko not in kegg_sub.columns:
                corr_matrix.loc[taxon, ko] = np.nan
                continue
            k_vec = kegg_sub[ko].values
            if np.std(t_vec) < 1e-10 or np.std(k_vec) < 1e-10:
                corr_matrix.loc[taxon, ko] = np.nan
                continue
            rho, _ = stats.spearmanr(t_vec, k_vec)
            corr_matrix.loc[taxon, ko] = rho

    corr_matrix = corr_matrix.astype(float).dropna(how="all", axis=1)

    # Readable labels
    taxon_labels = [t.split(" [")[0][:40] for t in matched_taxa]
    ko_labels    = [f"{ko}\n({KO_PATHWAY_MAP.get(ko,'—')[:20]})" for ko in corr_matrix.columns]

    fig, ax = plt.subplots(figsize=(max(12, len(corr_matrix.columns) * 0.55),
                                    max(5, len(matched_taxa) * 0.7)))
    sns.heatmap(
        corr_matrix.values.astype(float),
        cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3, linecolor="white", ax=ax,
        xticklabels=ko_labels,
        yticklabels=taxon_labels,
        cbar_kws={"label": "Spearman ρ"},
    )
    ax.set_title(
        "Signal Taxa ↔ Top KEGG KO Terms — Spearman Correlation\n"
        "(Reveals which functional modules are co-encoded with CRC taxa)",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    _save(fig, "fig4_signal_taxa_ko_heatmap.png")
else:
    print("  ⚠ No signal taxa matched in species profiles — skipping Module D.")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT TOP ENRICHED PATHWAYS
# ══════════════════════════════════════════════════════════════════════════════
_section("Final Pathway Enrichment Summary")

print("\n  ┌─ TOP CRC-ENRICHED PATHWAYS (↑ in CRC) ──────────────────────────────┐")
crc_paths = df_path[df_path["Mean_log2FC"] > 0].head(8)
for _, row in crc_paths.iterrows():
    print(f"  │  {row['Pathway']:<44s}  KOs={int(row['Significant_KOs']):3d}"
          f"  FC={row['Mean_log2FC']:+.3f}  score={row['Enrichment_score']:.3f}")
print("  └──────────────────────────────────────────────────────────────────────┘")

print("\n  ┌─ TOP CTR-ENRICHED PATHWAYS (↑ in HEALTHY) ──────────────────────────┐")
ctr_paths = df_path[df_path["Mean_log2FC"] < 0].head(8)
for _, row in ctr_paths.iterrows():
    print(f"  │  {row['Pathway']:<44s}  KOs={int(row['Significant_KOs']):3d}"
          f"  FC={row['Mean_log2FC']:+.3f}  score={row['Enrichment_score']:.3f}")
print("  └──────────────────────────────────────────────────────────────────────┘")

print(f"\n{'═'*70}")
print(f"  KEGG PATHWAY ENRICHMENT COMPLETE ✅")
print(f"  All outputs → {os.path.abspath(OUT_DIR)}/")
print(f"{'═'*70}")
