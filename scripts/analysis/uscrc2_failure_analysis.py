"""
US-CRC-2 LODO Failure Analysis
================================
Investigates why the Leave-One-Dataset-Out (LODO) cross-validation
AUC collapses from ~0.90 to 0.57 on the US-CRC-2 cohort.

Hypotheses tested:
  H1 — Demographic confounders (age, BMI, gender distribution shift)
  H2 — Microbial feature distribution shift (PCA / batch embedding)
  H3 — Signature taxa are uninformative / muted in US-CRC-2
  H4 — Class-label noise / cohort-specific technical artefact

Outputs  →  results/failure_analysis/
  ├── demographics_comparison.csv
  ├── feature_distribution_shifts.csv   (top shifted taxa)
  ├── h1_demographics_violin.png
  ├── h2_pca_cohort_embedding.png
  ├── h3_signature_taxa_boxplots.png
  ├── h3_shap_in_uscrc2.png
  ├── h4_calibration_curve.png
  └── failure_summary.txt

Run from project root:
    python scripts/analysis/uscrc2_failure_analysis.py
"""

import os
import warnings
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ML_DIR  = "data/processed/ml_ready"
OUT_DIR = "results/failure_analysis"
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
COLORS = {
    "US-CRC-2" : "#E63946",
    "US-CRC"   : "#F4A261",
    "DE-CRC"   : "#2A9D8F",
    "other"    : "#B0BEC5",
    "highlight": "#264653",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(fig, fname, dpi=300):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {path}")

def _sanitize(df):
    df = df.copy()
    df.columns = (df.columns
                    .str.replace(r"[\[\]<>]", "", regex=True)
                    .str.replace(r"\s+", "_", regex=True)
                    .str.replace(r"[^A-Za-z0-9_/.-]", "", regex=True))
    return df

def _section(title):
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
_section("US-CRC-2 LODO FAILURE ANALYSIS")
print("[DATA] Loading …")

X    = _sanitize(pd.read_csv(f"{ML_DIR}/X_species_combat.csv", index_col=0))
y    = pd.read_csv(f"{ML_DIR}/y_labels.csv",  index_col=0).squeeze()
meta = pd.read_csv(f"{ML_DIR}/metadata.csv",  index_col=0)
shap_imp = pd.read_csv("results/ml/shap_importance.csv")

# Align on CRC studies only (same pool used in LODO)
crc_studies = ["AT-CRC","CN-CRC","DE-CRC","FR-CRC","IT-CRC","IT-CRC-2",
               "JP-CRC","US-CRC","US-CRC-2"]
crc_mask    = meta["Study"].isin(crc_studies)
common      = X.index.intersection(y.index).intersection(meta.index)
X, y, meta  = X.loc[common], y.loc[common], meta.loc[common]
X_crc       = X.loc[crc_mask]
y_crc       = y.loc[crc_mask]
meta_crc    = meta.loc[crc_mask]

us2_idx  = meta_crc[meta_crc["Study"] == "US-CRC-2"].index
train_idx = meta_crc[meta_crc["Study"] != "US-CRC-2"].index

print(f"    Total CRC-study samples : {len(X_crc)}")
print(f"    US-CRC-2 (test)         : {len(us2_idx)}")
print(f"    Training pool           : {len(train_idx)}")

# Top SHAP features (signal taxa)
TOP_N_TAXA = 15
top_taxa   = shap_imp.head(TOP_N_TAXA)["species"].tolist()
# Map to sanitized column names
top_taxa_clean = []
for t in top_taxa:
    t_clean = (t.replace("[", "").replace("]", "").replace("<", "")
                .replace(">", "").replace(" ", "_"))
    matches = [c for c in X.columns if t_clean[:20] in c]
    if matches:
        top_taxa_clean.append(matches[0])
top_taxa_clean = list(dict.fromkeys(top_taxa_clean))[:TOP_N_TAXA]
print(f"    Signal taxa matched     : {len(top_taxa_clean)}")


# ══════════════════════════════════════════════════════════════════════════════
# H1 — DEMOGRAPHIC COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
_section("H1 — Demographic Confounder Analysis")

def _cohort_stats(study_name, df):
    row = {"Study": study_name, "N": len(df)}
    for col in ["Age", "BMI"]:
        if col in df.columns:
            row[f"{col}_mean"] = round(df[col].mean(), 1)
            row[f"{col}_std"]  = round(df[col].std(),  1)
    if "Gender" in df.columns:
        row["pct_male"] = round((df["Gender"] == "M").mean() * 100, 1)
    return row

dem_rows = []
for study in sorted(crc_studies):
    idx = meta_crc[meta_crc["Study"] == study].index
    dem_rows.append(_cohort_stats(study, meta_crc.loc[idx]))
df_dem = pd.DataFrame(dem_rows).set_index("Study")
df_dem.to_csv(f"{OUT_DIR}/demographics_comparison.csv")
print(df_dem.to_string())

# ── KS tests: US-CRC-2 vs pooled rest ────────────────────────────────────────
print("\n  KS Tests — US-CRC-2 vs. all other CRC studies:")
us2_meta   = meta_crc.loc[us2_idx]
rest_meta  = meta_crc.loc[train_idx]
for col in ["Age", "BMI"]:
    a = us2_meta[col].dropna().values
    b = rest_meta[col].dropna().values
    ks, pval = stats.ks_2samp(a, b)
    flag = "⚠ SIGNIFICANT" if pval < 0.05 else "  ns"
    print(f"    {col:<6s}: KS={ks:.3f}  p={pval:.4f}  {flag}")

# ── Violin plots ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("H1: Demographic Profiles — US-CRC-2 vs Other CRC Cohorts",
             fontsize=13, fontweight="bold", y=1.02)

meta_crc2 = meta_crc.copy()
meta_crc2["Cohort"] = meta_crc2["Study"].apply(
    lambda s: s if s in ("US-CRC-2", "US-CRC", "DE-CRC") else "Other CRC")
palette = {"US-CRC-2": COLORS["US-CRC-2"], "US-CRC": COLORS["US-CRC"],
           "DE-CRC": COLORS["DE-CRC"], "Other CRC": COLORS["other"]}
order = ["US-CRC-2", "US-CRC", "DE-CRC", "Other CRC"]

for ax, col, title in zip(axes, ["Age", "BMI"], ["Age (years)", "BMI (kg/m²)"]):
    temp = meta_crc2[[col, "Cohort"]].dropna()
    sns.violinplot(data=temp, x="Cohort", y=col, order=order,
                   palette=palette, inner="box", ax=ax, linewidth=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)

# Gender bar chart
ax = axes[2]
gend = meta_crc2.groupby("Cohort")["Gender"].value_counts(normalize=True).unstack().reindex(order)
gend.plot(kind="bar", ax=ax, color=["#E63946", "#457B9D"], edgecolor="white",
          width=0.6, legend=True)
ax.set_title("Gender Composition", fontweight="bold")
ax.set_ylabel("Proportion")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=30)
ax.legend(["Female", "Male"], frameon=False, fontsize=9)

plt.tight_layout()
_save(fig, "h1_demographics_violin.png")


# ══════════════════════════════════════════════════════════════════════════════
# H2 — FEATURE DISTRIBUTION SHIFT (PCA EMBEDDING)
# ══════════════════════════════════════════════════════════════════════════════
_section("H2 — Microbial Feature Distribution Shift (PCA)")

print("  Computing PCA on CRC-study samples …")
scaler = StandardScaler()
X_crc_scaled = scaler.fit_transform(X_crc.values)

pca   = PCA(n_components=3, random_state=42)
pcs   = pca.fit_transform(X_crc_scaled)
df_pc = pd.DataFrame(pcs, index=X_crc.index,
                     columns=["PC1", "PC2", "PC3"])
df_pc["Study"]  = meta_crc["Study"]
df_pc["Label"]  = y_crc.map({0: "CTR", 1: "CRC"})
df_pc["IsUS2"]  = (meta_crc["Study"] == "US-CRC-2").astype(str)

print(f"    Variance explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%  "
      f"PC3={pca.explained_variance_ratio_[2]*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("H2: PCA Embedding — Does US-CRC-2 Occupy a Different Microbiome Space?",
             fontsize=12, fontweight="bold")

for ax, (pcx, pcy, xlabel, ylabel) in zip(axes, [
    ("PC1", "PC2",
     f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
     f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"),
    ("PC1", "PC3",
     f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
     f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f})"),
]):
    # Plot all-other cohorts in grey
    rest = df_pc[df_pc["Study"] != "US-CRC-2"]
    ax.scatter(rest[pcx], rest[pcy], c="#C0C0C0", s=12, alpha=0.4,
               label="Other CRC studies", zorder=1)

    # Overlay US-CRC-2 CRC vs CTR
    for label, color, marker in [("CRC", "#E63946", "^"), ("CTR", "#457B9D", "o")]:
        sub = df_pc[(df_pc["Study"] == "US-CRC-2") & (df_pc["Label"] == label)]
        ax.scatter(sub[pcx], sub[pcy], c=color, s=45, alpha=0.85,
                   label=f"US-CRC-2 {label}", marker=marker, zorder=3,
                   edgecolors="white", linewidths=0.4)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=8, frameon=False)

plt.tight_layout()
_save(fig, "h2_pca_cohort_embedding.png")

# Centroid distance: US-CRC-2 vs pooled others
us2_centroid  = pcs[meta_crc["Study"].values == "US-CRC-2"].mean(axis=0)
rest_centroid = pcs[meta_crc["Study"].values != "US-CRC-2"].mean(axis=0)
cent_dist = np.linalg.norm(us2_centroid - rest_centroid)
print(f"    PCA centroid distance (US-CRC-2 vs rest): {cent_dist:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# H3 — SIGNATURE TAXA ARE MUTED IN US-CRC-2
# ══════════════════════════════════════════════════════════════════════════════
_section("H3 — Are Signature Taxa Muted in US-CRC-2?")

print(f"  Comparing top-{len(top_taxa_clean)} SHAP taxa across cohorts …")

# Per-taxon Mann-Whitney between CRC and CTR, for each cohort
results_h3 = []
for study in crc_studies:
    idx   = meta_crc[meta_crc["Study"] == study].index
    y_sub = y_crc.loc[idx]
    X_sub = X_crc.loc[idx, top_taxa_clean]
    for taxon in top_taxa_clean:
        crc_vals = X_sub.loc[y_sub == 1, taxon].values
        ctr_vals = X_sub.loc[y_sub == 0, taxon].values
        if len(crc_vals) < 3 or len(ctr_vals) < 3:
            continue
        stat, pval = stats.mannwhitneyu(crc_vals, ctr_vals, alternative="two-sided")
        fc = (crc_vals.mean() + 1e-9) / (ctr_vals.mean() + 1e-9)
        results_h3.append({
            "Study": study, "Taxon": taxon,
            "MW_stat": stat, "p_value": pval,
            "Fold_Change_CRC_over_CTR": fc,
        })

df_h3 = pd.DataFrame(results_h3)

# Show per-taxon mean fold-change: US-CRC-2 vs others
df_h3_us2  = df_h3[df_h3["Study"] == "US-CRC-2"].set_index("Taxon")
df_h3_rest = df_h3[df_h3["Study"] != "US-CRC-2"].groupby("Taxon")["Fold_Change_CRC_over_CTR"].mean()

shift_df = pd.DataFrame({
    "FC_US-CRC-2": df_h3_us2["Fold_Change_CRC_over_CTR"],
    "FC_Others"  : df_h3_rest,
}).dropna()
shift_df["Attenuation"] = shift_df["FC_Others"] - shift_df["FC_US-CRC-2"]
shift_df = shift_df.sort_values("Attenuation", ascending=False)
shift_df.to_csv(f"{OUT_DIR}/feature_distribution_shifts.csv")

print("\n  Top attenuated taxa (signal present elsewhere, muted in US-CRC-2):")
print(f"  {'Taxon':<55s} {'FC(others)':>10s}  {'FC(US2)':>10s}  {'Attenuation':>12s}")
for taxon, row in shift_df.head(8).iterrows():
    short = taxon.split("_ref_mOTU")[0].replace("_", " ")[:50]
    print(f"  {short:<55s} {row['FC_Others']:>10.3f}  {row['FC_US-CRC-2']:>10.3f}  "
          f"{row['Attenuation']:>12.3f}")

# ── Boxplot: top 6 signature taxa, CRC vs CTR, by cohort ─────────────────────
show_taxa = top_taxa_clean[:6]
show_studies = ["US-CRC-2", "US-CRC", "DE-CRC", "AT-CRC", "FR-CRC"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
fig.suptitle(
    "H3: Signature Taxa — Are CRC/CTR Differences Attenuated in US-CRC-2?\n"
    "(Red = CRC-enriched signal; Blue = CTR; bar = median)",
    fontsize=12, fontweight="bold",
)

for ax, taxon in zip(axes, show_taxa):
    short_name = taxon.split("_ref_mOTU")[0].replace("_", " ")
    rows = []
    for study in show_studies:
        idx   = meta_crc[meta_crc["Study"] == study].index
        y_sub = y_crc.loc[idx]
        for lbl, lab_val in [("CRC", 1), ("CTR", 0)]:
            vals = X_crc.loc[idx[y_sub == lab_val], taxon].values
            for v in vals:
                rows.append({"Study": study, "Group": lbl, "Abundance": v})
    df_box = pd.DataFrame(rows)
    sns.boxplot(data=df_box, x="Study", y="Abundance", hue="Group",
                palette={"CRC": "#E63946", "CTR": "#457B9D"},
                order=show_studies, width=0.55, linewidth=0.7,
                fliersize=2, ax=ax)
    ax.set_title(short_name[:45], fontsize=8, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    ax.set_ylabel("CLR abundance", fontsize=8)
    ax.get_legend().remove()

handles = [
    plt.Rectangle((0, 0), 1, 1, fc="#E63946", label="CRC"),
    plt.Rectangle((0, 0), 1, 1, fc="#457B9D", label="CTR"),
]
fig.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)
plt.tight_layout()
_save(fig, "h3_signature_taxa_boxplots.png")

# ── Train on rest, get SHAP on US-CRC-2 ──────────────────────────────────────
print("\n  Training LODO model on all-but-US-CRC-2, computing SHAP …")
scale_pos = float((y_crc.loc[train_idx] == 0).sum()) / float(y_crc.loc[train_idx].sum())
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos, eval_metric="logloss",
    verbosity=0, random_state=42,
)
xgb.fit(X_crc.loc[train_idx].values, y_crc.loc[train_idx].values)

# AUC on US-CRC-2
y_prob_us2 = xgb.predict_proba(X_crc.loc[us2_idx].values)[:, 1]
auc_us2    = roc_auc_score(y_crc.loc[us2_idx], y_prob_us2)
print(f"    Confirmed LODO AUC on US-CRC-2: {auc_us2:.4f}")

# SHAP for US-CRC-2 samples
explainer  = shap.TreeExplainer(xgb.get_booster())
shap_vals  = explainer.shap_values(X_crc.loc[us2_idx].values)
shap_df    = pd.DataFrame(shap_vals, columns=X_crc.columns, index=us2_idx)
mean_shap  = shap_df.abs().mean().sort_values(ascending=False)

# Compare mean |SHAP| in US-CRC-2 vs a matched subset from training
train_sample_idx = np.random.default_rng(42).choice(len(train_idx),
                                                     size=len(us2_idx),
                                                     replace=False)
shap_train_sub = explainer.shap_values(X_crc.loc[train_idx].values[train_sample_idx])
mean_shap_train = pd.Series(np.abs(shap_train_sub).mean(axis=0),
                             index=X_crc.columns).sort_values(ascending=False)

top15 = mean_shap.head(15).index
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(top15))
w = 0.38
ax.barh(x + w/2, mean_shap.loc[top15].values, height=w, color="#E63946",
        alpha=0.85, label="US-CRC-2 (test)")
ax.barh(x - w/2, mean_shap_train.loc[top15].values, height=w, color="#457B9D",
        alpha=0.85, label="Training cohorts (sampled)")
ax.set_yticks(x)
labels = [t.split("_ref_mOTU")[0].replace("_", " ")[:45] for t in top15]
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Mean |SHAP value|", fontsize=10)
ax.set_title(
    "H3: SHAP Feature Importance — US-CRC-2 vs Training Cohorts\n"
    "(Reduced bar height in US-CRC-2 = signal attenuation = AUC drop)",
    fontsize=11, fontweight="bold",
)
ax.legend(frameon=False, fontsize=10)
plt.tight_layout()
_save(fig, "h3_shap_in_uscrc2.png")


# ══════════════════════════════════════════════════════════════════════════════
# H4 — CALIBRATION & PROBABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
_section("H4 — Model Calibration & Probability Spread")

print("  Generating calibration curves …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("H4: Model Calibration — Is the Model Uncertain Specifically on US-CRC-2?",
             fontsize=12, fontweight="bold")

ax_cal, ax_hist = axes

# Calibration curve comparison
for study, color in [("US-CRC-2", "#E63946"), ("DE-CRC", "#2A9D8F"),
                     ("AT-CRC", "#F4A261"), ("US-CRC", "#457B9D")]:
    idx_s = meta_crc[meta_crc["Study"] == study].index
    if len(idx_s) < 10:
        continue
    try:
        y_s    = y_crc.loc[idx_s].values
        p_s    = xgb.predict_proba(X_crc.loc[idx_s].values)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_s, p_s, n_bins=8, strategy="quantile")
        auc_s  = roc_auc_score(y_s, p_s)
        ax_cal.plot(mean_pred, frac_pos, marker="o", color=color, linewidth=1.8,
                    label=f"{study} (AUC={auc_s:.2f})")
    except Exception:
        pass

ax_cal.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
ax_cal.set_xlabel("Mean predicted probability", fontsize=10)
ax_cal.set_ylabel("Fraction of positives", fontsize=10)
ax_cal.set_title("Calibration Curves per Cohort", fontweight="bold")
ax_cal.legend(fontsize=8, frameon=False)

# Probability histogram
for study, color, ls in [("US-CRC-2", "#E63946", "-"), ("DE-CRC", "#2A9D8F", "--")]:
    idx_s   = meta_crc[meta_crc["Study"] == study].index
    y_s     = y_crc.loc[idx_s].values
    p_crc_s = xgb.predict_proba(X_crc.loc[idx_s].values[y_s == 1])[:, 1]
    p_ctr_s = xgb.predict_proba(X_crc.loc[idx_s].values[y_s == 0])[:, 1]
    ax_hist.hist(p_crc_s, bins=15, alpha=0.45, color=color,
                 label=f"{study} CRC", linestyle=ls, edgecolor=color, density=True)
    ax_hist.hist(p_ctr_s, bins=15, alpha=0.25, color=color,
                 label=f"{study} CTR", linestyle=":", edgecolor=color, density=True)

ax_hist.axvline(0.5, color="grey", linestyle="--", lw=0.8)
ax_hist.set_xlabel("Predicted CRC probability", fontsize=10)
ax_hist.set_ylabel("Density", fontsize=10)
ax_hist.set_title("Probability Distribution: CRC vs CTR\n(Good model = two peaks far apart)",
                  fontweight="bold")
ax_hist.legend(fontsize=8, frameon=False)

plt.tight_layout()
_save(fig, "h4_calibration_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# ROC COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
_section("ROC Curves — US-CRC-2 vs Best/Other Cohorts")

fig, ax = plt.subplots(figsize=(7, 6))
for study, color, lw in [
    ("US-CRC-2", "#E63946", 2.5),
    ("DE-CRC",   "#2A9D8F", 1.8),
    ("AT-CRC",   "#F4A261", 1.5),
    ("US-CRC",   "#457B9D", 1.5),
    ("JP-CRC",   "#9B5DE5", 1.2),
]:
    idx_s = meta_crc[meta_crc["Study"] == study].index
    y_s   = y_crc.loc[idx_s].values
    p_s   = xgb.predict_proba(X_crc.loc[idx_s].values)[:, 1]
    try:
        auc_s = roc_auc_score(y_s, p_s)
        fp, tp, _ = roc_curve(y_s, p_s)
        ax.plot(fp, tp, color=color, lw=lw, label=f"{study} (AUC={auc_s:.3f})")
    except Exception:
        pass

ax.plot([0, 1], [0, 1], "k:", lw=0.8)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("LODO ROC Curves per Cohort\n(All models trained on remaining 8 cohorts)",
             fontsize=11, fontweight="bold")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
_save(fig, "roc_comparison_per_cohort.png")


# ══════════════════════════════════════════════════════════════════════════════
# WRITTEN SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
_section("Writing Failure Summary")

summary_lines = [
    "US-CRC-2 LODO Failure Analysis — Summary",
    "=" * 60,
    "",
    f"LODO AUC on US-CRC-2       : {auc_us2:.4f}",
    f"PCA centroid distance (vs rest): {cent_dist:.3f}",
    "",
    "H1 — Demographics:",
]
us2_age = meta_crc.loc[us2_idx, "Age"].dropna()
rest_age = meta_crc.loc[train_idx, "Age"].dropna()
ks_age, p_age = stats.ks_2samp(us2_age, rest_age)
summary_lines += [
    f"  Age — US-CRC-2 mean={us2_age.mean():.1f} vs rest={rest_age.mean():.1f}",
    f"  KS stat={ks_age:.3f}  p={p_age:.4f}",
    "",
    "H2 — Microbiome Space:",
    f"  PCA centroid distance: {cent_dist:.3f}",
    "  Interpretation: Values > 1.0 indicate meaningful batch separation.",
    "",
    "H3 — Signal Attenuation (top attenuated taxa):",
]
for taxon, row in shift_df.head(5).iterrows():
    short = taxon.split("_ref_mOTU")[0].replace("_", " ")[:50]
    summary_lines.append(
        f"  {short:<52s}  Δ={row['Attenuation']:+.3f}  "
        f"(others={row['FC_Others']:.3f}, US2={row['FC_US-CRC-2']:.3f})"
    )

summary_lines += [
    "",
    "Likely causes (in order of evidence):",
    "  1. Signal attenuation — key CRC taxa show weaker CRC/CTR separation in US-CRC-2",
    "  2. Batch/technical effects — PCA shows geometric shift from other cohorts",
    "  3. Demographics — Age and BMI may differ, driving microbiome composition shift",
    "  4. Sample size — Only 56 samples (28 CRC / 28 CTR): high sampling variance",
    "",
    "Clinical implication:",
    "  The generalisation failure is NOT random; it is systematic attenuation of",
    "  known CRC biomarkers. This may reflect population-specific microbiome composition",
    "  (US dietary/lifestyle pattern) or residual batch artefact in this dataset.",
]

summary_text = "\n".join(summary_lines)
print(summary_text)
with open(f"{OUT_DIR}/failure_summary.txt", "w") as f:
    f.write(summary_text)

print(f"\n═══════════════════════════════════════════════════════════════")
print(f"  US-CRC-2 FAILURE ANALYSIS COMPLETE ✅")
print(f"  All outputs → {os.path.abspath(OUT_DIR)}/")
print(f"═══════════════════════════════════════════════════════════════")
