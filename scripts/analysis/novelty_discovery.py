"""
Novelty Discovery Pipeline — CRC Microbiome Meta-Analysis
==========================================================
Four independent biological-insight modules that mine the trained
XGBoost / LightGBM models and batch-corrected data for novel mechanisms.

Modules
-------
  1. Microbial Epistasis          — SHAP Interaction Values → top synergistic pairs
  2. Probiotic Antagonists        — Spearman correlations in healthy gut
  3. Sub-Phenotyping              — Age-stratified biomarkers (Early vs Late Onset)
  4. Non-Monotonic Pathogenicity  — SHAP Dependence Plots for top features

Outputs  →  results/novelty/
  ├── module1_shap_interactions_top5.csv
  ├── module1_shap_interaction_heatmap.png
  ├── module2_probiotic_antagonists.csv
  ├── module2_correlation_heatmap.png
  ├── module3_early_onset_importances.csv
  ├── module3_late_onset_importances.csv
  ├── module3_differential_biomarkers.csv
  ├── module3_feature_importance_comparison.png
  ├── module4_shap_dependence_<feature>.png  (×3)
  └── module4_pdp_<feature>.png              (×3)

Usage
-----
    python scripts/analysis/novelty_discovery.py

Run from the project root (crc_microbiome_project/).
"""

import os
import sys
import warnings
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

warnings.filterwarnings("ignore")

# ── Aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#264653"]

# ── Config ───────────────────────────────────────────────────────────────────
# Cap for SHAP interaction computation; set to None to use all samples.
SHAP_INTERACTION_MAX_SAMPLES = 500
SHAP_DEPENDENCE_MAX_SAMPLES  = 1000   # for Module 4 dependence plots

ML_DIR  = "data/processed/ml_ready"
OUT_DIR = "results/novelty"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _sanitize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same column-name sanitisation used in ensemble_model.py."""
    df = df.copy()
    df.columns = (
        df.columns
          .str.replace(r"[\[\]<>]", "", regex=True)
          .str.replace(r"\s+", "_", regex=True)
          .str.replace(r"[^A-Za-z0-9_/.-]", "", regex=True)
    )
    return df


def _short_name(col: str, maxlen: int = 40) -> str:
    """Return a readable truncated feature label."""
    # Strip mOTU suffixes and keep genus-species
    parts = col.split("_ref_mOTU")[0].replace("_", " ")
    return parts if len(parts) <= maxlen else parts[:maxlen - 1] + "…"


def _save_fig(fig: plt.Figure, fname: str, dpi: int = 300) -> None:
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved → {path}")


def _section(title: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    print("\n[DATA] Loading batch-corrected features, labels, and metadata …")
    X    = _sanitize_cols(pd.read_csv(f"{ML_DIR}/X_species_combat.csv", index_col=0))
    y    = pd.read_csv(f"{ML_DIR}/y_labels.csv", index_col=0).squeeze()
    meta = pd.read_csv(f"{ML_DIR}/metadata.csv", index_col=0)

    # Align on common samples (safeguard)
    common = X.index.intersection(y.index).intersection(meta.index)
    X, y, meta = X.loc[common], y.loc[common], meta.loc[common]

    print(f"    Samples : {X.shape[0]}  (CRC={int(y.sum())}, CTR={int((y==0).sum())})")
    print(f"    Features: {X.shape[1]}")
    print(f"    Meta Age NaNs: {meta['Age'].isna().sum()}")
    return X, y, meta


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — MICROBIAL EPISTASIS (SHAP INTERACTION VALUES)
# ══════════════════════════════════════════════════════════════════════════════
def module1_epistasis(X: pd.DataFrame, y: pd.Series) -> None:
    _section("MODULE 1 — Microbial Epistasis (SHAP Interaction Values)")

    # ── Train XGBoost on full dataset ────────────────────────────────────────
    print("  [1.1] Training XGBoost classifier on full dataset …")
    scale_pos = float((y == 0).sum()) / float(y.sum())
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        verbosity=0, random_state=42,
    )
    xgb.fit(X.values, y.values)
    print("      XGBoost trained.")

    # ── SHAP Interaction Values ───────────────────────────────────────────────
    print("  [1.2] Computing SHAP Interaction Values …")
    if SHAP_INTERACTION_MAX_SAMPLES and X.shape[0] > SHAP_INTERACTION_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=SHAP_INTERACTION_MAX_SAMPLES, replace=False)
        X_shap = X.iloc[idx]
        print(f"      Using a random subsample of {SHAP_INTERACTION_MAX_SAMPLES} samples "
              f"(full set = {X.shape[0]}) for speed.")
    else:
        X_shap = X

    explainer   = shap.TreeExplainer(xgb.get_booster())
    shap_inter  = explainer.shap_interaction_values(X_shap.values)
    # shap_inter shape: (n_samples, n_features, n_features)
    # Mean absolute interaction strength across samples
    mean_inter  = np.abs(shap_inter).mean(axis=0)          # (n_feat, n_feat)
    np.fill_diagonal(mean_inter, 0)                         # zero-out self-interactions

    # ── Extract top synergistic pairs ────────────────────────────────────────
    print("  [1.3] Extracting top synergistic pairs …")
    feat_names = X.columns.tolist()
    n_feat     = len(feat_names)

    records = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            records.append({
                "Feature_A"         : feat_names[i],
                "Feature_B"         : feat_names[j],
                "Mean_Abs_Interact" : mean_inter[i, j],
            })

    df_inter = (
        pd.DataFrame(records)
          .sort_values("Mean_Abs_Interact", ascending=False)
          .reset_index(drop=True)
    )

    top5 = df_inter.head(5)
    print("\n  ┌─ TOP 5 SYNERGISTIC BACTERIAL PAIRS ─────────────────────────────────┐")
    for rank, row in top5.iterrows():
        print(f"  │ #{rank+1:02d}  {_short_name(row.Feature_A):<38s}"
              f" ✕  {_short_name(row.Feature_B):<38s}"
              f"  [{row.Mean_Abs_Interact:.5f}]")
    print("  └──────────────────────────────────────────────────────────────────────┘")

    top5.to_csv(f"{OUT_DIR}/module1_shap_interactions_top5.csv", index=False)

    # ── Heatmap: top-30 features interaction sub-matrix ──────────────────────
    print("  [1.4] Plotting interaction heatmap …")

    # Pick top 25 features by their total interaction mass
    top_feat_idx = np.argsort(mean_inter.sum(axis=1))[::-1][:25]
    sub_mat  = mean_inter[np.ix_(top_feat_idx, top_feat_idx)]
    sub_labs = [_short_name(feat_names[i], maxlen=30) for i in top_feat_idx]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        sub_mat, xticklabels=sub_labs, yticklabels=sub_labs,
        cmap="YlOrRd", ax=ax, linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Mean |SHAP Interaction|"},
    )
    ax.set_title(
        "SHAP Pairwise Interaction Values — Top 25 Features\n"
        "(Off-diagonal = synergistic contribution to CRC probability)",
        fontsize=12, fontweight="bold", pad=12,
    )
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    _save_fig(fig, "module1_shap_interaction_heatmap.png")

    # Store XGBoost model for reuse in Module 4
    return xgb


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — PROBIOTIC ANTAGONISTS (COMPETITIVE EXCLUSION)
# ══════════════════════════════════════════════════════════════════════════════
def module2_probiotics(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> None:
    _section("MODULE 2 — Probiotic Antagonists (Competitive Exclusion)")

    # ── Filter to Healthy Controls only ──────────────────────────────────────
    ctr_mask = (y == 0)
    X_ctr    = X.loc[ctr_mask]
    print(f"  Healthy Control samples: {X_ctr.shape[0]}")

    # ── Identify known CRC pathogens in the feature set ──────────────────────
    pathogen_keywords = ["Fusobacterium", "Parvimonas", "Peptostreptococcus"]
    pathogen_cols     = [
        c for c in X.columns
        if any(kw in c for kw in pathogen_keywords)
    ]

    if not pathogen_cols:
        print("  ⚠ WARNING: No pathogen columns found. Skipping Module 2.")
        return

    print(f"  Known pathogens detected ({len(pathogen_cols)}):")
    for p in pathogen_cols:
        print(f"    • {_short_name(p)}")

    # ── Spearman correlations: pathogens vs. all others ───────────────────────
    print("  [2.1] Computing Spearman correlations in healthy gut …")
    other_cols = [c for c in X.columns if c not in pathogen_cols]

    rho_records = []
    for pathogen in pathogen_cols:
        path_vec = X_ctr[pathogen].values
        for other in other_cols:
            other_vec = X_ctr[other].values
            # Skip if no variance
            if np.std(path_vec) == 0 or np.std(other_vec) == 0:
                continue
            rho, pval = stats.spearmanr(path_vec, other_vec)
            rho_records.append({
                "Pathogen"    : pathogen,
                "Species"     : other,
                "Spearman_rho": rho,
                "p_value"     : pval,
            })

    df_corr = (
        pd.DataFrame(rho_records)
          .sort_values("Spearman_rho", ascending=True)   # most negative first
          .reset_index(drop=True)
    )

    # ── Mean rho across all pathogens per species ─────────────────────────────
    mean_rho = (
        df_corr.groupby("Species")["Spearman_rho"]
               .mean()
               .sort_values()
               .reset_index()
               .rename(columns={"Spearman_rho": "Mean_Spearman_rho"})
    )
    # Keep only negative correlations → antagonists / potential probiotics
    antagonists = mean_rho[mean_rho["Mean_Spearman_rho"] < 0].head(10)

    top5_antag = antagonists.head(5)
    print("\n  ┌─ TOP 5 POTENTIAL PROBIOTIC ANTAGONISTS ─────────────────────────────┐")
    for _, row in top5_antag.iterrows():
        print(f"  │  {_short_name(row.Species):<55s}  rho = {row.Mean_Spearman_rho:+.4f}")
    print("  └──────────────────────────────────────────────────────────────────────┘")

    antagonists.to_csv(f"{OUT_DIR}/module2_probiotic_antagonists.csv", index=False)

    # ── Heatmap: top antagonist species vs pathogens ──────────────────────────
    print("  [2.2] Plotting correlation heatmap …")
    top10_species = antagonists["Species"].tolist()

    heat_data = []
    for pathogen in pathogen_cols:
        row_vals = []
        for sp in top10_species:
            subset = df_corr[(df_corr["Pathogen"] == pathogen) & (df_corr["Species"] == sp)]
            row_vals.append(subset["Spearman_rho"].values[0] if len(subset) else np.nan)
        heat_data.append(row_vals)

    heat_df = pd.DataFrame(
        heat_data,
        index   = [_short_name(p, 35) for p in pathogen_cols],
        columns = [_short_name(s, 35) for s in top10_species],
    )

    fig, ax = plt.subplots(figsize=(14, max(4, len(pathogen_cols) * 1.5)))
    sns.heatmap(
        heat_df, cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.4, linecolor="white", ax=ax,
        cbar_kws={"label": "Spearman ρ"},
    )
    ax.set_title(
        "Spearman Correlations: Known CRC Pathogens vs. Top Antagonist Species\n"
        "(Healthy Controls only — negative ρ = competitive exclusion signal)",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    _save_fig(fig, "module2_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — SUB-PHENOTYPING (DEMOGRAPHIC-SPECIFIC BIOMARKERS)
# ══════════════════════════════════════════════════════════════════════════════
def module3_subphenotype(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> None:
    _section("MODULE 3 — Sub-Phenotyping (Age-Stratified Biomarkers)")

    # ── Drop samples with missing Age ─────────────────────────────────────────
    valid_idx = meta.index[meta["Age"].notna()]
    dropped   = len(X) - len(valid_idx)
    if dropped:
        print(f"  ⚠ Dropped {dropped} samples with missing Age.")
    X_v    = X.loc[valid_idx]
    y_v    = y.loc[valid_idx]
    meta_v = meta.loc[valid_idx]

    # ── Age-stratify ──────────────────────────────────────────────────────────
    early_mask = meta_v["Age"] < 50
    late_mask  = meta_v["Age"] >= 50

    X_early, y_early = X_v.loc[early_mask], y_v.loc[early_mask]
    X_late,  y_late  = X_v.loc[late_mask],  y_v.loc[late_mask]

    print(f"  Early-Onset (<50 yrs) : {X_early.shape[0]} samples  "
          f"(CRC={int(y_early.sum())}, CTR={int((y_early==0).sum())})")
    print(f"  Late-Onset  (≥50 yrs) : {X_late.shape[0]}  samples  "
          f"(CRC={int(y_late.sum())}, CTR={int((y_late==0).sum())})")

    def _train_rf(X_sub: pd.DataFrame, y_sub: pd.Series, label: str):
        """Train a Random Forest and return normalised feature importances."""
        if y_sub.nunique() < 2:
            print(f"  ⚠ {label}: only one class present — skipping.")
            return None
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        rf.fit(X_sub.values, y_sub.values)
        imp = pd.Series(rf.feature_importances_, index=X_sub.columns)
        imp = imp.sort_values(ascending=False)
        return imp

    # ── Train per-group models ────────────────────────────────────────────────
    print("\n  [3.1] Training Random Forest — Early-Onset group …")
    imp_early = _train_rf(X_early, y_early, "Early-Onset")

    print("  [3.2] Training Random Forest — Late-Onset group …")
    imp_late  = _train_rf(X_late,  y_late,  "Late-Onset")

    if imp_early is None or imp_late is None:
        print("  ⚠ Cannot proceed — insufficient class diversity.")
        return

    # ── Save importances ──────────────────────────────────────────────────────
    imp_early.to_frame("Importance").to_csv(f"{OUT_DIR}/module3_early_onset_importances.csv")
    imp_late.to_frame("Importance").to_csv(f"{OUT_DIR}/module3_late_onset_importances.csv")

    # ── Differential biomarkers ───────────────────────────────────────────────
    #  "Early-specific" = high rank in early, low rank in late (and vice-versa)
    all_feats  = X.columns.tolist()
    rank_early = imp_early.rank(ascending=False)   # lower rank = more important
    rank_late  = imp_late.rank(ascending=False)

    diff_df = pd.DataFrame({
        "Imp_Early" : imp_early,
        "Imp_Late"  : imp_late,
        "Rank_Early": rank_early,
        "Rank_Late" : rank_late,
    })
    diff_df["Rank_Diff_Early_Minus_Late"] = diff_df["Rank_Late"] - diff_df["Rank_Early"]
    diff_df.sort_values("Rank_Diff_Early_Minus_Late", ascending=False, inplace=True)

    print("\n  ┌─ TOP 5 EARLY-ONSET SPECIFIC BIOMARKERS ─────────────────────────────┐")
    for feat, row in diff_df.head(5).iterrows():
        print(f"  │  {_short_name(feat):<50s} "
              f" EarlyRank={int(row.Rank_Early):4d}  LateRank={int(row.Rank_Late):4d}")
    print("  └──────────────────────────────────────────────────────────────────────┘")

    print("\n  ┌─ TOP 5 LATE-ONSET SPECIFIC BIOMARKERS ──────────────────────────────┐")
    for feat, row in diff_df.tail(5).iloc[::-1].iterrows():
        print(f"  │  {_short_name(feat):<50s} "
              f" EarlyRank={int(row.Rank_Early):4d}  LateRank={int(row.Rank_Late):4d}")
    print("  └──────────────────────────────────────────────────────────────────────┘")

    diff_df.to_csv(f"{OUT_DIR}/module3_differential_biomarkers.csv")

    # ── Comparison bar chart ──────────────────────────────────────────────────
    print("  [3.3] Plotting feature importance comparison …")
    TOP_N = 15
    # Union of top-N from each group
    union_feats = list(
        dict.fromkeys(
            imp_early.head(TOP_N).index.tolist() +
            imp_late.head(TOP_N).index.tolist()
        )
    )

    comp_df = pd.DataFrame({
        "Early-Onset": imp_early.reindex(union_feats).values,
        "Late-Onset" : imp_late.reindex(union_feats).values,
    }, index=[_short_name(f, 35) for f in union_feats])

    fig, ax = plt.subplots(figsize=(12, 8))
    x   = np.arange(len(comp_df))
    w   = 0.38
    ax.barh(x + w/2, comp_df["Early-Onset"], height=w, color=PALETTE[0],
            label="Early-Onset (<50)", alpha=0.85)
    ax.barh(x - w/2, comp_df["Late-Onset"],  height=w, color=PALETTE[1],
            label="Late-Onset (≥50)",  alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(comp_df.index, fontsize=8)
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=10)
    ax.set_title(
        "Age-Stratified Feature Importances — Early vs Late Onset CRC\n"
        f"(Top {TOP_N} features from each group shown)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    _save_fig(fig, "module3_feature_importance_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — NON-MONOTONIC PATHOGENICITY (SHAP DEPENDENCE + PDP)
# ══════════════════════════════════════════════════════════════════════════════
def module4_pdp(X: pd.DataFrame, y: pd.Series, xgb_model: XGBClassifier) -> None:
    _section("MODULE 4 — Non-Monotonic Pathogenicity (SHAP Dependence / PDP)")

    # ── Top 3 features by XGBoost gain importance ─────────────────────────────
    imp_series = pd.Series(
        xgb_model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)
    top3_feats = imp_series.head(3).index.tolist()

    print("  Top 3 features by XGBoost importance:")
    for i, f in enumerate(top3_feats, 1):
        print(f"    {i}. {f}  (importance={imp_series[f]:.5f})")

    # ── SHAP values for all samples ───────────────────────────────────────────
    print("\n  [4.1] Computing SHAP values for dependence plots …")
    if SHAP_DEPENDENCE_MAX_SAMPLES and X.shape[0] > SHAP_DEPENDENCE_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], SHAP_DEPENDENCE_MAX_SAMPLES, replace=False)
        X_dep = X.iloc[idx]
    else:
        X_dep = X

    explainer  = shap.TreeExplainer(xgb_model.get_booster())
    shap_vals  = explainer.shap_values(X_dep.values)          # (n_samples, n_features)
    feat_names = X.columns.tolist()

    for feat in top3_feats:
        feat_idx  = feat_names.index(feat)
        feat_vals = X_dep[feat].values
        sv        = shap_vals[:, feat_idx]
        label     = _short_name(feat, maxlen=50)

        # ── SHAP Dependence Plot ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            feat_vals, sv,
            c=sv, cmap="RdBu_r", alpha=0.55, s=12, edgecolors="none",
        )
        # Smoothed trend (LOESS-like via rolling median)
        sort_idx    = np.argsort(feat_vals)
        x_sorted    = feat_vals[sort_idx]
        sv_sorted   = sv[sort_idx]
        window      = max(1, len(x_sorted) // 30)
        sv_rolling  = pd.Series(sv_sorted).rolling(window, center=True,
                                                     min_periods=1).mean().values
        ax.plot(x_sorted, sv_rolling, color="#111111", lw=2.0,
                label="Smoothed trend", zorder=5)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("SHAP value", fontsize=9)
        ax.set_xlabel(f"Feature abundance: {label}", fontsize=10)
        ax.set_ylabel("SHAP value (impact on CRC probability)", fontsize=10)
        ax.set_title(
            f"SHAP Dependence Plot — {label}\n"
            "(Look for thresholds, U-shapes, or step-functions)",
            fontsize=11, fontweight="bold",
        )
        ax.legend(frameon=False, fontsize=9)
        safe_fname = feat.replace("/", "-").replace(" ", "_")[:60]
        _save_fig(fig, f"module4_shap_dependence_{safe_fname}.png", dpi=300)

        # ── Standard Partial Dependence Plot (sklearn) ────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            disp = PartialDependenceDisplay.from_estimator(
                xgb_model, X.values,
                features=[feat_idx],
                feature_names=feat_names,
                kind="average",
                ax=ax,
                grid_resolution=80,
                random_state=42,
            )
            ax.set_title(
                f"Partial Dependence Plot — {label}\n"
                "(Average marginal effect on predicted CRC probability)",
                fontsize=11, fontweight="bold",
            )
            ax.set_xlabel(f"Feature abundance: {label}", fontsize=10)
            ax.set_ylabel("Partial dependence", fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"PDP failed:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        _save_fig(fig, f"module4_pdp_{safe_fname}.png", dpi=300)

    print("\n  ⓘ  Interpretation guide:")
    print("     • Sigmoid / step-function  → sharp colonisation threshold")
    print("     • U-shape (SHAP)           → intermediate abundance is protective")
    print("     • Monotone increase (SHAP) → linear dose-response pathogen")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   CRC MICROBIOME — NOVELTY DISCOVERY PIPELINE                       ║")
    print("║   4 Modules: Epistasis · Probiotics · Sub-Phenotyping · PDP         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Load shared data once
    X, y, meta = load_data()

    # Track which modules succeed
    results = {}

    # ── Module 1 ─────────────────────────────────────────────────────────────
    try:
        xgb_model = module1_epistasis(X, y)
        results["Module 1 — Epistasis"] = "✅ SUCCESS"
    except Exception:
        print(f"\n  ❌ Module 1 failed:\n{traceback.format_exc()}")
        results["Module 1 — Epistasis"] = "❌ FAILED"
        # Train a fallback XGBoost so Modules 3/4 can still run
        xgb_model = XGBClassifier(
            n_estimators=100, verbosity=0, random_state=42,
            eval_metric="logloss",
        )
        xgb_model.fit(X.values, y.values)

    # ── Module 2 ─────────────────────────────────────────────────────────────
    try:
        module2_probiotics(X, y, meta)
        results["Module 2 — Probiotics"] = "✅ SUCCESS"
    except Exception:
        print(f"\n  ❌ Module 2 failed:\n{traceback.format_exc()}")
        results["Module 2 — Probiotics"] = "❌ FAILED"

    # ── Module 3 ─────────────────────────────────────────────────────────────
    try:
        module3_subphenotype(X, y, meta)
        results["Module 3 — Sub-Phenotyping"] = "✅ SUCCESS"
    except Exception:
        print(f"\n  ❌ Module 3 failed:\n{traceback.format_exc()}")
        results["Module 3 — Sub-Phenotyping"] = "❌ FAILED"

    # ── Module 4 ─────────────────────────────────────────────────────────────
    try:
        module4_pdp(X, y, xgb_model)
        results["Module 4 — PDP / SHAP Dependence"] = "✅ SUCCESS"
    except Exception:
        print(f"\n  ❌ Module 4 failed:\n{traceback.format_exc()}")
        results["Module 4 — PDP / SHAP Dependence"] = "❌ FAILED"

    # ── Summary ──────────────────────────────────────────────────────────────
    bar = "═" * 70
    print(f"\n{bar}")
    print("  PIPELINE COMPLETE — Module Summary")
    print(bar)
    for mod, status in results.items():
        print(f"  {status}  {mod}")
    print(f"\n  All outputs saved to:  {os.path.abspath(OUT_DIR)}/")
    print(bar)


if __name__ == "__main__":
    main()
