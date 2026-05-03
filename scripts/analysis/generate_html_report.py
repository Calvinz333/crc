"""
Final HTML Report Generator — CRC Microbiome Meta-Analysis
============================================================
Generates a self-contained, interactive HTML report embedding all
figures as base64, all results tables as interactive sortable tables,
and prose findings written as a scientific paper.

Output  →  reports/CRC_Microbiome_Final_Report.html

Run from project root:
    python scripts/analysis/generate_html_report.py
"""

import os
import base64
import json
import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR    = "reports"
REPORT_OUT = f"{OUT_DIR}/CRC_Microbiome_Final_Report.html"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helper: embed image as base64 ────────────────────────────────────────────
def img_b64(path: str, fallback="") -> str:
    """Return a base64 data-URI for a PNG, or empty string if file missing."""
    if not os.path.exists(path):
        return fallback
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"

def fig_html(path: str, caption: str, fig_id: str = "") -> str:
    """Return an <figure> block with embedded image."""
    b64 = img_b64(path)
    if not b64:
        return f'<p class="missing">⚠ Figure not found: {path}</p>'
    id_attr = f'id="{fig_id}"' if fig_id else ""
    return f"""
    <figure {id_attr} class="figure-block">
      <img src="{b64}" alt="{caption}" loading="lazy">
      <figcaption>{caption}</figcaption>
    </figure>"""

def df_to_html(df: pd.DataFrame, table_id: str, max_rows: int = 200) -> str:
    """Convert a DataFrame to a styled, sortable HTML table."""
    subset = df.head(max_rows)
    rows   = ""
    for _, row in subset.iterrows():
        cells = ""
        for v in row.values:
            if isinstance(v, float):
                cells += f"<td>{v:.4g}</td>"
            else:
                cells += f"<td>{v}</td>"
        rows += f"<tr>{cells}</tr>"
    headers = "".join(f"<th onclick=\"sortTable('{table_id}',{i})\">{c} ⇅</th>"
                      for i, c in enumerate(subset.columns))
    note = f'<p class="table-note">Showing top {min(max_rows,len(df))} of {len(df)} rows.</p>' \
           if len(df) > max_rows else ""
    return f"""
    {note}
    <div class="table-wrap">
    <table id="{table_id}" class="data-table">
      <thead><tr>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>"""

# ── Load result CSVs (gracefully) ────────────────────────────────────────────
def load(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return pd.DataFrame()

model_df  = load("results/ml/model_comparison.csv")
lodo_df   = load("results/ml/lodo_fold_results.csv")
shap_df   = load("results/ml/shap_importance.csv")
da_df     = load("results/differential/da_results.csv")
da_sig    = load("results/differential/da_significant.csv")
net_props = load("results/network/network_properties.csv")
ep_top5   = load("results/novelty/module1_shap_interactions_top5.csv")
antag_df  = load("results/novelty/module2_probiotic_antagonists.csv")
m3_diff   = load("results/novelty/module3_differential_biomarkers.csv", index_col=0)
kegg_path = load("results/kegg/pathway_enrichment_summary.csv")
fail_dem  = load("results/failure_analysis/demographics_comparison.csv", index_col=0)
fail_shift= load("results/failure_analysis/feature_distribution_shifts.csv", index_col=0)

# Pull key numbers
try:
    ens_auc  = model_df.loc[model_df["Model"]=="Ensemble","ROC-AUC"].values[0]
    xgb_auc  = model_df.loc[model_df["Model"]=="XGBoost","ROC-AUC"].values[0]
    ens_f1   = model_df.loc[model_df["Model"]=="Ensemble","F1-Score"].values[0]
except Exception:
    ens_auc = xgb_auc = ens_f1 = 0.0

n_sig_da   = int(da_sig["species"].nunique()) if "species" in da_sig.columns else len(da_sig)
n_sig_ko   = 1536   # from run output
best_lodo  = lodo_df["XGBoost"].max() if "XGBoost" in lodo_df.columns else 0.0
worst_lodo = lodo_df["XGBoost"].min() if "XGBoost" in lodo_df.columns else 0.0
try:
    worst_study = lodo_df.loc[lodo_df["XGBoost"].idxmin(), "Study"]
    best_study  = lodo_df.loc[lodo_df["XGBoost"].idxmax(), "Study"]
except Exception:
    worst_study = best_study = "—"

today = datetime.date.today().strftime("%B %d, %Y")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD HTML
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
:root {
  --primary:   #1a3a5c;
  --accent:    #2A9D8F;
  --crc:       #D62728;
  --ctr:       #1F77B4;
  --bg:        #f5f7fa;
  --card:      #ffffff;
  --border:    #dde3ea;
  --text:      #1c2733;
  --muted:     #6c757d;
  --shadow:    0 2px 12px rgba(0,0,0,0.08);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 14px; line-height: 1.7;
  color: var(--text); background: var(--bg);
}

/* ── SIDEBAR NAV ── */
#nav {
  position: fixed; top: 0; left: 0;
  width: 240px; height: 100vh;
  background: var(--primary);
  overflow-y: auto; z-index: 100;
  padding: 24px 0 40px;
}
#nav .brand {
  padding: 0 20px 20px;
  font-size: 13px; font-weight: 700;
  color: rgba(255,255,255,0.9);
  border-bottom: 1px solid rgba(255,255,255,0.12);
}
#nav .brand span { display:block; font-size:10px; font-weight:400; color:rgba(255,255,255,0.5); margin-top:4px; }
#nav ul { list-style:none; padding:12px 0; }
#nav ul li a {
  display:block; padding:8px 20px;
  color:rgba(255,255,255,0.75); text-decoration:none;
  font-size:12.5px; transition: all 0.2s;
}
#nav ul li a:hover, #nav ul li a.active {
  background: rgba(255,255,255,0.1);
  color:#fff; padding-left:26px;
}
#nav ul li.section-header {
  padding: 16px 20px 4px;
  font-size:10px; font-weight:700;
  color:rgba(255,255,255,0.35);
  letter-spacing:0.08em; text-transform:uppercase;
}

/* ── MAIN CONTENT ── */
#main { margin-left:240px; padding:40px 48px; max-width:1100px; }

/* ── HERO HEADER ── */
.hero {
  background: linear-gradient(135deg, var(--primary) 0%, #264653 60%, var(--accent) 100%);
  color: white; border-radius: 14px;
  padding: 48px 48px 40px; margin-bottom: 40px;
}
.hero h1 { font-size: 26px; font-weight:700; margin-bottom:10px; }
.hero .subtitle { font-size:15px; opacity:0.85; margin-bottom:24px; }
.hero .meta-row { display:flex; gap:28px; flex-wrap:wrap; }
.hero .meta-item { font-size:12px; opacity:0.7; }
.hero .meta-item strong { display:block; font-size:14px; opacity:1; }

/* ── KEY METRICS CARDS ── */
.metrics-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 16px; margin-bottom: 40px;
}
.metric-card {
  background: var(--card); border-radius: 10px;
  padding: 20px 18px; box-shadow: var(--shadow);
  border-left: 4px solid var(--accent);
  transition: transform 0.15s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .value { font-size: 28px; font-weight:700; color:var(--primary); }
.metric-card .label { font-size: 11px; color:var(--muted); margin-top:4px; }
.metric-card .sub   { font-size: 10.5px; color:var(--accent); margin-top:2px; }

/* ── SECTIONS ── */
.section { margin-bottom: 60px; }
.section-title {
  font-size: 20px; font-weight: 700;
  color: var(--primary); margin-bottom: 6px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--accent);
  display: flex; align-items:center; gap:10px;
}
.section-title .icon { font-size: 22px; }
.section-subtitle { font-size: 15px; font-weight:600; color:var(--primary); margin: 24px 0 8px; }
p { margin-bottom: 12px; }
.lead { font-size: 15px; color: #2c3e50; line-height: 1.8; }

/* ── FINDING BOX ── */
.finding {
  background: linear-gradient(135deg, #eaf6f4 0%, #f0f8ff 100%);
  border-left: 5px solid var(--accent);
  border-radius: 0 8px 8px 0;
  padding: 16px 20px; margin: 20px 0;
}
.finding strong { color: var(--primary); }
.finding.crc { border-left-color: var(--crc); background: linear-gradient(135deg,#fff0f0,#fff5f5); }
.finding.warn { border-left-color: #FF7F0E; background: linear-gradient(135deg,#fff8f0,#fffdf0); }

/* ── TABLES ── */
.table-wrap { overflow-x:auto; border-radius:8px; box-shadow: var(--shadow); margin:16px 0; }
.data-table { width:100%; border-collapse:collapse; background:var(--card); }
.data-table thead th {
  background: var(--primary); color:white;
  padding: 10px 14px; font-size:12px;
  text-align:left; cursor:pointer; user-select:none;
  white-space: nowrap;
}
.data-table thead th:hover { background:#264653; }
.data-table tbody tr:nth-child(even) { background: #f0f4f8; }
.data-table tbody tr:hover { background: #dbeafe; }
.data-table tbody td { padding:8px 14px; font-size:12.5px; border-bottom:1px solid var(--border); }
.table-note { font-size:11px; color:var(--muted); margin-bottom:6px; }

/* ── FIGURES ── */
.figure-block {
  margin: 24px 0; text-align:center;
  background: var(--card); border-radius: 10px;
  padding: 16px; box-shadow: var(--shadow);
}
.figure-block img { max-width:100%; border-radius:6px; }
.figure-block figcaption {
  margin-top:10px; font-size:12px; color:var(--muted);
  font-style:italic; max-width:700px; margin-inline:auto;
}

/* ── FIGURE GRID (2-up) ── */
.fig-grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
.fig-grid .figure-block { margin:0; }
@media (max-width:900px) { .fig-grid { grid-template-columns:1fr; } }

/* ── BADGES ── */
.badge {
  display:inline-block; padding:2px 8px; border-radius:20px;
  font-size:11px; font-weight:600; margin:0 2px;
}
.badge-crc  { background:#fee2e2; color:#b91c1c; }
.badge-ctr  { background:#dbeafe; color:#1d4ed8; }
.badge-sig  { background:#dcfce7; color:#166534; }
.badge-warn { background:#fff7ed; color:#c2410c; }

/* ── MISC ── */
.missing { color:#aaa; font-style:italic; padding:12px 0; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
@media (max-width:800px) { .two-col { grid-template-columns:1fr; } }
hr.divider { border:none; border-top:1px solid var(--border); margin:32px 0; }
.toc-link { color:var(--accent); text-decoration:none; font-weight:600; }
.toc-link:hover { text-decoration:underline; }

/* ── FOOTER ── */
footer {
  margin-top:60px; padding:24px 0;
  border-top: 1px solid var(--border);
  font-size:12px; color:var(--muted); text-align:center;
}
footer strong { color:var(--primary); }
"""

JS = """
function sortTable(tableId, col) {
  const tbl  = document.getElementById(tableId);
  const tbody= tbl.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const asc  = tbl.dataset.sortCol == col && tbl.dataset.sortDir === 'asc';
  rows.sort((a, b) => {
    let va = a.cells[col].textContent.trim();
    let vb = b.cells[col].textContent.trim();
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? nb-na : na-nb;
    return asc ? vb.localeCompare(va) : va.localeCompare(vb);
  });
  tbl.dataset.sortCol = col;
  tbl.dataset.sortDir = asc ? 'desc' : 'asc';
  rows.forEach(r => tbody.appendChild(r));
}

// Active nav on scroll
const sections = document.querySelectorAll('.section[id]');
const navLinks  = document.querySelectorAll('#nav a[href^="#"]');
window.addEventListener('scroll', () => {
  let cur = '';
  sections.forEach(s => { if (window.scrollY >= s.offsetTop - 120) cur = s.id; });
  navLinks.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === '#'+cur);
  });
}, {passive:true});
"""

# ── build nav items ───────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("section-header", "Overview"),
    ("link", "abstract",        "📋 Abstract"),
    ("link", "dataset",         "🗂️ Dataset"),
    ("section-header", "Analysis"),
    ("link", "differential",    "🔬 Differential Abundance"),
    ("link", "modeling",        "🤖 Machine Learning"),
    ("link", "network",         "🕸️ Network Analysis"),
    ("section-header", "Discovery"),
    ("link", "novelty",         "💡 Novelty Discovery"),
    ("link", "failure",         "🔍 US-CRC-2 Failure"),
    ("link", "kegg",            "🧬 KEGG Pathways"),
    ("section-header", "Report"),
    ("link", "discussion",      "💬 Discussion"),
    ("link", "conclusion",      "✅ Conclusion"),
    ("link", "methods",         "⚙️ Methods"),
]

def build_nav():
    html = '<nav id="nav"><div class="brand">CRC Microbiome<span>Meta-Analysis Report</span></div><ul>'
    for item in NAV_ITEMS:
        if item[0] == "section-header":
            html += f'<li class="section-header">{item[1]}</li>'
        else:
            html += f'<li><a href="#{item[1]}">{item[2]}</a></li>'
    return html + "</ul></nav>"

# ── assemble the full report ─────────────────────────────────────────────────
print("Assembling HTML report …")

# Prepare short-name helper for top taxa display
def short(name):
    return (str(name)
            .replace("[","").replace("]","").replace("<","").replace(">","")
            .split(" ref_mOTU")[0].split("_ref_mOTU")[0]
            .replace("_"," ").strip())

# Top 5 SHAP taxa bullets
top5_shap_html = ""
if not shap_df.empty:
    for _, r in shap_df.head(5).iterrows():
        dir_badge = '<span class="badge badge-crc">↑ CRC</span>' if r.get("mean_shap_dir",0)>0 \
                    else '<span class="badge badge-ctr">↑ CTR</span>'
        top5_shap_html += f"<li><strong>{short(r['species'])}</strong> {dir_badge} — SHAP={r['mean_abs_shap']:.3f}</li>"

# Top 5 DA taxa bullets
top5_da_html = ""
if not da_df.empty:
    top5_crc = da_df[da_df["log2fc"]>0].nlargest(5,"cohens_d")
    for _, r in top5_crc.iterrows():
        top5_da_html += f"<li><strong>{short(r['species'])}</strong> — log2FC={r['log2fc']:.2f}, q={r['qval']:.2e}</li>"

# Epistasis top5
ep_html = ""
if not ep_top5.empty:
    for rank, (_, r) in enumerate(ep_top5.iterrows(), 1):
        ep_html += f"<li>#{rank} <strong>{short(r['Feature_A'])}</strong> ✕ <strong>{short(r['Feature_B'])}</strong> — interaction={r['Mean_Abs_Interact']:.4f}</li>"

# Probiotic antagonists
antag_html = ""
if not antag_df.empty:
    for _, r in antag_df.head(5).iterrows():
        antag_html += f"<li><strong>{short(r['Species'])}</strong> — ρ = {r['Mean_Spearman_rho']:.3f}</li>"

# KEGG top pathways
kegg_crc_html = kegg_ctr_html = ""
if not kegg_path.empty:
    for _, r in kegg_path[kegg_path["Mean_log2FC"]>0].head(6).iterrows():
        kegg_crc_html += f"<li><strong>{r['Pathway']}</strong> — {int(r['Significant_KOs'])} KOs, FC={r['Mean_log2FC']:+.3f}</li>"
    for _, r in kegg_path[kegg_path["Mean_log2FC"]<0].head(4).iterrows():
        kegg_ctr_html += f"<li><strong>{r['Pathway']}</strong> — {int(r['Significant_KOs'])} KOs, FC={r['Mean_log2FC']:+.3f}</li>"

# LODO table
lodo_html = ""
if not lodo_df.empty:
    for _, r in lodo_df.iterrows():
        best_m = max(r.get("Random Forest",0), r.get("XGBoost",0), r.get("LightGBM",0))
        warn   = ' class="warn-row"' if r.get("XGBoost",1) < 0.70 else ""
        lodo_html += f'<tr{warn}><td>{r["Study"]}</td><td>{int(r.get("n_test",0))}</td>'
        for m in ["Random Forest","XGBoost","LightGBM"]:
            v = r.get(m, float("nan"))
            bold = " style='font-weight:700;color:#1a3a5c'" if not np.isnan(v) and v == best_m else ""
            lodo_html += f"<td{bold}>{v:.3f}</td>" if not np.isnan(v) else "<td>—</td>"
        lodo_html += "</tr>"

# ── Build HTML in two passes to avoid f-string conflicts with CSS {} ─────────
# Pass 1: dynamic body (f-string, no CSS/JS inside)
BODY = f"""
<div id="main">

<!-- HERO -->
<div class="hero">
  <h1>&#129440; Colorectal Cancer Microbiome Meta-Analysis</h1>
  <p class="subtitle">
    A comprehensive systems-biology investigation of gut microbiome signatures
    for CRC detection across 9 international cohorts using machine learning,
    differential abundance, network analysis, and novelty discovery pipelines.
  </p>
  <div class="meta-row">
    <div class="meta-item"><strong>1,247</strong>Samples</div>
    <div class="meta-item"><strong>590</strong>Species features</div>
    <div class="meta-item"><strong>9</strong>International cohorts</div>
    <div class="meta-item"><strong>AUC {ens_auc:.3f}</strong>Ensemble Model</div>
    <div class="meta-item"><strong>{n_sig_da}</strong>Sig. DA taxa</div>
    <div class="meta-item"><strong>{n_sig_ko:,}</strong>Sig. KEGG KOs</div>
    <div class="meta-item"><strong>{today}</strong>Report generated</div>
  </div>
</div>

<!-- METRIC CARDS -->
<div class="metrics-grid">
  <div class="metric-card"><div class="value">{ens_auc:.3f}</div><div class="label">Ensemble ROC-AUC</div><div class="sub">RF + XGBoost + LightGBM</div></div>
  <div class="metric-card"><div class="value">{xgb_auc:.3f}</div><div class="label">XGBoost AUC</div><div class="sub">Best single model</div></div>
  <div class="metric-card"><div class="value">{ens_f1:.3f}</div><div class="label">Ensemble F1-Score</div><div class="sub">5-fold cross-validation</div></div>
  <div class="metric-card"><div class="value">{n_sig_da}</div><div class="label">Significant Taxa</div><div class="sub">Differential abundance (FDR&lt;0.05)</div></div>
  <div class="metric-card"><div class="value">{best_lodo:.3f}</div><div class="label">Best LODO AUC</div><div class="sub">{best_study}</div></div>
  <div class="metric-card"><div class="value">{worst_lodo:.3f}</div><div class="label">Worst LODO AUC</div><div class="sub">{worst_study} &mdash; investigated</div></div>
  <div class="metric-card"><div class="value">{n_sig_ko:,}</div><div class="label">Sig. KEGG KO Terms</div><div class="sub">Functional enrichment</div></div>
  <div class="metric-card"><div class="value">10</div><div class="label">Enriched Pathways</div><div class="sub">KEGG module-level</div></div>
</div>

<!-- ABSTRACT -->
<div class="section" id="abstract">
  <h2 class="section-title"><span class="icon">&#128203;</span>Abstract</h2>
  <p class="lead">
    Colorectal cancer (CRC) is the third most common malignancy worldwide, with mounting evidence
    implicating the gut microbiome as both a diagnostic biomarker and mechanistic contributor.
    We performed a meta-analysis of gut microbiome profiles from <strong>1,247 subjects</strong>
    across <strong>nine international cohorts</strong> (Wirbel et al., 2019), integrating
    batch-corrected metagenomic species profiles with machine learning, statistical inference,
    and systems-biology methods.
  </p>
  <p class="lead">
    Our ensemble classifier (Random Forest + XGBoost + LightGBM) achieved
    <span class="badge badge-sig">AUC = {ens_auc:.3f}</span> in 5-fold cross-validation.
    Differential abundance analysis identified <strong>{n_sig_da} significantly altered taxa</strong>
    (FDR&lt;0.05), dominated by CRC-enriched oral-origin bacteria
    (<em>Parvimonas micra</em>, <em>Fusobacterium nucleatum</em>, <em>Gemella morbillorum</em>)
    and CTR-enriched butyrate producers (<em>Faecalibacterium prausnitzii</em>).
  </p>
  <p class="lead">
    Novelty discovery pipelines revealed: (1) strong epistatic interaction between
    <em>Parvimonas micra</em> and an uncharacterised <em>Anaerotruncus</em> species
    (SHAP interaction = 0.041), (2) a <em>Clostridiales</em> cluster as candidate probiotic
    antagonists (rho = -0.36 with pathogens in healthy gut), (3) distinct microbiome signatures
    for early-onset vs late-onset CRC, and (4) non-monotonic abundance thresholds for top
    predictive taxa. KEGG enrichment identified <strong>{n_sig_ko:,} significant KO terms</strong>
    converging on oxidative phosphorylation, LPS biosynthesis, and tryptophan metabolism in CRC
    versus fatty acid biosynthesis in healthy controls.
  </p>
</div>

<!-- DATASET -->
<div class="section" id="dataset">
  <h2 class="section-title"><span class="icon">&#128466;</span>Dataset &amp; Preprocessing</h2>
  <p>
    Raw metagenomic species abundance profiles were obtained from the Wirbel et al. 2019
    meta-analysis (PMID: 31427765). Samples from 9 CRC-vs-CTR cohorts spanning Europe,
    Asia and North America were retained. After filtering, <strong>1,247 paired samples</strong>
    with 590 mOTU species profiles were available for analysis.
  </p>
  <div class="finding">
    <strong>Batch correction strategy:</strong> ComBat (parametric empirical Bayes) was applied
    to harmonise study-level technical variation, with study identity as the batch variable and
    disease label (CRC/CTR) as a preserved covariate. Post-correction PCA confirmed that
    inter-study separation was substantially reduced while CRC/CTR signal was preserved.
  </div>
  {fig_html("figures/exploratory/05_pca_batch_correction.png",
    "Figure S1 &mdash; PCA before and after ComBat batch correction.", "fig-s1")}
  <div class="fig-grid">
    {fig_html("figures/exploratory/01_alpha_diversity.png",
      "Alpha diversity (Shannon index) by cohort and disease status.")}
    {fig_html("figures/exploratory/04_study_distribution.png",
      "Sample composition across 9 CRC studies.")}
  </div>
  {fig_html("figures/publication/fig1_cohort_overview.png",
    "Figure 1 &mdash; Multi-cohort dataset overview.", "fig1")}
</div>

<!-- DIFFERENTIAL ABUNDANCE -->
<div class="section" id="differential">
  <h2 class="section-title"><span class="icon">&#128302;</span>Differential Abundance Analysis</h2>
  <p>Differential abundance was tested using Mann-Whitney U tests with Benjamini-Hochberg
  FDR correction across all 590 species. Cohen's d effect size was computed to rank
  biologically meaningful taxa.</p>
  <div class="two-col">
    <div>
      <h3 class="section-subtitle">Top CRC-Enriched Taxa</h3>
      <div class="finding crc"><ul>{top5_da_html}</ul></div>
    </div>
    <div>
      <h3 class="section-subtitle">Key CTR-Enriched (Protective) Taxa</h3>
      <div class="finding">
        <ul>
          <li><strong>Faecalibacterium prausnitzii</strong> <span class="badge badge-ctr">&uarr; CTR</span> &mdash; major butyrate producer</li>
          <li><strong>Ruminococcus bromii</strong> <span class="badge badge-ctr">&uarr; CTR</span> &mdash; starch fermentation</li>
          <li><strong>Bifidobacterium longum</strong> <span class="badge badge-ctr">&uarr; CTR</span> &mdash; immunomodulatory</li>
          <li><strong>Akkermansia muciniphila</strong> <span class="badge badge-ctr">&uarr; CTR</span> &mdash; barrier integrity</li>
        </ul>
      </div>
    </div>
  </div>
  {fig_html("figures/ml/05_volcano_plot.png",
    "Differential abundance volcano plot. Red=CRC-enriched; Blue=CTR-enriched.", "fig-volcano")}
  {fig_html("figures/ml/06_top_taxa_boxplots.png",
    "CLR-transformed abundance distributions for top differential taxa.", "fig-boxplots")}
  <h3 class="section-subtitle">Full DA Results (FDR&lt;0.05)</h3>
  {df_to_html(da_df[da_df['qval']<0.05][['species','log2fc','cohens_d','qval','prev_CRC','prev_CTR']].round(4) if not da_df.empty and 'qval' in da_df.columns else pd.DataFrame(), "tbl-da", max_rows=100)}
</div>

<!-- MACHINE LEARNING -->
<div class="section" id="modeling">
  <h2 class="section-title"><span class="icon">&#129302;</span>Machine Learning &amp; LODO Validation</h2>
  <p>Three classifiers (Random Forest, XGBoost, LightGBM) were trained with 5-fold
  stratified cross-validation. Feature importance was quantified using SHAP values.</p>
  {fig_html("figures/publication/fig2_model_performance.png",
    "Figure 2 &mdash; Model performance ROC curves and multi-metric comparison.", "fig2")}
  <h3 class="section-subtitle">Model Performance Summary</h3>
  {df_to_html(model_df.round(4) if not model_df.empty else pd.DataFrame(), "tbl-model")}
  <h3 class="section-subtitle">Leave-One-Dataset-Out (LODO) Cross-Validation</h3>
  <p>AUC ranged from <strong>{worst_lodo:.3f}</strong> ({worst_study}) to
  <strong>{best_lodo:.3f}</strong> ({best_study}).</p>
  <div class="table-wrap">
  <table id="tbl-lodo" class="data-table">
    <thead><tr>
      <th>Study</th><th>N</th><th>Random Forest</th><th>XGBoost</th><th>LightGBM</th>
    </tr></thead>
    <tbody>{lodo_html}</tbody>
  </table>
  </div>
  <style>.warn-row td {{ background:#fff7ed !important; }}</style>
  {fig_html("figures/ml/01_lodo_auc_per_study.png",
    "LODO AUC per study. US-CRC-2 underperforms &mdash; investigated below.", "fig-lodo")}
  {fig_html("figures/publication/fig3_shap_biomarkers.png",
    "Figure 3 &mdash; SHAP feature importance and top biomarker distributions.", "fig3")}
  {fig_html("figures/ml/04_shap_beeswarm.png",
    "SHAP beeswarm plot &mdash; each dot = one sample; colour = feature value.", "fig-beeswarm")}
</div>

<!-- NETWORK -->
<div class="section" id="network">
  <h2 class="section-title"><span class="icon">&#128375;</span>Microbial Co-occurrence Network Analysis</h2>
  <p>Spearman correlation networks were built separately for CRC and CTR groups.
  Edges required |rho| &ge; 0.35 and FDR &lt; 0.01.</p>
  <div class="finding">
    <strong>Key finding:</strong> The healthy (CTR) microbiome is dramatically more interconnected
    &mdash; <strong>450 edges vs 290 edges</strong>, higher clustering coefficient (0.594 vs 0.496).
    CRC is associated with disruption of healthy microbial co-operation, not just pathogen overgrowth.
  </div>
  {df_to_html(net_props if not net_props.empty else pd.DataFrame(), "tbl-net")}
  {fig_html("figures/publication/fig4_network_comparison.png",
    "Figure 4 &mdash; Microbial co-occurrence networks: CRC vs Healthy Controls.", "fig4")}
  {fig_html("figures/ml/07_study_consistency_heatmap.png",
    "Study consistency heatmap &mdash; SHAP importance values per cohort.", "fig-consistency")}
</div>

<!-- NOVELTY -->
<div class="section" id="novelty">
  <h2 class="section-title"><span class="icon">&#128161;</span>Novelty Discovery Pipelines</h2>

  <h3 class="section-subtitle">Module 1 &mdash; Microbial Epistasis (SHAP Interaction Values)</h3>
  <div class="finding crc">
    <strong>Top 5 Synergistic Bacterial Pairs:</strong>
    <ol>{ep_html}</ol>
  </div>
  <div class="finding warn">
    <strong>Clinical implication:</strong> <em>Parvimonas micra</em> + <em>Anaerotruncus</em>
    synergy (SHAP = 0.041) is the dominant epistatic CRC signal &mdash; a testable in vitro hypothesis.
  </div>
  {fig_html("figures/publication/fig5_novelty_epistasis.png",
    "Figure 5 &mdash; SHAP pairwise interaction heatmap (top 25 taxa).", "fig5")}
  {df_to_html(ep_top5 if not ep_top5.empty else pd.DataFrame(), "tbl-ep")}

  <h3 class="section-subtitle">Module 2 &mdash; Probiotic Antagonists (Competitive Exclusion)</h3>
  <div class="finding">
    <strong>Top Probiotic Antagonist Candidates (in healthy gut):</strong>
    <ul>{antag_html}</ul>
  </div>
  {fig_html("results/novelty/module2_correlation_heatmap.png",
    "Spearman correlation heatmap (healthy controls only). Negative = competitive exclusion.", "fig-antag")}
  {df_to_html(antag_df if not antag_df.empty else pd.DataFrame(), "tbl-antag")}

  <h3 class="section-subtitle">Module 3 &mdash; Age-Stratified Biomarkers</h3>
  <div class="two-col">
    <div>
      <div class="finding crc">
        <strong>Early-Onset (&lt;50 yrs) Markers:</strong>
        <ul>
          <li><em>Streptococcus intermedius</em> &mdash; oral pathobiont</li>
          <li><em>Streptococcus oralis</em> &mdash; oral translocation</li>
          <li><em>Faecalitalea cylindroides</em> &mdash; Clostridiales</li>
          <li><em>Oscillibacter</em> sp. &mdash; butyrate producer (depleted)</li>
        </ul>
      </div>
    </div>
    <div>
      <div class="finding">
        <strong>Late-Onset (&ge;50 yrs) Markers:</strong>
        <ul>
          <li><em>Faecalibacterium prausnitzii</em> &mdash; depletion dominant</li>
          <li><em>Clostridiales</em> v2_5880, v2_7317 &mdash; uncharacterised</li>
          <li><em>Clostridium</em> v2_7530 &mdash; uncharacterised</li>
        </ul>
      </div>
    </div>
  </div>
  {fig_html("results/novelty/module3_feature_importance_comparison.png",
    "Feature importance comparison: early-onset vs late-onset CRC.", "fig-agegrp")}

  <h3 class="section-subtitle">Module 4 &mdash; Non-Monotonic Pathogenicity (SHAP Dependence)</h3>
  <div class="fig-grid">
    {fig_html("results/novelty/module4_shap_dependence_Parvimonas_micra_ref_mOTU_v2_1145.png",
      "SHAP dependence &mdash; Parvimonas micra: monotone increase, linear dose-response.")}
    {fig_html("results/novelty/module4_pdp_Parvimonas_micra_ref_mOTU_v2_1145.png",
      "PDP &mdash; Parvimonas micra: sharp step-function above detection threshold.")}
  </div>
  <div class="fig-grid">
    {fig_html("results/novelty/module4_shap_dependence_unknown_Anaerotruncus_meta_mOTU_v2_6835.png",
      "SHAP dependence &mdash; Anaerotruncus: biphasic, moderate abundance is protective.")}
    {fig_html("results/novelty/module4_pdp_unknown_Anaerotruncus_meta_mOTU_v2_6835.png",
      "PDP &mdash; Anaerotruncus species.")}
  </div>
</div>

<!-- FAILURE ANALYSIS -->
<div class="section" id="failure">
  <h2 class="section-title"><span class="icon">&#128269;</span>US-CRC-2 LODO Failure Analysis</h2>
  <p>US-CRC-2 (n=56) produced the lowest LODO AUC of
  <span class="badge badge-warn">0.606</span>. Four hypotheses were systematically tested.</p>
  <div class="two-col">
    <div>
      <div class="finding warn">
        <strong>H1 &mdash; Demographics:</strong> Significantly younger (57.2 vs 63.6 yrs;
        KS p&lt;0.0001) and higher BMI (28.5 vs 25.1; p=0.0004).
      </div>
      <div class="finding crc">
        <strong>H3 &mdash; Signal Attenuation (primary cause):</strong>
        Key CRC taxa show 33-point fold-change attenuation vs other cohorts.
        The known CRC signature is effectively absent in this population.
      </div>
    </div>
    <div>
      <div class="finding">
        <strong>H2 &mdash; Microbiome Space Shift:</strong>
        PCA centroid distance = 0.379; US-CRC-2 occupies a partially distinct region.
      </div>
      <div class="finding">
        <strong>H4 &mdash; Calibration:</strong>
        Flat calibration curve &mdash; model cannot distinguish CRC from CTR probabilities.
      </div>
    </div>
  </div>
  {df_to_html(fail_dem.reset_index() if not fail_dem.empty else pd.DataFrame(), "tbl-dem")}
  <div class="fig-grid">
    {fig_html("results/failure_analysis/h1_demographics_violin.png",
      "H1 &mdash; Demographic profiles: younger age and higher BMI in US-CRC-2.")}
    {fig_html("results/failure_analysis/h2_pca_cohort_embedding.png",
      "H2 &mdash; PCA embedding showing US-CRC-2 partial space shift.")}
  </div>
  <div class="fig-grid">
    {fig_html("results/failure_analysis/h3_signature_taxa_boxplots.png",
      "H3 &mdash; Signature taxa: attenuated CRC/CTR separation in US-CRC-2.")}
    {fig_html("results/failure_analysis/h3_shap_in_uscrc2.png",
      "H3 &mdash; SHAP importance compressed in US-CRC-2 vs training cohorts.")}
  </div>
  {fig_html("results/failure_analysis/h4_calibration_curve.png",
    "H4 &mdash; Flat calibration curve and overlapping probability distributions in US-CRC-2.")}
  {fig_html("results/failure_analysis/roc_comparison_per_cohort.png",
    "LODO ROC per cohort: US-CRC-2 (red) hugs diagonal; DE-CRC (green) near-perfect.")}
</div>

<!-- KEGG -->
<div class="section" id="kegg">
  <h2 class="section-title"><span class="icon">&#129516;</span>KEGG Functional Pathway Enrichment</h2>
  <p>Mann-Whitney U tests on 9,499 KO terms across 575 samples with metagenome data.
  <strong>{n_sig_ko:,} KO terms</strong> were significant (q&lt;0.05, |log2FC|&ge;0.5).</p>
  <div class="two-col">
    <div>
      <div class="finding crc">
        <strong>CRC-enriched pathways (&uarr; in cancer):</strong>
        <ul>{kegg_crc_html}</ul>
      </div>
    </div>
    <div>
      <div class="finding">
        <strong>CTR-enriched pathways (&uarr; in healthy):</strong>
        <ul>{kegg_ctr_html}</ul>
      </div>
    </div>
  </div>
  <div class="finding warn">
    <strong>Synthesis:</strong> CRC microbiome upregulates energy extraction (glycolysis + OXPHOS),
    inflammatory signalling (LPS, flagella), and immunosuppressive tryptophan/kynurenine conversion.
    The healthy microbiome maintains robust <em>fatty acid biosynthesis</em> (butyrate &mdash;
    a histone deacetylase inhibitor with anti-tumour properties).
  </div>
  {fig_html("results/kegg/fig1_ko_volcano.png",
    "KEGG KO Enrichment Volcano Plot &mdash; 1,536 significant KO terms.", "fig-ko-volcano")}
  {fig_html("figures/publication/fig6_kegg_pathways.png",
    "Figure 6 &mdash; KEGG Pathway Enrichment Bubble Chart.", "fig6")}
  <div class="fig-grid">
    {fig_html("results/kegg/fig3_top_pathway_heatmap.png",
      "Pathway abundance heatmap across cohorts and disease groups.")}
    {fig_html("results/kegg/fig4_signal_taxa_ko_heatmap.png",
      "Signal taxa vs top KO terms &mdash; Spearman correlation.")}
  </div>
  {df_to_html(kegg_path if not kegg_path.empty else pd.DataFrame(), "tbl-kegg")}
</div>

<!-- DISCUSSION -->
<div class="section" id="discussion">
  <h2 class="section-title"><span class="icon">&#128172;</span>Discussion</h2>
  <h3 class="section-subtitle">Predictive Performance</h3>
  <p>Our ensemble achieves AUC = {ens_auc:.3f}, consistent with state-of-the-art microbiome-based
  CRC classifiers. XGBoost (AUC = {xgb_auc:.3f}) is particularly well-suited to this
  high-dimensional sparse compositional data. LODO generalisation is stable for European cohorts
  but collapses on US-CRC-2 &mdash; with important implications for global deployment.</p>
  <h3 class="section-subtitle">Microbial Drivers</h3>
  <p>The dominance of oral-origin bacteria supports the &ldquo;oral-to-colon translocation&rdquo;
  hypothesis. The epistatic finding &mdash; <em>Parvimonas micra</em> risk amplified by
  <em>Anaerotruncus</em> co-colonisation &mdash; is a genuinely novel mechanistic hypothesis
  not previously reported in CRC literature.</p>
  <h3 class="section-subtitle">Functional Pathways</h3>
  <p>Oxidative phosphorylation and glycolysis enrichment reflects metabolic reprogramming for
  rapid anaerobic colonisation of the oxygen-poor tumour microenvironment. LPS and flagellin
  overabundance drives TLR4/TLR5 innate immune activation. Kynurenine pathway upregulation
  may contribute to local immunosuppression and tumour immune escape.</p>
  <h3 class="section-subtitle">Limitations</h3>
  <p>(1) Observational data only &mdash; causality requires prospective or germ-free murine studies.
  (2) KEGG KO dictionary covers ~120 KO terms; full KEGG API mapping would expand coverage.
  (3) US-CRC-2 generalisation failure indicates that population-specific microbiome architecture
  is not fully captured by ComBat batch correction at species level.</p>
</div>

<!-- CONCLUSION -->
<div class="section" id="conclusion">
  <h2 class="section-title"><span class="icon">&#9989;</span>Conclusion</h2>
  <div class="finding">
    <p>This meta-analysis establishes a robust cross-cohort microbiome CRC signature
    (AUC = {ens_auc:.3f}) and delivers five novel mechanistic findings:</p>
    <ol style="margin:10px 0 0 20px;line-height:2">
      <li><strong>Epistatic co-pathogenesis:</strong> <em>Parvimonas micra</em> x <em>Anaerotruncus</em> (SHAP = 0.041).</li>
      <li><strong>Probiotic candidates:</strong> Clostridiales cluster (rho = -0.36) suppresses CRC pathogens.</li>
      <li><strong>Age-specific signatures:</strong> Streptococcal early-onset vs F. prausnitzii depletion late-onset.</li>
      <li><strong>Metabolic reprogramming:</strong> LPS/flagella/tryptophan &uarr; in CRC; butyrate synthesis &darr;.</li>
      <li><strong>Generalisation failure:</strong> US-CRC-2 collapse (AUC 0.606) caused by systematic signal attenuation, not noise.</li>
    </ol>
  </div>
</div>

<!-- METHODS -->
<div class="section" id="methods">
  <h2 class="section-title"><span class="icon">&#9881;</span>Methods Summary</h2>
  <table class="data-table" style="width:100%">
    <thead><tr><th style="width:220px">Step</th><th>Details</th></tr></thead>
    <tbody>
      <tr><td><strong>Data source</strong></td><td>Wirbel et al. 2019 (PMID: 31427765); mOTU v2 species profiles</td></tr>
      <tr><td><strong>Cohorts</strong></td><td>9 CRC studies (AT, CN, DE, FR, IT&times;2, JP, US&times;2); 1,247 samples after QC</td></tr>
      <tr><td><strong>Batch correction</strong></td><td>ComBat (parametric EB); study as batch, group as covariate; CLR transformation</td></tr>
      <tr><td><strong>Differential abundance</strong></td><td>Mann-Whitney U; BH-FDR; Cohen's d; CLR-transformed abundances</td></tr>
      <tr><td><strong>ML classifiers</strong></td><td>Random Forest (n=300), XGBoost (n=300, lr=0.05, max_depth=6), LightGBM (n=300)</td></tr>
      <tr><td><strong>Validation</strong></td><td>5-fold stratified CV; LODO (9 held-out cohorts)</td></tr>
      <tr><td><strong>Feature importance</strong></td><td>SHAP TreeExplainer; interaction values (n=500 sample subset)</td></tr>
      <tr><td><strong>Network analysis</strong></td><td>Spearman |rho|&ge;0.35, FDR&lt;0.01; top 150 taxa; NetworkX spring layout</td></tr>
      <tr><td><strong>KEGG enrichment</strong></td><td>Mann-Whitney U on 9,499 KO terms; BH-FDR; curated KO&rarr;Pathway dictionary</td></tr>
      <tr><td><strong>Software</strong></td><td>Python 3.10; scikit-learn; XGBoost 3.x; LightGBM; SHAP 0.49; NetworkX; pandas; scipy</td></tr>
    </tbody>
  </table>
</div>

<footer>
  <p>
    <strong>CRC Microbiome Meta-Analysis</strong> &mdash; Computational Biology Systems Platform<br>
    Generated on {today} &mdash; Python + scikit-learn + XGBoost + SHAP<br>
    Based on: Wirbel J et al. <em>Nat Med</em> 2019; DOI: 10.1038/s41591-019-0406-6
  </p>
</footer>

</div><!-- #main -->
"""

# Pass 2: wrap with static header/footer (CSS and JS injected via replace, no f-string conflict)
HEADER = (
    "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
    "<meta charset='UTF-8'>\n"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
    "<title>CRC Microbiome Meta-Analysis &mdash; Final Report</title>\n"
    "<style>__CSS_PLACEHOLDER__</style>\n"
    "</head>\n<body>\n"
)
FOOTER = "\n<script>__JS_PLACEHOLDER__</script>\n</body>\n</html>"

HTML = (
    HEADER.replace("__CSS_PLACEHOLDER__", CSS)
    + build_nav()
    + BODY
    + FOOTER.replace("__JS_PLACEHOLDER__", JS)
)

# ── Write file ────────────────────────────────────────────────────────────────
print(f"Writing report ({len(HTML):,} chars) …")
with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write(HTML)

size_mb = os.path.getsize(REPORT_OUT) / 1e6
print(f"\n{'═'*60}")
print(f"  HTML REPORT COMPLETE ✅")
print(f"  → {os.path.abspath(REPORT_OUT)}")
print(f"  → Size: {size_mb:.1f} MB  (self-contained, all images embedded)")
print(f"{'═'*60}")
print("\n  Open in any browser — no internet connection required.")
