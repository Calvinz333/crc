# Colorectal Cancer (CRC) Microbiome Analysis Pipeline

This repository contains a comprehensive machine learning and statistical analysis pipeline for identifying microbial signatures associated with Colorectal Cancer (CRC) across multiple geographic cohorts. The project leverages cross-cohort validation, ensemble machine learning models, and advanced feature importance interpretations (SHAP) to discover robust, generalizable microbiome biomarkers.

## 🧬 Project Overview

The core objective of this project is to build a robust, interpretable machine learning pipeline that can differentiate CRC patients from healthy controls based on their gut microbiome profiles. 

**Key Methodologies:**
* **Leave-One-Dataset-Out (LODO) Cross-Validation:** To ensure models generalize across different geographic cohorts rather than simply memorizing dataset-specific batch effects.
* **Ensemble Machine Learning:** Utilizing advanced gradient boosting frameworks (XGBoost, LightGBM) and Random Forests.
* **Explainable AI (SHAP):** Extracting real-world importance and directionality of specific microbial taxa.
* **Microbial Network Analysis:** Mapping the co-occurrence networks in healthy vs. CRC disease states.
* **Pathway Enrichment:** Identifying functional KEGG pathways associated with structural shifts.

## 📁 Repository Structure

* `scripts/` - Core analysis modules.
  * `preprocessing/` - Batch correction and data normalization.
  * `modeling/` - LODO cross-validation and ensemble model training.
  * `analysis/` - Differential abundance, network analysis, KEGG pathway enrichment, novelty discovery, and figure generation.
* `reports/` - Output directory for HTML analysis reports, manuscript drafts, and generated PDFs.
* `figures/` - Output directory for exploratory, ML, and publication-ready visualizations.
* `notebooks/` - Jupyter notebooks for interactive exploratory data analysis.
* `setup_env.sh` - Shell script to initialize the project directory and Conda environment.
* `qiime2-amplicon-2024.10.yml` - Conda environment definition for reproducibility.

*(Note: The `data/` and `models/` directories are intentionally excluded from version control to protect data privacy and save repository space.)*

## ⚙️ Setup & Installation

This project relies on the QIIME2 framework alongside a standard data science Python stack.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Calvinz333/crc.git
   cd crc
   ```

2. **Set up the Conda Environment:**
   You can recreate the exact environment used for this analysis using the provided shell script or YAML file:
   ```bash
   bash setup_env.sh
   # OR
   conda env create -f qiime2-amplicon-2024.10.yml
   conda activate qiime2-amplicon-2024.10
   ```

## 🚀 Pipeline Execution

The pipeline is entirely modular. Scripts can be executed independently, assuming the requisite pre-processed data is available in the `data/` directory.

1. **Exploratory Analysis & Preprocessing:** 
   `python scripts/analysis/exploratory_analysis.py`
   `python scripts/preprocessing/batch_correction.py`
2. **Differential Abundance & Network Analysis:**
   `python scripts/analysis/differential_abundance.py`
   `python scripts/analysis/network_analysis.py`
3. **Machine Learning Modeling:**
   `python scripts/modeling/lodo_cv.py`
   `python scripts/modeling/ensemble_model.py`
4. **Figure & Report Generation:**
   `python scripts/analysis/generate_pub_figures.py`
   `python scripts/analysis/generate_html_report.py`
   `python scripts/analysis/make_pdf_v2.py`

## 📊 Key Outputs
The pipeline automatically structures outputs into the `reports/` and `figures/` folders, generating publication-ready ROC curves, SHAP beeswarm plots, taxonomic boxplots, geographic attenuation analyses, and a compiled interactive HTML document summarizing the findings.
