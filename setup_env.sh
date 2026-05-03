#!/bin/bash
set -e

# Define variables
OS_TYPE=$(uname -s)
QIIME2_VER="2024.10"
ENV_NAME="qiime2-amplicon-${QIIME2_VER}"

# Determine correct URL based on OS
if [ "$OS_TYPE" == "Darwin" ]; then
    echo "Detected macOS system."
    # User confirmed Intel Mac, so we use the standard osx-64 URL.
    # Correct URL for QIIME2 2024.10 Amplicon on macOS (Intel)
    ENV_URL="https://data.qiime2.org/distro/amplicon/qiime2-amplicon-2024.10-py310-osx-conda.yml"
elif [ "$OS_TYPE" == "Linux" ]; then
    echo "Detected Linux system."
    ENV_URL="https://data.qiime2.org/distro/amplicon/qiime2-amplicon-${QIIME2_VER}-py39-linux-conda.yml"
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

echo "Downloading environment file from: $ENV_URL"
curl -L "$ENV_URL" -o "${ENV_NAME}.yml"

# Fix dependency conflict: remove deblur and sortmerna (deblur requires sortmerna 2.0 which fails on some Mac Intel setups)
# We use DADA2 anyway as per the guide.
echo "Removing deblur and sortmerna from environment file to avoid conflicts..."
sed -i '' '/deblur/d' "${ENV_NAME}.yml"
sed -i '' '/sortmerna/d' "${ENV_NAME}.yml"

echo "Creating Conda environment: $ENV_NAME"
# Create env from file
conda env create -n "$ENV_NAME" --file "${ENV_NAME}.yml"

echo "Installing additional dependencies via pip into $ENV_NAME..."
# We use 'conda run' to execute pip inside the new environment without having to activate it 
# (which can be tricky in scripts depending on shell setup)
conda run -n "$ENV_NAME" pip install \
    scikit-learn \
    xgboost \
    lightgbm \
    shap \
    imbalanced-learn \
    pandas \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    biopython \
    openpyxl

echo "Environment setup complete!"
echo "To activate, run: conda activate $ENV_NAME"
