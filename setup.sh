#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create data directory for dataset
mkdir -p data

# Install Python dependencies
pip install -r requirements.txt

echo "Setup complete. Place ghc_train.tsv and ghc_test.tsv in data/ then run: python train_svm.py"
