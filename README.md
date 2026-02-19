# SVM demo: hate speech detection

Minimal demo for training a binary SVM classifier on the **Gab Hate Corpus**.  
*Warning: the dataset contains graphic and hateful content.*

**Background:** [docs/SVM.md](docs/SVM.md) — SVMs and single/binary labels. [docs/Rebalancing.md](docs/Rebalancing.md) — rebalancing techniques (random over/under, SMOTE, ADASYN, class weighting).

## Setup

1. **Create and activate a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/macOS
   # Windows: venv\Scripts\activate
   ```
2. Run the setup script (installs deps into the active env):
   ```bash
   bash setup.sh
   ```
3. Download the dataset from [OSF (Gab Hate Corpus)](https://osf.io/edua3/) and place **ghc_train.tsv** and **ghc_test.tsv** in the **data/** folder.

## Scripts and solutions

- **scripts/** — Partially empty scripts with instructions in docstrings. Implement these to practice the pipeline.
- **solutions/** — Reference implementations. Run or compare against these if stuck (e.g. `python solutions/train_svm.py` from repo root).

## Train the model

From repo root: `python scripts/train_svm.py` (or `cd scripts` then `python train_svm.py`).  
The pipeline loads TSV from `data/`, builds binary labels, TF-IDF, LinearSVC, and prints classification report and confusion matrix.

## Optional: label distribution

To inspect binary and multiclass label counts: `python scripts/current_distribution.py` (or run `solutions/current_distribution.py` for the reference). Reads from `data/ghc_train.tsv` and `data/ghc_test.tsv`.

## Reference

Kennedy, B., Atari, M., Davani, A.M. et al. (2022) Introducing the Gab Hate Corpus: defining and applying hate-based rhetoric to social media posts at scale. *Lang Resources & Evaluation* 56, 79–108.
