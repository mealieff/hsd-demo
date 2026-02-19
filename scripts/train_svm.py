"""
Train a binary SVM classifier on the Gab Hate Corpus.

INSTRUCTIONS
------------
1. load_data(path): Read the TSV at path with pandas (sep='\t'). Check that columns
   'text', 'hd', 'cv', 'vo' exist; if not, raise an error. Return the DataFrame.

2. binary_label(row): Map a row to a single binary label. Treat [hd, cv, vo] as
   [0,0,0] or [0,0,1] -> 'non-hate'; any other combination -> 'hate'. Return the string.

3. main(): (a) Define DATA_DIR = "data", TRAIN_FILE and TEST_FILE under it.
   (b) If either file is missing, print a message and return.
   (c) Load train and test with load_data; add a 'y' column using binary_label on
       columns ['hd','cv','vo'].
   (d) Build X_train, y_train, X_test, y_test from the 'text' and 'y' columns
       (fill NaN text with "").
   (e) Fit a TfidfVectorizer (e.g. max_features=50000, sublinear_tf=True, min_df=2)
       on X_train; transform train and test to get X_train_tf, X_test_tf.
   (f) Fit a LinearSVC(max_iter=2000, random_state=42) on X_train_tf, y_train.
   (g) Predict on X_test_tf and print classification_report and confusion_matrix.

See solutions/train_svm.py for a reference implementation.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "ghc_train.tsv")
TEST_FILE = os.path.join(DATA_DIR, "ghc_test.tsv")


def load_data(path):
    """Load TSV; ensure columns text, hd, cv, vo exist. Return DataFrame."""
    raise NotImplementedError("Implement: read CSV with sep='\t', validate columns, return df.")


def binary_label(row):
    """Return 'non-hate' for [000] or [001], else 'hate'. Row has hd, cv, vo."""
    raise NotImplementedError("Implement: map row[['hd','cv','vo']] to 'non-hate' or 'hate'.")


def main():
    """Load data, add binary y, TF-IDF, train LinearSVC, print report and confusion matrix."""
    raise NotImplementedError("Implement the full pipeline as described in the module docstring.")


if __name__ == "__main__":
    main()
