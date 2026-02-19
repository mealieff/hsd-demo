"""
Train a binary SVM classifier on the Gab Hate Corpus.
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
    df = pd.read_csv(path, sep="\t")
    for col in ("text", "hd", "cv", "vo"):
        if col not in df.columns:
            raise FileNotFoundError(
                f"{path} must have columns: text, hd, cv, vo. See data/README.md."
            )
    return df


def binary_label(row):
    """Non-hate = [000] or [001]; hate = anything else (VO alone is not counted as hate)."""
    t = row[["hd", "cv", "vo"]].tolist()
    return "non-hate" if t in [[0, 0, 0], [0, 0, 1]] else "hate"


def main():
    if not os.path.isfile(TRAIN_FILE) or not os.path.isfile(TEST_FILE):
        print(
            f"Missing data. Place ghc_train.tsv and ghc_test.tsv in {DATA_DIR}/ (see data/README.md)."
        )
        return

    print("Loading data...")
    train = load_data(TRAIN_FILE)
    test = load_data(TEST_FILE)
    train["y"] = train[["hd", "cv", "vo"]].apply(binary_label, axis=1)
    test["y"] = test[["hd", "cv", "vo"]].apply(binary_label, axis=1)

    X_train = train["text"].astype(str).fillna("")
    y_train = train["y"]
    X_test = test["text"].astype(str).fillna("")
    y_test = test["y"]

    print("Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=50000, sublinear_tf=True, min_df=2)
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)

    print("Training SVM (LinearSVC)...")
    clf = LinearSVC(max_iter=2000, random_state=42)
    clf.fit(X_train_tf, y_train)

    y_pred = clf.predict(X_test_tf)
    print("\nTest set performance:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
