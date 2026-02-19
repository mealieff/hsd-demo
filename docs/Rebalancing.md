# Rebalancing techniques

Imbalanced classes (e.g. few hate, many non-hate) push models toward predicting the majority. Rebalancing the training set or the loss fixes that. All helpers live in **scripts/rebalance.py**.

| Technique | What it does | Use when |
|-----------|--------------|----------|
| **Random oversample** | Duplicate minority examples until class sizes match. Works with sparse X. | You have sparse features (TF-IDF) and don’t mind duplicates. |
| **Random undersample** | Drop majority examples until class sizes match. No new points. | You’re okay discarding data; keeps runtime small. |
| **SMOTE** | Add synthetic minority points by interpolating between neighbors. Needs **dense** X. | You want synthetic diversity and can afford dense X (imblearn). |
| **ADASYN** | Like SMOTE; more synthetic points where minority is harder to separate. Dense X. | Same as SMOTE, with focus on hard regions. |
| **Class weighting** | Reweight loss so minority errors cost more. No resampling. | You want to keep all data and avoid changing sample count. |

**Code (apply to training data only; never resample the test set):**

- `random_oversample(X, y)` / `random_undersample(X, y)` or `balanced_resample(X, y, strategy='over'|'under')`
- `smote_resample(X, y, k_neighbors=5)` / `adasyn_resample(X, y, n_neighbors=5)` — X dense, e.g. `X.toarray()`
- `LinearSVC(class_weight=svm_class_weight(y_train), ...)` for weighting only

**Inspect imbalance:** `class_counts(y)` and `class_indices(y)`.

---

**Documentation**

- **scikit-learn:** [resample](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html) · [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) · [compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
- **imbalanced-learn:** [over_sampling](https://imbalanced-learn.org/stable/references/over_sampling.html) · [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) · [ADASYN](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html)
