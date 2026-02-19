"""
Dataset balancing utilities for binary (and optionally multiclass) imbalanced data.
- Random over/undersampling: sklearn only (works with dense and sparse X).
- SMOTE / ADASYN: imbalanced-learn (imblearn); require dense X for large feature spaces.
"""
import numpy as np
from sklearn.utils import resample

try:
    from scipy.sparse import issparse, vstack as sparse_vstack
except ImportError:
    issparse = lambda x: False
    sparse_vstack = None


def class_indices(y):
    """Return a dict mapping each unique label to indices where y equals that label."""
    y = np.asarray(y)
    return {label: np.flatnonzero(y == label) for label in np.unique(y)}


def class_counts(y):
    """Return a dict mapping each unique label to its count."""
    return {k: len(ind) for k, ind in class_indices(y).items()}


def _resample_class(X, y, indices, n_samples, replace, random_state):
    """Resample one class by index; return (X_part, y_part)."""
    i = resample(
        indices, replace=replace, n_samples=n_samples, random_state=random_state
    )
    X_part = X[i] if not issparse(X) else X[i].copy()
    y_part = np.asarray(y)[i]
    return X_part, y_part


def _stack(X_parts, y_parts):
    """Concatenate (X_parts, y_parts) handling sparse X."""
    y_out = np.concatenate(y_parts, axis=0)
    if issparse(X_parts[0]):
        X_out = sparse_vstack(X_parts)
    else:
        X_out = np.vstack(X_parts)
    return X_out, y_out


def random_oversample(X, y, random_state=None):
    """
    Oversample minority class(es) so each class has as many samples as the majority.
    Minority samples are drawn with replacement.
    """
    idx = class_indices(np.asarray(y))
    if len(idx) == 0:
        return X, y
    n_max = max(len(ind) for ind in idx.values())
    X_parts, y_parts = [], []
    for label, ind in idx.items():
        n = len(ind)
        X_part, y_part = _resample_class(
            X, y, ind, n_samples=n_max, replace=(n < n_max), random_state=random_state
        )
        X_parts.append(X_part)
        y_parts.append(y_part)
    return _stack(X_parts, y_parts)


def random_undersample(X, y, random_state=None):
    """
    Undersample majority class(es) so each class has as many samples as the minority.
    Majority samples are drawn without replacement.
    """
    idx = class_indices(np.asarray(y))
    if len(idx) == 0:
        return X, y
    n_min = min(len(ind) for ind in idx.values())
    X_parts, y_parts = [], []
    for label, ind in idx.items():
        X_part, y_part = _resample_class(
            X, y, ind, n_samples=n_min, replace=False, random_state=random_state
        )
        X_parts.append(X_part)
        y_parts.append(y_part)
    return _stack(X_parts, y_parts)


def balanced_resample(
    X, y, strategy="over", random_state=None
):
    """
    Resample (X, y) to balance class counts.

    strategy : 'over' | 'under'
        - 'over': oversample minority to majority size (random_oversample).
        - 'under': undersample majority to minority size (random_undersample).
    """
    if strategy == "over":
        return random_oversample(X, y, random_state=random_state)
    if strategy == "under":
        return random_undersample(X, y, random_state=random_state)
    raise ValueError("strategy must be 'over' or 'under'")


def smote_resample(X, y, random_state=None, k_neighbors=5, **kwargs):
    """
    Oversample minority class(es) using SMOTE (Synthetic Minority Over-sampling).
    Requires imbalanced-learn. X must be dense (use .toarray() on sparse if needed).
    """
    from imblearn.over_sampling import SMOTE
    X = np.asarray(X) if not issparse(X) else X.toarray()
    y = np.asarray(y).ravel()
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors, **kwargs)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def adasyn_resample(X, y, random_state=None, n_neighbors=5, **kwargs):
    """
    Oversample minority class(es) using ADASYN (Adaptive Synthetic Sampling).
    Requires imbalanced-learn. X must be dense (use .toarray() on sparse if needed).
    """
    from imblearn.over_sampling import ADASYN
    X = np.asarray(X) if not issparse(X) else X.toarray()
    y = np.asarray(y).ravel()
    adasyn = ADASYN(random_state=random_state, n_neighbors=n_neighbors, **kwargs)
    X_res, y_res = adasyn.fit_resample(X, y)
    return X_res, y_res


def svm_class_weight(y):
    """
    Return class_weight for use with LinearSVC (or SVC) to balance by loss weighting.
    Usage: LinearSVC(class_weight=svm_class_weight(y_train), ...).
    Avoids resampling; use when you want to keep all data and reweight the loss.
    """
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y).ravel())
    w = compute_class_weight(
        "balanced", classes=np.unique(y_enc), y=y_enc
    )
    return dict(zip(le.classes_, w))
