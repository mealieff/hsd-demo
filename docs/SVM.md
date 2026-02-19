# SVMs: single and binary labels

## What is an SVM?

A **Support Vector Machine (SVM)** is a classifier that finds a separating boundary (e.g. a hyperplane in feature space) so that the margin between the two classes is as large as possible. Points closest to the boundary are the “support vectors.” SVMs work well with high-dimensional features (e.g. TF-IDF over many terms) and are robust when classes are well separated.

## Single label per example

In **single-label** classification, each example has exactly one label. The model predicts one class per instance.

- **Binary**: two classes (e.g. “hate” vs “non-hate”). One binary label per example; the SVM learns one decision boundary.
- **Multiclass**: more than two classes. Each example still has one label. Common approaches:
  - **One-vs-rest (OvR)**: train one binary SVM per class (that class vs all others), then assign the class whose classifier gives the highest score.
  - **One-vs-one (OvO)**: train a binary SVM for every pair of classes; prediction by majority vote.

This demo uses **binary single-label**: each post has one target, “hate” or “non-hate,” and we train a single binary SVM.

## Binary labels in this demo

The dataset has three binary indicators per row: **hd** (hate), **cv** (calls for violence), **vo** (vulgar). We map them to a **single binary label**:

- **non-hate**: `[hd=0, cv=0, vo=0]` or `[0,0,1]` (vulgar only).
- **hate**: any other combination (e.g. hate only, hate+violence, etc.).

So we reduce three binary columns to one binary target; the SVM then learns to predict that single label (hate vs non-hate) from the text features (e.g. TF-IDF).

## Summary

| Setting        | Meaning                          | This demo        |
|----------------|----------------------------------|------------------|
| Single label   | One label per example            | Yes (one target) |
| Binary         | Two classes                      | Yes (hate / non-hate) |
| SVM role       | One binary classifier            | LinearSVC on TF-IDF |

To extend to **multiclass** (e.g. 8 combinations of hd/cv/vo), you would keep one label per example but use more than two classes and train with something like `LinearSVC(..., multi_class='ovr')` or an OvO scheme.
