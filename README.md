# hsd-demo

## Overview

This repository contains code and data for an ongoing project in hate speech detection which aims to mitigate high false positive rates due to class imbalance. We focus on resampling techniques that aim to rebalance binary and multiclass distributions by focusing exclusively on data preprocessing. We validate the resampling techniques with binary and multiclass (OVR) SVM classification. Warning: the data contains graphic and hateful content. 

## Key Files

### Code

- **`getdistrib.py`**
  This script aims to read in the test and train files and provide counts for binary labels and multiclass labels along test and training splits. 

### Data Files
- **`ghc_train.tsv`** and **`ghc_test.tsv`**  
  Training and testing datasets from the Gab Hate Speech Corpus: https://osf.io/edua3/. Each column in this file is populated with 'text' (Gab posts), 'hd', 'cv' and 'vo' (populated with 1s and 0s). 

### First Preprocessing Step
Run `getdistrib.py` to generate label counts for binary and multilabel class distributions. 

  
