# hsd-demo

## Overview

This repository contains code and data for an ongoing project in hate speech detection which aims to mitigate high false positive rates due to class imbalance. We focus on resampling techniques that aim to rebalance binary and multiclass distributions by focusing exclusively on data preprocessing. We validate the resampling techniques with binary and multiclass (OVR) SVM classification. 

_Warning:_ the data contains graphic and hateful content. 

## Key Files

### Code

- **`getdistrib.py`**
  This script aims to read in the test and train files and provide counts for binary labels and multiclass labels along test and training splits. 

### Data Files
- **`ghc_train.tsv`** and **`ghc_test.tsv`**  
  Training and testing datasets from the Gab Hate Speech Corpus: https://osf.io/edua3/. Each column in this file is populated with 'text' (Gab posts), 'hd' (hate speech), 'cv' (calls for violence), and 'vo' (vulgar language). Each tab-delimited column is populated with 1s and 0s if the category of offensive language is present. 

### References
Kennedy, B., Atari, M., Davani, A.M. et al. (2022) Introducing the Gab Hate Corpus: defining and applying hate-based rhetoric to social media posts at scale. Lang Resources & Evaluation 56, 79–108.

### First Preprocessing Step
Run `getdistrib.py` to generate label counts for binary, which gathers label distributions into two categories: hate and non hate, and multilabel class distributions which considers every possible label distribution. This script reports distributions for training and test sets. 

  
