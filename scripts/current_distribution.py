"""
Report binary and multiclass label distributions for the train and test TSVs.

INSTRUCTIONS
------------
1. load_and_validate_dataset(file_path): Read the TSV with pandas (sep='\t').
   Check that columns 'text', 'hd', 'cv', 'vo' exist; if not, raise an error. Return DataFrame.

2. classify_binary_label(row): Given a row with columns hd, cv, vo, return 'non-hate'
   if the triple is [0,0,0] or [0,0,1], else 'hate'.

3. classify_multiclass_label(row): Return a string of the triple (e.g. '010' for hd=0,cv=1,vo=0).

4. count_label_combinations(data, dataset_name): Print multiclass counts for the 8
   combinations {}; {HD}; {CV}; {VO}; {HD,CV}; {HD,VO}; {CV,VO}; {HD,CV,VO} (map each
   to the corresponding [hd,cv,vo] and count in data).

5. print_label_distribution(data, dataset_name): Assume data has 'binary_label' and
   'multiclass_label'. Print value_counts for both, then call count_label_combinations.

6. main(): Use data_dir = "data", train/test paths = data_dir/ghc_train.tsv, ghc_test.tsv.
   If files missing, print message and return. Load both; add binary_label and
   multiclass_label columns; call print_label_distribution for train and test.

See solutions/current_distribution.py for a reference implementation.
"""
import pandas as pd


def load_and_validate_dataset(file_path):
    """Load TSV and validate columns text, hd, cv, vo. Return DataFrame."""
    raise NotImplementedError("Implement: read CSV, check columns, return data.")


def classify_binary_label(row):
    """Return 'non-hate' for [000] or [001], else 'hate'."""
    raise NotImplementedError("Implement binary label from row hd, cv, vo.")


def classify_multiclass_label(row):
    """Return string of hd,cv,vo (e.g. '101')."""
    raise NotImplementedError("Implement: concatenate hd, cv, vo as string.")


def count_label_combinations(data, dataset_name):
    """Print counts for each of the 8 hd/cv/vo combinations, labeled as {}; {HD}; etc."""
    raise NotImplementedError("Implement: loop 8 combinations, count in data, print.")


def print_label_distribution(data, dataset_name):
    """Print binary and multiclass value_counts, then count_label_combinations."""
    raise NotImplementedError("Implement: print value_counts for binary_label and multiclass_label, then count_label_combinations.")


def main():
    """Load train/test from data/, add labels, print distributions for both."""
    raise NotImplementedError("Implement as described in the module docstring.")


if __name__ == "__main__":
    main()
