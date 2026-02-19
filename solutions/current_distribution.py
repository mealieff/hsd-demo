# This script extracts binary and multiclass labels per test/train file
import pandas as pd

#  This reads in the dataset and ensure the columns are correct
def load_and_validate_dataset(file_path):
    data = pd.read_csv(file_path, sep='\t')
    required_columns = {'text', 'hd', 'cv', 'vo'}
    if not required_columns.issubset(data.columns):
        raise Exception(f"The input file {file_path} must contain 'text', 'hd', 'cv', and 'vo' columns.")
    return data

# This classifies row as 'non-hate' or 'hate' based on binary labels so that VO (vulgar language) is not looped in which hate speech.
# Binary nonhate  = [000] or [001]; Binary hate = everything else
def classify_binary_label(row):
    return 'non-hate' if row.tolist() in [[0, 0, 0], [0, 0, 1]] else 'hate'

# This classififies representation of multiclass labels by considering each row as a series of three 0 and 1 combinations.
def classify_multiclass_label(row):
    return ''.join(map(str, row.tolist()))

def count_label_combinations(data, dataset_name):
    print(f"\n{dataset_name} Specific Multiclass Label Counts:")
    label_map = {
        '{}': [0, 0, 0],
        '{HD}': [1, 0, 0],
        '{CV}': [0, 1, 0],
        '{VO}': [0, 0, 1],
        '{HD, CV}': [1, 1, 0],
        '{HD, VO}': [1, 0, 1],
        '{CV, VO}': [0, 1, 1],
        '{HD, CV, VO}': [1, 1, 1]
    }
    for label, combination in label_map.items():
        count = (data[['hd', 'cv', 'vo']].values.tolist().count(combination))
        print(f"{label}: {count}")

def print_label_distribution(data, dataset_name):
    print(f"\n{dataset_name} Label Distributions:")
    print("Binary Labels:")
    print(data['binary_label'].value_counts())
    print("\nMulticlass Labels:")
    print(data['multiclass_label'].value_counts())
    count_label_combinations(data, dataset_name)

def main():
    import os
    data_dir = "data"
    train_file = os.path.join(data_dir, "ghc_train.tsv")
    test_file = os.path.join(data_dir, "ghc_test.tsv")
    if not os.path.isfile(train_file) or not os.path.isfile(test_file):
        print(f"Place ghc_train.tsv and ghc_test.tsv in {data_dir}/ (see data/README.md).")
        return

    # Load the datasets  
    print("Loading datasets...")
    train_data = load_and_validate_dataset(train_file)
    test_data = load_and_validate_dataset(test_file)

    # Apply classification as either binary or multiclass
    train_data['binary_label'] = train_data[['hd', 'cv', 'vo']].apply(classify_binary_label, axis=1)
    test_data['binary_label'] = test_data[['hd', 'cv', 'vo']].apply(classify_binary_label, axis=1)
    train_data['multiclass_label'] = train_data[['hd', 'cv', 'vo']].apply(classify_multiclass_label, axis=1)
    test_data['multiclass_label'] = test_data[['hd', 'cv', 'vo']].apply(classify_multiclass_label, axis=1)

    # Print thedistributions
    print_label_distribution(train_data, "Training Set")
    print_label_distribution(test_data, "Test Set")

if __name__ == "__main__":
    main()
