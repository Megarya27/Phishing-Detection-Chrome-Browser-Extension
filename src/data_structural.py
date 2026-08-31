# This program loads the UCI Phishing Websites datasets and 
# produces the 70:15:15 train/validation/test split used by structural baselines
# (XGBoost, SVM) and by structural stream of hybrid model.


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from global_config import UCI_ARFF, RANDOM_SEED, STRUCTURAL_FEATURES

STRUCTURAL_LABEL_COL = "Result"  # -1 = phishing, 1 = legitimate in the UCI encoding

def load_uci_arff(path=UCI_ARFF) -> pd.DataFrame:
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Find line where data starts and move to the directly following line
    for i, line in enumerate(lines):
        if line.strip().lower() == "@data":
            data_start_index = i + 1
            break

    # Load the data directly using pandas, skipping ARFF metadata headers
    columns = STRUCTURAL_FEATURES + [STRUCTURAL_LABEL_COL]
    return pd.read_csv(path, skiprows=data_start_index, names=columns, header=None)

def to_binary_label(series: pd.Series) -> pd.Series:
    # UCI encodes phishing as -1 and legitimate as 1. This function changes so:
    # 1 = phishing
    # 0 = legitimate
    return series.map({-1: 1, 1: 0})

def load_structural_splits(path=UCI_ARFF, seed=RANDOM_SEED):
    # Loads dataset and splits it so 70% Train, 15% Validation, and 15% Test.
    # The split is stratified. This means ratio of phishing to legit sites
    # is preserved across all three sets.

    df = load_uci_arff(path)
    X = df[STRUCTURAL_FEATURES].astype(float)
    y = to_binary_label(df[STRUCTURAL_LABEL_COL])

    # 1st split: 70% for Training, 30% for Validation & Test
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    
    # 2nd split: Divide the remaining 30% equally into 15% Validation and 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.50, stratify=y_val_test, random_state=seed)
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,}

if __name__ == "__main__":
    splits = load_structural_splits()
    for name, arr in splits.items():
        if hasattr(arr, "shape"):
            print(name, arr.shape)
        else:
            print(name, len(arr))
    print("Train label balance:\n", splits["y_train"].value_counts(normalize=True))