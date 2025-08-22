import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
from source.IO import load_spectra


def train_test_split(data_dir: Path, test_size: float = 0.2, random_state: int = 42):
    """
    Loads the spectra and labels from the given data directory,
    then splits them into training and test sets.
    Saves the split data into a new directory called "train_test_split" in data_dir.
    """
    loaded = load_spectra(data_dir)
    spectra: pd.DataFrame = loaded["spectra"]
    labels: pd.DataFrame = loaded["labels"]

    spectra_train, spectra_test, labels_train, labels_test = sk_train_test_split(
        spectra, labels, test_size=test_size, random_state=random_state
    )

    # save the split data
    out_dir = Path(data_dir) / "train_test_split"
    os.makedirs(out_dir, exist_ok=True)

    # export spectra_train
    path_spectra_train = Path(out_dir) / "spectra_train.csv"
    path_spectra_test = Path(out_dir) / "spectra_test.csv"
    spectra_train.to_csv(path_spectra_train, index=False)
    spectra_test.to_csv(path_spectra_test, index=False)
    print(
        f"Saved spectra train to {path_spectra_train} with shape {spectra_train.shape}"
    )
    path_labels_train = Path(out_dir) / "labels_train.csv"
    path_labels_test = Path(out_dir) / "labels_test.csv"
    labels_train.to_csv(path_labels_train, index=False)
    labels_test.to_csv(path_labels_test, index=False)
    print(f"Saved labels train to {path_labels_train} with shape {labels_train.shape}")


if __name__ == "__main__":
    data_dir = Path("data") / "cleaned_up_version"
    print(f"Splitting data in {data_dir} into train and test sets.")
    train_test_split(data_dir, test_size=0.2, random_state=42)
