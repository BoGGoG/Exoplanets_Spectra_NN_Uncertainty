import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_spectra(data_dir: Path, verbose: bool = True):
    path_labels = data_dir / "labels.csv"
    path_spectra = data_dir / "spectra.csv"
    path_wavelengths = data_dir / "wavelengths.npy"

    spectra = pd.read_csv(path_spectra, index_col=0)
    labels = pd.read_csv(path_labels)
    wavelengths = np.load(path_wavelengths)

    print(f"Loaded spectra with shape: {spectra.shape}")
    print(f"Loaded labels with shape: {labels.shape}")
    print(f"Loaded wavelengths with shape: {wavelengths.shape}")

    return {"spectra": spectra, "labels": labels, "wavelengths": wavelengths}


if __name__ == "__main__":
    data_dir = Path("data") / "cleaned_up_version"
    loaded = load_spectra(data_dir)
    spectra = loaded["spectra"]
    labels = loaded["labels"]
    wavelengths = loaded["wavelengths"]

    print(labels.head())
    print(labels.columns)

    plot_dir = Path("data") / "cleaned_up_version_plots"
    os.makedirs(plot_dir, exist_ok=True)
    for i in range(20):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(spectra.iloc[i, :].values, label=f"{i=}")
        plt.xlabel("bin")
        plt.ylabel("Intensity?")
        title_labels = [f"{l:.4}" for l in labels.iloc[i, :].values if not pd.isna(l)]
        title_labels = ", ".join(title_labels)
        plt.title(f"Spectrum {i}: {title_labels}")
        path = plot_dir / f"spectrum_{i}.png"
        plt.savefig(path)
