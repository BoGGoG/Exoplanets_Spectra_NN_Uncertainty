import os
import shutil
from pathlib import Path

import lightning as L
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, Dataset, random_split

from source.IO import read_config
from source.models.Model01 import CNNTimeSeriesRegressor, Model_Lit, model_registry
from source.training.optimizers import cosine_decay_scheduler
from source.utils import var_names
from scipy.interpolate import interp1d

plt.style.use(["science", "grid"])
mpl.rcParams.update(
    {
        # Fonts
        "font.family": "sans-serif",  # "sans-serif" or "serif"
        "font.serif": [
            "Times New Roman"
        ],  # Or ["Computer Modern Roman"] for LaTeX style
        "mathtext.fontset": "cm",  # Computer Modern for math
        "font.size": 14,  # Base font size (good for papers)
        # Axes
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "axes.linewidth": 1.8,
        # Ticks
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        # Legend
        "legend.fontsize": 14,
        "legend.frameon": False,
        # Lines
        "lines.linewidth": 3.2,
        "lines.markersize": 6,
        # Savefig
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

if __name__ == "__main__":
    wasp39b_file = Path("data") / "observation_wasp39b.npy"
    ensemble_savedir = (
        Path("runs")
        / "2025_09_10"
        / "out"
        / "ensemble"
        / "ensemble_3"
        / "models"
        / "ckpt"
    )
    plots_savedir = ensemble_savedir.parent.parent / "plots" / "wasp39b"
    os.makedirs(plots_savedir, exist_ok=True)
    models_paths = os.listdir(ensemble_savedir)
    models_paths = [f for f in models_paths if f.endswith(".ckpt")]
    models_paths = sorted(models_paths)
    models_paths = [ensemble_savedir / f for f in models_paths]

    models = []
    for m in models_paths:
        print(f"Loading model from {m}")
        model = Model_Lit.load_from_checkpoint(checkpoint_path=m)
        models.append(model)

    print(f"Loaded {len(models)} models")

    wasp39b_data = np.load(wasp39b_file, allow_pickle=True)
    wasp39b_wavelengths = wasp39b_data[:, 0]
    wasp39b_flux = wasp39b_data[:, 1]
    train_wavelengths_file = Path("data") / "cleaned_up_version" / "wavelengths.npy"
    train_wavelengths = np.load(train_wavelengths_file)

    f_interp = interp1d(
        wasp39b_wavelengths,
        wasp39b_flux,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )
    # todo: try savgol filter for smoothing
    wasp39b_flux_resampled = f_interp(train_wavelengths)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plt.plot(
        wasp39b_wavelengths,
        wasp39b_flux,
        color="b",
        label="WASP-39b Observation",
        marker="o",
        markersize=2,
        alpha=0.3,
        linewidth=1.5,
    )
    plt.plot(
        train_wavelengths,
        wasp39b_flux_resampled,
        color="r",
        label="Resampled",
        marker="o",
        linestyle="-",
        markersize=4,
        linewidth=1.5,
    )
    plt.xlim(0, 8)
    plt.ylim(0.0205, 0.0229)
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("flux")
    plt.legend()
    plotpath = plots_savedir / "wasp39b_resampled.png"
    plt.savefig(plotpath)
    plt.close()
    print(f"Saved plot to {plotpath}")

    # mo
    scaler_mean = np.array(
        [
            1190.83726355,
            -6.00609932,
            -6.50023774,
            -5.99933525,
            -4.49630322,
            -6.49604898,
        ]
    )
    scaler_var = np.array(
        [
            4.40644566e05,
            3.00482659e00,
            2.08444992e00,
            3.02831506e00,
            7.46910864e-01,
            2.07242987e00,
        ]
    )

    # we only have a single observation
    mus_pred = []
    sigmas2_pred = []
    x_obs = torch.tensor(wasp39b_flux_resampled, dtype=torch.float32).unsqueeze(0)
    print(x_obs.shape)

    device = torch.device("cpu")
    for model in models:
        model = model.to(device)
        model.eval()
        prediction = model(x_obs[:, 1:])
        mus_pred.append(prediction[0].detach().cpu().numpy())
        sigmas2_pred.append(prediction[1].detach().cpu().numpy())

    mus_pred = np.array(mus_pred).squeeze(1)
    sigmas2_pred = np.array(sigmas2_pred).squeeze(1)
    print(mus_pred.shape, sigmas2_pred.shape)
    print("mus_pred:", mus_pred)
    print("sigmas2_pred:", sigmas2_pred)

    # rescale back to original
    mus_pred_rescaled = mus_pred * np.sqrt(scaler_var) + scaler_mean
    sigmas2_pred_rescaled = sigmas2_pred * scaler_var

    mus_mean = np.mean(mus_pred_rescaled, axis=0)
    aleatoric_sigma2 = np.mean(sigmas2_pred_rescaled, axis=0)
    epistemic_sigma2 = np.var(mus_pred_rescaled, axis=0)

    print("Predictions:")
    for i, var in enumerate(var_names):
        print(
            f"{var:<10}: {mus_mean[i]:.3f} (+/- {np.sqrt(aleatoric_sigma2[i]):.3f} (aleat.) +/- {np.sqrt(epistemic_sigma2[i]):.3f} (epist.))"
        )

    # save to a text file in plots_savedir
    summary_file = plots_savedir / "wasp39b_inference_ensemble.txt"
    with open(summary_file, "w") as f:
        f.write("WASP-39b Inference with Ensemble of Models\n")
        f.write("=" * 50 + "\n")
        for i, var in enumerate(var_names):
            f.write(
                f"{var:<10}: {mus_mean[i]:.3f} (+/- {np.sqrt(aleatoric_sigma2[i]):.3f} (aleat.) +/- {np.sqrt(epistemic_sigma2[i]):.3f} (epist.))\n"
            )

    # for each variable, plot a point (the mu) together with error bars (aleatoric and epistemic)
    # also name the variables on the x-axis
    predictions_forestano = [0.0, -6.68, -4.50, -7.98, -2.8, -10.36]
    # predictions_literature = [0., -5.94, -6.59, -5.3, -4.25, -6]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="constrained")
    for i, var in enumerate(var_names[1:]):
        i += 1  # skip the first variable
        plt.errorbar(
            i,
            mus_mean[i],
            yerr=np.sqrt(aleatoric_sigma2[i]),
            fmt="o",
            color="b",
            label="Aleatoric Uncertainty" if i == 1 else "",
            capsize=5,
            markersize=4,
            linewidth=2,
        )
        plt.errorbar(
            i,
            mus_mean[i],
            yerr=np.sqrt(epistemic_sigma2[i]),
            fmt="o",
            color="r",
            label="Epistemic Uncertainty" if i == 1 else "",
            capsize=5,
            linewidth=2,
            markersize=4,
        )
        plt.scatter(
            i,
            predictions_forestano[i],
            color="g",
            marker="x",
            s=100,
            label="Forestano et al. (2025)" if i == 1 else "",
        )
    # remove ticks and add variable names
    plt.legend()
    plt.title("WASP-39b Inference with Ensemble of Models")
    plt.xticks(range(1, len(var_names)), var_names[1:], rotation=45, ha="right")
    plt.ylabel("log concentration")
    plotpath = plots_savedir / "wasp39b_inference_ensemble.png"
    plt.savefig(plotpath)
    plt.close()
    print(f"Saved plot to {plotpath}")
