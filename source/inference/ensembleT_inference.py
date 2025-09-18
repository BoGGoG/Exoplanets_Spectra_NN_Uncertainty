"""
Like ensemble_inference.py, but for the models that only predict T
"""

import argparse
import configparser
import os
import shutil
import textwrap
from datetime import date
from pathlib import Path

import lightning as L
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna
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
from source.models.ModelT import CNNTimeSeriesRegressorT, ModelT_Lit
from source.training.optimizers import cosine_decay_scheduler
from source.utils import var_names


def load_models(config):
    model_class = config["MODEL"]["model_class"]
    ensemble_directory = (
        Path(config["ENSEMBLE_INFERENCE"]["ENSEMBLE_SAVEDIR"]) / "models" / "ckpt"
    )
    models_names = os.listdir(ensemble_directory)

    # only files that end with .ckpt
    models_names = [m for m in models_names if m.endswith(".ckpt")]

    models = []
    for m in models_names:
        print(f"Loading model from {ensemble_directory / m}")
        model = ModelT_Lit.load_from_checkpoint(checkpoint_path=ensemble_directory / m)
        model.scaler_mean = model.hparams["scaler_mean"]
        model.scaler_var = model.hparams["scaler_var"]
        models.append(model)
    return models


class InferenceDataSet(Dataset):
    def __init__(self, spectra):
        self.spectra = spectra

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        """returns a tuple (spectrum, label), but label is a dummy value here"""
        return torch.tensor(self.spectra[idx], dtype=torch.float32), torch.tensor(
            0, dtype=torch.float32
        )


def darken_cmap(cmap, factor=0.7):
    new_cmap = cmap(np.linspace(0, 1, 256))
    new_cmap[:, :3] *= factor  # scale RGB channels
    return mpl.colors.ListedColormap(new_cmap)


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
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
                                    """),
    )
    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", required=True
    )
    predict = False # only set to True if you want to run the predictions again or for the first time
    args = parser.parse_args()
    config = read_config(args.config)
    if predict:
        models = load_models(config)
        print(f"Loaded {len(models)} models for ensemble inference.")

    inference_data_path = config["ENSEMBLE_INFERENCE"]["inference_data_path"]
    print(f"Loading inference data from {inference_data_path}")
    nrows = config.getint("ENSEMBLE_INFERENCE", "n_events")
    nrows = None if nrows == -1 else nrows
    spectra = pd.read_csv(
        inference_data_path,
        # index_col=0,
        nrows=nrows,
    ).values
    print(f"Loaded inference data with shape {spectra.shape}")
    inference_dataset = InferenceDataSet(spectra)
    inference_dataloader = DataLoader(
        inference_dataset, batch_size=256, shuffle=False, drop_last=False
    )
    predictions_dir = (
        Path(config["ENSEMBLE_INFERENCE"]["ENSEMBLE_SAVEDIR"]) / "test_predictions"
    )
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / "ensemble_predictions.npz"

    if predict:
        trainer = Trainer()
        predictions = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for m in models:
            m.to(device)
            pred = trainer.predict(m, dataloaders=inference_dataloader)
            m.to("cpu")
            mus_pred = torch.cat([p[0].cpu() for p in pred]).numpy()
            sigmas2_pred = torch.cat([p[1].cpu() for p in pred]).numpy()  # sigmas^2
            output = np.array([mus_pred, sigmas2_pred])
            predictions.append(output)
        predictions = np.array(predictions)
        predictions = predictions.transpose(
            0, 2, 3, 1
        )  # [model, sample, output_dim, (mu,sigma)]

        # save predictions
        np.savez_compressed(predictions_path, predictions=predictions)
        print(f"Saved predictions to {predictions_path}")

    # load predictions
    predictions = np.load(predictions_path)["predictions"]


    # rescale all predictons

    # raise NotImplementedError

    # for the predictions, take the mean of the mus
    # for the aleatoric variance, take the mean of the sigmas^2
    # for the epistemic variance, take the variance of the mus
    y_pred = predictions[:, :, :, 0].mean(axis=0)
    aleatoric_var = predictions[:, :, :, 1].mean(axis=0)
    epistemic_var = predictions[:, :, :, 0].var(axis=0)

    # since here we are only predicting T, the third dimension has size 1
    y_pred = y_pred[:, 0]
    aleatoric_var = aleatoric_var[:, 0]
    epistemic_var = epistemic_var[:, 0]

    # -----------------------------------------------
    # EVALUATION
    plots_dir = Path(config["ENSEMBLE_INFERENCE"]["plot_dir"])
    os.makedirs(plots_dir, exist_ok=True)
    path_labels_test = config["ENSEMBLE_INFERENCE"]["inference_data_labels"]
    labels = pd.read_csv(path_labels_test, nrows=nrows).values
    labels = labels[:, 0]

    rmses = [] # for separate models
    for i in range(predictions.shape[0]):
        rmse_i = np.sqrt(np.mean((predictions[i,:,0,0] - labels) ** 2))
        rmses.append(rmse_i)

    rmses = np.array(rmses)
    rmse = np.sqrt(np.mean((y_pred - labels) ** 2))
    print(f"RMSE on inference data: {rmse:.4f}")
    print(f"RMSEs of individual models: {rmses}")

    # plot rmses of individual models and the ensemble
    color_ensemble = "red"
    color_models = "blue"
    plt.figure(figsize=(10, 6))
    for i, rmse_i in enumerate(rmses):
        x = 0
        y = rmse_i
        plt.scatter(
            x,
            y / rmse,
            color=color_models,
            alpha=0.4,
            s=500,
            label="indiv. models" if i == 0 else None,
        )
    plt.scatter(
        x, rmse / rmse, color=color_ensemble, label="Ensemble", edgecolor="black", s=500
    )
    plt.ylabel("Normalized RMSE")
    plt.title(f"Normalized RMSE (ensemble RMSE=1) of {len(rmses)} Individual Models and Ensemble")
    plt.legend()
    plot_path = plots_dir / "rmse_ensemble_vs_individual_models.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved RMSE plot to {plot_path}")

    # histogram of individual diffs
    diffs = y_pred - labels
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(diffs, bins=100, density=True, alpha=0.7, color="blue", edgecolor="black")
    plt.xlabel(r"$y_\text{pred} - y_\text{true}$")
    plt.ylabel("Density")
    plt.title("Histogram of Prediction Errors")
    plot_path = plots_dir / "histogram_prediction_errors.png"
    plt.savefig(plot_path)
    print(f"Saved histogram of prediction errors to {plot_path}")
    plt.close()

    # histogram of individual absolute diffs
    abs_diffs = np.abs(diffs)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(abs_diffs, bins=100, density=True, alpha=0.7, color="blue", edgecolor="black")
    loss_mean = abs_diffs.mean()
    loss_median = np.median(abs_diffs)
    plt.axvline(loss_mean, color="red", linestyle="dashed", linewidth=1, label=f"Mean: {loss_mean:.4f}")
    plt.axvline(loss_median, color="green", linestyle="dashed", linewidth=1, label=f"Median: {loss_median:.4f}")
    plt.xlabel(r"$|y_\text{pred} - y_\text{true}|$")
    plt.ylabel("Density")
    plt.title("Histogram of Absolute Prediction Errors")
    plt.legend()
    plot_path = plots_dir / "histogram_absolute_prediction_errors.png"
    plt.savefig(plot_path)
    print(f"Saved histogram of absolute prediction errors to {plot_path}")
    plt.close()
    
    # plot true vs prediction
    # color by absolute difference between true and predicted
    abs_diffs = np.abs(y_pred - labels)
    fig, ax = plt.subplots(figsize=(10, 10))

    norm_T = Normalize(vmin=abs_diffs.min(), vmax=abs_diffs.max())
    colors = abs_diffs
    sc = ax.scatter(
            labels, y_pred, c=colors, cmap=darken_cmap(plt.get_cmap("jet")), norm=norm_T, alpha=0.7, edgecolor="black")
    # diagonal line
    ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=2)
    ax.set_xlabel(r"$T_\text{true}$")
    ax.set_ylabel(r"$T_\text{pred}$")
    cbar = plt.colorbar(sc, ax=ax)
    plt.title("Predicted vs True Values Colored by Absolute Error")
    plot_path = plots_dir / "true_vs_pred_colored_by_error.png"
    plt.savefig(plot_path)
    print(f"Saved true vs predicted plot to {plot_path}")
    plt.close()
