import argparse
import configparser
import os
import shutil
import textwrap
from datetime import date
from pathlib import Path

import lightning as L
import numpy as np
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch import Trainer

from source.IO import read_config
from source.models.Model01 import CNNTimeSeriesRegressor, Model_Lit, model_registry
from source.training.optimizers import cosine_decay_scheduler
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from source.utils import var_names
import matplotlib as mpl
import scienceplots


def load_models(config):
    model_class = config["MODEL"]["model_class"]
    ensemble_directory = (
        Path(config["ENSEMBLE_INFERENCE"]["ENSEMBLE_SAVEDIR"]) / "models" / "ckpt"
    )
    models_names = os.listdir(ensemble_directory)

    models = []
    for m in models_names:
        print(f"Loading model from {ensemble_directory / m}")
        model = Model_Lit.load_from_checkpoint(checkpoint_path=ensemble_directory / m)
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
    args = parser.parse_args()
    config = read_config(args.config)
    models = load_models(config)
    print(f"Loaded {len(models)} models for ensemble inference.")

    inference_data_path = config["ENSEMBLE_INFERENCE"]["inference_data_path"]
    print(f"Loading inference data from {inference_data_path}")
    nrows = config.getint("ENSEMBLE_INFERENCE", "n_events")
    nrows = None if nrows == -1 else nrows
    spectra = pd.read_csv(
        inference_data_path,
        index_col=0,
        nrows=nrows,
    ).values
    print(f"Loaded inference data with shape {spectra.shape}")
    inference_dataset = InferenceDataSet(spectra)
    inference_dataloader = DataLoader(
        inference_dataset, batch_size=256, shuffle=False, drop_last=False
    )

    trainer = Trainer()
    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m in models:
        m.to(device)
        m.scaler_mean = np.array(
            [
                1190.83726355,
                -6.00609932,
                -6.50023774,
                -5.99933525,
                -4.49630322,
                -6.49604898,
            ]
        )
        m.scaler_var = np.array(
            [
                4.40644566e05,
                3.00482659e00,
                2.08444992e00,
                3.02831506e00,
                7.46910864e-01,
                2.07242987e00,
            ]
        )
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
    print(f"Predictions shape from all models: {predictions.shape}")
    print(predictions[:, 0, :])

    # for the predictions, take the mean of the mus
    # for the aleatoric variance, take the mean of the sigmas^2
    # for the epistemic variance, take the variance of the mus
    y_pred = predictions[:, :, :, 0].mean(axis=0)
    aleatoric_var = predictions[:, :, :, 1].mean(axis=0)
    epistemic_var = predictions[:, :, :, 0].var(axis=0)

    # -----------------------------------------------
    # EVALUATION
    plots_dir = Path(config["ENSEMBLE_INFERENCE"]["plot_dir"])
    os.makedirs(plots_dir, exist_ok=True)
    # load true values and compare
    path_labels_test = config["ENSEMBLE_INFERENCE"]["inference_data_labels"]
    labels_train = pd.read_csv(path_labels_test, nrows=nrows).values

    print(f"{y_pred.shape=}, {labels_train.shape=}")

    # RMSE for each output dimsions for each model separately
    rmses = []
    for i in range(predictions.shape[0]):
        rmse_i = np.sqrt(np.mean((predictions[i, :, :, 0] - labels_train) ** 2, axis=0))
        rmses.append(rmse_i)
    rmses = np.array(rmses)

    # calculate the RMSE for each output dimension
    rmse = np.sqrt(np.mean((y_pred - labels_train) ** 2, axis=0))
    print(f"RMSE for each output dimension: {rmse}")

    # -----------------------------------------------
    # plot rmses of the individual models and the ensemble. x axis is the output dimension, y is the rmse
    color_ensemble = "red"
    color_models = "blue"
    plt.figure(figsize=(10, 6))
    for i, rmse_i in enumerate(rmses):
        x = np.arange(len(rmse_i))
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
    plt.xticks(
        x,
        [
            models[0].labels_names[i] if models[0].labels_names else var_names[i]
            for i in x
        ],
    )
    plt.xlabel("Output Dimension")
    plt.ylabel("Normalized RMSE")
    plt.title("Normalized RMSE (ensemble RMSE=1) of Individual Models and Ensemble")
    plt.legend()
    plt.grid()
    plot_path = plots_dir / "rmse_ensemble_vs_individual_models.png"
    plt.savefig(plot_path)
    print(f"Saved RMSE plot to {plot_path}")

    diff = y_pred - labels_train
    indiv_losses_per_var = np.abs(diff)

    # -----------------------------------------------
    # PLOT DIFF HISTOGRAMS FOR EACH OUTPUT DIMENSION
    n_vars = y_pred.shape[1]
    y_scale = "linear"
    x_scale = "linear"
    lossname = "RMSE"
    ranges = [
        [0, 100],  # T
        [0, 0.6],  # X_H2O
        [0, 0.25],  # X_CO2
        [0, 0.8],  # X_CH4
        [0, 0.2],  # X_CO
        [0, 0.75],  # X_NH3
    ]

    fig, axs = plt.subplots(2, n_vars // 2, figsize=(2.5 * n_vars, 7))
    # make axs 1d array
    axs = axs.flatten()
    for i in range(n_vars):
        if y_scale == "linear" and lossname == "RMSE":
            axs[i].hist(
                indiv_losses_per_var[:, i],
                bins=200,
                label=f"{models[0].labels_names[i] if models[0].labels_names else var_names[i]}",
                density=True,
                range=ranges[i],
            )
        else:
            axs[i].hist(
                indiv_losses_per_var[:, i],
                bins=200,
                label=f"{models[0].labels_names[i] if models[0].labels_names else var_names[i]}",
                density=True,
            )
        axs[i].set_xscale(x_scale)
        axs[i].set_yscale(y_scale)
        # axs[i].set_xlabel(lossname if i == n_vars - 1 else "")
        axs[i].set_xlabel(lossname)
        axs[i].set_ylabel("Count")
        axs[i].legend(fontsize="large")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    # plt.suptitle("Histogram of Individual Losses per Variable")
    plot_path = plots_dir / "histogram_individual_losses_per_variable.png"
    plt.savefig(plot_path)
    print(f"Saved histogram of individual losses per variable to {plot_path}")

    # -----------------------------------------------
    # Evaluate uncertainty estimates
    # We want to do the following:
    # Create bins for y_true
    # For those bins and for each output dimension, calculate the mean aleatoric and epistemic uncertainty
    # also calculate the variance of the labels in those bins
    print(
        f"{y_pred.shape=}, {labels_train.shape=}, {aleatoric_var.shape=}, {epistemic_var.shape=}"
    )
    y_test = labels_train
    n_vars = y_test.shape[1]
    # fig, axs = plt.subplots(1, n_vars, figsize=(5 * n_vars, 5))
    fig, axs = plt.subplots(2, n_vars // 2, figsize=(2.5 * n_vars, 7))
    axs = axs.flatten()
    n_bins = 25

    for i_pred_var in range(n_vars):
        bins = np.linspace(
            min(y_test[:, i_pred_var]), max(y_test[:, i_pred_var]), n_bins
        )
        bin_labels_stds = []  # std of true values in each bin
        bin_sigmas = []  # mean aleatoric uncertainty in each bin
        bin_epistemics = []  # mean epistemic uncertainty in each bin
        bin_rmse = []  # error (rmse) in each bin
        bin_centers = []
        prediction_diffs = y_pred[:, i_pred_var] - y_test[:, i_pred_var]
        for i in range(len(bins) - 1):
            bin_mask = (y_test[:, i_pred_var] >= bins[i]) & (
                y_test[:, i_pred_var] < bins[i + 1]
            )
            if np.any(bin_mask):
                bin_labels_std = np.std(y_test[bin_mask, i_pred_var])
                bin_sigma = np.sqrt(np.mean(aleatoric_var[bin_mask, i_pred_var]))
                bin_epistemic = np.sqrt(np.mean(epistemic_var[bin_mask, i_pred_var]))
                bin_rmse_i = np.sqrt(np.mean(prediction_diffs[bin_mask] ** 2))
                bin_labels_stds.append(bin_labels_std)
                bin_sigmas.append(bin_sigma)
                bin_epistemics.append(bin_epistemic)
                bin_rmse.append(bin_rmse_i)
                bin_center = 0.5 * (bins[i] + bins[i + 1])
                bin_centers.append(bin_center)
            else:
                bin_labels_stds.append(np.nan)
                bin_sigmas.append(np.nan)
                bin_epistemics.append(np.nan)
                bin_rmse.append(np.nan)
                bin_center = 0.5 * (bins[i] + bins[i + 1])
                bin_centers.append(bin_center)
        bin_centers = np.array(bin_centers)
        bin_labels_std = np.array(bin_labels_stds)
        bin_sigmas = np.array(bin_sigmas)
        bin_epistemics = np.array(bin_epistemics)
        bin_rmse = np.array(bin_rmse)

        plt.sca(axs[i_pred_var])
        var_name = (
            models[0].labels_names[i_pred_var]
            if models[0].labels_names
            # else f"variable_{i_pred_var}"
            else var_names[i_pred_var]
        )

        plt.plot(
            bin_centers, bin_labels_std, label="labels std", color="gray", marker="x"
        )
        plt.plot(
            bin_centers,
            bin_sigmas,
            label=r"Aleatoric Unc. $\sigma_\mathrm{A}$",
            color="orange",
            marker="o",
        )
        plt.plot(
            bin_centers,
            bin_epistemics,
            label=r"Epistemic Unc. $\sigma_\mathrm{E}$",
            color="blue",
            marker="^",
        )
        # sum of aleatoric and epistemic
        plt.plot(
            bin_centers,
            np.sqrt(bin_sigmas**2 + bin_epistemics**2),
            label=r"Total Unc. $\sqrt{\sigma_\mathrm{A}^2 + \sigma_\mathrm{E}^2}$",
            color="black",
            marker="D",
            ls="--",
        )
        plt.plot(bin_centers, bin_rmse, label="RMSE", color="red", marker="s")
        plt.xlabel(f"True {var_name}")
        plt.ylabel(f"{var_name} Uncert./RMSE")
        plt.legend() if i_pred_var == 0 else None
        # plt.grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    model_class = config["MODEL"]["model_class"]
    # plt.suptitle(
    #     f"Uncertainty Estimates for ensemble of {len(models)} {model_class} models"
    # )
    plot_path = plots_dir / "uncertainty_estimates_vs_true_std_horizontal.png"
    plt.savefig(plot_path)
    plt.close()

    ################ Errors means and stds
    # For each variable, calculate the MAE in in bins based on the true value of that variable.
    # Also calculate the mean predicted sigma in those bins and plot as error bar on the mean.
    fig, axs = plt.subplots(2, n_vars // 2, figsize=(2.5 * n_vars, 7))
    # make axs 1d array
    axs = axs.flatten()
    y_lims = [
        [0, 2000],
        [0, 3.0],
        [0, 3.0],
        [0, 3.0],
        [0, 3.0],
        [0, 3.0],
        [0, 3.0],
    ]

    n_bins = 25

    for i_pred_var in range(n_vars):
        bins = np.linspace(
            min(y_test[:, i_pred_var]), max(y_test[:, i_pred_var]), n_bins
        )
        bin_labels_stds = []  # std of true values in each bin
        bin_sigmas = []  # mean aleatoric uncertainty in each bin
        bin_epistemics = []  # mean epistemic uncertainty in each bin
        bin_rmse = []  # error (rmse) in each bin
        bin_centers = []
        prediction_diffs = y_pred[:, i_pred_var] - y_test[:, i_pred_var]
        for i in range(len(bins) - 1):
            bin_mask = (y_test[:, i_pred_var] >= bins[i]) & (
                y_test[:, i_pred_var] < bins[i + 1]
            )
            if np.any(bin_mask):
                bin_labels_std = np.std(y_test[bin_mask, i_pred_var])
                bin_sigma = np.sqrt(np.mean(aleatoric_var[bin_mask, i_pred_var]))
                bin_epistemic = np.sqrt(np.mean(epistemic_var[bin_mask, i_pred_var]))
                bin_rmse_i = np.sqrt(np.mean(prediction_diffs[bin_mask] ** 2))
                bin_labels_stds.append(bin_labels_std)
                bin_sigmas.append(bin_sigma)
                bin_epistemics.append(bin_epistemic)
                bin_rmse.append(bin_rmse_i)
                bin_center = 0.5 * (bins[i] + bins[i + 1])
                bin_centers.append(bin_center)
            else:
                bin_labels_stds.append(np.nan)
                bin_sigmas.append(np.nan)
                bin_epistemics.append(np.nan)
                bin_rmse.append(np.nan)
                bin_center = 0.5 * (bins[i] + bins[i + 1])
                bin_centers.append(bin_center)
        bin_centers = np.array(bin_centers)
        bin_labels_std = np.array(bin_labels_stds)
        bin_sigmas = np.array(bin_sigmas)
        bin_epistemics = np.array(bin_epistemics)
        bin_rmse = np.array(bin_rmse)

        plt.sca(axs[i_pred_var])
        var_name = (
            models[0].labels_names[i_pred_var]
            if models[0].labels_names
            # else f"variable_{i_pred_var}"
            else var_names[i_pred_var]
        )

        plt.sca(axs[i_pred_var])
        var_name = (
            models[0].labels_names[i_pred_var]
            if models[0].labels_names
            else var_names[i_pred_var]
        )
        plt.plot(
            bin_centers,
            bin_rmse,
            label="RMSE" if i_pred_var == 0 else None,
            color="red",
            marker="o",
            lw=2,
            markersize=5,
        )
        plt.errorbar(
            bin_centers + (bin_centers[1] - bin_centers[0]) * 0.08,
            bin_rmse,
            yerr=bin_sigmas,
            color="orange",
            label="Aleatoric Std Dev" if i_pred_var == 0 else None,
            capsize=5,
            elinewidth=2,
            fmt="none",  # disable connecting lines between error bars
            # alpha=0.6,
        )
        plt.errorbar(
            bin_centers - (bin_centers[1] - bin_centers[0]) * 0.08,
            bin_rmse,
            yerr=bin_epistemics,
            color="blue",
            label="Epistemic Std Dev" if i_pred_var == 0 else None,
            capsize=5,
            elinewidth=2,
            fmt="none",  # disable connecting lines between error bars
            # alpha=0.6,
        )
        plt.ylim(y_lims[i_pred_var])
        plt.xlabel(f"True Values of {var_name}")
        plt.ylabel(
            r"RMSE with pred. $\sigma$"
            if (i_pred_var == 0 or i_pred_var == 3)
            else None
        )
        # plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    path = plots_dir / "errors_with_errorbars_horizontal.png"
    plt.savefig(path)
    plt.close(fig)
