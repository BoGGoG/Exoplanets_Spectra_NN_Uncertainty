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
from source.models.Model01 import CNNTimeSeriesRegressor, Model_Lit, model_registry
from source.training.optimizers import cosine_decay_scheduler
from source.utils import var_names


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
    predict = False
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

        # save predictions
        np.savez_compressed(predictions_path, predictions=predictions)

    # load predictions
    predictions = np.load(predictions_path)["predictions"]

    # raise NotImplementedError

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
        [var_names[i] for i in x],
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
    # PLOT RMSE HISTOGRAMS FOR EACH OUTPUT DIMENSION
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
                label=f"{var_names[i]}",
                density=True,
                range=ranges[i],
            )
        else:
            axs[i].hist(
                indiv_losses_per_var[:, i],
                bins=200,
                label=f"{var_names[i]}",
                density=True,
            )
        axs[i].set_xscale(x_scale)
        axs[i].set_yscale(y_scale)
        # axs[i].set_xlabel(lossname if i == n_vars - 1 else "")
        axs[i].set_xlabel(lossname)
        axs[i].set_ylabel("Count")
        # add text with mean of the loss to the plot with a vertical line
        loss_mean = np.mean(indiv_losses_per_var[:, i])
        loss_median = np.median(indiv_losses_per_var[:, i])
        axs[i].axvline(
            loss_mean, color="orange", linestyle="--", label=f"Mean={loss_mean:.4}"
        )
        axs[i].axvline(
            loss_median, color="red", linestyle="--", label=f"Median={loss_median:.4}"
        )
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
        var_name = var_names[i_pred_var]

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
    plot_path = plots_dir / "uncertainty_estimates_vs_true_std.png"
    plt.savefig(plot_path)
    plt.close()

    ########################################################################################
    # same, but only RMSE and epistemic uncertainty.
    # Epistemic with different axis on the right side
    fig, axs = plt.subplots(
        2, n_vars // 2, figsize=(2.5 * n_vars, 7), layout="constrained"
    )
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
        var_name = var_names[i_pred_var]

        # plt.plot(
        #     bin_centers,
        #     bin_sigmas,
        #     label=r"Aleatoric Unc. $\sigma_\mathrm{A}$",
        #     color="orange",
        #     marker="o",
        # )
        plt.plot(bin_centers, bin_rmse, label="RMSE", color="red", marker="s")
        plt.xlabel(f"True {var_name}")
        plt.ylabel(f"{var_name} RMSE", color="red")
        plt.tick_params(axis="y", labelcolor="red")
        twin_ax = axs[i_pred_var].twinx()
        twin_ax.plot(
            bin_centers,
            bin_epistemics,
            label=r"Epistemic Unc. $\sigma_\mathrm{E}$",
            color="blue",
            marker="^",
        )
        twin_ax.set_ylabel(f"{var_name} Epistemic Unc.", color="blue")
        twin_ax.tick_params(axis="y", labelcolor="blue")
        # sum of aleatoric and epistemic
        # plt.plot(
        #     bin_centers,
        #     np.sqrt(bin_sigmas**2 + bin_epistemics**2),
        #     label=r"Total Unc. $\sqrt{\sigma_\mathrm{A}^2 + \sigma_\mathrm{E}^2}$",
        #     color="black",
        #     marker="D",
        #     ls="--",
        # )
        lines, labels = axs[i_pred_var].get_legend_handles_labels()
        lines2, labels2 = twin_ax.get_legend_handles_labels()
        twin_ax.legend(lines + lines2, labels + labels2) if i_pred_var == 0 else None
        # plt.grid()
    plot_path = plots_dir / "uncertainty_estimates_vs_rmse_twin.png"
    plt.savefig(plot_path)
    plt.close()

    ########################################################################################
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
        var_name = var_names[i_pred_var]

        plt.sca(axs[i_pred_var])
        var_name = var_names[i_pred_var]
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

    ########################################################################################

    # plot all true vs predicted values for each variable
    # the color of the dots should be the absolute difference between true and predicted
    # on a colormap from blue (small difference) to red (large difference)
    abs_diffs = np.abs(y_pred - y_test)
    # fig, axs = plt.subplots(2, n_vars // 2, figsize=(2.5 * n_vars, 7))
    fig, axs = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4))
    # axs = axs.flatten()

    norm_first = Normalize(vmin=abs_diffs[:, 0].min(), vmax=abs_diffs[:, 0].max())
    norm_last = Normalize(vmin=abs_diffs[:, -1].min(), vmax=abs_diffs[:, -1].max())
    y_lims = [-9, -3]

    for i_pred_var in range(n_vars):
        colors = abs_diffs[:, i_pred_var]
        if i_pred_var == 0:
            norm = norm_first
        else:
            norm = norm_last
        sc = axs[i_pred_var].scatter(
            y_test[:, i_pred_var],
            y_pred[:, i_pred_var],
            c=colors,
            # cmap="coolwarm",
            cmap="jet",
            norm=norm,
            s=0.5,
            # alpha=0.5,
        )
        axs[i_pred_var].set_xlabel(r"$y_\text{t}$")
        # if i_pred_var == 0 or i_pred_var == 1:
        #     axs[i_pred_var].set_ylabel(r"$y_\text{p}$")
        # axs[i_pred_var].xaxis.set_major_locator(plt.MaxNLocator(9))
        # axs[i_pred_var].xaxis.set_minor_locator(plt.MaxNLocator(9))
        axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
        # Add colorbar for first and last variable
        if i_pred_var == 0 or i_pred_var == n_vars - 1:
            cbar = fig.colorbar(sc, ax=axs[i_pred_var])
            # cbar.set_label("Absolute Error", fontsize=12)
        if i_pred_var == 0:
            y_lims_T = [y_test[:, i_pred_var].min(), y_test[:, i_pred_var].max()]
            axs[i_pred_var].set_ylim(y_lims_T)
            axs[i_pred_var].set_xlim(y_lims_T)
        else:
            axs[i_pred_var].set_ylim(y_lims)
            axs[i_pred_var].set_xlim(y_lims)
        if i_pred_var > 1:  # no ytick labels for all but first two plots
            axs[i_pred_var].set_yticklabels([])

    plt.suptitle(r"$n_\text{test}=$" + f"{y_test.shape[0]}, Ensemble of {15} models")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    # path = plots_dir / "true_vs_pred_scatter_two_rows_with_alpha.png"
    # path = plots_dir / "true_vs_pred_scatter_two_rows.png"
    # path = plots_dir / "true_vs_pred_scatter_one_row_with_alpha.png"
    path = plots_dir / "true_vs_pred_scatter_one_row.png"
    plt.savefig(path)
    print(f"Saved true vs predicted scatter plot to {path}")
    plt.close(fig)

    ########################################################################################
    # scatter plot of error vs predicted uncertainty for each variable
    # the color of the dots should be the absolute difference between prediction error and predicted uncertainty
    # fig, axs = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4), layout="constrained")
    fig, axs = plt.subplots(
        2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
    )
    axs = axs.flatten()
    pred_errors_all = y_pred - y_test
    pred_uncs_all = np.sqrt(aleatoric_var)
    abs_diffs = np.abs(pred_errors_all) - pred_uncs_all
    norm_first = Normalize(
        vmin=abs_diffs[:, 0].min() * 0.7, vmax=abs_diffs[:, 0].max() * 0.7
    )
    norm_last = Normalize(
        vmin=abs_diffs[:, -1].min() * 0.7, vmax=abs_diffs[:, -1].max() * 0.7
    )

    dark_turbo = darken_cmap(plt.cm.turbo, 0.7)
    for i_pred_var in range(n_vars):
        if i_pred_var == 0:
            norm = norm_first
        else:
            norm = norm_last
        pred_errors = y_pred[:, i_pred_var] - y_test[:, i_pred_var]
        var_name = var_names[i_pred_var]
        pred_unc = np.sqrt(aleatoric_var[:, i_pred_var])
        colors = np.abs(pred_errors) - pred_unc
        sc = axs[i_pred_var].scatter(
            pred_unc,
            pred_errors,
            c=colors,
            s=1,
            # cmap="jet",
            # cmap="turbo",
            # cmap="berlin",
            # cmap="vanimo",
            # cmap="magma",
            cmap=dark_turbo,
            norm=norm,
        )
        # plot diagonal line from 0,0 to max of x and max and min of y
        ax_x_max = pred_unc.max()
        axs[i_pred_var].plot(
            [0, ax_x_max],
            [0, ax_x_max],
            ls="--",
            color="gray",
            lw=1,
            label="1:1 line",
        )
        axs[i_pred_var].plot([0, ax_x_max], [0, -ax_x_max], ls="--", color="gray", lw=1)
        if i_pred_var == 0:
            axs[i_pred_var].set_ylabel(r"error = $y_\text{pred} - y_\text{true}$")
        if i_pred_var == 0 or i_pred_var == n_vars - 1:
            cbar = fig.colorbar(sc, ax=axs[i_pred_var])
            # cbar.set_label("Absolute Error", fontsize=12)
        # if i_pred_var > 1:  # no ytick labels for all but first two plots
        #     axs[i_pred_var].set_yticklabels([])
        axs[i_pred_var].set_xlabel(r"Aleatoric Unc. $\sigma_\mathrm{A}$")
        axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
        axs[i_pred_var].legend()
    # plt.tight_layout()
    plt.suptitle(
        r"Pred. error vs. aleatoric uncertainty, color=$|\mathrm{error}-\sigma_\mathrm{A}|$, n_test="
        + f"{y_test.shape[0]}"
    )
    plt.subplots_adjust(wspace=0.2)
    plot_path = plots_dir / "error_vs_aleatoric_uncertainty.png"
    plt.savefig(plot_path)
    print(f"Saved error vs aleatoric uncertainty plot to {plot_path}")
    plt.close(fig)

    # same with epistemic uncertainty
    fig, axs = plt.subplots(
        2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
    )
    axs = axs.flatten()
    pred_errors_all = y_pred - y_test
    pred_uncs_all = np.sqrt(epistemic_var)
    abs_diffs = np.abs(pred_errors_all) - pred_uncs_all
    norm_first = Normalize(
        vmin=abs_diffs[:, 0].min() * 0.7, vmax=abs_diffs[:, 0].max() * 0.7
    )
    norm_last = Normalize(
        vmin=abs_diffs[:, -1].min() * 0.7, vmax=abs_diffs[:, -1].max() * 0.7
    )

    dark_turbo = darken_cmap(plt.cm.turbo, 0.7)
    for i_pred_var in range(n_vars):
        if i_pred_var == 0:
            norm = norm_first
        else:
            norm = norm_last
        pred_errors = y_pred[:, i_pred_var] - y_test[:, i_pred_var]
        var_name = var_names[i_pred_var]
        pred_unc = np.sqrt(epistemic_var[:, i_pred_var])
        colors = np.abs(pred_errors) - pred_unc
        sc = axs[i_pred_var].scatter(
            pred_unc,
            pred_errors,
            c=colors,
            s=1,
            # cmap="jet",
            # cmap="turbo",
            # cmap="berlin",
            # cmap="vanimo",
            # cmap="magma",
            cmap=dark_turbo,
            norm=norm,
        )
        # plot diagonal line from 0,0 to max of x and max and min of y
        ax_x_max = pred_unc.max()
        axs[i_pred_var].plot(
            [0, ax_x_max],
            [0, ax_x_max],
            ls="--",
            color="gray",
            lw=1,
            label="1:1 line",
        )
        axs[i_pred_var].plot([0, ax_x_max], [0, -ax_x_max], ls="--", color="gray", lw=1)
        if i_pred_var == 0:
            axs[i_pred_var].set_ylabel(r"error = $y_\text{pred} - y_\text{true}$")
        if i_pred_var == 0 or i_pred_var == n_vars - 1:
            cbar = fig.colorbar(sc, ax=axs[i_pred_var])
            # cbar.set_label("Absolute Error", fontsize=12)
        # if i_pred_var > 1:  # no ytick labels for all but first two plots
        #     axs[i_pred_var].set_yticklabels([])
        axs[i_pred_var].set_xlabel(r"Epistemic Unc. $\sigma_\mathrm{E}$")
        axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
        axs[i_pred_var].legend()
    # plt.tight_layout()
    plt.suptitle(
        r"Pred. error vs. epistemic uncertainty, color=$|\mathrm{error}-\sigma_\mathrm{E}|$, n_test="
        + f"{y_test.shape[0]}"
    )
    plt.subplots_adjust(wspace=0.2)
    plot_path = plots_dir / "error_vs_epistemic_uncertainty.png"
    plt.savefig(plot_path)
    print(f"Saved error vs epistemic uncertainty plot to {plot_path}")
    plt.close(fig)

    # same with full uncertainty (aleatoric^2 + epistemic^2)^1/2
    fig, axs = plt.subplots(
        2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
    )
    axs = axs.flatten()
    pred_errors_all = y_pred - y_test
    pred_uncs_all = np.sqrt(epistemic_var + aleatoric_var)
    abs_diffs = np.abs(pred_errors_all) - pred_uncs_all
    norm_first = Normalize(
        vmin=abs_diffs[:, 0].min() * 0.7, vmax=abs_diffs[:, 0].max() * 0.7
    )
    norm_last = Normalize(
        vmin=abs_diffs[:, -1].min() * 0.7, vmax=abs_diffs[:, -1].max() * 0.7
    )

    dark_turbo = darken_cmap(plt.cm.turbo, 0.7)
    for i_pred_var in range(n_vars):
        if i_pred_var == 0:
            norm = norm_first
        else:
            norm = norm_last
        pred_errors = y_pred[:, i_pred_var] - y_test[:, i_pred_var]
        var_name = var_names[i_pred_var]
        pred_unc = np.sqrt(epistemic_var[:, i_pred_var] + aleatoric_var[:, i_pred_var])
        colors = np.abs(pred_errors) - pred_unc
        sc = axs[i_pred_var].scatter(
            pred_unc,
            pred_errors,
            c=colors,
            s=1,
            # cmap="jet",
            # cmap="turbo",
            # cmap="berlin",
            # cmap="vanimo",
            # cmap="magma",
            cmap=dark_turbo,
            norm=norm,
        )
        # plot diagonal line from 0,0 to max of x and max and min of y
        ax_x_max = pred_unc.max()
        axs[i_pred_var].plot(
            [0, ax_x_max],
            [0, ax_x_max],
            ls="--",
            color="gray",
            lw=1,
            label="1:1 line",
        )
        axs[i_pred_var].plot([0, ax_x_max], [0, -ax_x_max], ls="--", color="gray", lw=1)
        if i_pred_var == 0:
            axs[i_pred_var].set_ylabel(r"error = $y_\text{pred} - y_\text{true}$")
        if i_pred_var == 0 or i_pred_var == n_vars - 1:
            cbar = fig.colorbar(sc, ax=axs[i_pred_var])
            # cbar.set_label("Absolute Error", fontsize=12)
        # if i_pred_var > 1:  # no ytick labels for all but first two plots
        #     axs[i_pred_var].set_yticklabels([])
        axs[i_pred_var].set_xlabel(
            r"$\sqrt{\sigma_\mathrm{E}^2 + \sigma_\mathrm{A}^2}$"
        )
        axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
        axs[i_pred_var].legend()
    # plt.tight_layout()
    plt.suptitle(
        r"Pred. error vs. ep.+al. uncertainty, color=$|\mathrm{error}-\sigma_\mathrm{E+A}|$, n_test="
        + f"{y_test.shape[0]}"
    )
    plot_path = plots_dir / "error_vs_full_uncertainty.png"
    plt.savefig(plot_path)
    print(f"Saved error vs full uncertainty plot to {plot_path}")
    plt.close(fig)

    ###################################################################################################
    # scatter plot of sigma_aleatoric vs sigma_epistemic for each variable
    fig, axs = plt.subplots(
        2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
    )
    axs = axs.flatten()
    for i_pred_var in range(n_vars):
        axs[i_pred_var].scatter(
            np.sqrt(aleatoric_var[:, i_pred_var]),
            np.sqrt(epistemic_var[:, i_pred_var]),
            s=1,
            color="blue",
        )
        axs[i_pred_var].set_xlabel(r"Aleatoric Unc. $\sigma_\mathrm{A}$")
        axs[i_pred_var].set_ylabel(r"Epistemic Unc. $\sigma_\mathrm{E}$")
        axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
    plot_path = plots_dir / "aleatoric_vs_epistemic_uncertainty.png"
    plt.savefig(plot_path)
    print(f"Saved aleatoric vs epistemic uncertainty plot to {plot_path}")
    plt.close(fig)

    ###################################################################################################
    # scatter plot of true value vs RMSE/predicted uncertainty for each variable
    # colorcoding: true value of the Temperature (blue=low, red=high)
    color_names = ["temp", "H2O", "CO2", "CH4", "CO", "NH3"]
    for i_color in range(n_vars):
        fig, axs = plt.subplots(
            2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
        )
        axs = axs.flatten()
        color_vals = y_test[:, i_color]
        cmap = plt.get_cmap("jet")

        for i_pred_var in range(n_vars):
            y_plot = (y_pred[:, i_pred_var] - y_test[:, i_pred_var]) / np.sqrt(
                aleatoric_var[:, i_pred_var]
            )
            x_plot = y_test[:, i_pred_var]
            axs[i_pred_var].axhline(1, ls="--", color="gray", lw=2)
            axs[i_pred_var].axhline(-1, ls="--", color="gray", lw=2)
            axs[i_pred_var].axhspan(-1, 1, color="lightgreen", alpha=0.5)
            y_max_plot = y_plot.max()
            y_min_plot = y_plot.min()
            axs[i_pred_var].axhspan(1, y_max_plot, color="red", alpha=0.2)
            axs[i_pred_var].axhspan(y_min_plot, -1, color="red", alpha=0.2)
            axs[i_pred_var].scatter(x_plot, y_plot, s=1, c=color_vals, cmap=cmap)
            axs[i_pred_var].set_xlabel(r"True Value")
            axs[i_pred_var].set_ylabel(
                r"($y_\text{pred} - y_\text{true})$ / Aleatoric Unc."
            )
            axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
        plt.suptitle("True Value vs. Error/Aleatoric Uncertainty")
        plot_path = (
            plots_dir
            / "true_vs_rmse_scatter"
            / f"true_value_vs_rmse_over_aleatoric_uncertainty_color_{color_names[i_color]}.png"
        )
        os.makedirs(plot_path.parent, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Saved true value vs rmse/aleatoric uncertainty plot to {plot_path}")
        plt.close(fig)

    # same for epistemic
    for i_color in range(n_vars):
        fig, axs = plt.subplots(
            2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
        )
        axs = axs.flatten()
        color_vals = y_test[:, i_color]
        cmap = plt.get_cmap("jet")

        for i_pred_var in range(n_vars):
            y_plot = (y_pred[:, i_pred_var] - y_test[:, i_pred_var]) / np.sqrt(
                epistemic_var[:, i_pred_var]
            )
            x_plot = y_test[:, i_pred_var]
            axs[i_pred_var].axhline(1, ls="--", color="gray", lw=2)
            axs[i_pred_var].axhline(-1, ls="--", color="gray", lw=2)
            axs[i_pred_var].axhspan(-1, 1, color="lightgreen", alpha=0.5)
            y_max_plot = y_plot.max()
            y_min_plot = y_plot.min()
            axs[i_pred_var].axhspan(1, y_max_plot, color="red", alpha=0.2)
            axs[i_pred_var].axhspan(y_min_plot, -1, color="red", alpha=0.2)
            axs[i_pred_var].scatter(x_plot, y_plot, s=1, c=color_vals, cmap=cmap)
            axs[i_pred_var].set_xlabel(r"True Value")
            axs[i_pred_var].set_ylabel(
                r"($y_\text{pred} - y_\text{true})$ / Epistemic Unc."
            )
            axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
        plt.suptitle("True Value vs. Error/Epistemic Uncertainty")
        plot_path = (
            plots_dir
            / "true_vs_rmse_scatter"
            / f"true_value_vs_rmse_over_epistemic_uncertainty_color_{color_names[i_color]}.png"
        )
        os.makedirs(plot_path.parent, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Saved true value vs rmse/epistemic uncertainty plot to {plot_path}")
        plt.close(fig)

    # same for full uncertainty
    fig, axs = plt.subplots(
        2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
    )
    axs = axs.flatten()
    for i_pred_var in range(n_vars):
        y_plot = (y_pred[:, i_pred_var] - y_test[:, i_pred_var]) / np.sqrt(
            epistemic_var[:, i_pred_var] + aleatoric_var[:, i_pred_var]
        )
        x_plot = y_test[:, i_pred_var]
        axs[i_pred_var].axhline(1, ls="--", color="gray", lw=2)
        axs[i_pred_var].axhline(-1, ls="--", color="gray", lw=2)
        axs[i_pred_var].axhspan(-1, 1, color="lightgreen", alpha=0.5)
        y_max_plot = y_plot.max()
        y_min_plot = y_plot.min()
        axs[i_pred_var].axhspan(1, y_max_plot, color="red", alpha=0.2)
        axs[i_pred_var].axhspan(y_min_plot, -1, color="red", alpha=0.2)
        axs[i_pred_var].scatter(x_plot, y_plot, s=1, color="blue")
        axs[i_pred_var].set_xlabel(r"True Value")
        axs[i_pred_var].set_ylabel(
            r"($y_\text{pred} - y_\text{true}) / \sqrt{\sigma_\mathrm{E}^2 + \sigma_\mathrm{A}^2}$"
        )
        axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
    plt.suptitle("True Value vs. Error/(Ep.+Al.) Uncertainty")
    plot_path = plots_dir / "true_value_vs_rmse_over_full_uncertainty.png"
    plt.savefig(plot_path)
    print(f"Saved true value vs rmse/ep.+al. uncertainty plot to {plot_path}")
    plt.close(fig)

    # same plot for aleatoric uncertainty, but only those where the temperature prediction is bad
    # bad means absolute error > 500
    for i_color in range(n_vars):
        fig, axs = plt.subplots(
            2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
        )
        axs = axs.flatten()
        color_vals = y_test[:, i_color]
        cmap = plt.get_cmap("jet")

        for i_pred_var in range(n_vars):
            y_plot = (y_pred[:, i_pred_var] - y_test[:, i_pred_var]) / np.sqrt(
                aleatoric_var[:, i_pred_var]
            )
            x_plot = y_test[:, i_pred_var]
            T_pred = y_pred[:, 0]
            mask_bad_T = np.abs(T_pred - y_test[:, 0]) > 500
            x_plot = x_plot[mask_bad_T]
            y_plot = y_plot[mask_bad_T]
            axs[i_pred_var].axhline(1, ls="--", color="gray", lw=2)
            axs[i_pred_var].axhline(-1, ls="--", color="gray", lw=2)
            axs[i_pred_var].axhspan(-1, 1, color="lightgreen", alpha=0.5)
            y_max_plot = y_plot.max()
            y_min_plot = y_plot.min()
            axs[i_pred_var].axhspan(1, y_max_plot, color="red", alpha=0.2)
            axs[i_pred_var].axhspan(y_min_plot, -1, color="red", alpha=0.2)
            axs[i_pred_var].scatter(
                x_plot, y_plot, s=12, c=color_vals[mask_bad_T], cmap=cmap
            )
            axs[i_pred_var].set_xlabel(r"True Value")
            axs[i_pred_var].set_ylabel(
                r"($y_\text{pred} - y_\text{true})$ / Aleatoric Unc."
            )
            axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
        plt.suptitle("True Value vs. Error/Aleatoric Uncertainty (only bad Tpred)")
        plot_path = (
            plots_dir
            / "true_vs_rmse_scatter_bad_Tpred"
            / f"true_value_vs_rmse_over_aleatoric_uncertainty_color_{color_names[i_color]}_only_bad_Tpred.png"
        )
        os.makedirs(plot_path.parent, exist_ok=True)
        plt.savefig(plot_path)
        print(
            f"Saved true value vs rmse/aleatoric uncertainty (only bad Tpred) plot to {plot_path}"
        )
        plt.close(fig)

    # same for epistemic
    for i_color in range(n_vars):
        fig, axs = plt.subplots(
            2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
        )
        axs = axs.flatten()
        color_vals = y_test[:, i_color]
        cmap = plt.get_cmap("jet")

        for i_pred_var in range(n_vars):
            y_plot = (y_pred[:, i_pred_var] - y_test[:, i_pred_var]) / np.sqrt(
                epistemic_var[:, i_pred_var]
            )
            x_plot = y_test[:, i_pred_var]
            T_pred = y_pred[:, 0]
            mask_bad_T = np.abs(T_pred - y_test[:, 0]) > 500
            x_plot = x_plot[mask_bad_T]
            y_plot = y_plot[mask_bad_T]
            axs[i_pred_var].axhline(1, ls="--", color="gray", lw=2)
            axs[i_pred_var].axhline(-1, ls="--", color="gray", lw=2)
            axs[i_pred_var].axhspan(-1, 1, color="lightgreen", alpha=0.5)
            y_max_plot = y_plot.max()
            y_min_plot = y_plot.min()
            axs[i_pred_var].axhspan(1, y_max_plot, color="red", alpha=0.2)
            axs[i_pred_var].axhspan(y_min_plot, -1, color="red", alpha=0.2)
            axs[i_pred_var].scatter(
                x_plot, y_plot, s=12, c=color_vals[mask_bad_T], cmap=cmap
            )
            axs[i_pred_var].set_xlabel(r"True Value")
            axs[i_pred_var].set_ylabel(
                r"($y_\text{pred} - y_\text{true})$ / Epistemic Unc."
            )
            axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
        plt.suptitle("True Value vs. Error/Epistemic Uncertainty (only bad Tpred)")
        plot_path = (
            plots_dir
            / "true_vs_rmse_scatter_bad_Tpred"
            / f"true_value_vs_rmse_over_epistemic_uncertainty_color_{color_names[i_color]}_only_bad_Tpred.png"
        )
        os.makedirs(plot_path.parent, exist_ok=True)
        plt.savefig(plot_path)
        print(
            f"Saved true value vs rmse/epistemic uncertainty (only bad Tpred) plot to {plot_path}"
        )
        plt.close(fig)

    # same plot, but for each target variable, the color is chosen according to the largest value of the other variables (except temperature)
    fig, axs = plt.subplots(
        2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
    )
    axs = axs.flatten()
    color_vals = y_test
    cmap = plt.get_cmap("jet")

    for i_pred_var in range(n_vars):
        y_plot = (y_pred[:, i_pred_var] - y_test[:, i_pred_var]) / np.sqrt(
            aleatoric_var[:, i_pred_var]
        )
        x_plot = y_test[:, i_pred_var]
        axs[i_pred_var].axhline(1, ls="--", color="gray", lw=2)
        axs[i_pred_var].axhline(-1, ls="--", color="gray", lw=2)
        axs[i_pred_var].axhspan(-1, 1, color="lightgreen", alpha=0.5)
        y_max_plot = y_plot.max()
        y_min_plot = y_plot.min()
        axs[i_pred_var].axhspan(1, y_max_plot, color="red", alpha=0.2)
        axs[i_pred_var].axhspan(y_min_plot, -1, color="red", alpha=0.2)
        # all except the current variable and temperature
        color_i = color_vals[:, [j for j in range(1, n_vars) if j != i_pred_var]]
        color_i = color_i.max(axis=1)
        sc = axs[i_pred_var].scatter(x_plot, y_plot, s=1, c=color_i, cmap=cmap)
        fig.colorbar(sc, ax=axs[i_pred_var], label="Largest other conc.")
        axs[i_pred_var].set_xlabel(r"True Value")
        axs[i_pred_var].set_ylabel(
            r"($y_\text{pred} - y_\text{true})$ / Aleatoric Unc."
        )
        axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
    plt.suptitle(
        "True Value vs. Error/Aleatoric Uncertainty (color=largest other concentration)"
    )
    plot_path = (
        plots_dir
        / "true_vs_rmse_scatter"
        / "true_value_vs_rmse_over_aleatoric_uncertainty_color_largest_other_concentration.png"
    )
    os.makedirs(plot_path.parent, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Saved true value vs rmse/aleatoric uncertainty plot to {plot_path}")
    plt.close(fig)

    # same plot, but for each target variable, the color is chosen according to the sum other variables (except temperature)
    fig, axs = plt.subplots(
        2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
    )
    axs = axs.flatten()
    color_vals = y_test
    cmap = plt.get_cmap("jet")

    for i_pred_var in range(n_vars):
        y_plot = (y_pred[:, i_pred_var] - y_test[:, i_pred_var]) / np.sqrt(
            aleatoric_var[:, i_pred_var]
        )
        x_plot = y_test[:, i_pred_var]
        axs[i_pred_var].axhline(1, ls="--", color="gray", lw=2)
        axs[i_pred_var].axhline(-1, ls="--", color="gray", lw=2)
        axs[i_pred_var].axhspan(-1, 1, color="lightgreen", alpha=0.5)
        y_max_plot = y_plot.max()
        y_min_plot = y_plot.min()
        axs[i_pred_var].axhspan(1, y_max_plot, color="red", alpha=0.2)
        axs[i_pred_var].axhspan(y_min_plot, -1, color="red", alpha=0.2)
        # all except the current variable and temperature
        color_i = color_vals[:, [j for j in range(1, n_vars) if j != i_pred_var]]
        color_i = color_i.sum(axis=1)
        sc = axs[i_pred_var].scatter(x_plot, y_plot, s=1, c=color_i, cmap=cmap)
        fig.colorbar(sc, ax=axs[i_pred_var], label="Sum of other conc.")
        axs[i_pred_var].set_xlabel(r"True Value")
        axs[i_pred_var].set_ylabel(
            r"($y_\text{pred} - y_\text{true})$ / Aleatoric Unc."
        )
        axs[i_pred_var].set_title(f"{var_names[i_pred_var]}")
    plt.suptitle(
        "True Value vs. Error/Aleatoric Uncertainty (color=sum of other concentrations)"
    )
    plot_path = (
        plots_dir
        / "true_vs_rmse_scatter"
        / "true_value_vs_rmse_over_aleatoric_uncertainty_color_sum_other_concentrations.png"
    )
    os.makedirs(plot_path.parent, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Saved true value vs rmse/aleatoric uncertainty plot to {plot_path}")
    plt.close(fig)

    ##########################################################################
    # error/sigma histograms for each variable
    fig, axs = plt.subplots(
        2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
    )
    axs = axs.flatten()
    for i_pred_var in range(n_vars):
        y_plot = (y_pred[:, i_pred_var] - y_test[:, i_pred_var]) / np.sqrt(
            epistemic_var[:, i_pred_var]
        )
        axs[i_pred_var].axvspan(-1, 1, color="lightgreen", alpha=0.5)
        x_max_plot = 5
        axs[i_pred_var].axvspan(1, x_max_plot, color="red", alpha=0.2)
        axs[i_pred_var].axvspan(-x_max_plot, -1, color="red", alpha=0.2)
        counts, bins, _ = axs[i_pred_var].hist(
            y_plot, bins=100, range=[-5, 5], color="blue", alpha=0.7
        )
        axs[i_pred_var].set_xlabel(
            r"($y_\text{pred} - y_\text{true}) / \sigma_\text{E}$"
        )
        if i_pred_var == 0 or i_pred_var == 3:
            axs[i_pred_var].set_ylabel("Count")
        n_in_1_sigma = counts[(bins[:-1] >= -1) & (bins[:-1] <= 1)].sum()
        axs[i_pred_var].set_title(
            f"{var_names[i_pred_var]}, {100 * n_in_1_sigma / counts.sum():.2f}\\%"
            + r" in $\pm 1 \sigma_\text{E}$"
        )
        # draw a normal distribution with mean 0 and std 1
        xs = np.linspace(-5, 5, 200)
        ys = np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi)
        axs[i_pred_var].plot(
            xs,
            ys * counts.max() / ys.max(),
            color="red",
            lw=2,
            label=r"Gauss., $\sigma=1$",
        )
        axs[i_pred_var].legend(loc="upper left")

    plt.suptitle("Histogram of Error/Epistemic Uncertainty")
    plot_path = plots_dir / "histogram_error_over_epistemic_uncertainty.png"
    plt.savefig(plot_path)
    print(f"Saved histogram of error/epistemic uncertainty plot to {plot_path}")
    plt.close(fig)

    # same for aleatoric
    fig, axs = plt.subplots(
        2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
    )
    axs = axs.flatten()
    for i_pred_var in range(n_vars):
        y_plot = (y_pred[:, i_pred_var] - y_test[:, i_pred_var]) / np.sqrt(
            aleatoric_var[:, i_pred_var]
        )
        axs[i_pred_var].axvspan(-1, 1, color="lightgreen", alpha=0.5)
        x_max_plot = 5
        axs[i_pred_var].axvspan(1, x_max_plot, color="red", alpha=0.2)
        axs[i_pred_var].axvspan(-x_max_plot, -1, color="red", alpha=0.2)
        counts, bins, _ = axs[i_pred_var].hist(
            y_plot, bins=100, range=[-5, 5], color="blue", alpha=0.7
        )
        axs[i_pred_var].set_xlabel(
            r"($y_\text{pred} - y_\text{true}) / \sigma_\text{A}$"
        )
        if i_pred_var == 0 or i_pred_var == 3:
            axs[i_pred_var].set_ylabel("Count")
        n_in_1_sigma = counts[(bins[:-1] >= -1) & (bins[:-1] <= 1)].sum()
        axs[i_pred_var].set_title(
            f"{var_names[i_pred_var]}, {100 * n_in_1_sigma / counts.sum():.2f}\\%"
            + r" in $\pm 1 \sigma_\text{A}$"
        )
        # draw a normal distribution with mean 0 and std 1
        xs = np.linspace(-5, 5, 200)
        ys = np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi)
        axs[i_pred_var].plot(
            xs,
            ys * counts.max() / ys.max(),
            color="red",
            lw=2,
            label=r"Gauss., $\sigma=1$",
        )
        axs[i_pred_var].legend(loc="upper left")
    plt.suptitle("Histogram of Error/Aleatoric Uncertainty")
    plot_path = plots_dir / "histogram_error_over_aleatoric_uncertainty.png"
    plt.savefig(plot_path)
    print(f"Saved histogram of error/aleatoric uncertainty plot to {plot_path}")
    plt.close(fig)

    # same for full uncertainty
    fig, axs = plt.subplots(
        2, n_vars // 2, figsize=(2.5 * n_vars, 8), layout="constrained"
    )
    axs = axs.flatten()
    for i_pred_var in range(n_vars):
        y_plot = (y_pred[:, i_pred_var] - y_test[:, i_pred_var]) / np.sqrt(
            aleatoric_var[:, i_pred_var] + epistemic_var[:, i_pred_var]
        )
        axs[i_pred_var].axvspan(-1, 1, color="lightgreen", alpha=0.5)
        x_max_plot = 5
        axs[i_pred_var].axvspan(1, x_max_plot, color="red", alpha=0.2)
        axs[i_pred_var].axvspan(-x_max_plot, -1, color="red", alpha=0.2)
        counts, bins, _ = axs[i_pred_var].hist(
            y_plot, bins=100, range=[-5, 5], color="blue", alpha=0.7
        )
        axs[i_pred_var].set_xlabel(
            r"($y_\text{pred} - y_\text{true}) / \sqrt{\sigma_\text{A}^2 + \sigma_\text{E}^2}$"
        )
        if i_pred_var == 0 or i_pred_var == 3:
            axs[i_pred_var].set_ylabel("Count")
        n_in_1_sigma = counts[(bins[:-1] >= -1) & (bins[:-1] <= 1)].sum()
        axs[i_pred_var].set_title(
            f"{var_names[i_pred_var]}, {100 * n_in_1_sigma / counts.sum():.2f}\\%"
            + r" in $\pm 1 \sigma_\text{A+E}$"
        )
        # draw a normal distribution with mean 0 and std 1
        xs = np.linspace(-5, 5, 200)
        ys = np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi)
        axs[i_pred_var].plot(
            xs,
            ys * counts.max() / ys.max(),
            color="red",
            lw=2,
            label=r"Gauss., $\sigma=1$",
        )
        axs[i_pred_var].legend(loc="upper left")
    plt.suptitle("Histogram of Error/Ep. + Al. Uncertainty")
    plot_path = plots_dir / "histogram_error_over_full_uncertainty.png"
    plt.savefig(plot_path)
    print(f"Saved histogram of error/full uncertainty plot to {plot_path}")
    plt.close(fig)
