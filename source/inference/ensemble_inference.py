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


def load_models(config):
    model_class = config["MODEL"]["model_class"]
    ensemble_directory = (
        Path(config["ENSEMBLE_INFERENCE"]["ENSEMBLE_SAVEDIR"]) / "models" / "ckpt"
    )
    models_names = os.listdir(ensemble_directory)

    models = []
    for m in models_names:
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
        x, [models[0].labels_names[i] if models[0].labels_names else i for i in x]
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

    fig, axs = plt.subplots(n_vars, 1, figsize=(8, 3 * n_vars))
    for i in range(n_vars):
        if y_scale == "linear" and lossname == "RMSE":
            axs[i].hist(
                indiv_losses_per_var[:, i],
                bins=200,
                label=f"{models[0].labels_names[i] if models[0].labels_names else i}",
                density=True,
                range=ranges[i],
            )
        else:
            axs[i].hist(
                indiv_losses_per_var[:, i],
                bins=200,
                label=f"{models[0].labels_names[i] if models[0].labels_names else i}",
                density=True,
            )
        axs[i].set_xscale(x_scale)
        axs[i].set_yscale(y_scale)
        axs[i].set_xlabel(lossname if i == n_vars - 1 else "")
        axs[i].set_ylabel("Count")
        axs[i].legend(fontsize="large")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Histogram of Individual Losses per Variable")
    plot_path = plots_dir / "histogram_individual_losses_per_variable.png"
    plt.savefig(plot_path)
    print(f"Saved histogram of individual losses per variable to {plot_path}")
