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
    for m in models[:3]:
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
        sigmas_pred = torch.cat([p[1].cpu() for p in pred]).numpy()
        output = np.array([mus_pred, sigmas_pred])
        predictions.append(output)
    predictions = np.array(predictions)
    predictions = predictions.transpose(
        0, 2, 3, 1
    )  # [model, sample, output_dim, (mu,sigma)]
    print(f"Predictions shape from all models: {predictions.shape}")
    print(predictions[:, 0, :])
