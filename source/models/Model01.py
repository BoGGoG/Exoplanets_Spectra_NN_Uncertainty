from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from source.IO import load_spectra


class MeanStdNormalizer:
    """
    Subtracts the mean and divides by the standard deviation for each spectrum.
    This normalizes the spectra to have a mean of 0 and a standard deviation of
    1.
    """

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + self.epsilon)


class Model_01(nn.Module):
    """
    Multivariate regression model:
    Spectrum (1D tensor) -> MLP -> Output (1D tensor)
    """

    def __init__(self, hparams, verbose=False):
        super(Model_01, self).__init__()
        self.verbose = verbose
        self.in_length = hparams["in_length"]
        # self.normalizer = nn.LayerNorm(self.in_length)
        self.normalizer = MeanStdNormalizer()
        self.out_features = hparams["n_out_features"]

        current_dim = self.in_length
        fc_hidden_dims = [50, 50]
        self.fc = nn.Sequential()
        for i, hidden_dim in enumerate(fc_hidden_dims):
            self.fc.add_module(f"fc_{i}", nn.Linear(current_dim, hidden_dim))
            self.fc.add_module(f"fc_{i}_act", nn.ReLU())
            self.fc.add_module(f"fc_{i}_dropout", nn.Dropout(0.1))
            current_dim = hidden_dim
        self.fc.add_module(f"fc_last", nn.Linear(current_dim, self.out_features))

    def forward(self, x):
        x = self.normalizer(x)
        x = self.fc(x)

        return x


class Model_01_Lit(L.LightningModule):
    """
    Lightning wrapper for Model_01.
    """

    def __init__(self, hparams, verbose=False):
        super(Model_01_Lit, self).__init__()
        self.model = Model_01(hparams, verbose=verbose)
        self.save_hyperparameters(hparams)
        self.setup_performed_train = False
        self.setup_performed_test = False
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()
        self.test_step_outputs = []
        self.labels_names = None

    def setup(self, stage=None):
        path_spectra_train = hparams["data_dir"] / "spectra_train.csv"
        path_labels_train = hparams["data_dir"] / "labels_train.csv"
        path_spectra_test = hparams["data_dir"] / "spectra_test.csv"
        path_labels_test = hparams["data_dir"] / "labels_test.csv"

        if (stage == "fit" or stage is None) and not self.setup_performed_train:
            spectra_train = pd.read_csv(
                path_spectra_train, index_col=0, nrows=hparams["n_load_train"]
            ).values
            labels_train = pd.read_csv(path_labels_train, nrows=hparams["n_load_train"])
            self.labels_names = labels_train.columns.tolist()
            labels_train = labels_train.values
            labels_train = self.scaler.fit_transform(labels_train)
            print(f"Loaded training spectra with shape: {spectra_train.shape}")
            print(f"Loaded training labels with shape: {labels_train.shape}")
            print(f"{spectra_train[0:3]}")
            print(f"{labels_train[0:3]}")

            spectra_train = torch.tensor(spectra_train, dtype=torch.float32)
            labels_train = torch.tensor(labels_train, dtype=torch.float32)

            # train val split
            generator = torch.Generator().manual_seed(42)
            train_ds = torch.utils.data.TensorDataset(spectra_train, labels_train)
            train_ds, val_ds = torch.utils.data.random_split(
                train_ds, [0.8, 0.2], generator=generator
            )
            self.train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=hparams["batch_size"],
                shuffle=True,
                drop_last=False,
            )
            self.val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=hparams["val_batch_size"],
                shuffle=False,
                drop_last=False,
            )
            self.setup_performed_train = True

        if stage == "test" and not self.setup_performed_test:
            spectra_test = pd.read_csv(
                path_spectra_test, index_col=0, nrows=hparams["n_load_train"]
            ).values
            labels_test = pd.read_csv(
                path_labels_test, nrows=hparams["n_load_train"]
            ).values
            labels_test = self.scaler.transform(labels_test)

            spectra_test = torch.tensor(spectra_test, dtype=torch.float32)
            labels_test = torch.tensor(labels_test, dtype=torch.float32)
            test_ds = torch.utils.data.TensorDataset(spectra_test, labels_test)
            self.test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=hparams["val_batch_size"],
                shuffle=False,
                drop_last=False,
            )
            self.setup_performed_test = True

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log(
            "losses/train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log(
            "losses/val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y, y_hat)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        out_dict = {
            "x": x,
            "y": y,
            "y_hat": y_hat,
        }
        self.test_step_outputs.append(out_dict)
        return loss

    def on_test_epoch_end(self):
        x = torch.cat([x["x"] for x in self.test_step_outputs], dim=0)
        y = torch.cat([x["y"] for x in self.test_step_outputs], dim=0)
        y_hat = torch.cat([x["y_hat"] for x in self.test_step_outputs], dim=0)

        figs = []
        loss_f = torch.nn.MSELoss(reduction="none")
        indiv_losses_per_var = loss_f(y, y_hat)
        indiv_losses = indiv_losses_per_var.mean(dim=1).detach().cpu().numpy()
        indiv_losses_per_var = indiv_losses_per_var.detach().cpu().numpy()

        # histogram of individual losses
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.hist(indiv_losses, bins=50)
        plt.xlabel("Loss")
        plt.ylabel("Count")
        plt.title("Histogram of Individual Losses")
        self.logger.experiment.add_figure("indiv_losses_hist", fig)
        plt.close(fig)

        # for each variable, plot the loss histogram
        n_vars = y.shape[1]
        fig, axs = plt.subplots(n_vars, 1, figsize=(5, 3 * n_vars))
        for i in range(n_vars):
            axs[i].hist(
                indiv_losses_per_var[:, i],
                bins=100,
                label=f"{self.labels_names[i] if self.labels_names else i}",
            )
            axs[i].set_xlabel("Loss")
            axs[i].set_ylabel("Count")
            axs[i].legend()
        plt.suptitle("Histogram of Individual Losses per Variable")
        plt.tight_layout()
        self.logger.experiment.add_figure("indiv_losses_per_var_hist", fig)

        # self.logger.experiment.add_figure("peaks", figs)

    def configure_optimizers(self):
        """ToDo: Scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    data_dir = Path("data") / "cleaned_up_version"
    n_load = 1_000

    hparams = {
        "data_dir": data_dir / "train_test_split",
        "n_load_train": n_load,
        "batch_size": 32,
        "val_batch_size": 512,
        "in_length": 50,
        "n_out_features": 6,
    }

    lit_logdir = Path("lightning_logs") / "Model_01"
    logger = TensorBoardLogger(
        lit_logdir, name="Model_01", default_hp_metric=False, log_graph=True
    )
    model = Model_01_Lit(hparams)
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        logger=logger,
    )
    trainer.fit(model)
    trainer.test(model)
