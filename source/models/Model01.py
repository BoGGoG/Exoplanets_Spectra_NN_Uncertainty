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


class NLLLossMultivariate:
    """
    Negative Log Likelihood with uncertainty estimation for multivariate regression.
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def forward(self, y_pred, sigmas2_pred, y_true):
        """
        y_pred shape: [batch, n_out_features]
        sigmas_pred shape: [batch, n_out_features], the sigma^2 predicted by the model for each variable
        y_true shape: [batch, n_out_features]

        Assuming Gaussian distribution for each output variable.
        For each variable, we compute the NLL:
        0.5 log(sigma^2) + 0.5 * ((y_true - mu)^2 / sigma^2)
        """
        assert y_pred.shape == sigmas2_pred.shape == y_true.shape
        eps = 1e-8  # small value to avoid log(0)

        losses = 0.5 * torch.log(sigmas2_pred + eps) + 0.5 * (
            (y_true - y_pred) ** 2
        ) / (sigmas2_pred + eps)
        if self.reduction == "mean":
            losses = losses.sum(dim=1)  # sum over output features
            loss = losses.mean()
        elif self.reduction == "sum":
            losses = losses.sum(dim=1)  # sum over output features
            loss = losses.sum()
        else:
            loss = losses  # no reduction

        return loss

    def __call__(self, y_pred, sigmas2_pred, y_true):
        return self.forward(y_pred, sigmas2_pred, y_true)


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
        last_fc_layer_dim = current_dim
        # self.fc.add_module(f"fc_last", nn.Linear(current_dim, self.out_features))

        self.mus_model = nn.Sequential()
        mus_model_dims = [20]
        current_dim = last_fc_layer_dim
        for i, hidden_dim in enumerate(mus_model_dims):
            self.mus_model.add_module(f"mus_fc_{i}", nn.Linear(current_dim, hidden_dim))
            self.mus_model.add_module(f"mus_fc_{i}_act", nn.ReLU())
            self.mus_model.add_module(f"mus_fc_{i}_dropout", nn.Dropout(0.1))
            current_dim = hidden_dim
        self.mus_model.add_module(
            f"mus_fc_last", nn.Linear(current_dim, self.out_features)
        )

        self.sigmas_model = nn.Sequential()
        sigmas_model_dims = [20]
        current_dim = last_fc_layer_dim
        for i, hidden_dim in enumerate(sigmas_model_dims):
            self.sigmas_model.add_module(
                f"sigmas_fc_{i}", nn.Linear(current_dim, hidden_dim)
            )
            self.sigmas_model.add_module(f"sigmas_fc_{i}_act", nn.ReLU())
            self.sigmas_model.add_module(f"sigmas_fc_{i}_dropout", nn.Dropout(0.1))
            current_dim = hidden_dim
        self.sigmas_model.add_module(
            f"sigmas_fc_last", nn.Linear(current_dim, self.out_features)
        )
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def forward(self, x):
        x = self.normalizer(x)
        x = self.fc(x)
        mus = self.mus_model(x)
        sigmas2 = self.sigmas_model(x)  # sigma^2
        sigmas2 = self.softplus(sigmas2)  # ensure positive variance

        return {"mus": mus, "sigmas2": sigmas2}


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
        self.criterion = NLLLossMultivariate(reduction="mean")
        self.pred_criterion = nn.MSELoss(reduction="mean")
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
        y_pred = out["mus"]
        sigma2_pred = out["sigmas2"]

        loss = self.criterion(y_pred, sigma2_pred, y)
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
        y_pred = out["mus"]
        sigma2_pred = out["sigmas2"]

        loss = self.criterion(y_pred, sigma2_pred, y)
        pred_loss = self.pred_criterion(y_pred, y)
        self.log(
            "losses/val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "losses/val_pred_loss",
            pred_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        y_pred = out["mus"]
        sigma2_pred = out["sigmas2"]
        loss = self.criterion(y_pred, sigma2_pred, y)
        pred_loss = self.pred_criterion(y_pred, y)

        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_pred_loss",
            pred_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        out_dict = {
            "x": x,
            "y": y,
            "y_pred": y_pred,
            "sigmas2_pred": sigma2_pred,
        }
        self.test_step_outputs.append(out_dict)
        return loss

    def on_test_epoch_end(self):
        x = torch.cat([x["x"] for x in self.test_step_outputs], dim=0)
        y = torch.cat([x["y"] for x in self.test_step_outputs], dim=0)
        y_pred = torch.cat([x["y_pred"] for x in self.test_step_outputs], dim=0)
        sigmas2_pred = torch.cat(
            [x["sigmas2_pred"] for x in self.test_step_outputs], dim=0
        )

        nllloss_f_indiv = NLLLossMultivariate(reduction=None)
        indiv_nllosses_per_variable = nllloss_f_indiv(y_pred, sigmas2_pred, y)
        indiv_nllosses = indiv_nllosses_per_variable.mean(dim=1).detach().cpu().numpy()
        indiv_nllosses_per_variable = indiv_nllosses_per_variable.detach().cpu().numpy()
        lossf_mse = torch.nn.MSELoss(reduction="none")
        indiv_mse_per_var = lossf_mse(y, y_pred)
        indiv_rmse = torch.sqrt(indiv_mse_per_var.mean(dim=1)).detach().cpu().numpy()
        indiv_rmse_per_var = torch.sqrt(indiv_mse_per_var).detach().cpu().numpy()
        plot_indiv_losses_hists(self, indiv_rmse, lossname="RMSE")
        plot_indiv_losses_hists_per_variable(self, indiv_rmse_per_var, lossname="RMSE")
        plot_indiv_losses_hists(self, indiv_nllosses, lossname="NLL")
        plot_indiv_losses_hists_per_variable(
            self, indiv_nllosses_per_variable, lossname="NLL"
        )

    def configure_optimizers(self):
        """ToDo: Scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def plot_indiv_losses_hists(model, indiv_losses, lossname="Loss"):
    # histogram of individual losses
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.hist(indiv_losses, bins=50)
    plt.xlabel(lossname)
    plt.ylabel("Count")
    plt.title("Histogram of Individual Losses")
    model.logger.experiment.add_figure(f"indiv_{lossname}_hist", fig)
    plt.close(fig)


def plot_indiv_losses_hists_per_variable(model, indiv_losses_per_var, lossname="Loss"):
    # for each variable, plot the loss histogram
    n_vars = indiv_losses_per_var.shape[1]
    fig, axs = plt.subplots(n_vars, 1, figsize=(8, 3 * n_vars))
    for i in range(n_vars):
        axs[i].hist(
            indiv_losses_per_var[:, i],
            bins=100,
            label=f"{model.labels_names[i] if model.labels_names else i}",
        )
        axs[i].set_xlabel(lossname if i == n_vars - 1 else "")
        axs[i].set_ylabel("Count")
        axs[i].legend(fontsize="large")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Histogram of Individual Losses per Variable")
    model.logger.experiment.add_figure(f"indiv_{lossname}_per_var_hist", fig)
    plt.close(fig)


def test_NLLLoss():
    # test NLL loss
    criterion_indiv = NLLLossMultivariate(reduction=None)
    criterion = NLLLossMultivariate(reduction="mean")
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y_true = torch.tensor([[1.0, 2.5], [3.1, 3.0]], dtype=torch.float32)
    sigmas2_pred = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)

    loss = criterion_indiv(y_pred, sigmas2_pred, y_true)
    print("Loss without reduction:", loss)

    loss_mean = criterion.forward(y_pred, sigmas2_pred, y_true)
    print("Loss with mean reduction:", loss_mean)


if __name__ == "__main__":
    data_dir = Path("data") / "cleaned_up_version"
    n_load = 5_000

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
        max_epochs=10,
        accelerator="auto",
        devices=1,
        logger=logger,
    )
    trainer.fit(model)
    trainer.test(model)
