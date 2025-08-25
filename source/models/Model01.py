from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
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

    def __init__(self, reduction: str = "mean", alpha: float = 1.0):
        self.reduction = reduction
        self.alpha = alpha

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

        losses = 0.5 * torch.log(sigmas2_pred + eps) + self.alpha * 0.5 * (
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
        fc_hidden_dims = [300, 300]
        self.fc = nn.Sequential()
        self.act = nn.ELU()
        for i, hidden_dim in enumerate(fc_hidden_dims):
            self.fc.add_module(f"fc_{i}", nn.Linear(current_dim, hidden_dim))
            # self.fc.add_module(f"fc_{i}_batchnorm", nn.BatchNorm1d(hidden_dim))
            self.fc.add_module(f"fc_{i}_act", self.act)
            self.fc.add_module(f"fc_{i}_dropout", nn.Dropout(0.1))
            current_dim = hidden_dim
        last_fc_layer_dim = current_dim
        # self.fc.add_module(f"fc_last", nn.Linear(current_dim, self.out_features))

        # for each feature, make a separate MLP that outputs the mean mu and variance sigma2 (with softplus)
        self.feature_heads = nn.ModuleList()
        feature_heads_dims = [200, 200]
        for i in range(self.out_features):
            feature_head = nn.Sequential()
            current_dim = last_fc_layer_dim
            for j, hidden_dim in enumerate(feature_heads_dims):
                feature_head.add_module(
                    f"feature_head_{i}_fc_{j}", nn.Linear(current_dim, hidden_dim)
                )
                feature_head.add_module(f"feature_head_{i}_fc_{j}_act", self.act)
                feature_head.add_module(
                    f"feature_head_{i}_fc_{j}_dropout", nn.Dropout(0.1)
                )
                current_dim = hidden_dim
            feature_head.add_module(
                f"feature_head_{i}_fc_last", nn.Linear(current_dim, 2)
            )  # output mu and sigma^2
            self.feature_heads.append(feature_head)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def forward(self, x):
        x = self.normalizer(x)
        x = self.fc(x)
        feature_outputs = []
        for feature_head in self.feature_heads:
            feature_output = feature_head(x)
            # apply softplus to the second output (sigma^2)
            feature_outputs.append(feature_output)
        mus = torch.stack(
            [fo[:, 0] for fo in feature_outputs], dim=1
        )  # shape: [batch, n_out_features]
        sigmas2 = torch.stack(
            [self.softplus(fo[:, 1]) for fo in feature_outputs], dim=1
        )  # shape: [batch, n_out_features]

        return {"mus": mus, "sigmas2": sigmas2}


class Model_02(nn.Module):
    """
    Multivariate regression model with GRU
    Spectrum (1D tensor) -> GRU, MLP -> Output (1D tensor)
    """

    def __init__(self, hparams, verbose=False):
        super(Model_02, self).__init__()
        self.verbose = verbose
        self.in_length = hparams["in_length"]
        # self.normalizer = nn.LayerNorm(self.in_length)
        self.normalizer = MeanStdNormalizer()
        self.out_features = hparams["n_out_features"]
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hparams["gru_hidden_size"],
            num_layers=hparams["gru_num_layers"],
            batch_first=True,
            dropout=hparams["gru_dropout"] if hparams["gru_num_layers"] > 1 else 0.0,
            bidirectional=True,
        )
        gru_output_dim = hparams["gru_hidden_size"] * 2  # bidirectional
        self.act = nn.ELU()

        current_dim = self.in_length
        fc_hidden_dims = [300, 300]
        self.fc = nn.Sequential()
        for i, hidden_dim in enumerate(fc_hidden_dims):
            self.fc.add_module(f"fc_{i}", nn.Linear(current_dim, hidden_dim))
            # self.fc.add_module(f"fc_{i}_batchnorm", nn.BatchNorm1d(hidden_dim))
            self.fc.add_module(f"fc_{i}_act", self.act)
            self.fc.add_module(f"fc_{i}_dropout", nn.Dropout(0.1))
            current_dim = hidden_dim
        last_fc_layer_dim = current_dim
        self.fc.add_module(f"fc_last", nn.Linear(current_dim, gru_output_dim))

        # for each feature, make a separate MLP that outputs the mean mu and variance sigma2 (with softplus)
        self.feature_heads = nn.ModuleList()
        feature_heads_dims = [200, 200]
        for i in range(self.out_features):
            feature_head = nn.Sequential()
            current_dim = gru_output_dim
            for j, hidden_dim in enumerate(feature_heads_dims):
                feature_head.add_module(
                    f"feature_head_{i}_fc_{j}", nn.Linear(current_dim, hidden_dim)
                )
                feature_head.add_module(f"feature_head_{i}_fc_{j}_act", self.act)
                feature_head.add_module(
                    f"feature_head_{i}_fc_{j}_dropout", nn.Dropout(0.1)
                )
                current_dim = hidden_dim
            feature_head.add_module(
                f"feature_head_{i}_fc_last", nn.Linear(current_dim, 2)
            )  # output mu and sigma^2
            self.feature_heads.append(feature_head)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def forward(self, x):
        x = self.normalizer(x)
        # x = self.fc(x)
        x_enc_fc = self.fc(x)
        x_enc_gru = self.gru(x.unsqueeze(-1))[0][:, -1, :]  # take last output
        x = x_enc_fc + x_enc_gru
        feature_outputs = []
        for feature_head in self.feature_heads:
            feature_output = feature_head(x)
            # apply softplus to the second output (sigma^2)
            feature_outputs.append(feature_output)
        mus = torch.stack(
            [fo[:, 0] for fo in feature_outputs], dim=1
        )  # shape: [batch, n_out_features]
        sigmas2 = torch.stack(
            [self.softplus(fo[:, 1]) for fo in feature_outputs], dim=1
        )  # shape: [batch, n_out_features]

        return {"mus": mus, "sigmas2": sigmas2}


class CNNTimeSeriesRegressor(nn.Module):
    def __init__(
        self,
        hparams: dict,
        verbose: bool = False,
    ):
        """
        Pure CNN encoder for time series regression (sequence → scalar).

        Args:
            in_channels: number of input channels (features per timestep).
            num_blocks: how many Conv blocks to stack.
            base_channels: starting number of channels in first conv block.
            hidden_dim: size of hidden layer in final MLP head.
        """
        super().__init__()

        in_channels = hparams["in_channels"]
        num_blocks = hparams["num_blocks"]
        base_channels = hparams["base_channels"]
        self.out_features = hparams["n_out_features"]
        self.act = nn.ELU()
        self.normalizer = MeanStdNormalizer()
        self.input_length = hparams["in_length"]
        self.dropout_val = hparams["dropout"]

        layers = []
        channels = in_channels
        for i in range(num_blocks):
            out_channels = base_channels * (2**i)  # 64 → 128 → 256 ...
            dilation = 2**i  # 1, 2, 4 ...
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=out_channels,
                        kernel_size=hparams["cnn_enc_kernel_size"],
                        stride=hparams["cnn_enc_stride"],  # downsample
                        padding=dilation * 3,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.SiLU(inplace=True),
                    nn.Dropout(self.dropout_val),
                )
            )
            channels = out_channels

        self.cnn_encoder = nn.Sequential(*layers)

        fc_hidden_dims = hparams["fc_enc_hidden_dims"]
        self.fc_encoder = nn.Sequential()
        current_dim = self.input_length
        for i, hidden_dim in enumerate(fc_hidden_dims):
            self.fc_encoder.add_module(f"fc_{i}", nn.Linear(current_dim, hidden_dim))
            # self.fc.add_module(f"fc_{i}_batchnorm", nn.BatchNorm1d(hidden_dim))
            self.fc_encoder.add_module(f"fc_{i}_act", self.act)
            self.fc_encoder.add_module(f"fc_{i}_dropout", nn.Dropout(self.dropout_val))
            current_dim = hidden_dim
        # output same dimension as cnn_enc
        self.fc_encoder.add_module(f"fc_last", nn.Linear(current_dim, 2 * channels))

        # for each feature, make a separate MLP that outputs the mean mu and variance sigma2 (with softplus)
        self.feature_heads = nn.ModuleList()
        feature_heads_dims = hparams["feature_heads_dims"]
        for i in range(self.out_features):
            feature_head = nn.Sequential()
            current_dim = 2 * channels
            for j, hidden_dim in enumerate(feature_heads_dims):
                feature_head.add_module(
                    f"feature_head_{i}_fc_{j}", nn.Linear(current_dim, hidden_dim)
                )
                feature_head.add_module(f"feature_head_{i}_fc_{j}_act", self.act)
                feature_head.add_module(
                    f"feature_head_{i}_fc_{j}_dropout", nn.Dropout(self.dropout_val)
                )
                current_dim = hidden_dim
            feature_head.add_module(
                f"feature_head_{i}_fc_last", nn.Linear(current_dim, 2)
            )  # output mu and sigma^2
            self.feature_heads.append(feature_head)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def forward(self, x):
        """
        x: (batch, channels, seq_len)
        """
        x = self.normalizer(x)
        x = x.unsqueeze(1)  # add channel dim
        z = self.cnn_encoder(x)  # (B, C, L')

        # Global average & max pooling
        avg_pool = torch.mean(z, dim=-1)
        max_pool, _ = torch.max(z, dim=-1)
        cnn_enc = torch.cat([avg_pool, max_pool], dim=1)

        fc_enc = self.fc_encoder(x.squeeze(1))  # (B, hidden_dim)

        enc = cnn_enc + fc_enc

        feature_outputs = []
        for feature_head in self.feature_heads:
            feature_output = feature_head(enc)
            # apply softplus to the second output (sigma^2)
            feature_outputs.append(feature_output)

        mus = torch.stack(
            [fo[:, 0] for fo in feature_outputs], dim=1
        )  # shape: [batch, n_out_features]
        sigmas2 = torch.stack(
            [self.softplus(fo[:, 1]) for fo in feature_outputs], dim=1
        )  # shape: [batch, n_out_features]

        return {"mus": mus, "sigmas2": sigmas2}


class Model_Lit(L.LightningModule):
    """
    Lightning wrapper for Model_01.
    """

    def __init__(self, hparams, verbose=False):
        super(Model_Lit, self).__init__()
        self.model = hparams["model_class"](hparams, verbose=verbose)
        self.alpha = hparams["alpha"]
        self.save_hyperparameters(hparams)
        self.setup_performed_train = False
        self.setup_performed_test = False
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.criterion = NLLLossMultivariate(reduction="mean", alpha=self.alpha)
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
                num_workers=4,
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
        x = torch.cat([x["x"] for x in self.test_step_outputs], dim=0).detach().cpu()
        y = torch.cat([x["y"] for x in self.test_step_outputs], dim=0).detach().cpu()
        y_pred = (
            torch.cat([x["y_pred"] for x in self.test_step_outputs], dim=0)
            .detach()
            .cpu()
        )
        sigmas2_pred = (
            torch.cat([x["sigmas2_pred"] for x in self.test_step_outputs], dim=0)
            .detach()
            .cpu()
        )

        nllloss_f_indiv = NLLLossMultivariate(reduction=None, alpha=self.alpha)
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
        plot_std_vs_pred_std(model, y, y_pred, sigmas2_pred)

    def configure_optimizers(self):
        """ToDo: Scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
        lr_scheduler_dic = {
            "scheduler": scheduler,
            "monitor": "losses/val_loss",
            "frequency": 1,
        }
        out_dict = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_dic,
        }
        return out_dict


def plot_std_vs_pred_std(model, y_test, y_pred, sigmas2_pred):
    y_test = y_test.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    sigmas2_pred = sigmas2_pred.detach().cpu().numpy()
    n_vars = y_test.shape[1]

    fig, axs = plt.subplots(n_vars, 1, figsize=(8, 3 * n_vars))

    for i_pred_var in range(n_vars):
        bins = np.linspace(min(y_test[:, i_pred_var]), max(y_test[:, i_pred_var]), 50)
        bin_stds = []
        bin_sigmas = []
        bin_errors = []
        diffs = y_test[:, i_pred_var] - y_pred[:, i_pred_var]
        for i in range(len(bins) - 1):
            bin_mask = (y_test[:, i_pred_var] >= bins[i]) & (
                y_test[:, i_pred_var] < bins[i + 1]
            )
            if np.any(bin_mask):
                bin_sigmas.append(np.mean(sigmas2_pred[bin_mask, i_pred_var]))
                bin_stds.append(np.std(diffs[bin_mask]))
            else:
                bin_stds.append(np.nan)
                bin_sigmas.append(np.nan)
        bin_stds = np.array(bin_stds)
        bin_sigmas = np.array(bin_sigmas)

        plt.sca(axs[i_pred_var])
        var_name = (
            model.labels_names[i_pred_var]
            if model.labels_names
            else f"variable_{i_pred_var}"
        )
        plt.scatter(
            bins[:-1], bin_stds, label=f"True Std {var_name}", color="blue", marker="o"
        )
        plt.scatter(
            bins[:-1],
            bin_sigmas,
            label=f"Predicted Std {var_name}",
            color="orange",
            marker="x",
        )
        plt.xlabel("True Values")
        plt.ylabel("Standard Deviation")
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Errors vs Sigma Predictions")

    model.logger.experiment.add_figure(f"std_vs_pred_std_indiv_variables", fig)
    plt.close(fig)


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


def train_model_01():
    """
    This function should just for for training a Model_01_Lit.
    """
    data_dir = Path("data") / "cleaned_up_version"
    n_load = None

    hparams = {
        "model_class": Model_01,
        "data_dir": data_dir / "train_test_split",
        "n_load_train": n_load,
        "batch_size": 32,
        "val_batch_size": 512,
        "in_length": 50,
        "n_out_features": 6,
        "alpha": 20.0,
    }

    lit_logdir = Path("lightning_logs") / "Model_01"
    logger = TensorBoardLogger(
        lit_logdir, name="Model_01", default_hp_metric=False, log_graph=True
    )
    model = Model_01_Lit(hparams)
    callbacks = [LearningRateMonitor(logging_interval="step")]
    trainer = L.Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model)
    trainer.test(model)


def train_model_02():
    """
    This function should just for for training a Model_02_Lit.
    """
    data_dir = Path("data") / "cleaned_up_version"
    n_load = 10_000

    hparams = {
        "model_class": Model_02,
        "data_dir": data_dir / "train_test_split",
        "n_load_train": n_load,
        "batch_size": 32,
        "val_batch_size": 512,
        "in_length": 50,
        "n_out_features": 6,
        "gru_hidden_size": 32,
        "gru_num_layers": 3,
        "gru_dropout": 0.1,
        "alpha": 1.0,
    }

    lit_logdir = Path("lightning_logs") / "Model_02"
    logger = TensorBoardLogger(
        lit_logdir, name="Model_02", default_hp_metric=False, log_graph=True
    )
    model = Model_Lit(hparams)
    callbacks = [LearningRateMonitor(logging_interval="step")]
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    data_dir = Path("data") / "cleaned_up_version"
    n_load = None

    hparams = {
        "model_class": CNNTimeSeriesRegressor,
        "data_dir": data_dir / "train_test_split",
        "n_load_train": n_load,
        "batch_size": 32,
        "val_batch_size": 512,
        "in_length": 50,
        "n_out_features": 6,
        "in_channels": 1,
        "num_blocks": 3,
        "base_channels": 32,
        "cnn_enc_kernel_size": 7,
        "cnn_enc_stride": 2,
        "fc_enc_hidden_dims": [300, 300],
        "feature_heads_dims": [300, 200],
        "alpha": 1.0,
        "dropout": 0.1,
    }

    lit_logdir = Path("lightning_logs") / hparams["model_class"].__name__
    logger = TensorBoardLogger(
        lit_logdir,
        name=hparams["model_class"].__name__,
        default_hp_metric=False,
        log_graph=True,
    )
    model = Model_Lit(hparams)
    callbacks = [LearningRateMonitor(logging_interval="step")]
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model)
    trainer.test(model)
