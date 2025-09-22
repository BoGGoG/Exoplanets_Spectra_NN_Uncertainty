"""
Quantum model very similar to CNNTimeSeriesRegressor, but with quantum layers acting one the encoding.
"""

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
from collections import namedtuple
import pennylane as qml
from typing import Optional
from source.models.Model01 import MeanStdNormalizer, NLLLossMultivariate, Model_Lit


# model output as named tuple instead of dict, because of some logging issues with dicts
# I think if one has self.example_input_array and wants to log the graph, it needs to be a tuple and not a dict
ModelOutput = namedtuple("ModelOutput", ["mus", "sigmas2"])
torch.set_float32_matmul_precision("highest")  # or "high"


def make_quantum_layer(
    n_qubits: int, n_layers: int, q_out: int, diff_method: str = "best"
):
    """
    Quantum feature extractor with:
      - data re-uploading (inputs injected each layer),
      - alternating entanglement (ring on even layers, ladder/pairing on odd),
      - measurements in X, Y, Z per qubit (then ZZ correlators if more outputs needed).

    Returns a qml.qnn.TorchLayer that accepts shape [batch, n_qubits] and returns [batch, q_out].
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # inputs: vector length n_qubits (one shot / one sample)
        # initial encoding (can be small — we re-upload in each layer)
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # variational blocks with data re-uploading and alternating entanglers
        for l in range(n_layers):
            # parametric single-qubit rotations (trainable)
            for i in range(n_qubits):
                # Rot is expressive (3 params per qubit)
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

            # alternating entanglement:
            if (l % 2) == 0:
                # even layer: ring entangler 0-1,1-2,...,n-1-0
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            else:
                # odd layer: ladder / pair entangler (0-1,2-3,4-5,...). wrap last if odd count
                i = 0
                while i + 1 < n_qubits:
                    qml.CNOT(wires=[i, i + 1])
                    i += 2
                if n_qubits % 2 == 1:
                    # connect last qubit with qubit 0 to avoid isolation
                    qml.CNOT(wires=[n_qubits - 1, 0])

            # data re-uploading: re-inject the classical inputs (as small rotations)
            for i in range(n_qubits):
                qml.RZ(
                    inputs[i] * 0.5, wires=i
                )  # small extra encoding in a different axis

        # build outputs: X, Y, Z per qubit (ordered per qubit)
        outputs = []
        for i in range(n_qubits):
            outputs.append(qml.expval(qml.PauliX(i)))
            outputs.append(qml.expval(qml.PauliY(i)))
            outputs.append(qml.expval(qml.PauliZ(i)))

        # if more outputs required than 3*n_qubits, add ZZ correlators
        base = 3 * n_qubits
        if q_out > base:
            extra_needed = q_out - base
            k = 0
            while k < extra_needed:
                a = k % n_qubits
                b = (a + 1) % n_qubits
                outputs.append(qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)))
                k += 1

        # return only up to q_out features (if caller set q_out smaller)
        return outputs[:q_out]

    x = torch.zeros(n_qubits)  # dummy input
    w = torch.zeros((n_layers, n_qubits, 3), requires_grad=True)

    qml.draw_mpl(circuit)(x, w)
    plt.show()

    # Tell TorchLayer how big the trainable tensor "weights" is
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    return qlayer


class QNN_01(nn.Module):
    def __init__(
        self,
        hparams: dict,
        verbose: bool = False,
    ):
        """
        CNN + dense encoder for time series regression (sequence → scalar).
        Quantum layers added after CNN encoder.

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

        # Quantum part
        self.n_qubits = hparams["n_qubits"]
        self.latent_dim = 2 * channels  # dimension of the input to the quantum layer
        self.projector = nn.Linear(
            self.latent_dim, self.n_qubits, bias=True
        )  # linear layer to project to n_qubits
        self.qlayer = make_quantum_layer(
            n_qubits=self.n_qubits, n_layers=4, q_out=4 * self.n_qubits
        )

        # for each feature, make a separate MLP that outputs the mean mu and variance sigma2 (with softplus)
        self.feature_heads = nn.ModuleList()
        feature_heads_dims = hparams["feature_heads_dims"]
        for i in range(self.out_features):
            feature_head = nn.Sequential()
            current_dim = 4 * self.n_qubits
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

        # Optional: initialize projector small to avoid huge angles at start
        with torch.no_grad():
            if hasattr(self.projector, "weight"):
                self.projector.weight.mul_(0.1)
            if hasattr(self.projector, "bias") and self.projector.bias is not None:
                self.projector.bias.zero_()

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

        angles = self.projector(enc)  # project to n_qubits
        angles = torch.tanh(angles) * np.pi  # map to [-pi, pi]
        # q_feats = self.qlayer(angles)  # [B, q_out]
        q_feats = torch.stack([self.qlayer(a) for a in angles], dim=0)  # no batching

        feature_outputs = []
        for feature_head in self.feature_heads:
            feature_output = feature_head(q_feats)
            # apply softplus to the second output (sigma^2)
            feature_outputs.append(feature_output)

        mus = torch.stack(
            [fo[:, 0] for fo in feature_outputs], dim=1
        )  # shape: [batch, n_out_features]
        sigmas2 = torch.stack(
            [self.softplus(fo[:, 1]) for fo in feature_outputs], dim=1
        )  # shape: [batch, n_out_features]

        return ModelOutput(mus=mus, sigmas2=sigmas2)

    def reset_weights(self, generator=None):
        """
        Model has linear layers as well as convolutional layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, generator=generator, gain=1.0)
                # nn.init.kaiming_uniform(m.weight, generator=generator, a=0, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(
                    m.weight, generator=generator, a=0, nonlinearity="relu"
                )
                nn.init.zeros_(m.bias)

class QNN_NoQ(nn.Module):
    def __init__(
        self,
        hparams: dict,
        verbose: bool = False,
    ):
        """
        CNN + dense encoder for time series regression (sequence → scalar).
        Same dimension reduction as for QNN_01, but no quantum layers.
        This is to test if the quantum layer actually does anything or if it's just the dimension reduction.

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

        # Quantum part
        self.n_qubits = hparams["n_qubits"]
        self.latent_dim = 2 * channels  # dimension of the input to the quantum layer
        self.projector = nn.Linear(
            self.latent_dim, self.n_qubits, bias=True
        )  # linear layer to project to n_qubits
        # self.qlayer = make_quantum_layer(
        #     n_qubits=self.n_qubits, n_layers=4, q_out=4 * self.n_qubits
        # )
        self.qlayer_substitute = nn.Linear(self.n_qubits, 4 * self.n_qubits)

        # for each feature, make a separate MLP that outputs the mean mu and variance sigma2 (with softplus)
        self.feature_heads = nn.ModuleList()
        feature_heads_dims = hparams["feature_heads_dims"]
        for i in range(self.out_features):
            feature_head = nn.Sequential()
            current_dim = 4 * self.n_qubits
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

        # Optional: initialize projector small to avoid huge angles at start
        with torch.no_grad():
            if hasattr(self.projector, "weight"):
                self.projector.weight.mul_(0.1)
            if hasattr(self.projector, "bias") and self.projector.bias is not None:
                self.projector.bias.zero_()

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

        angles = self.projector(enc)  # project to n_qubits
        angles = torch.tanh(angles) * np.pi  # map to [-pi, pi]
        # q_feats = self.qlayer(angles)  # [B, q_out]
        # q_feats = torch.stack([self.qlayer(a) for a in angles], dim=0)  # no batching
        # q_feats = angles  # just use the angles as features, no quantum layer
        q_feats = self.qlayer_substitute(angles)  # substitute quantum layer with linear layer

        feature_outputs = []
        for feature_head in self.feature_heads:
            feature_output = feature_head(q_feats)
            # apply softplus to the second output (sigma^2)
            feature_outputs.append(feature_output)

        mus = torch.stack(
            [fo[:, 0] for fo in feature_outputs], dim=1
        )  # shape: [batch, n_out_features]
        sigmas2 = torch.stack(
            [self.softplus(fo[:, 1]) for fo in feature_outputs], dim=1
        )  # shape: [batch, n_out_features]

        return ModelOutput(mus=mus, sigmas2=sigmas2)

    def reset_weights(self, generator=None):
        """
        Model has linear layers as well as convolutional layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, generator=generator, gain=1.0)
                # nn.init.kaiming_uniform(m.weight, generator=generator, a=0, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(
                    m.weight, generator=generator, a=0, nonlinearity="relu"
                )
                nn.init.zeros_(m.bias)


def test_QNN_01():
    data_dir = Path("data") / "cleaned_up_version"
    n_load = 500

    hparams = {
        "model_class": QNN_01,
        "data_dir": data_dir / "train_test_split",
        "n_load_train": n_load,
        "batch_size": 32,
        "val_batch_size": 128,
        "in_length": 51,
        "n_out_features": 6,
        "in_channels": 1,
        "num_blocks": 3,
        "base_channels": 64,
        "cnn_enc_kernel_size": 7,
        "cnn_enc_stride": 2,
        "fc_enc_hidden_dims": [450],
        "feature_heads_dims": [300, 500, 400],
        "alpha": 1.0,
        "dropout": 0.1,
        "n_qubits": 8,
        "lr": 1e-3,
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
        max_epochs=1,
        # accelerator="gpu",
        accelerator="cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    data_dir = Path("data") / "cleaned_up_version"
    n_load = 5000

    hparams = {
        "model_class": QNN_01,
        "data_dir": data_dir / "train_test_split",
        "n_load_train": n_load,
        "batch_size": 32,
        "val_batch_size": 128,
        "in_length": 51,
        "n_out_features": 6,
        "in_channels": 1,
        "num_blocks": 3,
        "base_channels": 64,
        "cnn_enc_kernel_size": 7,
        "cnn_enc_stride": 2,
        "fc_enc_hidden_dims": [450],
        "feature_heads_dims": [300, 500, 400],
        "alpha": 1.0,
        "dropout": 0.1,
        "n_qubits": 8, # not used here for quantum, but for dimension reduction
        "lr": 1e-3,
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
        max_epochs=50,
        # accelerator="gpu",
        accelerator="cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model)
    trainer.test(model)
