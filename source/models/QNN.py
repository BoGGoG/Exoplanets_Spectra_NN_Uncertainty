"""
Quantum model very similar to CNNTimeSeriesRegressor, but with quantum layers acting one the encoding.
"""

import math
from collections import namedtuple
from pathlib import Path
from typing import Optional

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy.util.langhelpers import repr_tuple_names
import pennylane as qml
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

from source.models.Model01 import Model_Lit, NLLLossMultivariate
from source.models.normalizers import MeanStdNormalizer

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


def make_quantum_layer_52_to_8(
    n_qubits: int = 8, n_layers: int = 7, q_out: int = 24, diff_method: str = "best"
):
    """
    Quantum layer for 52 features using 8 qubits and 7 layers.
      - Splits 52 features into 7 chunks of 8 (pads last if needed).
      - Each chunk is encoded at its corresponding layer.
      - Each layer: Rot trainables + alternating entanglement + data re-uploading.
      - Measurements: X,Y,Z per qubit, plus ZZ correlators if needed.

    Maximum number of outputs with 8 qubits is 3*8 + (8*7)/2 = 52.
    Or more generally: 3*n_qubits + n_qubits*(n_qubits-1)/2.
    """

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # inputs: vector length 52
        # pad to multiple of n_qubits
        pad_len = math.ceil(len(inputs) / n_qubits) * n_qubits - len(inputs)
        padded = torch.cat([inputs, torch.zeros(pad_len, device=inputs.device)])

        # reshape into (n_layers, n_qubits)
        chunks = padded.reshape(n_layers, n_qubits)

        for l in range(n_layers):
            # data encoding for this chunk
            for i in range(n_qubits):
                qml.RY(chunks[l, i], wires=i)

            # variational rotations
            for i in range(n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

            # alternating entanglement
            if (l % 2) == 0:
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            else:
                i = 0
                while i + 1 < n_qubits:
                    qml.CNOT(wires=[i, i + 1])
                    i += 2
                if n_qubits % 2 == 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            # optional re-upload: inject features again via small Z-rotations
            for i in range(n_qubits):
                qml.RZ(chunks[l, i] * 0.5, wires=i)

        # build outputs
        # outputs = []
        # for i in range(n_qubits):
        #     outputs.append(qml.expval(qml.PauliX(i)))
        #     outputs.append(qml.expval(qml.PauliY(i)))
        #     outputs.append(qml.expval(qml.PauliZ(i)))
        #
        # base = 3 * n_qubits
        # if q_out > base:
        #     extra_needed = q_out - base
        #     k = 0
        #     while k < extra_needed:
        #         a = k % n_qubits
        #         b = (a + 1) % n_qubits
        #         outputs.append(qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)))
        #         k += 1
        outputs = []
        # 1. single-qubit Pauli X,Y,Z
        n_outputs = 0
        for i in range(n_qubits):  # this is ugly ...
            outputs.append(qml.expval(qml.PauliX(i)))
            n_outputs += 1
            if n_outputs >= q_out:  # don't compute more than needed
                return outputs[:q_out]
            outputs.append(qml.expval(qml.PauliY(i)))
            n_outputs += 1
            if n_outputs >= q_out:  # don't compute more than needed
                return outputs[:q_out]
            outputs.append(qml.expval(qml.PauliZ(i)))
            n_outputs += 1
            if n_outputs >= q_out:  # don't compute more than needed
                return outputs[:q_out]

        # 2. all unique ZZ correlators
        if n_outputs >= q_out:  # don't compute more than needed
            return outputs[:q_out]
        for a in range(n_qubits):
            for b in range(a + 1, n_qubits):
                outputs.append(qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)))
                n_outputs += 1
                if n_outputs >= q_out:  # don't compute more than needed
                    return outputs[:q_out]

        return outputs[:q_out]

    # Test draw
    # x = torch.zeros(52)
    # w = torch.zeros((n_layers, n_qubits, 3), requires_grad=True)
    # qml.draw_mpl(circuit)(x, w)
    # plt.show()

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    return qlayer


def make_time_series_quantum_layer_01(
    n_qubits: int = 8,
    input_length: int = 51,
    diff_method: str = "best",
):
    """
    Quantum layer for time series data using 8 qubits.
    Half of the qubits are designated as "data qubits" and the other half as "memory qubits".
    The time series is split into chunks of 4 (matching the number of data qubits).
    For each chunk:
    - The chunk is encoded into the data qubits via Ry rotations.
    - A Rot gate is applied to all qubits (data + memory).
    - The data qubits are entangled in a ring using CNOT gates.
    - Data re-uploading is performed with small Rz rotations on the data qubits.
    - Another Rot gate is applied to the data qubits.
    - The data qubits are entangled with the memory qubits using CRY gates (one per data-memory pair).
    - The memory qubits are entangled in a ring using CNOT gates.
    This process is repeated for all chunks.
    Finally, measurements are performed to produce the output features.
    Always 52 outputs for 8 qubits and other n_qubits is not tested and probably won't work.
    """

    dev = qml.device("default.qubit", wires=n_qubits)
    q_out = 52

    chunksize = 4  # half of n_qubits
    # how many layers do we need to encode the full input?
    n_layers = math.ceil(input_length / chunksize)
    data_qubits = list(range(chunksize))
    memory_qubits = list(range(chunksize, n_qubits))

    print(f"{n_layers=}")
    data_pairs = [(0, 1), (1, 2), (2, 3), (0, 2)]

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights, phi, reupload_scale, data_entangles, memory_entangles):
        # inputs: vector length 52
        # pad to multiple of n_qubits
        pad_len = math.ceil(len(inputs) / chunksize) * chunksize - len(inputs)
        padded = torch.cat([inputs, torch.zeros(pad_len, device=inputs.device)])

        # reshape into (n_layers, n_qubits)
        chunks = padded.reshape(n_layers, chunksize)

        ##### main circuit part ####
        for l in range(n_layers):
            for i in data_qubits:
                qml.RY(chunks[l, i], wires=i)

            for i in data_qubits + memory_qubits:
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

            # entangle data qubits in a ring
            # for i in range(len(data_qubits) - 1):
            #     qml.CNOT(wires=[data_qubits[i], data_qubits[i + 1]])
            # qml.CNOT(wires=[data_qubits[-1], data_qubits[0]])
            for p_idx, (a, b) in enumerate(data_pairs):
                qml.IsingZZ(data_entangles[l, p_idx], wires=[a, b])

            # data re-uploading
            for i in data_qubits:
                qml.RZ(chunks[l, i] * reupload_scale[l, i], wires=i)

            for i in data_qubits:
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

            qml.Barrier(wires=data_qubits, only_visual=True)

            # entangle data qubits with memory qubits, make angle learnable parameter
            # for i in range(len(data_qubits)):
            #     qml.CRY(wires=[data_qubits[i], memory_qubits[i]], phi=phi[l, i])
            for i, dq in enumerate(data_qubits):
                for j, mq in enumerate(memory_qubits):
                    qml.CRY(phi[l, i, j], wires=[dq, mq])

            # entangle memory qubits in a ring with IsingZZ
            # for i in range(len(memory_qubits) - 1):
            #     qml.CNOT(wires=[memory_qubits[i], memory_qubits[i + 1]])
            # qml.CNOT(wires=[memory_qubits[-1], memory_qubits[0]])
            for i in range(len(memory_qubits)):
                qml.IsingZZ(
                    memory_entangles[l, i],
                    wires=[
                        memory_qubits[i],
                        memory_qubits[(i + 1) % len(memory_qubits)],
                    ],
                )

            qml.Barrier(wires=data_qubits + memory_qubits, only_visual=True)

        # done with most of the circuit, now just read out
        outputs = []
        # 1. single-qubit Pauli X,Y,Z
        n_outputs = 0
        for i in range(n_qubits):  # this is ugly ...
            outputs.append(qml.expval(qml.PauliX(i)))
            n_outputs += 1
            outputs.append(qml.expval(qml.PauliY(i)))
            n_outputs += 1
            outputs.append(qml.expval(qml.PauliZ(i)))
            n_outputs += 1

        # 2. all unique ZZ correlators
        for a in range(n_qubits):
            for b in range(a + 1, n_qubits):
                outputs.append(qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)))
                n_outputs += 1

        return outputs  # [52]

    ## Test draw
    x = torch.zeros(51)
    weights = torch.zeros((n_layers, n_qubits, 3), requires_grad=True)
    phis = torch.zeros(
        (n_layers, len(data_qubits), len(memory_qubits)), requires_grad=True
    )
    reupload_scales = torch.ones((n_layers, len(data_qubits)), requires_grad=True)
    data_entangles = torch.zeros(((n_layers, len(data_pairs))), requires_grad=True)
    memory_entangles = torch.zeros(((n_layers, len(data_pairs))), requires_grad=True)

    # figs = qml.draw_mpl(circuit, max_length=60)(
    #     x, weights, phis, reupload_scales, data_entangles
    # )
    # for ii, f in enumerate(figs):
    #     path = Path("figures") / f"quantum_circuit_time_series_01_{ii}.png"
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     f[0].savefig(path)
    #     print(f"Saved circuit diagram to {path}")

    # specs of the circuit
    specs_fun = qml.specs(circuit)
    print(
        specs_fun(x, weights, phis, reupload_scales, data_entangles, memory_entangles)
    )

    weight_shapes = {
        "weights": (n_layers, n_qubits, 3),
        "phi": (
            n_layers,
            len(data_qubits),
            len(memory_qubits),
        ),  # one CRY angle per data→memory pair
        "reupload_scale": (
            n_layers,
            len(data_qubits),
        ),  # one scale factor per data qubit
        "data_entangles": (n_layers, len(data_pairs)),
        "memory_entangles": (n_layers, len(data_pairs)),
    }
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    return qlayer


def make_time_series_quantum_layer_02(
    n_qubits: int = 7,
    input_length: int = 7,
    input_channels: int = 4,
    diff_method: str = "best",
):
    """
    Very different model. Takes input of shape [4, 7], where 4 is the channel dim and 7 is the length.
    Every time step is encoded separately, but because there are 4 channels, we can encode 4 values at once.
    """

    dev = qml.device("default.qubit", wires=n_qubits)

    data_qubits = list(range(input_channels))
    memory_qubits = list(range(input_channels, n_qubits))
    data_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights, phi, reupload_scale, data_entangles, memory_entangles):
        inputs = inputs.reshape(input_channels, input_length)

        ##### main circuit part ####
        for l in range(input_length):
            # upload data
            for i in data_qubits:
                qml.RY(inputs[i, l], wires=i)

            for i in data_qubits + memory_qubits:
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

            # ring entanglement
            for p_idx, (a, b) in enumerate(data_pairs):
                qml.IsingZZ(data_entangles[l, p_idx], wires=[a, b])

            # data re-uploading
            for i in data_qubits:
                qml.RZ(inputs[i, l] * reupload_scale[l, i], wires=i)

            for i in data_qubits:
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

            qml.Barrier(wires=data_qubits, only_visual=True)

            # couple data qubits with memory qubits
            for i, dq in enumerate(data_qubits):
                for j, mq in enumerate(memory_qubits):
                    # qml.CRY(phi[i, j], wires=[dq, mq])
                    qml.IsingZZ(phi[l, i, j], wires=[dq, mq])

            for i in range(len(memory_qubits)):
                qml.IsingZZ(
                    memory_entangles[l, i],
                    wires=[
                        memory_qubits[i],
                        memory_qubits[(i + 1) % len(memory_qubits)],
                    ],
                )

            qml.Barrier(wires=data_qubits + memory_qubits, only_visual=True)

        # done with most of the circuit, now just read out
        outputs = []
        # 1. single-qubit Pauli X,Y,Z
        for i in range(n_qubits):  # this is ugly ...
            outputs.append(qml.expval(qml.PauliX(i)))
            outputs.append(qml.expval(qml.PauliY(i)))
            outputs.append(qml.expval(qml.PauliZ(i)))

        # 2. all unique ZZ correlators
        for a in range(n_qubits):
            for b in range(a + 1, n_qubits):
                outputs.append(qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)))

        return outputs  # [52]

    ## Test draw
    x = torch.zeros(4, 7)
    weights = torch.zeros((input_length, n_qubits, 3), requires_grad=True)
    phis = torch.zeros(
        (input_length, len(data_qubits), len(memory_qubits)), requires_grad=True
    )
    reupload_scales = torch.ones((input_length, len(data_qubits)), requires_grad=True)
    data_entangles = torch.zeros((input_length, len(data_pairs)), requires_grad=True)
    memory_entangles = torch.zeros((input_length, len(data_pairs)), requires_grad=True)

    figs = qml.draw_mpl(circuit, max_length=60)(
        x, weights, phis, reupload_scales, data_entangles, memory_entangles
    )
    for ii, f in enumerate(figs):
        path = Path("figures") / "QNN_04" / f"quantum_circuit_time_series_01_{ii}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        f[0].savefig(path)
        print(f"Saved circuit diagram to {path}")

    # specs of the circuit
    specs_fun = qml.specs(circuit)
    print(
        specs_fun(x, weights, phis, reupload_scales, data_entangles, memory_entangles)
    )

    weight_shapes = {
        "weights": (input_length, n_qubits, 3),
        "phi": (
            input_length,
            len(data_qubits),
            len(memory_qubits),
        ),  # one CRY angle per data→memory pair
        "reupload_scale": (
            input_length,
            len(data_qubits),
        ),  # one scale factor per data qubit
        "data_entangles": (input_length, len(data_pairs)),
        "memory_entangles": (input_length, len(data_pairs)),
    }
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


class QNN_02(nn.Module):
    def __init__(
        self,
        hparams: dict,
        verbose: bool = False,
    ):
        """
        Completely different model.
        This QNN takes the full spectrum as input and inputs it to a quantum layer.
        The quantum layer cannot handle the full input with only 8 qubits, so we do
        a trick: We start with 8 inputs and at each layer we add 8 more as data re-uploading.
        """
        super().__init__()

        self.out_features = hparams["n_out_features"]
        self.act = nn.ELU()
        self.normalizer = MeanStdNormalizer()
        self.input_length = hparams["in_length"]
        self.dropout_val = hparams["dropout"]
        self.enc = nn.Linear(self.input_length, self.input_length)
        self.use_linear_encoder = hparams["use_linear_encoder"]

        # Quantum part
        self.n_qubits = hparams["n_qubits"]
        self.qlayer = make_quantum_layer_52_to_8(
            n_qubits=self.n_qubits, n_layers=7, q_out=hparams["n_q_out"]
        )
        # linear block that mixes the outputs of the quantum layer a bit
        self.linear = nn.Sequential()
        self.enc_dim = hparams["encoding_dim"]
        current_dim = hparams["n_q_out"]
        for i in range(len(hparams["linear_dims"])):
            self.linear.add_module(
                f"linear_fc_{i}", nn.Linear(current_dim, hparams["linear_dims"][i])
            )
            self.linear.add_module(f"linear_fc_{i}_act", self.act)
            self.linear.add_module(
                f"linear_fc_{i}_dropout", nn.Dropout(self.dropout_val)
            )
            current_dim = hparams["linear_dims"][i]
        self.linear.add_module("linear_fc_last", nn.Linear(current_dim, self.enc_dim))
        self.linear.add_module("linear_fc_last_act", self.act)
        self.linear.add_module("linear_fc_last_dropout", nn.Dropout(self.dropout_val))

        # for each feature, make a separate MLP that outputs the mean mu and variance sigma2 (with softplus)
        self.feature_heads = nn.ModuleList()
        feature_heads_dims = hparams["feature_heads_dims"]
        for i in range(self.out_features):
            feature_head = nn.Sequential()
            current_dim = self.enc_dim
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
        x: (batch, seq_len)
        """
        x = self.normalizer(x)
        if self.use_linear_encoder:
            x = self.enc(x)
        angles = torch.tanh(x) * np.pi  # map to [-pi, pi]
        q_feats = torch.stack(
            [self.qlayer(a) for a in angles], dim=0
        )  # no batching, so loop over batch
        q_feats = torch.squeeze(q_feats)
        q_feats = self.linear(q_feats)

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


class QNN_03(nn.Module):
    def __init__(
        self,
        hparams: dict,
        verbose: bool = False,
    ):
        """
        Completely different model.
        This QNN takes the full spectrum as input and inputs it to a quantum layer.
        The quantum layer cannot handle the full input with only 8 qubits, so we do
        a trick: We start with 8 inputs and at each layer we add 8 more as data re-uploading.
        """
        super().__init__()

        self.out_features = hparams["n_out_features"]
        self.act = nn.ELU()
        self.normalizer = MeanStdNormalizer()
        self.input_length = hparams["in_length"]
        self.dropout_val = hparams["dropout"]
        self.enc = nn.Linear(self.input_length, self.input_length)
        self.use_linear_encoder = hparams["use_linear_encoder"]

        # Quantum part
        self.n_qubits = hparams["n_qubits"]
        self.qlayer = make_time_series_quantum_layer_01(
            n_qubits=self.n_qubits, input_length=self.input_length
        )
        # linear block that mixes the outputs of the quantum layer a bit
        self.linear = nn.Sequential()
        self.enc_dim = hparams["encoding_dim"]
        current_dim = 52
        for i in range(len(hparams["linear_dims"])):
            self.linear.add_module(
                f"linear_fc_{i}", nn.Linear(current_dim, hparams["linear_dims"][i])
            )
            self.linear.add_module(f"linear_fc_{i}_act", self.act)
            self.linear.add_module(
                f"linear_fc_{i}_dropout", nn.Dropout(self.dropout_val)
            )
            current_dim = hparams["linear_dims"][i]
        self.linear.add_module("linear_fc_last", nn.Linear(current_dim, self.enc_dim))
        self.linear.add_module("linear_fc_last_act", self.act)
        self.linear.add_module("linear_fc_last_dropout", nn.Dropout(self.dropout_val))

        # for each feature, make a separate MLP that outputs the mean mu and variance sigma2 (with softplus)
        self.feature_heads = nn.ModuleList()
        feature_heads_dims = hparams["feature_heads_dims"]
        for i in range(self.out_features):
            feature_head = nn.Sequential()
            current_dim = self.enc_dim
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

        # custom init for quantum layer parameters
        for name, param in self.qlayer.named_parameters():
            if "reupload_scale" in name:
                torch.nn.init.constant_(param, 1.0)  # start with scaling = 1
            elif "phi" in name:
                torch.nn.init.normal_(param, mean=np.pi / 4.0, std=0.05)
            elif "data_entangles" in name:
                torch.nn.init.normal_(param, mean=np.pi / 4.0, std=0.05)  # tiny angles
            elif "memory_entangles" in name:
                torch.nn.init.normal_(param, mean=np.pi / 4.0, std=0.05)  # tiny angles
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        x = self.normalizer(x)
        if self.use_linear_encoder:
            x = self.enc(x)
        angles = torch.tanh(x) * np.pi  # map to [-pi, pi]
        q_feats = torch.stack(
            [self.qlayer(a) for a in angles], dim=0
        )  # no batching, so loop over batch
        q_feats = torch.squeeze(q_feats)
        q_feats = self.linear(q_feats)

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


class QNN_04(nn.Module):
    def __init__(
        self,
        hparams: dict,
        verbose: bool = False,
    ):
        """
        First, a CNN reduces the input length from 51 to (8, channels).
        Then, the QRNN takes the outputs of the CNN.
        """
        super().__init__()

        self.out_features = hparams["n_out_features"]
        self.act = nn.ELU()
        self.normalizer = MeanStdNormalizer()
        self.input_length = hparams["in_length"]
        self.dropout_val = hparams["dropout"]
        self.enc = nn.Linear(self.input_length, self.input_length)
        self.use_linear_encoder = hparams["use_linear_encoder"]

        # CNN part
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1
            ),  # (B, 16, 26)
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
            ),  # (B, 32, 13)
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(
                in_channels=32, out_channels=4, kernel_size=3, stride=2, padding=1
            ),  # (B, 64, 7)
            nn.BatchNorm1d(4),
        )

        # Quantum part
        self.n_qubits = hparams["n_qubits"]
        self.qlayer = make_time_series_quantum_layer_02(
            n_qubits=self.n_qubits, input_length=7
        )
        # linear block that mixes the outputs of the quantum layer a bit
        self.linear = nn.Sequential()
        self.enc_dim = hparams["encoding_dim"]
        current_dim = (
            self.n_qubits * 3 + (self.n_qubits * (self.n_qubits - 1)) // 2
        )  # X, Y, Z and ZZ correlators from QNN_04
        for i in range(len(hparams["linear_dims"])):
            self.linear.add_module(
                f"linear_fc_{i}", nn.Linear(current_dim, hparams["linear_dims"][i])
            )
            self.linear.add_module(f"linear_fc_{i}_act", self.act)
            self.linear.add_module(
                f"linear_fc_{i}_dropout", nn.Dropout(self.dropout_val)
            )
            current_dim = hparams["linear_dims"][i]
        self.linear.add_module("linear_fc_last", nn.Linear(current_dim, self.enc_dim))
        self.linear.add_module("linear_fc_last_act", self.act)
        self.linear.add_module("linear_fc_last_dropout", nn.Dropout(self.dropout_val))

        # for each feature, make a separate MLP that outputs the mean mu and variance sigma2 (with softplus)
        self.feature_heads = nn.ModuleList()
        feature_heads_dims = hparams["feature_heads_dims"]
        for i in range(self.out_features):
            feature_head = nn.Sequential()
            current_dim = self.enc_dim
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

        # custom init for quantum layer parameters
        for name, param in self.qlayer.named_parameters():
            if "reupload_scale" in name:
                torch.nn.init.constant_(param, 0.5)  # start with scaling = 1
            elif "phi" in name:
                torch.nn.init.normal_(param, mean=np.pi / 4.0, std=0.05)
            elif "data_entangles" in name:
                torch.nn.init.normal_(param, mean=np.pi / 4.0, std=0.05)  # tiny angles
            elif "memory_entangles" in name:
                torch.nn.init.normal_(param, mean=np.pi / 4.0, std=0.05)  # tiny angles
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        x = self.normalizer(x)
        cnn_out = self.cnn(x.unsqueeze(1))  # (B, 4, 7)
        angles = torch.tanh(cnn_out) * np.pi  # map to [-pi, pi]
        q_feats = torch.stack(
            [self.qlayer(a.flatten()) for a in angles], dim=0
        )  # no batching, so loop over batch
        q_feats = self.linear(q_feats)

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
        q_feats = self.qlayer_substitute(
            angles
        )  # substitute quantum layer with linear layer

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


def test_QNN_01():
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
        "n_qubits": 8,  # not used here for quantum, but for dimension reduction
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


def test_QNN_02():
    data_dir = Path("data") / "cleaned_up_version"
    n_load = 500

    hparams = {
        "model_class": QNN_02,
        "data_dir": data_dir / "train_test_split",
        "n_load_train": n_load,
        "batch_size": 32,
        "val_batch_size": 128,
        "in_length": 51,
        "n_out_features": 6,
        "linear_dims": [100, 100],
        "encoding_dim": 52,
        "feature_heads_dims": [300, 500, 400],
        "use_linear_encoder": False,
        "n_q_out": 32,
        "alpha": 1.0,
        "dropout": 0.1,
        "n_qubits": 8,  # not used here for quantum, but for dimension reduction
        "lr": 1e-3,
    }

    lit_logdir = Path("lightning_logs") / hparams["model_class"].__name__
    logger = TensorBoardLogger(
        lit_logdir,
        name=hparams["model_class"].__name__,
        default_hp_metric=False,
        log_graph=True,
    )
    model = Model_Lit(
        hparams
    )  # first need to load Model_Lit. Not loaded usually because of circular imports
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


if __name__ == "__main__":
    data_dir = Path("data") / "cleaned_up_version"
    n_load = 1_000

    hparams = {
        "model_class": QNN_04,
        "data_dir": data_dir / "train_test_split",
        "n_load_train": n_load,
        "batch_size": 32,
        "val_batch_size": 128,
        "in_length": 51,
        "n_out_features": 6,
        "linear_dims": [150],
        "encoding_dim": 68,
        "feature_heads_dims": [350, 400],
        "use_linear_encoder": True,
        # "n_q_out": 32,
        "alpha": 1.0,
        "dropout": 0.1,
        "n_qubits": 7,  # not used here for quantum, but for dimension reduction
        "lr": 1e-3,
    }

    lit_logdir = Path("lightning_logs") / hparams["model_class"].__name__
    logger = TensorBoardLogger(
        lit_logdir,
        name=hparams["model_class"].__name__,
        default_hp_metric=False,
        log_graph=True,
    )
    model = Model_Lit(
        hparams
    )  # first need to load Model_Lit. Not loaded usually because of circular imports
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

    # qlayer = make_time_series_quantum_layer_01(n_qubits=8, input_length=51)
