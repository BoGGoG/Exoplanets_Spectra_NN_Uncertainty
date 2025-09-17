"""
hpopt script specifically for Temperature only model
"""

import argparse
import configparser
import os
import textwrap
from datetime import datetime
from pathlib import Path

import lightning as L
import optuna
from pandas.core import base
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback

from source.models.ModelT import CNNTimeSeriesRegressorT, ModelT_Lit
from source.IO import read_config


# torch.set_float32_matmul_precision("high")  # or medium/high/highest
# ignore optuna UserWarning
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)


def setup_CNNTimeSeriesRegressorT_hparams(
    trial: optuna.trial.Trial, config: configparser.ConfigParser
) -> dict:
    n_load_train = config.getint("MODEL", "n_events_hpopt")
    epochs = config.getint("MODEL", "epochs_hpopt")
    rundir = config.get("DIRECTORIES", "rundir")
    data_dir = Path("data") / "cleaned_up_version"

    max_norm_clip = config.getfloat("MODEL", "max_norm_clip")

    k_batchsize = trial.suggest_int(
        "k_batchsize", 3, 6
    )  # scale lr when scaling batchsize
    swa_lrs = 1e-5
    swa_start = config.getint("MODEL", "swa_start_hpopt")
    dropout_val = trial.suggest_categorical("dropout_val", [0.1, 0.3, 0.5])
    # activation = trial.suggest_categorical(
    #     "activation", ["relu", "elu", "gelu", "silu"]
    # )
    cnn_channels = trial.suggest_categorical("cnn_channels", [16, 32, 64])
    cnn_enc_kernel_size = trial.suggest_int("cnn_enc_kernel_size", 3, 7, step=2)
    cnn_enc_stride = trial.suggest_int("cnn_enc_stride", 1, 5)
    cnn_num_blocks = trial.suggest_int("cnn_num_blocks", 2, 5)
    fc_enc_n_layers = trial.suggest_int("fc_enc_n_layers", 1, 3)
    fc_enc_hidden_dims = []
    for i in range(fc_enc_n_layers):
        fc_enc_hidden_dims.append(
            trial.suggest_int(f"fc_enc_hidden_dim_{i}", 50, 500, step=50)
        )
    feature_head_n_layers = trial.suggest_int("feature_head_n_layers", 1, 3)
    feature_heads_dims = []
    for i in range(feature_head_n_layers):
        feature_heads_dims.append(
            trial.suggest_int(f"feature_heads_dim_{i}", 50, 500, step=50)
        )
    hparams = {
        "model_class": config.get("MODEL", "model_class"),
        "data_dir": data_dir / "train_test_split",
        "n_load_train": n_load_train,
        "batch_size": 2**k_batchsize,
        "val_batch_size": 512,
        "in_length": 51,
        "n_out_features": 1,
        "in_channels": 1,
        "lr": 1e-3 * (2**k_batchsize) / 32,  # scale lr with batch size
        "num_blocks": cnn_num_blocks,
        "base_channels": cnn_channels,
        "cnn_enc_kernel_size": cnn_enc_kernel_size,
        "cnn_enc_stride": cnn_enc_stride,
        "fc_enc_hidden_dims": fc_enc_hidden_dims,
        "feature_heads_dims": feature_heads_dims,
        "alpha": config.getfloat("MODEL", "alpha"),
        "dropout": dropout_val,
        "swa_lrs": swa_lrs,
        "swa_start": swa_start,
        "max_norm_clip": max_norm_clip,
        "noise": 0.0,
    }

    return hparams


model_hparams_registry = {
    "CNNTimeSeriesRegressorT": setup_CNNTimeSeriesRegressorT_hparams,
}


class OptunaPruningCB(PyTorchLightningPruningCallback, L.pytorch.Callback):
    """
    Fix for `import lightning.pytorch` vs `import pytorch_lightning as pl` (old) bug.
    From https://github.com/optuna/optuna/issues/4689

    Should be fixed with optuna update some time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def setup_hparams(trial: optuna.trial.Trial, config: configparser.ConfigParser) -> dict:
    model_class = config["MODEL"]["model_class"]

    try:
        hparams_setup_func = model_hparams_registry[model_class]
    except KeyError:
        raise KeyError(
            f"Model class {model_class} not found in model_hparams_registry. Add it there and a function for hparams setup."
        )
    hparams = hparams_setup_func(trial, config)
    return hparams


class Objective:
    """
    Objective class to use with optuna.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, trial: optuna.trial.Trial) -> float:
        return objective_(trial, self.config)


def objective_(trial: optuna.trial.Trial, config) -> float:
    """
    Objective function to use with optuna. Trains a model with some random
    (from optuna chosen) hyperparameters and returns the test loss.

    config: dictionary with options:
        accelerator: "cpu" or "gpu"
        precision: "32" or "16-mixed"
        data_path: path to the data
        n_load_train: how many events to load
        epochs: how many epochs to train
    """
    model_class = config.get("MODEL", "model_class")

    rundir = config.get("DIRECTORIES", "rundir")
    hparams = setup_hparams(trial, config)
    print(f"Running trial {trial.number} with hparams: {hparams}")
    model = ModelT_Lit(hparams, verbose=True)
    lit_logdir = os.path.join(rundir, "out", "final", "lightning_logs")
    logger = TensorBoardLogger(
        lit_logdir, name=model_class, default_hp_metric=False, log_graph=True
    )
    val_ckeckpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val:.6f}",
        monitor="losses/val_loss",
        mode="min",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor="losses/val_loss", patience=100)
    swa = StochasticWeightAveraging(
        swa_lrs=hparams["swa_lrs"],
        swa_epoch_start=hparams["swa_start"],  # 50% of total epochs
    )
    pruner_cb = OptunaPruningCB(trial, monitor="losses/val_loss")
    callbacks = [val_ckeckpoint, lr_monitor, early_stopping, swa, pruner_cb]
    trainer = L.Trainer(
        accelerator=config.get("GENERAL", "accelerator"),
        max_epochs=config.getint("MODEL", "epochs_hpopt"),
        max_steps=-1,
        logger=logger,
        default_root_dir=os.path.join(rundir, "out", "final", "models", model_class),
        callbacks=callbacks,
        gradient_clip_val=hparams["max_norm_clip"],
        precision=config.get("GENERAL", "precision"),
    )
    tuner = L.pytorch.tuner.Tuner(trainer)
    tune_lr = True
    if tune_lr:
        lr_find_results = tuner.lr_find(model)
        fig = lr_find_results.plot(suggest=True)
        logger.experiment.add_figure("lr_find", fig)
        new_lr = lr_find_results.suggestion()
        model.hparams.lr = new_lr

    trainer.fit(model)
    best_model_path = val_ckeckpoint.best_model_path
    best_val_loss = val_ckeckpoint.best_model_score.item()

    trial.set_user_attr("model_path", best_model_path)
    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr("Model type", model_class)
    # Path is not JSON serializable, so convert to str
    hparams_save = hparams.copy()
    hparams_save["data_dir"] = str(hparams_save["data_dir"])
    trial.set_user_attr("hparams", hparams_save)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr("num_params", num_params)
    trainer.test(model, ckpt_path=best_model_path)
    history_str = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}: HPOpt, trial number: {trial.number}, loss: {trainer.callback_metrics.get('test_loss', 'NaN')}mm\n"
    model.model_history = model.model_history.append(history_str)
    model.hparams["model_history"] = model.model_history

    logger.log_hyperparams(
        hparams_save,
        {
            # "losses/val_loss": best_val_loss,
            "losses/test_loss": trainer.callback_metrics.get(
                "test_loss", torch.tensor(float("nan"))
            ).item(),
        },
    )

    return trainer.callback_metrics["test_loss"].item()


def do_hp_optimization(config: configparser.ConfigParser) -> optuna.study.Study:
    # read config
    rundir = config["DIRECTORIES"]["rundir"]
    model_class = config["MODEL"]["model_class"]
    optuna_log_file = os.path.join(
        rundir, "out", "final", "optuna", f"{model_class}.db"
    )
    os.makedirs(os.path.dirname(optuna_log_file), exist_ok=True)
    optuna_log_db = f"sqlite:///{optuna_log_file}"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=5, interval_steps=1
    )
    lit_logdir = os.path.join(rundir, "out", "final", "lightning_logs")
    # print_start_hpopt(optuna_log_db, lit_logdir, config)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=5, interval_steps=1
    )
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model_class}",
        storage=optuna_log_db,
        load_if_exists=True,
        pruner=pruner,
    )

    catches = (
        ValueError,
        KeyError,
        AssertionError,
    )  # errors that should not stop optimization
    catches = []
    study.optimize(
        Objective(config),
        n_trials=config.getint("MODEL", "n_trials_hpopt"),
        timeout=config.getint("MODEL", "timeout_hpopt"),
        catch=catches,
        show_progress_bar=True,
        # gc_after_trial=True,
    )


def main():
    print("Hyperparameter optimization for Exoplanets model")
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for Exoplanets model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
                                    In config file (first is recommended):
                                    [GENERAL]
                                    """),
    )
    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", required=True
    )
    args = parser.parse_args()
    config = read_config(args.config)
    do_hp_optimization(config)


if __name__ == "__main__":
    main()
