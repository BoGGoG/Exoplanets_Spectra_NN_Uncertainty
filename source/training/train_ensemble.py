"""
Do this after hyperparameter optimization and instead of continue.py
Will train the same model (best hyperparameters) multiple times with different seeds and average the results.
"""

import argparse
import configparser
import os
import shutil
import textwrap
from datetime import date

import lightning as L
import numpy as np
import optuna
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

from source.IO import read_config
from source.models.Model01 import CNNTimeSeriesRegressor, Model_Lit, model_registry
from source.training.optimizers import cosine_decay_scheduler
import torch.nn.init as init
import shutil

torch.set_float32_matmul_precision("highest")  # or "high"


def get_best_model_from_trials(study_name="", storage="", q=0.03):
    """
    same as get_best_hparams_from_trials, but returns the best model path instead of the best hyperparameters
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = study.get_trials()
    trials = [
        trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    trials = sorted(trials, key=lambda trial: trial.value)
    trials = trials[:20]
    # print 10 best trial number with loss and num_params sorted by loss
    print("10 best trials:")
    for trial in trials:
        print(
            "Trial number: {}, Loss: {:.8f}, Num params: {}".format(
                trial.number, trial.value, trial.user_attrs["num_params"]
            )
        )
    # get trials in best q quantile
    trials_losses = [trial.value for trial in trials]
    quantile = np.quantile(trials_losses, q)
    trials = [trial for trial in trials if trial.value <= quantile]
    # of those trials, use the one with the lowest number of parameters
    trials = sorted(trials, key=lambda trial: trial.user_attrs["num_params"])
    # best_trial = trials[0]
    best_trial = trials[-1]
    # get best hyperparameters
    hparams = best_trial.user_attrs["hparams"]
    print("Choosing best model according to the following criteria:")
    print("Loss in best {}% quantile: {}".format(q * 100, quantile))
    print("Lowest number of parameters: {}".format(best_trial.user_attrs["num_params"]))
    print("Best hyperparameters:")
    for key, value in hparams.items():
        print("{}: {}".format(key, value))
    return best_trial.user_attrs["model_path"]


def train_ensemble_of_models(
    best_model_path, config: configparser.ConfigParser
) -> tuple[str, int]:
    """
    Load the best model from hyperopt and continue training it n_models times with different seeds.
    Restart training with new initialization of the parameters of the model, but same hyperparameters.
    """
    model_class = config.get("MODEL", "model_class")
    run_dir = config.get("DIRECTORIES", "rundir")
    n_models = config.getint("ENSEMBLE", "n_ensemble_models")

    out_dir = config.get("ENSEMBLE", "ensemble_savedir")
    lit_logdir_base = out_dir or os.path.join(
        config.get("DIRECTORIES", "rundir"), "out", "ensemble"
    )
    # if lit_logdir exists, ask and delete it
    if os.path.exists(lit_logdir_base):
        input_str = input(f"Lit_logdir {lit_logdir_base} exists. Delete it? (y/n) ")
        if input_str.lower() == "y":
            shutil.rmtree(lit_logdir_base)
        else:
            raise ValueError("Lit_logdir exists. Exiting.")

    ensemble = []
    test_losses = []
    ckpt_paths = []
    print(f"\nSTARTING TRAINING OF ENSEMBLE OF {n_models} MODELS\n\n")
    for i in range(n_models):
        print(f"Model {i + 1}/{n_models} training started.")
        model = Model_Lit.load_from_checkpoint(best_model_path)
        print("loaded model")

        seed = 1234 + i
        # initialize parameters of model from scratch using Kaiming normal initialization
        # the model needs to have a function `reset_weights`
        generator = torch.Generator(device=model.device).manual_seed(seed)
        torch.manual_seed(seed)
        model.model.reset_weights(generator=generator)

        # TRAINING
        n_events = config.getint("ENSEMBLE", "n_events")
        n_events = n_events if n_events > 0 else None  # None = All
        model.hparams["n_load_train"] = n_events
        batch_size = config.getint(
            "ENSEMBLE",
            "batch_size",
            fallback=model.hparams["batch_size"],
        )
        if batch_size == 1:
            batch_size = model.hparams["batch_size"]
            accumulator = GradientAccumulationScheduler(
                scheduling={0: 64, 5: 32, 10: 16, 15: 8, 20: 4, 25: 2, 30: 1}
            )
        model.hparams["batch_size"] = batch_size
        model.batch_size = batch_size
        model.hparams["epochs"] = config.getint("ENSEMBLE", "epochs")
        model.num_events = (num_events := config.getint("MODEL", "n_events"))
        model.verbose = True
        model.data_dir = model.hparams["data_dir"]
        model.lr = config.getfloat("ENSEMBLE", "lr")

        swa_lr = config.getfloat("ENSEMBLE", "swa_lr")
        swa_start = config.getint("ENSEMBLE", "swa_start")
        model.hparams["swa_lr"] = swa_lr
        model.hparams["swa_start"] = swa_start

        #### MODIFY OPTIMIZER AND LR SCHEDULE HERE ####
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=model.lr, weight_decay=0.01
        )
        for param_group in optimizer.param_groups:
            param_group["initial_lr"] = model.lr
        scheduler = cosine_decay_scheduler(
            optimizer,
            warmup_steps=3,
            total_steps=swa_start,
            steps_per_cycle=15,
            decay_factor=0.6,
            cycle_stretch=1.5,
        )
        # print("Using cosine annealing with hard restarts schedule with warmup")
        # scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=10, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        model.external_optimizer: torch.optim.Optimizer = optimizer
        model.external_scheduler: torch.optim.lr_scheduler.LRScheduler = scheduler

        summary = ModelSummary(model, max_depth=-1)
        lit_logdir = os.path.join(
            lit_logdir_base,
            "lightning_logs",
            f"ensemble_{config.get('MODEL', 'model_class')}_{i}",
        )

        logger = TensorBoardLogger(
            lit_logdir,
            name=config.get("MODEL", "model_class"),
            default_hp_metric=False,
            log_graph=True,
        )
        val_ckeckpoint = ModelCheckpoint(
            filename="{epoch}-{step}-{val_loss:.5f}",
            monitor="losses/val_loss",
            mode="min",
            save_top_k=1,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stopping = EarlyStopping(
            monitor="losses/val_loss",
            patience=(patience := 100),
        )
        model.hparams["patience"] = patience
        swa = StochasticWeightAveraging(swa_lrs=swa_lr, swa_epoch_start=swa_start)
        callbacks = [val_ckeckpoint, lr_monitor, early_stopping, swa]
        try:
            callbacks.append(accumulator)
            print("Using gradient accumulation")
        except:
            pass
        trainer = L.Trainer(
            accelerator=config.get("GENERAL", "accelerator"),
            max_epochs=config.getint("ENSEMBLE", "epochs"),
            logger=logger,
            default_root_dir=os.path.join(
                run_dir,
                "out",
                "final",
                "models",
                config.get("MODEL", "model_class"),
            ),
            callbacks=callbacks,
            gradient_clip_val=model.hparams["max_norm_clip"],
            precision=config.get("GENERAL", "precision"),
        )
        trainer.fit(model)  # , ckpt_path=best_model_path)
        trainer.test(model, ckpt_path=val_ckeckpoint.best_model_path)

        # best checkpoint path
        best_path = val_ckeckpoint.best_model_path
        best_loss = val_ckeckpoint.best_model_score.item()
        test_loss = trainer.callback_metrics["test_loss"]
        print(f"Test loss {i}: {test_loss}")

        # FINISHED TRAINING
        # move model to cpu
        model.to("cpu")
        ensemble.append(model)
        test_losses.append(test_loss.item())
        ckpt_paths.append(best_path)
    print(f"{len(ensemble)=}")

    return ensemble, ckpt_paths, test_losses


def main():
    parser = argparse.ArgumentParser(
        description="Train an ensemble of models with the best hyperparameters found by hyperopt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
                                    """),
    )
    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", required=True
    )
    args = parser.parse_args()
    config = read_config(args.config)
    model_class = config["MODEL"]["model_class"]

    run_dir = config["DIRECTORIES"]["rundir"]
    db_file_path = os.path.join(run_dir, "out", "final", "optuna", f"{model_class}.db")
    optuna_log_dir = f"sqlite:///{db_file_path}"
    if config.get("MODEL", "use_pretrained").lower() != "false":
        print(
            "Using pretrained model from {}".format(
                config.get("MODEL", "use_pretrained")
            )
        )
        best_model_path = config.get("MODEL", "use_pretrained")
    else:
        print("Using best model from hyperopt")
        best_model_path = get_best_model_from_trials(
            study_name=model_class,
            storage=optuna_log_dir,
            q=config.getfloat("MODEL", "q"),
        )

    models, ckpts, losses = train_ensemble_of_models(
        best_model_path,
        config,
    )

    print("Ensemble training complete.")
    print("Test losses of individual models in ensemble:")
    print(losses)

    # save ensemble to disk

    out_dir = config.get("ENSEMBLE", "ensemble_savedir")
    if out_dir:
        models_save_dir = out_dir / "models"
    else:
        models_save_dir = os.path.join(
            config.get("DIRECTORIES", "rundir"), "out", "ensemble", "models"
        )
    os.makedirs(os.path.join(models_save_dir, "pt"), exist_ok=True)
    os.makedirs(os.path.join(models_save_dir, "ckpt"), exist_ok=True)

    paths_pt = []
    paths_ckpt = []

    for i, model in enumerate(models):
        path_pt = os.path.join(
            models_save_dir, "pt", f"{config.get('MODEL', 'model_class')}_{i}.pt"
        )
        path_cpt = os.path.join(
            models_save_dir, "ckpt", f"{config.get('MODEL', 'model_class')}_{i}.ckpt"
        )

        model.eval()
        torch.save(model.model, path_pt)
        shutil.copy(ckpts[i], path_cpt)

    print("Ensemble models saved to disk:")
    print(*paths_pt, sep="\n")
    print(*paths_ckpt, sep="\n")

    #     # save best model to new path
    # path_onnx = os.path.join(
    #     run_dir,
    #     "out",
    #     "final",
    #     "models",
    #     config.get("MODEL", "model_class"),
    #     "best_model_retrained_loss_{:.8f}.onxx".format(test_loss),
    # )
    # path_pt = os.path.join(
    #     run_dir,
    #     "out",
    #     "final",
    #     "models",
    #     config.get("MODEL", "model_class"),
    #     "best_model_retrained_loss_{:.8f}.pt".format(test_loss),
    # )
    # path_cpt = os.path.join(
    #     run_dir,
    #     "out",
    #     "final",
    #     "models",
    #     config.get("MODEL", "model_class"),
    #     "best_model_retrained_loss_{:.8f}.ckpt".format(test_loss),
    # )
    # if not os.path.exists(os.path.dirname(path_onnx)):
    #     os.makedirs(os.path.dirname(path_onnx))
    # model = model_registry[config.get("MODEL", "model_class")].load_from_checkpoint(
    #     best_model_retrained_path
    # )
    # shutil.copy(best_model_retrained_path, path_cpt)
    # model.eval()
    # torch.save(model.model, path_pt)
    #
    # # save model as onnx
    # x = torch.randn(
    #     1,
    #     7,
    #     config.getint("GENERAL", "input_size"),
    #     requires_grad=False,
    #     device=accelerator,
    # )
    # torch_out = model(x)
    # # Export the model
    # torch.onnx.export(
    #     model,  # model being run
    #     x,  # model input (or a tuple for multiple inputs)
    #     path_onnx,  # where to save the model (can be a file or file-like object)
    #     export_params=True,  # store the trained parameter weights inside the model file
    #     opset_version=config.getint(
    #         "MODEL", "opset_version_onnx"
    #     ),  # the ONNX version to export the model to
    #     do_constant_folding=True,  # whether to execute constant folding for optimization
    #     input_names=["input"],  # the model's input names
    #     output_names=["output"],  # the model's output names
    #     dynamic_axes={"input": {0: "batch_size"}},  # variable length axes
    # )
    # # torch.save(model.model, path)
    # print("Best model (pytorch) saved to: {}".format(path_pt))
    # print("Best model (lightning) saved to: {}".format(path_cpt))
    # print("onnx model saved to: {}".format(path_onnx))
    # # print("Load with `model = torch.load(\"{}\")`".format(path))
    # print("Lightning (Tensorboard) logs saved to: {}".format(logger.log_dir))
    # print("Load with `tensorboard --logdir={}`".format(logger.log_dir))


if __name__ == "__main__":
    main()
