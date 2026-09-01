import gc
import json
import random
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader

from bookrec.data import (
    RATING_COLUMN,
    create_id_mappings,
    encode_interactions,
    load_dataset,
    split_interactions,
)
from bookrec.explicit.datasets import ExplicitDataset
from bookrec.explicit.hyperparameters import ARCHITECTURES
from bookrec.explicit.metrics import rmse
from bookrec.explicit.model import ExplicitRecommenderMLP
from bookrec.training import train_loop, validation_loop


SEED = 42
NUM_TRIALS = 25
EPOCHS_PER_TRIAL = 20
BATCH_SIZE = 1_024
MODEL_NAME = "mlp"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    ratings = load_dataset()
    explicit_ratings = ratings[ratings[RATING_COLUMN] > 0].reset_index(drop=True)
    train, validation, _ = split_interactions(explicit_ratings, seed=SEED)

    user_to_index, item_to_index = create_id_mappings(train)
    train_data = encode_interactions(train, user_to_index, item_to_index)
    validation_data = encode_interactions(
        validation,
        user_to_index,
        item_to_index,
    )
    train_loader = DataLoader(
        ExplicitDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        ExplicitDataset(validation_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial) -> float:
        set_seed(SEED)
        architecture = trial.suggest_categorical(
            "architecture",
            list(ARCHITECTURES),
        )
        embedding_dim = trial.suggest_categorical(
            "embedding_dim",
            [8, 16, 32, 64],
        )
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-5,
            3e-3,
            log=True,
        )
        weight_decay = trial.suggest_categorical(
            "weight_decay",
            [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        )

        model = ExplicitRecommenderMLP(
            num_users=len(user_to_index),
            num_items=len(item_to_index),
            embedding_dim=embedding_dim,
            hidden_dims=ARCHITECTURES[architecture],
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS_PER_TRIAL,
        )
        loss_function = nn.MSELoss()
        best_rmse = float("inf")

        try:
            for epoch in range(EPOCHS_PER_TRIAL):
                train_loop(
                    train_loader,
                    model,
                    loss_function,
                    optimizer,
                    device,
                )
                scores = validation_loop(
                    validation_loader,
                    model,
                    [rmse],
                    device,
                )
                validation_rmse = scores["rmse"]
                best_rmse = min(best_rmse, validation_rmse)
                trial.report(validation_rmse, step=epoch)
                scheduler.step()

                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_rmse
        finally:
            del loss_function, scheduler, optimizer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    artifact_directory = Path("artifacts/explicit") / MODEL_NAME
    artifact_directory.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="explicit_mlp",
        direction="minimize",
        storage=f"sqlite:///{artifact_directory / 'hpo.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        ),
    )
    remaining_trials = max(0, NUM_TRIALS - len(study.trials))
    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=True,
        )

    best_hyperparameters = study.best_params.copy()
    architecture = best_hyperparameters.pop("architecture")
    best_hyperparameters["hidden_dims"] = list(ARCHITECTURES[architecture])
    output_path = artifact_directory / "best_hparams.json"
    output_path.write_text(json.dumps(best_hyperparameters, indent=2))

    print(f"Best validation RMSE: {study.best_value:.6f}")
    print(f"Best hyperparameters: {best_hyperparameters}")
    print(f"Saved hyperparameters to {output_path}")


if __name__ == "__main__":
    main()
