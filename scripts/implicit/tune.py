import gc
import json
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from bookrec.data import (
    create_id_mappings,
    encode_interactions,
    load_dataset,
    split_interactions,
)
from bookrec.implicit.datasets import ImplicitDataset, SampledRankingDataset
from bookrec.implicit.hyperparameters import ARCHITECTURES
from bookrec.implicit.metrics import ndcg_at_50
from bookrec.implicit.model import ImplicitRecommenderMLP
from bookrec.training import train_loop, validation_loop


SEED = 42
NUM_TRIALS = 25
EPOCHS_PER_TRIAL = 12
EVALUATION_CANDIDATES = 1_000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    interactions = load_dataset()
    train, validation, _ = split_interactions(interactions, seed=SEED)

    user_to_index, item_to_index = create_id_mappings(train)
    train_data = encode_interactions(train, user_to_index, item_to_index)
    validation_data = encode_interactions(
        validation,
        user_to_index,
        item_to_index,
    )
    known_interactions = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )

    train_dataset = ImplicitDataset(
        train_data,
        known_interactions,
        num_items=len(item_to_index),
        negatives_per_positive=4,
    )
    validation_dataset = SampledRankingDataset(
        validation_data,
        known_interactions,
        num_items=len(item_to_index),
        num_candidates=EVALUATION_CANDIDATES,
        seed=SEED,
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
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
            [16, 32, 64],
        )
        dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-4,
            3e-3,
            log=True,
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            1e-7,
            1e-3,
            log=True,
        )

        model = ImplicitRecommenderMLP(
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
        loss_function = nn.BCEWithLogitsLoss()
        best_ndcg = -float("inf")

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
                    [ndcg_at_50],
                    device,
                )
                validation_ndcg = scores["ndcg_at_50"]
                best_ndcg = max(best_ndcg, validation_ndcg)
                trial.report(validation_ndcg, step=epoch)
                scheduler.step()

                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_ndcg
        finally:
            del loss_function, scheduler, optimizer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    artifact_directory = Path("artifacts/implicit")
    artifact_directory.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="implicit_mlp",
        direction="maximize",
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

    print(f"Best validation NDCG@50: {study.best_value:.6f}")
    print(f"Best hyperparameters: {best_hyperparameters}")
    print(f"Saved hyperparameters to {output_path}")


if __name__ == "__main__":
    main()
