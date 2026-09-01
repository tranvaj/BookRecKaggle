import argparse
import gc
import json
from pathlib import Path

import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bookrec.data import (
    create_id_mappings,
    encode_interactions,
    load_dataset,
    split_interactions,
)
from bookrec.implicit.baselines import ALSBaseline
from bookrec.implicit.datasets import (
    SampledRankingDataset,
    collate_implicit_batch,
)
from bookrec.implicit.hyperparameters import DEFAULT_ALS_HYPERPARAMETERS
from bookrec.implicit.metrics import ndcg_at_50
from bookrec.training import validation_loop


SEED = 42
NUM_TRIALS = 25
EVALUATION_CANDIDATES = 1_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-trials",
        type=int,
        default=NUM_TRIALS,
        help="Target total number of trials in the resumable study.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/implicit/als"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_trials < 1:
        raise ValueError("--num-trials must be at least 1")

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
    validation_dataset = SampledRankingDataset(
        validation_data,
        context_interactions=train_data,
        all_interactions=known_interactions,
        num_items=len(item_to_index),
        num_candidates=EVALUATION_CANDIDATES,
        seed=SEED,
        context_mode="full",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_implicit_batch,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial) -> float:
        hyperparameters = {
            "factors": trial.suggest_categorical(
                "factors",
                [16, 32, 64, 128],
            ),
            "regularization": trial.suggest_float(
                "regularization",
                1e-3,
                1.0,
                log=True,
            ),
            "alpha": trial.suggest_float(
                "alpha",
                1.0,
                100.0,
                log=True,
            ),
            "iterations": trial.suggest_categorical(
                "iterations",
                [10, 20, 40],
            ),
        }
        model = ALSBaseline.fit(
            train_data,
            num_users=len(user_to_index),
            num_items=len(item_to_index),
            seed=args.seed,
            **hyperparameters,
        )
        try:
            scores = validation_loop(
                validation_loader,
                model,
                [ndcg_at_50],
                device,
            )
            validation_ndcg = scores["ndcg_at_50"]
            print(
                f"Trial {trial.number + 1}: "
                f"validation NDCG@50={validation_ndcg:.6f}; "
                f"hyperparameters={hyperparameters}"
            )
            return validation_ndcg
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="implicit_als_v1",
        direction="maximize",
        storage=f"sqlite:///{args.artifact_dir / 'hpo.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    finished_trials = sum(trial.state.is_finished() for trial in study.trials)
    if not study.trials:
        study.enqueue_trial(DEFAULT_ALS_HYPERPARAMETERS)

    remaining_trials = max(0, args.num_trials - finished_trials)
    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=True,
        )

    best_hyperparameters = study.best_params
    output_path = args.artifact_dir / "best_hparams.json"
    output_path.write_text(
        json.dumps(best_hyperparameters, indent=2) + "\n"
    )
    print(f"Best validation NDCG@50: {study.best_value:.6f}")
    print(f"Best hyperparameters: {best_hyperparameters}")
    print(f"Saved hyperparameters to {output_path}")


if __name__ == "__main__":
    main()
