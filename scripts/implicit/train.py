import argparse
import random
from pathlib import Path

import numpy as np
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
from bookrec.implicit.datasets import (
    ImplicitDataset,
    SampledRankingDataset,
    collate_implicit_batch,
)
from bookrec.implicit.evaluation import RANKING_METRICS
from bookrec.implicit.hyperparameters import load_hyperparameters
from bookrec.implicit.model import (
    HISTORY_MLP_ARCHITECTURE_VERSION,
    MODEL_REGISTRY,
    create_implicit_model,
)
from bookrec.training import train as train_model


SPLIT_SEED = 42
DEFAULT_TRAINING_SEED = 42
ARTIFACT_ROOT = Path("artifacts/implicit")
EVALUATION_CANDIDATES = 1_000
EPOCHS = 100
BATCH_SIZE = 512
NEGATIVES_PER_POSITIVE = 8
HISTORY_CONTEXT_MODE = "mixed"
HISTORY_SINGLETON_PROBABILITY = 0.5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=MODEL_REGISTRY,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help="Random seed for initialization, shuffling, and negative sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Checkpoint directory. Defaults to artifacts/implicit/<model>. "
            "Set this for independently trained ensemble members."
        ),
    )
    parser.add_argument(
        "--hyperparameters-path",
        type=Path,
        help=(
            "Hyperparameter JSON. Defaults to the canonical "
            "artifacts/implicit/<model>/best_hparams.json."
        ),
    )
    return parser.parse_args()


def train_implicit_model(
    model_name: str,
    seed: int = DEFAULT_TRAINING_SEED,
    output_dir: Path | None = None,
    hyperparameters_path: Path | None = None,
) -> Path:
    """Train one implicit model and return its portable checkpoint path."""
    if model_name not in MODEL_REGISTRY:
        choices = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {choices}")

    set_seed(seed)
    canonical_artifact_directory = ARTIFACT_ROOT / model_name
    artifact_directory = output_dir or canonical_artifact_directory
    selected_hyperparameters_path = (
        hyperparameters_path
        or canonical_artifact_directory / "best_hparams.json"
    )
    hyperparameters = load_hyperparameters(selected_hyperparameters_path)
    print(f"Training hyperparameters: {hyperparameters}")
    print(f"Training seed: {seed}; output: {artifact_directory}")
    interactions = load_dataset()
    train, validation, _ = split_interactions(
        interactions,
        seed=SPLIT_SEED,
    )

    user_to_index, item_to_index = create_id_mappings(train)
    train_data = encode_interactions(train, user_to_index, item_to_index)
    training_item_counts = torch.bincount(
        torch.tensor(train_data["item"].to_numpy(), dtype=torch.long),
        minlength=len(item_to_index),
    )
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
        negatives_per_positive=NEGATIVES_PER_POSITIVE,
        require_nonempty_history=model_name == "history_mlp",
        history_mode=(
            HISTORY_CONTEXT_MODE if model_name == "history_mlp" else "full"
        ),
        singleton_probability=HISTORY_SINGLETON_PROBABILITY,
    )
    validation_dataset = SampledRankingDataset(
        validation_data,
        context_interactions=train_data,
        all_interactions=known_interactions,
        num_items=len(item_to_index),
        num_candidates=EVALUATION_CANDIDATES,
        seed=SPLIT_SEED,
        context_mode="full",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_implicit_batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_implicit_batch,
    )
    model = create_implicit_model(
        model_name,
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        hyperparameters=hyperparameters,
    ).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
    )
    epochs = EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    artifact_directory.mkdir(parents=True, exist_ok=True)
    model, validation_scores, _, _ = train_model(
        model=model,
        train_dl=train_loader,
        val_dl=validation_loader,
        loss=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        val_metrics=RANKING_METRICS,
        save_val_metric="ndcg_at_50",
        device=device,
        output_path=artifact_directory / "best_model.pt",
        load_best_model=True,
        early_stopping_patience=5,
        metric_mode="max",
    )

    checkpoint_path = artifact_directory / "model_with_mappings.pt"
    torch.save(
        {
            "model_type": model_name,
            "architecture_version": (
                HISTORY_MLP_ARCHITECTURE_VERSION
                if model_name == "history_mlp"
                else 1
            ),
            "model_state_dict": model.state_dict(),
            "user_to_index": user_to_index,
            "item_to_index": item_to_index,
            "training_item_counts": training_item_counts,
            "hyperparameters": hyperparameters,
            "validation_metrics": max(
                validation_scores,
                key=lambda scores: scores["ndcg_at_50"],
            ),
            "split_seed": SPLIT_SEED,
            "training_seed": seed,
            "history_training": (
                {
                    "context_mode": HISTORY_CONTEXT_MODE,
                    "singleton_probability": HISTORY_SINGLETON_PROBABILITY,
                }
                if model_name == "history_mlp"
                else None
            ),
        },
        checkpoint_path,
    )
    print(f"Saved trained {model_name} model to {artifact_directory}")
    return checkpoint_path


def main():
    args = parse_args()
    train_implicit_model(
        model_name=args.model,
        seed=args.seed,
        output_dir=args.output_dir,
        hyperparameters_path=args.hyperparameters_path,
    )


if __name__ == "__main__":
    main()
