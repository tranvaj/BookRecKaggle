import argparse
import random
from pathlib import Path

import numpy as np
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
from bookrec.explicit.evaluation import RATING_METRICS
from bookrec.explicit.hyperparameters import load_hyperparameters
from bookrec.explicit.model import ExplicitRecommenderMLP
from bookrec.training import train as train_model


SPLIT_SEED = 42
DEFAULT_TRAINING_SEED = 42
EPOCHS = 100
BATCH_SIZE = 1024
MODEL_NAME = "mlp"
ARTIFACT_ROOT = Path("artifacts/explicit")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hyperparameters-path", type=Path)
    return parser.parse_args()


def train_explicit_model(
    seed: int = DEFAULT_TRAINING_SEED,
    output_dir: Path | None = None,
    hyperparameters_path: Path | None = None,
) -> Path:
    """Train one explicit MLP and return its portable checkpoint path."""
    set_seed(seed)
    canonical_artifact_directory = ARTIFACT_ROOT / MODEL_NAME
    artifact_directory = output_dir or canonical_artifact_directory
    selected_hyperparameters_path = (
        hyperparameters_path
        or canonical_artifact_directory / "best_hparams.json"
    )
    hyperparameters = load_hyperparameters(selected_hyperparameters_path)
    print(f"Training hyperparameters: {hyperparameters}")
    print(f"Training seed: {seed}; output: {artifact_directory}")
    ratings = load_dataset()
    explicit_ratings = ratings[ratings[RATING_COLUMN] > 0].reset_index(drop=True)
    train, validation, _ = split_interactions(
        explicit_ratings,
        seed=SPLIT_SEED,
    )

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
    model = ExplicitRecommenderMLP(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        embedding_dim=hyperparameters["embedding_dim"],
        hidden_dims=hyperparameters["hidden_dims"],
        dropout=hyperparameters["dropout"],
    ).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    artifact_directory.mkdir(parents=True, exist_ok=True)
    model, validation_scores, _, _ = train_model(
        model=model,
        train_dl=train_loader,
        val_dl=validation_loader,
        loss=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        val_metrics=RATING_METRICS,
        save_val_metric="rmse",
        device=device,
        output_path=artifact_directory / "best_model.pt",
        load_best_model=True,
        early_stopping_patience=5,
        metric_mode="min",
    )

    checkpoint_path = artifact_directory / "model_with_mappings.pt"
    torch.save(
        {
            "model_type": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "user_to_index": user_to_index,
            "item_to_index": item_to_index,
            "hyperparameters": hyperparameters,
            "validation_metrics": min(
                validation_scores,
                key=lambda scores: scores["rmse"],
            ),
            "split_seed": SPLIT_SEED,
            "training_seed": seed,
        },
        checkpoint_path,
    )
    print(f"Saved trained explicit model to {artifact_directory}")
    return checkpoint_path


def main():
    args = parse_args()
    train_explicit_model(
        seed=args.seed,
        output_dir=args.output_dir,
        hyperparameters_path=args.hyperparameters_path,
    )


if __name__ == "__main__":
    main()
