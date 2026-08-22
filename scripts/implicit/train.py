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
from bookrec.implicit.datasets import ImplicitDataset, SampledRankingDataset
from bookrec.implicit.evaluation import RANKING_METRICS
from bookrec.implicit.hyperparameters import load_hyperparameters
from bookrec.implicit.model import ImplicitRecommenderMLP
from bookrec.training import train as train_model


SEED = 42
EVALUATION_CANDIDATES = 1_000
EPOCHS = 12
BATCH_SIZE = 512


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    artifact_directory = Path("artifacts/implicit")
    hyperparameters = load_hyperparameters(
        artifact_directory / "best_hparams.json"
    )
    print(f"Training hyperparameters: {hyperparameters}")
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
        negatives_per_positive=8,
    )
    validation_dataset = SampledRankingDataset(
        validation_data,
        known_interactions,
        num_items=len(item_to_index),
        num_candidates=EVALUATION_CANDIDATES,
        seed=SEED,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    model = ImplicitRecommenderMLP(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        embedding_dim=hyperparameters["embedding_dim"],
        hidden_dims=hyperparameters["hidden_dims"],
        dropout=hyperparameters["dropout"],
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
    model, _, _, _ = train_model(
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
        early_stopping_patience=4,
        metric_mode="max",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_to_index": user_to_index,
            "item_to_index": item_to_index,
            "hyperparameters": hyperparameters,
        },
        artifact_directory / "model_with_mappings.pt",
    )
    print(f"Saved trained MLP to {artifact_directory}")


if __name__ == "__main__":
    main()
