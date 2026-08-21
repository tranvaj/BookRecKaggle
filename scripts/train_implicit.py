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
from bookrec.implicit.evaluation import RANKING_METRICS, evaluate_sampled_ranking
from bookrec.implicit.model import ImplicitRecommenderMLP
from bookrec.training import train


SEED = 42
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
    train, validation, test = split_interactions(interactions, seed=SEED)

    user_to_index, item_to_index = create_id_mappings(train)
    train_data = encode_interactions(train, user_to_index, item_to_index)
    validation_data = encode_interactions(
        validation,
        user_to_index,
        item_to_index,
    )
    test_data = encode_interactions(test, user_to_index, item_to_index)
    all_interactions = pd.concat(
        [train_data, validation_data, test_data],
        ignore_index=True,
    )

    train_dataset = ImplicitDataset(
        train_data,
        all_interactions,
        num_items=len(item_to_index),
        negatives_per_positive=4,
    )
    validation_dataset = SampledRankingDataset(
        validation_data,
        all_interactions,
        num_items=len(item_to_index),
        num_candidates=EVALUATION_CANDIDATES,
        seed=SEED,
    )
    test_dataset = SampledRankingDataset(
        test_data,
        all_interactions,
        num_items=len(item_to_index),
        num_candidates=EVALUATION_CANDIDATES,
        seed=SEED + 1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = ImplicitRecommenderMLP(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
    ).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    artifact_directory = Path("artifacts/implicit")
    artifact_directory.mkdir(parents=True, exist_ok=True)
    model, _, _, _ = train(
        model=model,
        train_dl=train_loader,
        val_dl=validation_loader,
        loss=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        val_metrics=RANKING_METRICS,
        save_val_metric="ndcg_at_10",
        device=device,
        output_path=artifact_directory / "best_model.pt",
        load_best_model=True,
        early_stopping_patience=3,
        metric_mode="max",
    )

    test_scores = evaluate_sampled_ranking(model, test_loader, device)
    print(f"Test metrics: {test_scores}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_to_index": user_to_index,
            "item_to_index": item_to_index,
        },
        artifact_directory / "model_with_mappings.pt",
    )


if __name__ == "__main__":
    main()
