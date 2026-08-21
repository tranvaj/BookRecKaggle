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
from bookrec.explicit.baselines import evaluate_mean_baselines
from bookrec.explicit.datasets import ExplicitDataset
from bookrec.explicit.evaluation import RATING_METRICS, evaluate_ratings
from bookrec.explicit.model import ExplicitRecommenderMLP
from bookrec.training import train


SEED = 42


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
    train, validation, test = split_interactions(explicit_ratings, seed=SEED)

    user_to_index, item_to_index = create_id_mappings(train)
    train_data = encode_interactions(train, user_to_index, item_to_index)
    validation_data = encode_interactions(
        validation,
        user_to_index,
        item_to_index,
    )
    test_data = encode_interactions(test, user_to_index, item_to_index)

    print("Validation baselines:")
    print(evaluate_mean_baselines(train, validation))

    train_loader = DataLoader(
        ExplicitDataset(train_data),
        batch_size=1_024,
        shuffle=True,
    )
    validation_loader = DataLoader(
        ExplicitDataset(validation_data),
        batch_size=2_048,
        shuffle=False,
    )
    test_loader = DataLoader(
        ExplicitDataset(test_data),
        batch_size=2_048,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExplicitRecommenderMLP(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        global_mean=float(train[RATING_COLUMN].mean()),
    ).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    artifact_directory = Path("artifacts/explicit")
    artifact_directory.mkdir(parents=True, exist_ok=True)
    model, _, _, _ = train(
        model=model,
        train_dl=train_loader,
        val_dl=validation_loader,
        loss=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        val_metrics=RATING_METRICS,
        save_val_metric="rmse",
        device=device,
        output_path=artifact_directory / "best_model.pt",
        load_best_model=True,
        early_stopping_patience=3,
        metric_mode="min",
    )

    print(f"Test metrics: {evaluate_ratings(model, test_loader, device)}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_to_index": user_to_index,
            "item_to_index": item_to_index,
            "global_mean": float(train[RATING_COLUMN].mean()),
        },
        artifact_directory / "model_with_mappings.pt",
    )


if __name__ == "__main__":
    main()
