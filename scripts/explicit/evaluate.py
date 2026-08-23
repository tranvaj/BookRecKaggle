from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bookrec.data import (
    ITEM_COLUMN,
    RATING_COLUMN,
    USER_COLUMN,
    encode_interactions,
    load_dataset,
    split_interactions,
)
from bookrec.explicit.baselines import evaluate_mean_baselines
from bookrec.explicit.datasets import ExplicitDataset
from bookrec.explicit.evaluation import evaluate_ratings
from bookrec.explicit.hyperparameters import DEFAULT_HYPERPARAMETERS
from bookrec.explicit.model import ExplicitRecommenderMLP


SEED = 42
MODEL_NAME = "mlp"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        Path("artifacts/explicit") / MODEL_NAME / "model_with_mappings.pt",
        map_location="cpu",
        weights_only=False,
    )
    user_to_index = checkpoint["user_to_index"]
    item_to_index = checkpoint["item_to_index"]
    hyperparameters = checkpoint.get(
        "hyperparameters",
        DEFAULT_HYPERPARAMETERS,
    )

    ratings = load_dataset()
    explicit_ratings = ratings[ratings[RATING_COLUMN] > 0].reset_index(drop=True)
    train, _, test = split_interactions(explicit_ratings, seed=SEED)
    known_test = test[
        test[USER_COLUMN].isin(user_to_index)
        & test[ITEM_COLUMN].isin(item_to_index)
    ].reset_index(drop=True)
    test_data = encode_interactions(
        known_test,
        user_to_index,
        item_to_index,
    )
    test_loader = DataLoader(
        ExplicitDataset(test_data),
        batch_size=2_048,
        shuffle=False,
    )

    model = ExplicitRecommenderMLP(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        embedding_dim=hyperparameters["embedding_dim"],
        hidden_dims=tuple(hyperparameters["hidden_dims"]),
        dropout=hyperparameters["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Test baselines:")
    print(evaluate_mean_baselines(train, known_test))
    print(f"MLP test metrics: {evaluate_ratings(model, test_loader, device)}")


if __name__ == "__main__":
    main()
