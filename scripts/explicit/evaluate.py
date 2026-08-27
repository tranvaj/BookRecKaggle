import argparse
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
from bookrec.explicit.model import ExplicitMLPEnsemble, ExplicitRecommenderMLP


SEED = 42
MODEL_NAME = "mlp"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/explicit"),
    )
    parser.add_argument(
        "--ensemble-dir",
        type=Path,
        help=(
            "Directory containing seed_<n>/model_with_mappings.pt members. "
            "Defaults to <artifact-root>/mlp_ensemble."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.artifact_root / MODEL_NAME / "model_with_mappings.pt",
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

    ensemble_directory = (
        args.ensemble_dir or args.artifact_root / "mlp_ensemble"
    )
    member_paths = sorted(
        ensemble_directory.glob("seed_*/model_with_mappings.pt")
    )
    if len(member_paths) < 2:
        raise FileNotFoundError(
            f"Expected at least two ensemble members in {ensemble_directory}. "
            "Run 'python -m scripts.explicit.train_ensemble'."
        )

    members = []
    ensemble_hyperparameters = None
    for path in member_paths:
        member_checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        member_hyperparameters = member_checkpoint.get(
            "hyperparameters",
            DEFAULT_HYPERPARAMETERS,
        )
        if (
            member_checkpoint.get("model_type", MODEL_NAME) != MODEL_NAME
            or member_checkpoint["user_to_index"] != user_to_index
            or member_checkpoint["item_to_index"] != item_to_index
            or (
                ensemble_hyperparameters is not None
                and member_hyperparameters != ensemble_hyperparameters
            )
        ):
            raise ValueError(f"Incompatible ensemble member: {path}")
        ensemble_hyperparameters = member_hyperparameters
        member = ExplicitRecommenderMLP(
            num_users=len(user_to_index),
            num_items=len(item_to_index),
            embedding_dim=member_hyperparameters["embedding_dim"],
            hidden_dims=tuple(member_hyperparameters["hidden_dims"]),
            dropout=member_hyperparameters["dropout"],
        )
        member.load_state_dict(member_checkpoint["model_state_dict"])
        members.append(member)

    ensemble = ExplicitMLPEnsemble(members).to(device)
    print(
        f"MLP ENSEMBLE ({len(members)} members) test metrics: "
        f"{evaluate_ratings(ensemble, test_loader, device)}"
    )


if __name__ == "__main__":
    main()
