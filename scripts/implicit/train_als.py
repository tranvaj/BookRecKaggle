from pathlib import Path

import torch

from bookrec.data import encode_interactions, load_dataset, split_interactions
from bookrec.implicit.baselines import ALSBaseline


SEED = 42


def main():
    artifact_directory = Path("artifacts/implicit")
    checkpoint = torch.load(
        artifact_directory / "model_with_mappings.pt",
        map_location="cpu",
        weights_only=False,
    )
    user_to_index = checkpoint["user_to_index"]
    item_to_index = checkpoint["item_to_index"]

    interactions = load_dataset()
    train, _, _ = split_interactions(interactions, seed=SEED)
    train_data = encode_interactions(train, user_to_index, item_to_index)

    als_model = ALSBaseline.fit(
        train_data,
        num_users=len(user_to_index),
        num_items=len(item_to_index),
    )
    torch.save(
        {
            "user_factors": als_model.user_factors.cpu(),
            "item_factors": als_model.item_factors.cpu(),
        },
        artifact_directory / "als_model.pt",
    )
    print(f"Saved trained ALS model to {artifact_directory}")


if __name__ == "__main__":
    main()
