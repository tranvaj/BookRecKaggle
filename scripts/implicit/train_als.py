from pathlib import Path

import torch

from bookrec.data import (
    create_id_mappings,
    encode_interactions,
    load_dataset,
    split_interactions,
)
from bookrec.implicit.baselines import ALSBaseline


SEED = 42


def main():
    artifact_root = Path("artifacts/implicit")
    interactions = load_dataset()
    train, _, _ = split_interactions(interactions, seed=SEED)
    user_to_index, item_to_index = create_id_mappings(train)
    train_data = encode_interactions(train, user_to_index, item_to_index)

    als_model = ALSBaseline.fit(
        train_data,
        num_users=len(user_to_index),
        num_items=len(item_to_index),
    )
    artifact_directory = artifact_root / "als"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "user_factors": als_model.user_factors.cpu(),
            "item_factors": als_model.item_factors.cpu(),
        },
        artifact_directory / "model.pt",
    )
    print(f"Saved trained ALS model to {artifact_directory}")


if __name__ == "__main__":
    main()
