import argparse
from pathlib import Path

import torch

from bookrec.data import (
    create_id_mappings,
    encode_interactions,
    load_dataset,
    split_interactions,
)
from bookrec.implicit.baselines import ALSBaseline
from bookrec.implicit.hyperparameters import load_als_hyperparameters


SEED = 42


def train_als(
    output_dir: Path = Path("artifacts/implicit/als"),
    hyperparameters_path: Path | None = None,
    seed: int = SEED,
) -> Path:
    """Train ALS with selected parameters and return its checkpoint path."""
    selected_hyperparameters_path = (
        hyperparameters_path or output_dir / "best_hparams.json"
    )
    hyperparameters = load_als_hyperparameters(
        selected_hyperparameters_path
    )
    print(f"Training ALS hyperparameters: {hyperparameters}")
    print(f"Training seed: {seed}; output: {output_dir}")

    interactions = load_dataset()
    train, _, _ = split_interactions(interactions, seed=SEED)
    user_to_index, item_to_index = create_id_mappings(train)
    train_data = encode_interactions(train, user_to_index, item_to_index)

    als_model = ALSBaseline.fit(
        train_data,
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        seed=seed,
        **hyperparameters,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "user_factors": als_model.user_factors.cpu(),
            "item_factors": als_model.item_factors.cpu(),
            "hyperparameters": hyperparameters,
            "training_seed": seed,
            "split_seed": SEED,
        },
        checkpoint_path,
    )
    print(f"Saved trained ALS model to {checkpoint_path}")
    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/implicit/als"),
    )
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        help=(
            "Path to best_hparams.json. Defaults to "
            "<output-dir>/best_hparams.json."
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    train_als(
        output_dir=args.output_dir,
        hyperparameters_path=args.hyperparameters,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
