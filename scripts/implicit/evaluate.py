import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bookrec.data import encode_interactions, load_dataset, split_interactions
from bookrec.implicit.baselines import ALSBaseline, evaluate_ranking_baselines
from bookrec.implicit.datasets import SampledRankingDataset
from bookrec.implicit.evaluation import evaluate_sampled_ranking
from bookrec.implicit.model import ImplicitRecommenderMLP


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_directory = Path("artifacts/implicit")

    checkpoint = torch.load(
        artifact_directory / "model_with_mappings.pt",
        map_location="cpu",
        weights_only=False,
    )
    user_to_index = checkpoint["user_to_index"]
    item_to_index = checkpoint["item_to_index"]

    interactions = load_dataset()
    train, validation, test = split_interactions(interactions, seed=SEED)
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

    test_dataset = SampledRankingDataset(
        test_data,
        all_interactions,
        num_items=len(item_to_index),
        num_candidates=EVALUATION_CANDIDATES,
        seed=SEED + 1,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = ImplicitRecommenderMLP(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    als_checkpoint = torch.load(
        artifact_directory / "als_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    als_model = ALSBaseline(
        als_checkpoint["user_factors"],
        als_checkpoint["item_factors"],
    )

    baseline_scores = evaluate_ranking_baselines(
        train_data,
        test_loader,
        num_items=len(item_to_index),
        device=device,
        als_model=als_model,
    )
    print(f"Test baselines: {baseline_scores}")

    test_scores = evaluate_sampled_ranking(model, test_loader, device)
    print(f"MLP test metrics: {test_scores}")


if __name__ == "__main__":
    main()
