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
from bookrec.implicit.hyperparameters import DEFAULT_HYPERPARAMETERS
from bookrec.implicit.model import MODEL_REGISTRY, create_implicit_model


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
    artifact_root = Path("artifacts/implicit")
    checkpoints = {}
    for model_name in MODEL_REGISTRY:
        checkpoint_path = (
            artifact_root / model_name / "model_with_mappings.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing {model_name} checkpoint. Run "
                f"'python -m scripts.implicit.train --model {model_name}'."
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("model_type") != model_name:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} is not a {model_name} model"
            )
        checkpoints[model_name] = checkpoint

    reference_checkpoint = checkpoints["mlp"]
    user_to_index = reference_checkpoint["user_to_index"]
    item_to_index = reference_checkpoint["item_to_index"]
    for model_name, checkpoint in checkpoints.items():
        if (
            checkpoint["user_to_index"] != user_to_index
            or checkpoint["item_to_index"] != item_to_index
        ):
            raise ValueError(
                f"The {model_name} checkpoint uses different ID mappings"
            )

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

    als_checkpoint = torch.load(
        artifact_root / "als" / "model.pt",
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
    del als_model

    for model_name, checkpoint in checkpoints.items():
        hyperparameters = checkpoint.get(
            "hyperparameters",
            DEFAULT_HYPERPARAMETERS,
        )
        model = create_implicit_model(
            model_name,
            num_users=len(user_to_index),
            num_items=len(item_to_index),
            hyperparameters=hyperparameters,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_scores = evaluate_sampled_ranking(model, test_loader, device)
        print(f"{model_name.upper()} test metrics: {test_scores}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
