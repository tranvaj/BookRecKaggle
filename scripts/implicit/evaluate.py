import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bookrec.data import encode_interactions, load_dataset, split_interactions
from bookrec.implicit.baselines import ALSBaseline, evaluate_ranking_baselines
from bookrec.implicit.datasets import (
    SampledRankingDataset,
    collate_implicit_batch,
)
from bookrec.implicit.evaluation import evaluate_sampled_ranking
from bookrec.implicit.hyperparameters import DEFAULT_HYPERPARAMETERS
from bookrec.implicit.model import (
    HISTORY_MLP_ARCHITECTURE_VERSION,
    MODEL_REGISTRY,
    ImplicitProbabilityDeepEnsemble,
    create_implicit_model,
)


SEED = 42
EVALUATION_CANDIDATES = 1_000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/implicit"),
        help="Directory containing the baseline model checkpoints.",
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
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_root = args.artifact_root
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
        if (
            model_name == "history_mlp"
            and checkpoint.get("architecture_version")
            != HISTORY_MLP_ARCHITECTURE_VERSION
        ):
            raise ValueError(
                f"Checkpoint at {checkpoint_path} uses the obsolete "
                "history_mlp architecture. Retrain it."
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
    test_context = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )
    all_interactions = pd.concat(
        [train_data, validation_data, test_data],
        ignore_index=True,
    )

    singleton_test_dataset = SampledRankingDataset(
        test_data,
        context_interactions=test_context,
        all_interactions=all_interactions,
        num_items=len(item_to_index),
        num_candidates=EVALUATION_CANDIDATES,
        seed=SEED + 1,
        context_mode="singleton",
    )
    full_history_test_dataset = SampledRankingDataset(
        test_data,
        context_interactions=test_context,
        all_interactions=all_interactions,
        num_items=len(item_to_index),
        num_candidates=EVALUATION_CANDIDATES,
        seed=SEED + 1,
        context_mode="full",
    )
    singleton_test_loader = DataLoader(
        singleton_test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_implicit_batch,
    )
    full_history_test_loader = DataLoader(
        full_history_test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_implicit_batch,
    )

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
        singleton_test_loader,
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

        if model_name == "history_mlp":
            full_history_scores = evaluate_sampled_ranking(
                model,
                full_history_test_loader,
                device,
            )
            print(
                "HISTORY_MLP full-history test metrics: "
                f"{full_history_scores}"
            )
            singleton_scores = evaluate_sampled_ranking(
                model,
                singleton_test_loader,
                device,
            )
            print(
                "HISTORY_MLP singleton-context test metrics: "
                f"{singleton_scores}"
            )
        else:
            test_scores = evaluate_sampled_ranking(
                model,
                singleton_test_loader,
                device,
            )
            print(f"{model_name.upper()} test metrics: {test_scores}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble_directory = args.ensemble_dir or artifact_root / "mlp_ensemble"
    member_paths = sorted(
        ensemble_directory.glob("seed_*/model_with_mappings.pt")
    )
    if len(member_paths) < 2:
        raise FileNotFoundError(
            f"Expected at least two ensemble members in {ensemble_directory}. "
            "Run 'python -m scripts.implicit.train_ensemble'."
        )
    member_checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in member_paths
    ]
    ensemble_model_type = member_checkpoints[0].get("model_type")
    ensemble_hyperparameters = member_checkpoints[0]["hyperparameters"]
    members = []
    for path, checkpoint in zip(member_paths, member_checkpoints):
        if (
            checkpoint.get("model_type") != ensemble_model_type
            or checkpoint["user_to_index"] != user_to_index
            or checkpoint["item_to_index"] != item_to_index
            or checkpoint["hyperparameters"] != ensemble_hyperparameters
        ):
            raise ValueError(f"Incompatible ensemble member: {path}")
        if (
            ensemble_model_type == "history_mlp"
            and checkpoint.get("architecture_version")
            != HISTORY_MLP_ARCHITECTURE_VERSION
        ):
            raise ValueError(
                f"Obsolete history_mlp ensemble member: {path}. Retrain it."
            )
        member = create_implicit_model(
            ensemble_model_type,
            num_users=len(user_to_index),
            num_items=len(item_to_index),
            hyperparameters=ensemble_hyperparameters,
        )
        member.load_state_dict(checkpoint["model_state_dict"])
        members.append(member)
    ensemble_model = ImplicitProbabilityDeepEnsemble(members).to(device)
    ensemble_label = (
        f"{ensemble_model_type.upper()} PROBABILITY ENSEMBLE "
        f"({len(members)} members)"
    )
    if ensemble_model_type == "history_mlp":
        full_history_scores = evaluate_sampled_ranking(
            ensemble_model,
            full_history_test_loader,
            device,
        )
        print(
            f"{ensemble_label} full-history test metrics: "
            f"{full_history_scores}"
        )
        singleton_scores = evaluate_sampled_ranking(
            ensemble_model,
            singleton_test_loader,
            device,
        )
        print(
            f"{ensemble_label} singleton-context test metrics: "
            f"{singleton_scores}"
        )
    else:
        test_scores = evaluate_sampled_ranking(
            ensemble_model,
            singleton_test_loader,
            device,
        )
        print(f"{ensemble_label} test metrics: {test_scores}")


if __name__ == "__main__":
    main()
