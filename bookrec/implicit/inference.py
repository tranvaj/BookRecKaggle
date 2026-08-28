"""Load and run history-based implicit recommenders for inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from bookrec.implicit.model import (
    HISTORY_MLP_ARCHITECTURE_VERSION,
    ImplicitHistoryMLP,
    ImplicitProbabilityDeepEnsemble,
    create_implicit_model,
)


@dataclass(frozen=True)
class LoadedHistoryMLP:
    """A history model and the item metadata needed to use it."""

    model: ImplicitHistoryMLP | ImplicitProbabilityDeepEnsemble
    item_to_index: dict[object, int]
    index_to_item: tuple[object, ...]
    training_item_counts: torch.Tensor | None
    member_paths: tuple[Path, ...]
    device: torch.device

    @property
    def is_ensemble(self) -> bool:
        return len(self.member_paths) > 1


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_checkpoint_paths(artifact_path: Path) -> list[Path]:
    if artifact_path.is_file():
        return [artifact_path]

    direct_checkpoint = artifact_path / "model_with_mappings.pt"
    if direct_checkpoint.is_file():
        return [direct_checkpoint]

    member_paths = sorted(
        artifact_path.glob("seed_*/model_with_mappings.pt")
    )
    if member_paths:
        if len(member_paths) < 2:
            raise ValueError(
                f"An ensemble requires at least two members in {artifact_path}"
            )
        return member_paths

    raise FileNotFoundError(
        f"No model_with_mappings.pt checkpoint found in {artifact_path}"
    )


def _validate_checkpoint(checkpoint: dict, checkpoint_path: Path) -> None:
    if checkpoint.get("model_type") != "history_mlp":
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is not a history_mlp model"
        )
    if (
        checkpoint.get("architecture_version")
        != HISTORY_MLP_ARCHITECTURE_VERSION
    ):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} uses an obsolete history_mlp "
            "architecture. Retrain it."
        )

    required_keys = {
        "model_state_dict",
        "item_to_index",
        "hyperparameters",
    }
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is missing: {missing}"
        )


def _invert_item_mapping(
    item_to_index: dict[object, int],
) -> tuple[object, ...]:
    index_to_item: list[object | None] = [None] * len(item_to_index)
    for item_id, index in item_to_index.items():
        if not 0 <= index < len(index_to_item):
            raise ValueError("Item mapping indices must be consecutive")
        if index_to_item[index] is not None:
            raise ValueError("Item mapping indices must be unique")
        index_to_item[index] = item_id
    if any(item_id is None for item_id in index_to_item):
        raise ValueError("Item mapping indices must be consecutive")
    return tuple(index_to_item)


def load_history_mlp(
    artifact_path: str | Path,
    device: str | torch.device | None = None,
) -> LoadedHistoryMLP:
    """Load one history MLP or a compatible probability ensemble.

    ``artifact_path`` may be a checkpoint file, a single-model artifact
    directory, or an ensemble directory containing ``seed_*`` directories.
    """
    selected_device = _resolve_device(device)
    member_paths = _resolve_checkpoint_paths(Path(artifact_path))
    checkpoints = []
    for path in member_paths:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        _validate_checkpoint(checkpoint, path)
        checkpoints.append(checkpoint)

    reference = checkpoints[0]
    item_to_index = reference["item_to_index"]
    hyperparameters = reference["hyperparameters"]
    for path, checkpoint in zip(member_paths[1:], checkpoints[1:]):
        if (
            checkpoint["item_to_index"] != item_to_index
            or checkpoint["hyperparameters"] != hyperparameters
        ):
            raise ValueError(f"Incompatible history_mlp ensemble member: {path}")

    members = []
    num_users = len(reference.get("user_to_index", {})) or 1
    for checkpoint in checkpoints:
        member = create_implicit_model(
            "history_mlp",
            num_users=num_users,
            num_items=len(item_to_index),
            hyperparameters=hyperparameters,
        )
        member.load_state_dict(checkpoint["model_state_dict"])
        members.append(member)

    if len(members) == 1:
        model = members[0]
    else:
        model = ImplicitProbabilityDeepEnsemble(members)
    model = model.to(selected_device)
    model.eval()

    training_item_counts = reference.get("training_item_counts")
    if training_item_counts is not None:
        training_item_counts = training_item_counts.detach().cpu()

    return LoadedHistoryMLP(
        model=model,
        item_to_index=item_to_index,
        index_to_item=_invert_item_mapping(item_to_index),
        training_item_counts=training_item_counts,
        member_paths=tuple(member_paths),
        device=selected_device,
    )


def _history_model_num_items(model: nn.Module) -> int:
    if isinstance(model, ImplicitHistoryMLP):
        return model.item_embedding.num_embeddings
    if isinstance(model, ImplicitProbabilityDeepEnsemble) and all(
        isinstance(member, ImplicitHistoryMLP) for member in model.members
    ):
        return model.members[0].item_embedding.num_embeddings
    raise TypeError(
        "Expected an ImplicitHistoryMLP or an ensemble of history MLPs"
    )


def predict_history_probabilities(
    model: ImplicitHistoryMLP | ImplicitProbabilityDeepEnsemble,
    history_items: Sequence[int] | torch.Tensor,
    candidate_items: Sequence[int] | torch.Tensor | None = None,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 8_192,
) -> torch.Tensor:
    """Score candidates for one item history and return CPU probabilities.

    Item values are encoded item indices. If ``candidate_items`` is omitted,
    every item known by the model is scored in index order. History indices are
    deduplicated so the input retains binary-vector semantics.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    num_items = _history_model_num_items(model)
    history = torch.as_tensor(history_items, dtype=torch.long).flatten()
    if history.numel() == 0:
        raise ValueError("history_items must contain at least one item")
    history = torch.unique(history)

    if candidate_items is None:
        candidates = torch.arange(num_items, dtype=torch.long)
    else:
        candidates = torch.as_tensor(candidate_items, dtype=torch.long).flatten()

    for name, values in (
        ("history_items", history),
        ("candidate_items", candidates),
    ):
        if values.numel() and (
            values.min().item() < 0 or values.max().item() >= num_items
        ):
            raise IndexError(
                f"{name} must contain indices between 0 and {num_items - 1}"
            )

    if candidates.numel() == 0:
        return torch.empty(0, dtype=torch.float32)

    selected_device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )
    model = model.to(selected_device)
    model.eval()
    history = history.to(selected_device)
    history_offset = torch.tensor([0], device=selected_device)
    probabilities = []

    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            candidate_batch = candidates[start : start + batch_size]
            candidate_batch = candidate_batch.to(selected_device).unsqueeze(0)
            users = torch.zeros_like(candidate_batch)
            scores = model(
                users=users,
                items=candidate_batch,
                history_items=history,
                history_offset=history_offset,
            )
            if isinstance(model, ImplicitHistoryMLP):
                scores = torch.sigmoid(scores)
            probabilities.append(scores.squeeze(0).cpu())

    return torch.cat(probabilities)
