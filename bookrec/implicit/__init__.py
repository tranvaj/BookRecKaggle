"""Implicit-feedback recommendation components."""

from bookrec.implicit.inference import (
    LoadedHistoryMLP,
    load_history_mlp,
    predict_history_probabilities,
)
from bookrec.implicit.model import (
    MODEL_REGISTRY,
    ImplicitHistoryMLP,
    ImplicitRecommenderMLP,
    ImplicitRecommenderNeuMF,
    create_implicit_model,
)

__all__ = [
    "ImplicitHistoryMLP",
    "ImplicitRecommenderMLP",
    "ImplicitRecommenderNeuMF",
    "LoadedHistoryMLP",
    "MODEL_REGISTRY",
    "create_implicit_model",
    "load_history_mlp",
    "predict_history_probabilities",
]
