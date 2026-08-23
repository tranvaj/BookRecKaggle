"""Implicit-feedback recommendation components."""

from bookrec.implicit.model import (
    MODEL_REGISTRY,
    ImplicitRecommenderMLP,
    ImplicitRecommenderNeuMF,
    create_implicit_model,
)

__all__ = [
    "ImplicitRecommenderMLP",
    "ImplicitRecommenderNeuMF",
    "MODEL_REGISTRY",
    "create_implicit_model",
]
