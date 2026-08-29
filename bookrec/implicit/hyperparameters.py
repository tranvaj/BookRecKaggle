import json
from pathlib import Path


ARCHITECTURES = {
    "64-32": (64, 32),
    "128-64": (128, 64),
    "128-64-32": (128, 64, 32),
}

DEFAULT_HYPERPARAMETERS = {
    "embedding_dim": 32,
    "hidden_dims": (64, 32),
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
}

DEFAULT_ALS_HYPERPARAMETERS = {
    "factors": 32,
    "regularization": 0.05,
    "alpha": 10.0,
    "iterations": 20,
}


def load_hyperparameters(path: Path) -> dict:
    hyperparameters = DEFAULT_HYPERPARAMETERS.copy()
    if path.exists():
        hyperparameters.update(json.loads(path.read_text()))
    hyperparameters["hidden_dims"] = tuple(hyperparameters["hidden_dims"])
    return hyperparameters


def load_als_hyperparameters(path: Path) -> dict:
    """Load tuned ALS parameters, falling back to the existing defaults."""
    hyperparameters = DEFAULT_ALS_HYPERPARAMETERS.copy()
    if path.exists():
        hyperparameters.update(json.loads(path.read_text()))
    return hyperparameters
