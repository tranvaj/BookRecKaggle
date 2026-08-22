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


def load_hyperparameters(path: Path) -> dict:
    hyperparameters = DEFAULT_HYPERPARAMETERS.copy()
    if path.exists():
        hyperparameters.update(json.loads(path.read_text()))
    hyperparameters["hidden_dims"] = tuple(hyperparameters["hidden_dims"])
    return hyperparameters
