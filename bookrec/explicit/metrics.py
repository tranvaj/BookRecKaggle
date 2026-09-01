import numpy as np


def rmse(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - labels) ** 2)))


def mae(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - labels)))
