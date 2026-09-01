import torch

from bookrec.explicit.metrics import mae, rmse
from bookrec.training import validation_loop


RATING_METRICS = [rmse, mae]


def evaluate_ratings(model, data_loader, device: torch.device):
    return validation_loop(data_loader, model, RATING_METRICS, device)
