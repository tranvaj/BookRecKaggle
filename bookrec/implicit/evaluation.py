import torch

from bookrec.implicit.metrics import (
    binary_cross_entropy,
    ndcg_at_50,
    recall_at_50,
    recall_at_100
)
from bookrec.training import validation_loop


RANKING_METRICS = [
    recall_at_50,
    ndcg_at_50,
    recall_at_100,
    binary_cross_entropy,
]


def evaluate_sampled_ranking(model, data_loader, device: torch.device):
    return validation_loop(data_loader, model, RANKING_METRICS, device)
