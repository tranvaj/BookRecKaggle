import torch

from bookrec.implicit.metrics import (
    binary_cross_entropy,
    mean_reciprocal_rank,
    ndcg_at_10,
    recall_at_10,
)
from bookrec.training import validation_loop


RANKING_METRICS = [
    binary_cross_entropy,
    recall_at_10,
    ndcg_at_10,
    mean_reciprocal_rank,
]


def evaluate_sampled_ranking(model, data_loader, device: torch.device):
    return validation_loop(data_loader, model, RANKING_METRICS, device)
