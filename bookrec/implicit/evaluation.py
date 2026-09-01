import torch

from bookrec.implicit.metrics import (
    ndcg_at_50,
    recall_at_50,
    recall_at_100
)
from bookrec.training import validation_loop


RANKING_METRICS = [
    recall_at_50,
    ndcg_at_50,
    recall_at_100,
]


def evaluate_sampled_ranking(model, data_loader, device: torch.device):
    return validation_loop(data_loader, model, RANKING_METRICS, device)
