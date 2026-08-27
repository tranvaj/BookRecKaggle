import numpy as np
import pandas as pd
import torch
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from torch import nn

from bookrec.implicit.metrics import ndcg_at_50, recall_at_50, recall_at_100
from bookrec.training import validation_loop


BASELINE_METRICS = [recall_at_50, ndcg_at_50, recall_at_100]


class RandomBaseline(nn.Module):
    """Assign fresh random scores to candidate items."""

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        return torch.rand(items.shape, device=items.device)


class MostPopularBaseline(nn.Module):
    """Rank candidate items by their interaction count in the training set."""

    def __init__(self, item_scores: torch.Tensor):
        super().__init__()
        self.register_buffer("item_scores", item_scores.float())

    @classmethod
    def fit(
        cls,
        train_interactions: pd.DataFrame,
        num_items: int,
    ) -> "MostPopularBaseline":
        items = torch.tensor(
            train_interactions["item"].to_numpy(copy=True),
            dtype=torch.long,
        )
        counts = torch.bincount(items, minlength=num_items)
        return cls(torch.log1p(counts))

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_scores[items]


class ALSBaseline(nn.Module):
    """Implicit-feedback matrix factorization trained with ALS."""

    def __init__(
        self,
        user_factors: torch.Tensor,
        item_factors: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("user_factors", user_factors.float())
        self.register_buffer("item_factors", item_factors.float())

    @classmethod
    def fit(
        cls,
        train_interactions: pd.DataFrame,
        num_users: int,
        num_items: int,
        factors: int = 32,
        regularization: float = 0.05,
        alpha: float = 10.0,
        iterations: int = 20,
        seed: int = 42,
    ) -> "ALSBaseline":
        user_item_matrix = csr_matrix(
            (
                np.ones(len(train_interactions), dtype=np.float32),
                (
                    train_interactions["user"].to_numpy(),
                    train_interactions["item"].to_numpy(),
                ),
            ),
            shape=(num_users, num_items),
        )
        with threadpool_limits(limits=1, user_api="blas"):
            model = AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                alpha=alpha,
                iterations=iterations,
                random_state=seed,
                use_gpu=False,
            )
            model.fit(user_item_matrix)
        return cls(
            torch.from_numpy(model.user_factors),
            torch.from_numpy(model.item_factors),
        )

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        user_factors = self.user_factors[users]
        item_factors = self.item_factors[items]
        return (user_factors * item_factors).sum(dim=-1)


def evaluate_ranking_baselines(
    train_interactions: pd.DataFrame,
    data_loader,
    num_items: int,
    device: torch.device,
    als_model: ALSBaseline,
) -> dict[str, dict[str, float]]:
    baselines = {
        "random": RandomBaseline(),
        "most_popular": MostPopularBaseline.fit(
            train_interactions,
            num_items=num_items,
        ),
        "als": als_model,
    }
    return {
        name: validation_loop(data_loader, model, BASELINE_METRICS, device)
        for name, model in baselines.items()
    }
