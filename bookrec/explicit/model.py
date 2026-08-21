import torch
from torch import nn


class ExplicitRecommenderMLP(nn.Module):
    """Predict ratings from global, user, item, and interaction effects."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        global_mean: float,
        embedding_dim: int = 32,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_users < 1 or num_items < 1:
            raise ValueError("num_users and num_items must be positive")

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.register_buffer("global_mean", torch.tensor(float(global_mean)))

        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend(
                (nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (self.user_embedding(users), self.item_embedding(items)),
            dim=-1,
        )
        interaction = self.mlp(features).squeeze(-1)
        return (
            self.global_mean
            + self.user_bias(users).squeeze(-1)
            + self.item_bias(items).squeeze(-1)
            + interaction
        )
