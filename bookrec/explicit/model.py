import torch
from torch import nn


class ExplicitRecommenderMLP(nn.Module):
    """Predict ratings directly from learned user and item embeddings."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_users < 1 or num_items < 1:
            raise ValueError("num_users and num_items must be positive")

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

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

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (self.user_embedding(users), self.item_embedding(items)),
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)


class ExplicitMLPEnsemble(nn.Module):
    """Average rating predictions from independent MLP copies."""

    def __init__(self, members: list[ExplicitRecommenderMLP]):
        super().__init__()
        if len(members) < 2:
            raise ValueError("An ensemble requires at least two members")
        self.members = nn.ModuleList(members)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [member(users, items) for member in self.members],
            dim=0,
        ).mean(dim=0)
