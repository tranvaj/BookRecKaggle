import torch
from torch import nn


class ImplicitRecommenderNeuMF(nn.Module):
    """Combine GMF and MLP features to score user-item interactions."""

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

        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)

        layers: list[nn.Module] = []
        mlp_output_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend(
                (
                    nn.Linear(mlp_output_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            mlp_output_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(embedding_dim + mlp_output_dim, 1)

        for embedding in (
            self.gmf_user_embedding,
            self.gmf_item_embedding,
            self.mlp_user_embedding,
            self.mlp_item_embedding,
        ):
            nn.init.normal_(embedding.weight, std=0.05)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf_features = (
            self.gmf_user_embedding(users)
            * self.gmf_item_embedding(items)
        )
        mlp_input = torch.cat(
            (self.mlp_user_embedding(users), self.mlp_item_embedding(items)),
            dim=-1,
        )
        mlp_features = self.mlp(mlp_input)
        combined_features = torch.cat(
            (gmf_features, mlp_features),
            dim=-1,
        )
        return self.output_layer(combined_features).squeeze(-1)


class ImplicitRecommenderMLP(nn.Module):
    """Score user-item interaction likelihood with embedding features."""

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


MODEL_REGISTRY = {
    "mlp": ImplicitRecommenderMLP,
    "neumf": ImplicitRecommenderNeuMF,
}


def create_implicit_model(
    model_name: str,
    num_users: int,
    num_items: int,
    hyperparameters: dict,
) -> nn.Module:
    try:
        model_class = MODEL_REGISTRY[model_name]
    except KeyError as error:
        choices = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown implicit model '{model_name}'. Choose from: {choices}"
        ) from error

    return model_class(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=hyperparameters["embedding_dim"],
        hidden_dims=tuple(hyperparameters["hidden_dims"]),
        dropout=hyperparameters["dropout"],
    )
