import torch
from torch import nn
from torch.nn import functional as F


HISTORY_MLP_ARCHITECTURE_VERSION = 3


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

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
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

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (self.user_embedding(users), self.item_embedding(items)),
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)


class ImplicitHistoryMLP(nn.Module):
    """Score candidates through normalized, shared item interactions."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 32,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_items < 1:
            raise ValueError("num_items must be positive")

        self.item_embedding = nn.EmbeddingBag(
            num_items,
            embedding_dim,
            mode="sum",
            include_last_offset=False,
        )

        layers: list[nn.Module] = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                (nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        history_items: torch.Tensor,
        history_offset: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        del users
        history_features = F.normalize(
            self.item_embedding(history_items, history_offset),
            dim=-1,
        )
        if history_features.shape[0] != items.shape[0]:
            raise ValueError(
                "Expected one history offset for each grouped candidate sample"
            )

        candidate_features = F.normalize(
            F.embedding(items, self.item_embedding.weight),
            dim=-1,
        )
        if items.ndim == 1:
            expanded_history = history_features
        else:
            expanded_history = history_features
            for _ in range(items.ndim - 1):
                expanded_history = expanded_history.unsqueeze(1)
            expanded_history = expanded_history.expand(*items.shape, -1)

        features = expanded_history * candidate_features
        return self.mlp(features).squeeze(-1)


class ImplicitProbabilityDeepEnsemble(nn.Module):
    """Average probabilities from independent copies of one architecture."""

    def __init__(
        self,
        members: list[nn.Module],
    ):
        super().__init__()
        if len(members) < 2:
            raise ValueError("A deep ensemble requires at least two members")
        self.members = nn.ModuleList(members)

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                torch.sigmoid(member(users, items, **kwargs))
                for member in self.members
            ],
            dim=0,
        ).mean(dim=0)


MODEL_REGISTRY = {
    "mlp": ImplicitRecommenderMLP,
    "neumf": ImplicitRecommenderNeuMF,
    "history_mlp": ImplicitHistoryMLP,
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

    model_kwargs = {
        "num_items": num_items,
        "embedding_dim": hyperparameters["embedding_dim"],
        "hidden_dims": tuple(hyperparameters["hidden_dims"]),
        "dropout": hyperparameters["dropout"],
    }
    if model_name != "history_mlp":
        model_kwargs["num_users"] = num_users
    return model_class(**model_kwargs)
