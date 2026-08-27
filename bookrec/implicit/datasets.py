import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate


def collate_implicit_batch(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Collate grouped candidates and variable-length item histories."""
    histories = [sample["history_items"] for sample in samples]
    lengths = torch.tensor([len(history) for history in histories])
    offsets = torch.zeros(len(histories), dtype=torch.long)
    if len(histories) > 1:
        offsets[1:] = torch.cumsum(lengths[:-1], dim=0)
    nonempty_histories = [history for history in histories if len(history)]
    flattened_history = (
        torch.cat(nonempty_histories)
        if nonempty_histories
        else torch.empty(0, dtype=torch.long)
    )
    batch = default_collate(
        [
            {
                key: value
                for key, value in sample.items()
                if key != "history_items"
            }
            for sample in samples
        ]
    )
    batch["history_items"] = flattened_history
    batch["history_offset"] = offsets
    return batch


class ImplicitDataset(Dataset):
    """Positive interactions with dynamically sampled unobserved items."""

    def __init__(
        self,
        train_interactions: pd.DataFrame,
        all_interactions: pd.DataFrame,
        num_items: int,
        negatives_per_positive: int = 4,
        require_nonempty_history: bool = False,
    ):
        positive_pairs = (
            train_interactions[["user", "item"]]
            .drop_duplicates()
            .to_numpy()
        )
        self.history_by_user = (
            train_interactions.groupby("user")["item"].agg(set).to_dict()
        )
        if require_nonempty_history:
            positive_pairs = np.asarray(
                [
                    pair
                    for pair in positive_pairs
                    if len(self.history_by_user[int(pair[0])]) > 1
                ],
                dtype=np.int64,
            ).reshape(-1, 2)
        self.positive_pairs = positive_pairs
        self.seen_items = (
            all_interactions
            .groupby("user")["item"]
            .agg(set)
            .to_dict()
        )
        self.num_items = num_items
        self.negatives_per_positive = negatives_per_positive

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, index):
        user, positive_item = map(int, self.positive_pairs[index])
        seen = self.seen_items[user]
        history = sorted(self.history_by_user[user] - {positive_item})

        if self.num_items - len(seen) < self.negatives_per_positive:
            raise ValueError(
                f"User {user} does not have enough unobserved items"
            )

        negatives = []
        sampled = set()
        while len(negatives) < self.negatives_per_positive:
            candidate = torch.randint(self.num_items, size=(1,)).item()
            if candidate not in seen and candidate not in sampled:
                negatives.append(candidate)
                sampled.add(candidate)

        items = torch.tensor([positive_item, *negatives], dtype=torch.long)
        labels = torch.tensor(
            [1.0] + [0.0] * self.negatives_per_positive,
            dtype=torch.float32,
        )
        permutation = torch.randperm(len(items))

        return {
            "users": torch.full((len(items),), user, dtype=torch.long),
            "items": items[permutation],
            "label": labels[permutation],
            "history_items": torch.tensor(history, dtype=torch.long),
        }


class SampledRankingDataset(Dataset):
    """One fixed sampled candidate ranking per user for evaluation."""

    def __init__(
        self,
        held_out_interactions: pd.DataFrame,
        context_interactions: pd.DataFrame,
        all_interactions: pd.DataFrame,
        num_items: int,
        num_candidates: int = 1_000,
        seed: int = 42,
        context_mode: str = "full",
    ):
        if context_mode not in {"singleton", "full"}:
            raise ValueError("context_mode must be 'singleton' or 'full'")
        positives_by_user = (
            held_out_interactions
            .groupby("user")["item"]
            .agg(set)
            .to_dict()
        )
        if not positives_by_user:
            raise ValueError("Cannot evaluate without held-out interactions")
        context_by_user = (
            context_interactions.groupby("user")["item"].agg(set).to_dict()
        )

        seen_by_user = (
            all_interactions
            .groupby("user")["item"]
            .agg(set)
            .to_dict()
        )

        largest_positive_set = max(map(len, positives_by_user.values()))
        self.num_candidates = max(num_candidates, largest_positive_set)
        self.users = []
        self.histories = []
        self.candidate_items = []
        self.labels = []
        candidate_rng = np.random.default_rng(seed)
        context_rng = np.random.default_rng(seed + 1)

        for user, positive_items in positives_by_user.items():
            context_items = context_by_user.get(user, set()) - positive_items
            if not context_items:
                continue
            if context_mode == "singleton":
                history = [
                    int(context_rng.choice(sorted(context_items)))
                ]
            else:
                history = sorted(context_items)
            positives = sorted(positive_items)
            seen = seen_by_user[user]
            negatives_needed = self.num_candidates - len(positives)

            if num_items - len(seen) < negatives_needed:
                raise ValueError(
                    f"User {user} does not have enough unobserved items"
                )

            negatives = []
            sampled = set()
            while len(negatives) < negatives_needed:
                candidate = int(candidate_rng.integers(num_items))
                if candidate not in seen and candidate not in sampled:
                    negatives.append(candidate)
                    sampled.add(candidate)

            candidates = np.asarray(positives + negatives, dtype=np.int64)
            labels = np.asarray(
                [1.0] * len(positives) + [0.0] * len(negatives),
                dtype=np.float32,
            )
            permutation = candidate_rng.permutation(self.num_candidates)

            self.users.append(int(user))
            self.histories.append(np.asarray(history, dtype=np.int64))
            self.candidate_items.append(candidates[permutation])
            self.labels.append(labels[permutation])

        if not self.users:
            raise ValueError("Cannot evaluate without nonempty item histories")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return {
            "users": torch.full(
                (self.num_candidates,),
                self.users[index],
                dtype=torch.long,
            ),
            "items": torch.tensor(self.candidate_items[index], dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "history_items": torch.tensor(
                self.histories[index],
                dtype=torch.long,
            ),
        }
