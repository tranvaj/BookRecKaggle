import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ImplicitDataset(Dataset):
    """Positive interactions with dynamically sampled unobserved items."""

    def __init__(
        self,
        train_interactions: pd.DataFrame,
        all_interactions: pd.DataFrame,
        num_items: int,
        negatives_per_positive: int = 4,
    ):
        self.positive_pairs = (
            train_interactions[["user", "item"]]
            .drop_duplicates()
            .to_numpy()
        )
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
        }


class SampledRankingDataset(Dataset):
    """One fixed sampled candidate ranking per user for evaluation."""

    def __init__(
        self,
        held_out_interactions: pd.DataFrame,
        all_interactions: pd.DataFrame,
        num_items: int,
        num_candidates: int = 1_000,
        seed: int = 42,
    ):
        positives_by_user = (
            held_out_interactions
            .groupby("user")["item"]
            .agg(set)
            .to_dict()
        )
        if not positives_by_user:
            raise ValueError("Cannot evaluate without held-out interactions")

        seen_by_user = (
            all_interactions
            .groupby("user")["item"]
            .agg(set)
            .to_dict()
        )

        largest_positive_set = max(map(len, positives_by_user.values()))
        self.num_candidates = max(num_candidates, largest_positive_set)
        self.users = []
        self.candidate_items = []
        self.labels = []
        rng = np.random.default_rng(seed)

        for user, positive_items in positives_by_user.items():
            positives = list(positive_items)
            seen = seen_by_user[user]
            negatives_needed = self.num_candidates - len(positives)

            if num_items - len(seen) < negatives_needed:
                raise ValueError(
                    f"User {user} does not have enough unobserved items"
                )

            negatives = []
            sampled = set()
            while len(negatives) < negatives_needed:
                candidate = int(rng.integers(num_items))
                if candidate not in seen and candidate not in sampled:
                    negatives.append(candidate)
                    sampled.add(candidate)

            candidates = np.asarray(positives + negatives, dtype=np.int64)
            labels = np.asarray(
                [1.0] * len(positives) + [0.0] * len(negatives),
                dtype=np.float32,
            )
            permutation = rng.permutation(self.num_candidates)

            self.users.append(int(user))
            self.candidate_items.append(candidates[permutation])
            self.labels.append(labels[permutation])

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
        }
