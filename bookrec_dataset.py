import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class BookRecDataset(Dataset):
    """PyTorch dataset for pre-encoded user/book/rating interactions."""

    def __init__(self, data: pd.DataFrame):
        required_columns = {"user", "item", "rating"}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns: {missing}")

        self._users = torch.tensor(data["user"].to_numpy(), dtype=torch.long)
        self._items = torch.tensor(data["item"].to_numpy(), dtype=torch.long)
        self._ratings = torch.tensor(data["rating"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._users)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "users": self._users[index],
            "items": self._items[index],
            "ratings": self._ratings[index],
        }

class ImplicitDataset(Dataset):
    def __init__(
        self,
        train_interactions: pd.DataFrame,
        all_interactions: pd.DataFrame,
        num_items,
        negatives_per_positive=4,
        fixed_negatives: bool = False,
        seed: int = 42,
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
        self.fixed_negatives = fixed_negatives
        self.seed = seed

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, index):
        user, positive_item = self.positive_pairs[index]
        user = int(user)
        positive_item = int(positive_item)

        sampled_negatives = set()
        seen = self.seen_items[user]

        if self.num_items - len(seen) < self.negatives_per_positive:
            raise ValueError(
                f"User {user} does not have enough unobserved items"
            )

        generator = None

        if self.fixed_negatives:
            generator = torch.Generator()
            generator.manual_seed(self.seed + index)

        negatives = []
        sampled = set()

        while len(negatives) < self.negatives_per_positive:
            candidate = torch.randint(
                low=0,
                high=self.num_items,
                size=(1,),
                generator=generator,
            ).item()

            if candidate not in seen and candidate not in sampled:
                negatives.append(candidate)
                sampled.add(candidate)

        
        # Permute just in case
        items = torch.tensor(
            [positive_item, *negatives],
            dtype=torch.long,
        )

        labels = torch.tensor(
            [1.0] + [0.0] * self.negatives_per_positive,
            dtype=torch.float32,
        )

        permutation = torch.randperm(len(items))

        items = items[permutation]
        labels = labels[permutation]

        return {
            "users": torch.full(
                (len(items),),
                user,
                dtype=torch.long,
            ),
            "items": items,
            "label": labels,
        }


class SampledRankingDataset(Dataset):
    """One fixed sampled candidate ranking per user for evaluation."""

    def __init__(
        self,
        held_out_interactions: pd.DataFrame,
        all_interactions: pd.DataFrame,
        num_items: int,
        num_candidates: int = 100,
        seed: int = 42,
    ):
        positives_by_user = (
            held_out_interactions
            .groupby("user")["item"]
            .agg(set)
            .to_dict()
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
            "items": torch.tensor(
                self.candidate_items[index],
                dtype=torch.long,
            ),
            "label": torch.tensor(
                self.labels[index],
                dtype=torch.float32,
            ),
        }
