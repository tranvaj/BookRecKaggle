import pandas as pd
import torch
from torch.utils.data import Dataset


class ExplicitDataset(Dataset):
    """Pointwise user-item examples with explicit numeric ratings."""

    def __init__(self, interactions: pd.DataFrame):
        required_columns = {"user", "item", "rating"}
        missing_columns = required_columns.difference(interactions.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns: {missing}")

        self.users = torch.tensor(
            interactions["user"].to_numpy(),
            dtype=torch.long,
        )
        self.items = torch.tensor(
            interactions["item"].to_numpy(),
            dtype=torch.long,
        )
        self.ratings = torch.tensor(
            interactions["rating"].to_numpy(),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return {
            "users": self.users[index],
            "items": self.items[index],
            "label": self.ratings[index],
        }
