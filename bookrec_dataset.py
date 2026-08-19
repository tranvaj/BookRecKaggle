import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
class BookRecDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self._users = data["user"]
        self._items = data["item"]
        self._ratings = data["rating"]

    def __len__(self):
        return len(self._users)

    def __getitem__(self, index):
        res = {
            "users": torch.tensor(self._users[index], dtype=torch.long),
            "items": torch.tensor(self._items[index], dtype=torch.long),
            "ratings": torch.tensor(self._ratings[index], dtype=torch.long)
        }
        return res