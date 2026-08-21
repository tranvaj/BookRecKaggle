from dataclasses import dataclass

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter


USER_COLUMN = "User-ID"
ITEM_COLUMN = "ISBN"
RATING_COLUMN = "Book-Rating"
REQUIRED_COLUMNS = {USER_COLUMN, ITEM_COLUMN, RATING_COLUMN}


@dataclass(frozen=True)
class IdMappings:
    """Integer IDs learned from the training split."""

    users: dict[object, int]
    items: dict[object, int]


def load_ds(
    file_path: str = "Ratings.csv",
    url_path: str = "arashnic/book-recommendation-dataset",
) -> pd.DataFrame:
    """Load one file from the Book-Crossing Kaggle dataset."""
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        url_path,
        file_path,
    )


def get_splits(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    min_interactions: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible per-user train, validation, and test splits."""
    shuffled = df.groupby(USER_COLUMN).sample(frac=1, random_state=seed)
    pos = shuffled.groupby(USER_COLUMN).cumcount()
    n = shuffled.groupby(USER_COLUMN)[USER_COLUMN].transform("size")

    n_val = (n * val_frac).astype(int).clip(lower=1)
    n_test = (n * test_frac).astype(int).clip(lower=1)
    eligible = n >= min_interactions

    test_mask = eligible & (pos < n_test)
    val_mask = eligible & (pos >= n_test) & (pos < n_test + n_val)
    train_mask = ~(test_mask | val_mask)

    train_df = shuffled[train_mask].reset_index(drop=True)
    val_df = shuffled[val_mask].reset_index(drop=True)
    test_df = shuffled[test_mask].reset_index(drop=True)
    return train_df, val_df, test_df


def create_id_mappings(train_df):
    user_to_index = {
        user_id: index
        for index, user_id in enumerate(train_df[USER_COLUMN].unique())
    }

    item_to_index = {
        isbn: index
        for index, isbn in enumerate(train_df[ITEM_COLUMN].unique())
    }

    return user_to_index, item_to_index


def encode_ids(df, user_to_index, item_to_index):
    encoded = pd.DataFrame({
        "user": df[USER_COLUMN].map(user_to_index),
        "item": df[ITEM_COLUMN].map(item_to_index),
        "rating": df[RATING_COLUMN]
    })

    # Removes validation/test items that never occurred in training.
    encoded = encoded.dropna()

    encoded["user"] = encoded["user"].astype("int64")
    encoded["item"] = encoded["item"].astype("int64")
    encoded["rating"] = encoded["rating"].astype("int64")

    return encoded.reset_index(drop=True)