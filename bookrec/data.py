import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter


USER_COLUMN = "User-ID"
ITEM_COLUMN = "ISBN"
RATING_COLUMN = "Book-Rating"


def load_dataset(
    file_path: str = "Ratings.csv",
    dataset_handle: str = "arashnic/book-recommendation-dataset",
) -> pd.DataFrame:
    """Load one file from the Book-Crossing Kaggle dataset."""
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset_handle,
        file_path,
    )


def split_interactions(
    interactions: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    min_interactions: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible per-user train, validation, and test splits."""
    shuffled = interactions.groupby(USER_COLUMN).sample(frac=1, random_state=seed)
    position = shuffled.groupby(USER_COLUMN).cumcount()
    group_size = shuffled.groupby(USER_COLUMN)[USER_COLUMN].transform("size")

    validation_size = (group_size * val_fraction).astype(int).clip(lower=1)
    test_size = (group_size * test_fraction).astype(int).clip(lower=1)
    eligible = group_size >= min_interactions

    test_mask = eligible & (position < test_size)
    validation_mask = (
        eligible
        & (position >= test_size)
        & (position < test_size + validation_size)
    )
    train_mask = ~(test_mask | validation_mask)

    train = shuffled[train_mask].reset_index(drop=True)
    validation = shuffled[validation_mask].reset_index(drop=True)
    test = shuffled[test_mask].reset_index(drop=True)
    return train, validation, test


def create_id_mappings(
    train_interactions: pd.DataFrame,
) -> tuple[dict[object, int], dict[object, int]]:
    """Map training users and items to consecutive embedding indices."""
    user_to_index = {
        user_id: index
        for index, user_id in enumerate(train_interactions[USER_COLUMN].unique())
    }
    item_to_index = {
        isbn: index
        for index, isbn in enumerate(train_interactions[ITEM_COLUMN].unique())
    }
    return user_to_index, item_to_index


def encode_interactions(
    interactions: pd.DataFrame,
    user_to_index: dict[object, int],
    item_to_index: dict[object, int],
) -> pd.DataFrame:
    """Encode known users/items and drop cold-start interactions."""
    encoded = pd.DataFrame(
        {
            "user": interactions[USER_COLUMN].map(user_to_index),
            "item": interactions[ITEM_COLUMN].map(item_to_index),
            "rating": interactions[RATING_COLUMN],
        }
    ).dropna()

    encoded["user"] = encoded["user"].astype("int64")
    encoded["item"] = encoded["item"].astype("int64")
    encoded["rating"] = encoded["rating"].astype("float32")
    return encoded.reset_index(drop=True)
