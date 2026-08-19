import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from scipy.sparse import csr_matrix

def load_ds(file_path: str = "Ratings.csv", url_path: str = "arashnic/book-recommendation-dataset"):
    file_path = "Ratings.csv"
    df: pd.DataFrame = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        url_path,
        file_path,
    )
    return df


def get_splits(df: pd.DataFrame, val_frac=0.1, test_frac=0.1):
    shuffled = (
    df.groupby("User-ID")
      .sample(frac=1, random_state=42)
    )
    # Position of each rating within its user
    pos = shuffled.groupby("User-ID").cumcount()

    # Number of ratings each user has
    n = shuffled.groupby("User-ID")["User-ID"].transform("size")

    # How many ratings to allocate
    n_val = (n * val_frac).astype(int).clip(lower=1)
    n_test = (n * test_frac).astype(int).clip(lower=1)

    # Only split users with enough ratings
    eligible = n >= 5

    test_mask = eligible & (pos < n_test)

    val_mask = (
        eligible
        & (pos >= n_test)
        & (pos < n_test + n_val)
    )

    train_mask = ~(test_mask | val_mask)

    train_df = shuffled[train_mask]
    val_df = shuffled[val_mask]
    test_df = shuffled[test_mask]
    return train_df, val_df, test_df

def convert_df_to_csr(df: pd.DataFrame):
    user_codes, user_ids = pd.factorize(df["User-ID"])
    item_codes, item_ids = pd.factorize(df["ISBN"])

    expl_matrix = csr_matrix(
        (
            df["Book-Rating"],
            (user_codes, item_codes)
        ),
        shape=(len(user_ids), len(item_ids))
    )
    return expl_matrix