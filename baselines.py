from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_preprocessing import ITEM_COLUMN, RATING_COLUMN, USER_COLUMN


@dataclass(frozen=True)
class MeanBaselines:
    """Mean predictors fitted exclusively on training ratings."""

    global_mean: float
    user_means: pd.Series
    item_means: pd.Series

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "MeanBaselines":
        if train_df.empty:
            raise ValueError("Cannot fit baselines on an empty training set")

        return cls(
            global_mean=float(train_df[RATING_COLUMN].mean()),
            user_means=train_df.groupby(USER_COLUMN)[RATING_COLUMN].mean(),
            item_means=train_df.groupby(ITEM_COLUMN)[RATING_COLUMN].mean(),
        )

    def predict_global(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.global_mean, dtype=np.float64)

    def predict_user(self, df: pd.DataFrame) -> np.ndarray:
        return (
            df[USER_COLUMN]
            .map(self.user_means)
            .fillna(self.global_mean)
            .to_numpy(dtype=np.float64)
        )

    def predict_item(self, df: pd.DataFrame) -> np.ndarray:
        return (
            df[ITEM_COLUMN]
            .map(self.item_means)
            .fillna(self.global_mean)
            .to_numpy(dtype=np.float64)
        )


def regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("Actual and predicted ratings must have the same shape")
    if actual_array.size == 0:
        raise ValueError("Cannot evaluate an empty set")

    errors = predicted_array - actual_array
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
    }


def evaluate_mean_baselines(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate global-, user-, and item-mean predictions with RMSE and MAE."""
    baselines = MeanBaselines.fit(train_df)
    actual = eval_df[RATING_COLUMN]
    predictors = {
        "global_mean": baselines.predict_global,
        "user_mean": baselines.predict_user,
        "item_mean": baselines.predict_item,
    }
    rows = [
        {"model": name, **regression_metrics(actual, predict(eval_df))}
        for name, predict in predictors.items()
    ]
    return pd.DataFrame(rows).set_index("model")
