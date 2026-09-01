from dataclasses import dataclass

import numpy as np
import pandas as pd

from bookrec.data import ITEM_COLUMN, RATING_COLUMN, USER_COLUMN
from bookrec.explicit.metrics import mae, rmse


@dataclass(frozen=True)
class MeanBaselines:
    """Unregularized mean predictors fitted on training ratings."""

    global_mean: float
    user_means: pd.Series
    item_means: pd.Series

    @classmethod
    def fit(cls, train_interactions: pd.DataFrame) -> "MeanBaselines":
        return cls(
            global_mean=float(train_interactions[RATING_COLUMN].mean()),
            user_means=(
                train_interactions.groupby(USER_COLUMN)[RATING_COLUMN].mean()
            ),
            item_means=(
                train_interactions.groupby(ITEM_COLUMN)[RATING_COLUMN].mean()
            ),
        )

    def predict_global(self, interactions: pd.DataFrame) -> np.ndarray:
        return np.full(len(interactions), self.global_mean)

    def predict_user(self, interactions: pd.DataFrame) -> np.ndarray:
        return (
            interactions[USER_COLUMN]
            .map(self.user_means)
            .fillna(self.global_mean)
            .to_numpy()
        )

    def predict_item(self, interactions: pd.DataFrame) -> np.ndarray:
        return (
            interactions[ITEM_COLUMN]
            .map(self.item_means)
            .fillna(self.global_mean)
            .to_numpy()
        )


def evaluate_mean_baselines(
    train_interactions: pd.DataFrame,
    evaluation_interactions: pd.DataFrame,
) -> pd.DataFrame:
    baselines = MeanBaselines.fit(train_interactions)
    labels = evaluation_interactions[RATING_COLUMN].to_numpy()
    predictors = {
        "global_mean": baselines.predict_global,
        "user_mean": baselines.predict_user,
        "item_mean": baselines.predict_item,
    }
    return pd.DataFrame(
        [
            {
                "model": name,
                "rmse": rmse(labels, predict(evaluation_interactions)),
                "mae": mae(labels, predict(evaluation_interactions)),
            }
            for name, predict in predictors.items()
        ]
    ).set_index("model")
