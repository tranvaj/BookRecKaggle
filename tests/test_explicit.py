import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bookrec.explicit.baselines import MeanBaselines, evaluate_mean_baselines
from bookrec.explicit.datasets import ExplicitDataset
from bookrec.explicit.metrics import mae, rmse
from bookrec.explicit.model import ExplicitRecommenderMLP
from bookrec.training import train_loop


class ExplicitTests(unittest.TestCase):
    def setUp(self):
        self.encoded = pd.DataFrame(
            {
                "user": [0, 0, 1, 1],
                "item": [0, 1, 0, 1],
                "rating": [4.0, 6.0, 5.0, 7.0],
            }
        )
        self.raw = pd.DataFrame(
            {
                "User-ID": [10, 10, 20, 20],
                "ISBN": ["a", "b", "a", "b"],
                "Book-Rating": [4.0, 6.0, 5.0, 7.0],
            }
        )

    def test_dataset_returns_float_rating_label(self):
        sample = ExplicitDataset(self.encoded)[0]
        self.assertEqual(sample["label"].dtype, torch.float32)

    def test_model_trains_for_one_epoch(self):
        loader = DataLoader(ExplicitDataset(self.encoded), batch_size=2)
        model = ExplicitRecommenderMLP(2, 2, global_mean=5.5, embedding_dim=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = train_loop(
            loader,
            model,
            torch.nn.MSELoss(),
            optimizer,
            torch.device("cpu"),
        )

        self.assertTrue(all(np.isfinite(loss) for loss in losses))

    def test_metrics_and_mean_baselines(self):
        labels = np.array([1.0, 3.0])
        predictions = np.array([2.0, 3.0])
        self.assertAlmostEqual(rmse(labels, predictions), np.sqrt(0.5))
        self.assertAlmostEqual(mae(labels, predictions), 0.5)

        baselines = MeanBaselines.fit(self.raw)
        self.assertEqual(baselines.global_mean, 5.5)
        report = evaluate_mean_baselines(self.raw, self.raw)
        self.assertEqual(
            set(report.index),
            {"global_mean", "user_mean", "item_mean"},
        )


if __name__ == "__main__":
    unittest.main()
