import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bookrec.explicit.baselines import (
    MeanBaselines,
    evaluate_mean_baselines,
    fit_regularized_biases,
)
from bookrec.explicit.datasets import ExplicitDataset
from bookrec.explicit.hyperparameters import load_hyperparameters
from bookrec.explicit.metrics import mae, rmse
from bookrec.explicit.model import (
    ExplicitHybridRecommender,
    ExplicitRecommenderMLP,
)
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
        model = ExplicitRecommenderMLP(2, 2, embedding_dim=4)
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

    def test_regularized_biases_and_hybrid_predictions(self):
        global_mean, user_bias, item_bias = fit_regularized_biases(
            self.encoded,
            num_users=2,
            num_items=2,
            iterations=2,
        )
        mlp = ExplicitRecommenderMLP(
            num_users=2,
            num_items=2,
            embedding_dim=2,
            hidden_dims=(4,),
            dropout=0.0,
        )
        model = ExplicitHybridRecommender(
            global_mean,
            user_bias,
            item_bias,
            mlp,
        )

        predictions = model(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
        )

        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(torch.all((predictions >= 1) & (predictions <= 10)))

    def test_saved_hyperparameters_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_hparams.json"
            path.write_text(
                json.dumps(
                    {
                        "embedding_dim": 8,
                        "hidden_dims": [32, 16],
                        "learning_rate": 3e-4,
                    }
                )
            )
            hyperparameters = load_hyperparameters(path)

        self.assertEqual(hyperparameters["embedding_dim"], 8)
        self.assertEqual(hyperparameters["hidden_dims"], (32, 16))
        self.assertEqual(hyperparameters["learning_rate"], 3e-4)
        self.assertIn("dropout", hyperparameters)


if __name__ == "__main__":
    unittest.main()
