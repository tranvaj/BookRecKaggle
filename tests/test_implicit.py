import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bookrec.implicit.datasets import ImplicitDataset, SampledRankingDataset
from bookrec.implicit.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from bookrec.implicit.model import ImplicitRecommenderMLP
from bookrec.training import train_loop


class ImplicitTests(unittest.TestCase):
    def setUp(self):
        self.all_interactions = pd.DataFrame(
            {
                "user": [0, 0, 0, 1, 1],
                "item": [0, 1, 2, 3, 4],
                "rating": [0, 8, 0, 5, 0],
            }
        )

    def test_training_dataset_returns_one_positive_and_four_negatives(self):
        dataset = ImplicitDataset(
            self.all_interactions,
            self.all_interactions,
            num_items=20,
            negatives_per_positive=4,
        )

        sample = dataset[0]

        self.assertEqual(sample["users"].shape, (5,))
        self.assertEqual(sample["label"].sum().item(), 1.0)
        negative_items = sample["items"][sample["label"] == 0].tolist()
        self.assertTrue(set(negative_items).isdisjoint({0, 1, 2}))

    def test_ranking_dataset_contains_all_held_out_positives_per_user(self):
        held_out = pd.DataFrame(
            {"user": [0, 0, 1], "item": [1, 2, 3]}
        )
        dataset = SampledRankingDataset(
            held_out,
            self.all_interactions,
            num_items=20,
            num_candidates=10,
            seed=42,
        )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["label"].sum().item(), 2.0)
        self.assertEqual(dataset[1]["label"].sum().item(), 1.0)
        self.assertEqual(dataset[0]["items"].unique().numel(), 10)

    def test_ranking_metrics_average_across_users(self):
        labels = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
        logits = np.array([[3, 2, 1], [2, 3, 1]], dtype=np.float32)

        self.assertEqual(recall_at_k(labels, logits, k=1), 0.5)
        self.assertAlmostEqual(mean_reciprocal_rank(labels, logits), 0.75)
        self.assertAlmostEqual(ndcg_at_k(labels, logits, k=3), 0.8154648768)

    def test_model_trains_on_grouped_candidates(self):
        dataset = ImplicitDataset(
            self.all_interactions,
            self.all_interactions,
            num_items=20,
            negatives_per_positive=2,
        )
        loader = DataLoader(dataset, batch_size=2)
        model = ImplicitRecommenderMLP(2, 20, embedding_dim=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = train_loop(
            loader,
            model,
            torch.nn.BCEWithLogitsLoss(),
            optimizer,
            torch.device("cpu"),
        )

        self.assertTrue(all(np.isfinite(loss) for loss in losses))


if __name__ == "__main__":
    unittest.main()
