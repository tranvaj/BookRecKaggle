import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from baselines import MeanBaselines, evaluate_mean_baselines
from bookrec_dataset import BookRecDataset
from data_preprocessing import (
    build_id_mappings,
    encode_interactions,
    filter_known_interactions,
    get_splits,
    prepare_explicit_ratings,
)
from model import RecommenderMLP
from train import evaluate, train_one_epoch


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.ratings = pd.DataFrame(
            [
                {"User-ID": user, "ISBN": f"book-{item}", "Book-Rating": item + 1}
                for user in (10, 20)
                for item in range(5)
            ]
        )

    def test_prepare_explicit_ratings_removes_zero_and_invalid_ratings(self):
        raw = pd.DataFrame(
            {
                "User-ID": [1, 1, 1, 1],
                "ISBN": ["a", "b", "c", "d"],
                "Book-Rating": [0, 1, 10, 11],
            }
        )
        prepared = prepare_explicit_ratings(raw)
        self.assertEqual(prepared["Book-Rating"].tolist(), [1, 10])

    def test_default_split_keeps_training_examples_for_each_user(self):
        train, validation, test = get_splits(self.ratings)
        self.assertEqual((len(train), len(validation), len(test)), (6, 2, 2))
        self.assertEqual(set(train["User-ID"]), {10, 20})

    def test_encoding_reuses_training_mappings_and_drops_unknown_items(self):
        train = self.ratings.iloc[:5]
        mappings = build_id_mappings(train)
        evaluation = pd.DataFrame(
            {
                "User-ID": [10, 10],
                "ISBN": ["book-0", "unseen"],
                "Book-Rating": [5, 5],
            }
        )
        encoded = encode_interactions(evaluation, mappings)
        self.assertEqual(len(encoded), 1)
        self.assertEqual(encoded.loc[0, "item"], mappings.items["book-0"])

    def test_known_interactions_support_fair_model_comparison(self):
        train = self.ratings.iloc[:5]
        mappings = build_id_mappings(train)
        evaluation = pd.DataFrame(
            {
                "User-ID": [10, 10, 999],
                "ISBN": ["book-0", "unseen", "book-0"],
                "Book-Rating": [5, 5, 5],
            }
        )
        known = filter_known_interactions(evaluation, mappings)
        self.assertEqual(len(known), 1)
        self.assertEqual(known.loc[0, "ISBN"], "book-0")

    def test_dataset_uses_positional_storage_and_float_ratings(self):
        encoded = pd.DataFrame(
            {"user": [0], "item": [1], "rating": [7.0]},
            index=[99],
        )
        sample = BookRecDataset(encoded)[0]
        self.assertEqual(sample["users"].item(), 0)
        self.assertEqual(sample["ratings"].dtype, torch.float32)

    def test_user_mean_falls_back_to_global_mean(self):
        baselines = MeanBaselines.fit(self.ratings)
        evaluation = pd.DataFrame(
            {
                "User-ID": [10, 999],
                "ISBN": ["book-0", "unknown"],
                "Book-Rating": [1, 1],
            }
        )
        predictions = baselines.predict_user(evaluation)
        self.assertAlmostEqual(predictions[0], 3.0)
        self.assertAlmostEqual(predictions[1], baselines.global_mean)

    def test_baseline_report_contains_requested_unregularized_models(self):
        report = evaluate_mean_baselines(self.ratings, self.ratings)
        self.assertEqual(
            set(report.index),
            {"global_mean", "user_mean", "item_mean"},
        )
        self.assertTrue(np.isfinite(report[["rmse", "mae"]].to_numpy()).all())

    def test_model_trains_and_evaluates_one_batch(self):
        encoded = pd.DataFrame(
            {
                "user": [0, 0, 1, 1],
                "item": [0, 1, 0, 1],
                "rating": [4.0, 6.0, 5.0, 7.0],
            }
        )
        loader = DataLoader(BookRecDataset(encoded), batch_size=2)
        model = RecommenderMLP(2, 2, global_mean=5.5, embedding_dim=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_rmse = train_one_epoch(model, loader, optimizer, torch.device("cpu"))
        metrics = evaluate(model, loader, torch.device("cpu"))

        self.assertTrue(np.isfinite(train_rmse))
        self.assertEqual(set(metrics), {"rmse", "mae"})


if __name__ == "__main__":
    unittest.main()
