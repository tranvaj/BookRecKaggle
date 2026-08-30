import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from bookrec.catalog import BookCatalog, normalize_isbn
from bookrec.implicit.model import (
    HISTORY_MLP_ARCHITECTURE_VERSION,
    ImplicitHistoryMLP,
)
from bookrec.implicit.recommender import HistoryMLPRecommender


class BookCatalogTests(unittest.TestCase):
    def setUp(self):
        self.item_to_index = {
            "0000000001": 0,
            "0000000002": 1,
            "0000000003": 2,
            "0000000004": 3,
        }
        self.books = pd.DataFrame(
            {
                "ISBN": [
                    "0-000-00000-1",
                    "0000000002",
                    "0000000003",
                    "0000000004",
                    "not-in-model",
                ],
                "Book-Title": [
                    "Example Book",
                    "Example Book",
                    "Example Book (Movie Cover)",
                    "Another Book",
                    "Unknown Book",
                ],
                "Book-Author": ["A", "A", "A", "B", "C"],
            }
        )
        self.catalog = BookCatalog(
            self.books,
            self.item_to_index,
            training_item_counts=torch.tensor([3, 12, 20, 8]),
        )

    def test_isbn_normalization_and_model_lookup(self):
        self.assertEqual(normalize_isbn(" 0-000-00000-x "), "000000000X")
        self.assertEqual(self.catalog.item_index("0-000-00000-1"), 0)
        self.assertEqual(self.catalog.get_by_isbn("0000000001").title, "Example Book")
        with self.assertRaisesRegex(KeyError, "not supported"):
            self.catalog.item_index("9999999999")

    def test_duplicate_normalized_isbn_uses_most_supported_index(self):
        catalog = BookCatalog(
            pd.DataFrame(
                {
                    "ISBN": ["002542730X"],
                    "Book-Title": ["Politically Correct Bedtime Stories"],
                }
            ),
            {"002542730X": 0, "002542730x": 1},
            training_item_counts=torch.tensor([2, 7]),
        )

        self.assertEqual(catalog.item_index("002542730x"), 1)
        self.assertEqual(catalog.eligible_candidate_indices(1).tolist(), [1])

    def test_title_resolution_prefers_supported_exact_edition(self):
        resolved = self.catalog.resolve_title(
            " example   BOOK ",
            min_training_interactions=5,
        )
        variant = self.catalog.resolve_title(
            "Example Book (Movie",
            min_training_interactions=5,
        )

        self.assertEqual(resolved.isbn, "0000000002")
        self.assertEqual(variant.isbn, "0000000003")

    def test_candidates_require_metadata_and_training_support(self):
        candidates = self.catalog.eligible_candidate_indices(
            min_training_interactions=10
        )

        self.assertEqual(candidates.tolist(), [1, 2])
        self.assertEqual(self.catalog.model_item_count, 4)
        self.assertEqual(self.catalog.metadata_item_count, 4)

    def test_search_prioritizes_exact_title_then_support(self):
        results = self.catalog.search_titles(
            "Example Book",
            min_training_interactions=1,
        )

        self.assertEqual([result.item_index for result in results], [1, 0, 2])


class HistoryMLPRecommenderTests(unittest.TestCase):
    def _save_checkpoint(self, directory: Path, output_bias: float = 0.0) -> None:
        hyperparameters = {
            "embedding_dim": 2,
            "hidden_dims": (),
            "dropout": 0.0,
        }
        model = ImplicitHistoryMLP(
            num_items=5,
            embedding_dim=2,
            hidden_dims=(),
            dropout=0.0,
        )
        with torch.no_grad():
            model.item_embedding.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                        [1.0, 1.0],
                    ]
                )
            )
            model.mlp[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.mlp[0].bias.fill_(output_bias)

        torch.save(
            {
                "model_type": "history_mlp",
                "architecture_version": HISTORY_MLP_ARCHITECTURE_VERSION,
                "model_state_dict": model.state_dict(),
                "user_to_index": {"user": 0},
                "item_to_index": {
                    "0000000001": 0,
                    "0000000002": 1,
                    "0000000003": 2,
                    "0000000004": 3,
                    "0000000005": 4,
                },
                "training_item_counts": torch.tensor([20, 15, 10, 8, 12]),
                "hyperparameters": hyperparameters,
            },
            directory / "model_with_mappings.pt",
        )

    @staticmethod
    def _books() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ISBN": [
                    "0000000001",
                    "0000000002",
                    "0000000003",
                    "0000000004",
                    "0000000005",
                ],
                "Book-Title": [
                    "Query",
                    "Best Match",
                    "Orthogonal",
                    "Rare Match",
                    "Partial Match",
                ],
                "Book-Author": ["A", "B", "C", "D", "E"],
                "Publisher": ["P"] * 5,
            }
        )

    def test_recommender_ranks_metadata_and_excludes_history(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_directory = Path(temporary_directory)
            self._save_checkpoint(artifact_directory)
            recommender = HistoryMLPRecommender(
                artifact_directory,
                self._books(),
                device="cpu",
                min_candidate_interactions=5,
            )

            recommendations = recommender.recommend_by_isbn(
                ["0-000-00000-1", "0000000001"],
                top_k=3,
            )

        self.assertEqual(
            [recommendation["isbn"] for recommendation in recommendations],
            ["0000000002", "0000000005", "0000000003"],
        )
        self.assertEqual(
            [recommendation["rank"] for recommendation in recommendations],
            [1, 2, 3],
        )
        self.assertNotIn("item_index", recommendations[0])
        self.assertGreater(recommendations[0]["score"], recommendations[1]["score"])
        self.assertFalse(recommender.is_ensemble)
        self.assertEqual(recommender.member_count, 1)
        self.assertEqual(recommender.candidate_count, 5)

    def test_recommender_validates_queries_and_searches_titles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_directory = Path(temporary_directory)
            self._save_checkpoint(artifact_directory)
            recommender = HistoryMLPRecommender(
                artifact_directory,
                self._books(),
                device="cpu",
                min_candidate_interactions=5,
                min_query_interactions=10,
            )

            search_results = recommender.search_titles("match")
            title_recommendations = recommender.recommend_by_title(
                "Query",
                top_k=1,
            )

            with self.assertRaisesRegex(KeyError, "not supported"):
                recommender.recommend_by_isbn("9999999999")
            with self.assertRaisesRegex(ValueError, "at least one"):
                recommender.recommend_by_isbn([])

        self.assertEqual(
            [result["title"] for result in search_results],
            ["Best Match", "Partial Match", "Rare Match"],
        )
        self.assertEqual(title_recommendations[0]["isbn"], "0000000002")

    def test_recommender_loads_an_ensemble_once(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            ensemble_directory = Path(temporary_directory)
            for seed, output_bias in ((100, 0.0), (110, 0.5)):
                member_directory = ensemble_directory / f"seed_{seed}"
                member_directory.mkdir()
                self._save_checkpoint(member_directory, output_bias)

            recommender = HistoryMLPRecommender(
                ensemble_directory,
                self._books(),
                device="cpu",
                min_candidate_interactions=5,
            )
            recommendations = recommender.recommend_by_isbn(
                "0000000001",
                top_k=1,
            )

        self.assertTrue(recommender.is_ensemble)
        self.assertEqual(recommender.member_count, 2)
        self.assertEqual(recommendations[0]["isbn"], "0000000002")


if __name__ == "__main__":
    unittest.main()
