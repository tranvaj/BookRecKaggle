import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bookrec.ensemble import default_seeds, resolve_seeds
from bookrec.implicit.baselines import (
    ALSBaseline,
    MostPopularBaseline,
    RandomBaseline,
)
from bookrec.implicit.datasets import (
    ImplicitDataset,
    SampledRankingDataset,
    collate_implicit_batch,
)
from bookrec.implicit.hyperparameters import load_hyperparameters
from bookrec.implicit.inference import (
    load_history_mlp,
    predict_history_probabilities,
)
from bookrec.implicit.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from bookrec.implicit.model import (
    HISTORY_MLP_ARCHITECTURE_VERSION,
    MODEL_REGISTRY,
    ImplicitHistoryMLP,
    ImplicitProbabilityDeepEnsemble,
    ImplicitRecommenderMLP,
    create_implicit_model,
)
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
        positive_item = sample["items"][sample["label"] == 1].item()
        self.assertNotIn(positive_item, sample["history_items"].tolist())
        negative_items = sample["items"][sample["label"] == 0].tolist()
        self.assertTrue(set(negative_items).isdisjoint({0, 1, 2}))
        self.assertTrue(
            set(negative_items).isdisjoint(sample["history_items"].tolist())
        )

    def test_history_training_excludes_empty_context_examples(self):
        interactions = pd.DataFrame(
            {"user": [0, 1, 1], "item": [0, 1, 2]}
        )

        dataset = ImplicitDataset(
            interactions,
            interactions,
            num_items=5,
            negatives_per_positive=2,
            require_nonempty_history=True,
        )

        self.assertEqual(len(dataset), 2)
        self.assertTrue(
            all(len(dataset[index]["history_items"]) for index in range(2))
        )

    def test_mixed_history_training_samples_singleton_and_full_contexts(self):
        interactions = pd.DataFrame(
            {"user": [0, 0, 0, 0], "item": [0, 1, 2, 3]}
        )
        dataset = ImplicitDataset(
            interactions,
            interactions,
            num_items=10,
            negatives_per_positive=1,
            require_nonempty_history=True,
            history_mode="mixed",
            singleton_probability=0.5,
        )
        torch.manual_seed(7)

        history_lengths = {
            len(dataset[0]["history_items"])
            for _ in range(100)
        }

        self.assertEqual(history_lengths, {1, 3})

    def test_ranking_dataset_contains_all_held_out_positives_per_user(self):
        held_out = pd.DataFrame(
            {"user": [0, 0, 1], "item": [1, 2, 3]}
        )
        dataset = SampledRankingDataset(
            held_out,
            context_interactions=self.all_interactions,
            all_interactions=self.all_interactions,
            num_items=20,
            num_candidates=10,
            seed=42,
        )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["label"].sum().item(), 2.0)
        self.assertEqual(dataset[1]["label"].sum().item(), 1.0)
        self.assertEqual(dataset[0]["items"].unique().numel(), 10)

    def test_singleton_and_full_history_use_identical_rankings(self):
        context = pd.DataFrame(
            {"user": [0, 0, 0, 1, 1], "item": [0, 1, 4, 2, 3]}
        )
        held_out = pd.DataFrame({"user": [0, 1], "item": [5, 6]})
        all_interactions = pd.concat([context, held_out], ignore_index=True)
        common_arguments = {
            "held_out_interactions": held_out,
            "context_interactions": context,
            "all_interactions": all_interactions,
            "num_items": 20,
            "num_candidates": 10,
            "seed": 17,
        }

        singleton = SampledRankingDataset(
            **common_arguments,
            context_mode="singleton",
        )
        full = SampledRankingDataset(
            **common_arguments,
            context_mode="full",
        )

        for index in range(len(singleton)):
            singleton_sample = singleton[index]
            full_sample = full[index]
            self.assertTrue(
                torch.equal(singleton_sample["items"], full_sample["items"])
            )
            self.assertTrue(
                torch.equal(singleton_sample["label"], full_sample["label"])
            )
            self.assertEqual(len(singleton_sample["history_items"]), 1)
            held_out_items = set(
                held_out.loc[
                    held_out["user"] == singleton.users[index],
                    "item",
                ]
            )
            self.assertTrue(
                held_out_items.isdisjoint(full_sample["history_items"].tolist())
            )
            negative_items = full_sample["items"][
                full_sample["label"] == 0
            ].tolist()
            self.assertTrue(
                set(negative_items).isdisjoint(
                    full_sample["history_items"].tolist()
                )
            )

    def test_collator_flattens_histories_and_builds_offsets(self):
        samples = [
            {
                "users": torch.tensor([0, 0]),
                "items": torch.tensor([1, 2]),
                "label": torch.tensor([1.0, 0.0]),
                "history_items": torch.tensor([3, 4]),
            },
            {
                "users": torch.tensor([1, 1]),
                "items": torch.tensor([2, 3]),
                "label": torch.tensor([0.0, 1.0]),
                "history_items": torch.tensor([5]),
            },
        ]

        batch = collate_implicit_batch(samples)

        self.assertTrue(
            torch.equal(batch["history_items"], torch.tensor([3, 4, 5]))
        )
        self.assertTrue(
            torch.equal(batch["history_offset"], torch.tensor([0, 2]))
        )
        self.assertEqual(batch["items"].shape, (2, 2))

    def test_ranking_metrics_average_across_users(self):
        labels = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
        logits = np.array([[3, 2, 1], [2, 3, 1]], dtype=np.float32)

        self.assertEqual(recall_at_k(labels, logits, k=1), 0.5)
        self.assertAlmostEqual(mean_reciprocal_rank(labels, logits), 0.75)
        self.assertAlmostEqual(ndcg_at_k(labels, logits, k=3), 0.8154648768)

    def test_model_trains_on_grouped_candidates(self):
        hyperparameters = {
            "embedding_dim": 4,
            "hidden_dims": (8, 4),
            "dropout": 0.0,
        }

        for model_name in MODEL_REGISTRY:
            with self.subTest(model=model_name):
                dataset = ImplicitDataset(
                    self.all_interactions,
                    self.all_interactions,
                    num_items=20,
                    negatives_per_positive=2,
                    require_nonempty_history=model_name == "history_mlp",
                )
                loader = DataLoader(
                    dataset,
                    batch_size=2,
                    collate_fn=collate_implicit_batch,
                )
                model = create_implicit_model(
                    model_name,
                    num_users=2,
                    num_items=20,
                    hyperparameters=hyperparameters,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                losses = train_loop(
                    loader,
                    model,
                    torch.nn.BCEWithLogitsLoss(),
                    optimizer,
                    torch.device("cpu"),
                )

                self.assertTrue(all(np.isfinite(loss) for loss in losses))

    def test_sparse_history_sum_matches_dense_binary_projection(self):
        model = ImplicitHistoryMLP(
            num_items=6,
            embedding_dim=3,
            hidden_dims=(4,),
            dropout=0.0,
        )
        history_items = torch.tensor([0, 2, 3])
        history_offset = torch.tensor([0, 2])
        dense_histories = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        sparse_projection = model.item_embedding(
            history_items,
            history_offset,
        )
        dense_projection = dense_histories @ model.item_embedding.weight

        self.assertTrue(torch.allclose(sparse_projection, dense_projection))

    def test_history_model_uses_history_and_items_but_not_users(self):
        model = ImplicitHistoryMLP(
            num_items=5,
            embedding_dim=2,
            hidden_dims=(),
            dropout=0.0,
        )
        with torch.no_grad():
            model.item_embedding.weight.zero_()
            model.item_embedding.weight[0] = torch.tensor([1.0, 0.0])
            model.item_embedding.weight[1] = torch.tensor([0.0, 1.0])
            model.item_embedding.weight[2] = torch.tensor([1.0, 0.0])
            model.item_embedding.weight[3] = torch.tensor([0.0, 1.0])
            model.mlp[0].weight.fill_(1.0)
            model.mlp[0].bias.zero_()

        candidates = torch.tensor([[2, 3], [2, 3]])
        histories = torch.tensor([0, 1])
        offsets = torch.tensor([0, 1])
        first_users = torch.tensor([[0, 0], [1, 1]])
        other_users = torch.tensor([[4, 4], [3, 3]])

        scores = model(first_users, candidates, histories, offsets)
        other_user_scores = model(other_users, candidates, histories, offsets)

        self.assertEqual(scores.shape, candidates.shape)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.equal(scores, other_user_scores))
        self.assertNotEqual(scores[0, 0].item(), scores[1, 0].item())
        self.assertNotEqual(scores[0, 0].item(), scores[0, 1].item())
        self.assertEqual(model.mlp[0].in_features, 2)
        item_embedding_parameters = [
            name
            for name, _ in model.named_parameters()
            if name.endswith("item_embedding.weight")
        ]
        self.assertEqual(item_embedding_parameters, ["item_embedding.weight"])

    def test_most_popular_uses_training_interaction_counts(self):
        baseline = MostPopularBaseline.fit(
            self.all_interactions,
            num_items=6,
        )
        scores = baseline(
            users=torch.tensor([0, 0, 0]),
            items=torch.tensor([0, 3, 5]),
            history_items=torch.tensor([1]),
            history_offset=torch.tensor([0]),
        )

        self.assertEqual(scores[0].item(), scores[1].item())
        self.assertGreater(scores[0].item(), scores[2].item())

    def test_random_baseline_returns_scores_between_zero_and_one(self):
        users = torch.tensor([[0, 0, 0], [1, 1, 1]])
        items = torch.tensor([[2, 3, 4], [2, 3, 4]])

        history_arguments = {
            "history_items": torch.tensor([0, 1]),
            "history_offset": torch.tensor([0, 1]),
        }
        scores = RandomBaseline()(users, items, **history_arguments)

        self.assertEqual(scores.shape, items.shape)
        self.assertTrue(torch.all((scores >= 0) & (scores < 1)))

    def test_als_baseline_scores_grouped_candidates(self):
        baseline = ALSBaseline.fit(
            self.all_interactions,
            num_users=2,
            num_items=6,
            factors=2,
            iterations=2,
        )
        users = torch.tensor([[0, 0], [1, 1]])
        items = torch.tensor([[0, 3], [1, 4]])

        scores = baseline(
            users,
            items,
            history_items=torch.tensor([0, 1]),
            history_offset=torch.tensor([0, 1]),
        )

        self.assertEqual(scores.shape, items.shape)
        self.assertTrue(torch.isfinite(scores).all())

    def test_deep_ensemble_averages_member_probabilities(self):
        members = [
            ImplicitRecommenderMLP(
                num_users=2,
                num_items=4,
                embedding_dim=2,
                hidden_dims=(4,),
                dropout=0.0,
            )
            for _ in range(3)
        ]
        model = ImplicitProbabilityDeepEnsemble(members)
        users = torch.tensor([[0, 0, 0], [1, 1, 1]])
        items = torch.tensor([[0, 1, 2], [1, 2, 3]])

        scores = model(users, items)
        expected = torch.stack(
            [torch.sigmoid(member(users, items)) for member in members]
        ).mean(dim=0)

        self.assertEqual(scores.shape, items.shape)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.all((scores >= 0) & (scores <= 1)))
        self.assertTrue(torch.allclose(scores, expected))

    def test_deep_ensemble_requires_at_least_two_members(self):
        member = ImplicitRecommenderMLP(
            num_users=2,
            num_items=4,
            embedding_dim=2,
            hidden_dims=(4,),
            dropout=0.0,
        )

        with self.assertRaisesRegex(ValueError, "at least two members"):
            ImplicitProbabilityDeepEnsemble([member])

    def test_history_ensemble_averages_member_probabilities(self):
        members = [
            ImplicitHistoryMLP(
                num_items=5,
                embedding_dim=2,
                hidden_dims=(4,),
                dropout=0.0,
            )
            for _ in range(2)
        ]
        ensemble = ImplicitProbabilityDeepEnsemble(members)
        arguments = {
            "users": torch.tensor([[0, 0], [1, 1]]),
            "items": torch.tensor([[2, 3], [3, 4]]),
            "history_items": torch.tensor([0, 1, 2]),
            "history_offset": torch.tensor([0, 2]),
        }

        scores = ensemble(**arguments)
        expected = torch.stack(
            [torch.sigmoid(member(**arguments)) for member in members]
        ).mean(dim=0)

        self.assertTrue(torch.allclose(scores, expected))

    def test_history_inference_loads_and_scores_single_model(self):
        hyperparameters = {
            "embedding_dim": 2,
            "hidden_dims": (),
            "dropout": 0.0,
        }
        model = ImplicitHistoryMLP(
            num_items=3,
            embedding_dim=2,
            hidden_dims=(),
            dropout=0.0,
        )
        with torch.no_grad():
            model.item_embedding.weight.copy_(
                torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            )
            model.mlp[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.mlp[0].bias.zero_()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model_with_mappings.pt"
            torch.save(
                {
                    "model_type": "history_mlp",
                    "architecture_version": HISTORY_MLP_ARCHITECTURE_VERSION,
                    "model_state_dict": model.state_dict(),
                    "user_to_index": {"user": 0},
                    "item_to_index": {"a": 0, "b": 1, "c": 2},
                    "training_item_counts": torch.tensor([3, 2, 1]),
                    "hyperparameters": hyperparameters,
                },
                checkpoint_path,
            )

            loaded = load_history_mlp(Path(directory), device="cpu")
            probabilities = predict_history_probabilities(
                loaded.model,
                history_items=[0, 0],
                candidate_items=[1, 2],
                batch_size=1,
            )

        expected = torch.sigmoid(torch.tensor([1.0, 0.0]))
        self.assertFalse(loaded.is_ensemble)
        self.assertEqual(loaded.index_to_item, ("a", "b", "c"))
        self.assertTrue(torch.allclose(probabilities, expected))

    def test_history_inference_loads_ensemble_and_averages_probabilities(self):
        hyperparameters = {
            "embedding_dim": 2,
            "hidden_dims": (),
            "dropout": 0.0,
        }
        item_to_index = {"a": 0, "b": 1, "c": 2}

        with tempfile.TemporaryDirectory() as directory:
            ensemble_directory = Path(directory)
            for seed, bias in ((100, 0.0), (110, 2.0)):
                model = ImplicitHistoryMLP(
                    num_items=3,
                    embedding_dim=2,
                    hidden_dims=(),
                    dropout=0.0,
                )
                with torch.no_grad():
                    model.mlp[0].weight.zero_()
                    model.mlp[0].bias.fill_(bias)
                member_directory = ensemble_directory / f"seed_{seed}"
                member_directory.mkdir()
                torch.save(
                    {
                        "model_type": "history_mlp",
                        "architecture_version": (
                            HISTORY_MLP_ARCHITECTURE_VERSION
                        ),
                        "model_state_dict": model.state_dict(),
                        "user_to_index": {"user": 0},
                        "item_to_index": item_to_index,
                        "training_item_counts": torch.tensor([3, 2, 1]),
                        "hyperparameters": hyperparameters,
                    },
                    member_directory / "model_with_mappings.pt",
                )

            loaded = load_history_mlp(ensemble_directory, device="cpu")
            probabilities = predict_history_probabilities(
                loaded.model,
                history_items=[0],
                candidate_items=[1, 2],
            )

        expected_probability = (
            torch.sigmoid(torch.tensor(0.0))
            + torch.sigmoid(torch.tensor(2.0))
        ) / 2
        self.assertTrue(loaded.is_ensemble)
        self.assertEqual(len(loaded.member_paths), 2)
        self.assertTrue(
            torch.allclose(
                probabilities,
                expected_probability.repeat(2),
            )
        )

    def test_ensemble_seed_defaults_and_validation(self):
        self.assertEqual(default_seeds(4), [100, 110, 120, 130])
        self.assertEqual(resolve_seeds(3, [7, 17, 42]), [7, 17, 42])

        with self.assertRaisesRegex(ValueError, "Expected 3 seeds"):
            resolve_seeds(3, [7, 17])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            resolve_seeds(3, [7, 7, 17])

    def test_saved_hyperparameters_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_hparams.json"
            path.write_text(
                json.dumps(
                    {
                        "embedding_dim": 64,
                        "hidden_dims": [128, 64],
                        "dropout": 0.3,
                        "learning_rate": 5e-4,
                        "weight_decay": 1e-6,
                    }
                )
            )

            hyperparameters = load_hyperparameters(path)

        self.assertEqual(hyperparameters["embedding_dim"], 64)
        self.assertEqual(hyperparameters["hidden_dims"], (128, 64))


if __name__ == "__main__":
    unittest.main()
