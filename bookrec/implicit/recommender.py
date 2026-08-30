"""High-level serving interface for history-based book recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from bookrec.catalog import BookCatalog
from bookrec.implicit.inference import (
    LoadedHistoryMLP,
    load_history_mlp,
    predict_history_probabilities,
)


class HistoryMLPRecommender:
    """Load a history MLP once and serve ISBN- or title-based rankings.

    All long-lived state is created in ``__init__``: model members, mappings,
    metadata, and the eligible candidate-index tensor. Query histories and
    scores remain local to each method call, so one request cannot contaminate
    another request's history.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        books: pd.DataFrame | BookCatalog | str | Path,
        *,
        device: str | torch.device | None = None,
        min_candidate_interactions: int = 5,
        min_query_interactions: int = 20,
        inference_batch_size: int = 8_192,
    ) -> None:
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        if min_query_interactions < 0:
            raise ValueError("min_query_interactions must be non-negative")

        loaded = load_history_mlp(artifact_path, device=device)
        catalog = self._create_catalog(books, loaded)
        if not catalog.matches_item_mapping(loaded.item_to_index):
            raise ValueError(
                "BookCatalog item mapping does not match the loaded model"
            )

        candidate_indices = catalog.eligible_candidate_indices(
            min_candidate_interactions
        )
        if candidate_indices.numel() == 0:
            raise ValueError(
                "No recommendation candidates have metadata and meet the "
                "minimum training-interaction threshold"
            )

        self.loaded = loaded
        self.catalog = catalog
        self._candidate_indices = candidate_indices
        self.min_query_interactions = min_query_interactions
        self.inference_batch_size = inference_batch_size

    @property
    def is_ensemble(self) -> bool:
        return self.loaded.is_ensemble

    @property
    def member_count(self) -> int:
        return len(self.loaded.member_paths)

    @property
    def device(self) -> torch.device:
        return self.loaded.device

    @property
    def candidate_count(self) -> int:
        return len(self._candidate_indices)

    def recommend_by_isbn(
        self,
        history_isbns: str | Sequence[str],
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        """Recommend books from one or more exact ISBN editions.

        ISBNs may contain spaces or hyphens. Repeated ISBNs are removed to
        preserve the model's binary-history semantics. Every input ISBN is
        excluded from the returned candidates.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if isinstance(history_isbns, str):
            requested_isbns = [history_isbns]
        else:
            requested_isbns = list(history_isbns)
        if not requested_isbns:
            raise ValueError("history_isbns must contain at least one ISBN")

        history_indices = sorted(
            {self.catalog.item_index(isbn) for isbn in requested_isbns}
        )
        history_tensor = torch.tensor(history_indices, dtype=torch.long)
        candidates = self._candidate_indices[
            ~torch.isin(self._candidate_indices, history_tensor)
        ]
        if candidates.numel() == 0:
            return []

        scores = predict_history_probabilities(
            self.loaded.model,
            history_items=history_indices,
            candidate_items=candidates,
            device=self.loaded.device,
            batch_size=self.inference_batch_size,
        )
        result_count = min(top_k, len(candidates))
        top_positions = torch.topk(scores, k=result_count).indices

        recommendations = []
        for rank, position in enumerate(top_positions.tolist(), start=1):
            item_index = int(candidates[position])
            record = self.catalog.get_by_item_index(item_index)
            recommendation = record.to_dict()
            recommendation["rank"] = rank
            recommendation["score"] = float(scores[position])
            recommendations.append(recommendation)
        return recommendations

    def recommend_by_title(
        self,
        history_titles: str | Sequence[str],
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        """Resolve one or more supported titles, then recommend by ISBN."""
        if isinstance(history_titles, str):
            requested_titles = [history_titles]
        else:
            requested_titles = list(history_titles)
        if not requested_titles:
            raise ValueError("history_titles must contain at least one title")

        resolved = [
            self.catalog.resolve_title(
                title,
                min_training_interactions=self.min_query_interactions,
            )
            for title in requested_titles
        ]
        return self.recommend_by_isbn(
            [record.isbn for record in resolved],
            top_k=top_k,
        )

    def search_titles(
        self,
        query: str,
        limit: int = 10,
        min_training_interactions: int = 1,
    ) -> list[dict[str, object]]:
        """Return supported catalog matches suitable for an API search box."""
        records = self.catalog.search_titles(
            query,
            limit=limit,
            min_training_interactions=min_training_interactions,
        )
        return [record.to_dict() for record in records]

    @staticmethod
    def _create_catalog(
        books: pd.DataFrame | BookCatalog | str | Path,
        loaded: LoadedHistoryMLP,
    ) -> BookCatalog:
        if isinstance(books, BookCatalog):
            return books
        if isinstance(books, pd.DataFrame):
            return BookCatalog(
                books,
                loaded.item_to_index,
                loaded.training_item_counts,
            )
        return BookCatalog.from_csv(
            books,
            loaded.item_to_index,
            loaded.training_item_counts,
        )
