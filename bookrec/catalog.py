"""Book metadata and model-aware ISBN/title resolution."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import pandas as pd
import torch

from bookrec.data import ITEM_COLUMN


TITLE_COLUMN = "Book-Title"
AUTHOR_COLUMN = "Book-Author"
YEAR_COLUMN = "Year-Of-Publication"
PUBLISHER_COLUMN = "Publisher"
IMAGE_COLUMN = "Image-URL-M"
METADATA_COLUMNS = (
    ITEM_COLUMN,
    TITLE_COLUMN,
    AUTHOR_COLUMN,
    YEAR_COLUMN,
    PUBLISHER_COLUMN,
    IMAGE_COLUMN,
)


def normalize_isbn(isbn: object) -> str:
    """Return a comparison-safe ISBN while preserving ISBN-10 ``X`` digits."""
    if isbn is None or pd.isna(isbn):
        return ""
    return re.sub(r"[\s-]+", "", str(isbn)).upper()


def _normalize_title(title: object) -> str:
    if title is None or pd.isna(title):
        return ""
    return " ".join(str(title).split()).casefold()


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class CatalogBook:
    """Display metadata for one model-supported ISBN edition."""

    isbn: str
    title: str
    author: str | None
    year: str | None
    publisher: str | None
    image_url: str | None
    item_index: int
    training_interactions: int

    def to_dict(self) -> dict[str, object]:
        """Return the fields suitable for an API response."""
        return {
            "isbn": self.isbn,
            "title": self.title,
            "author": self.author,
            "year": self.year,
            "publisher": self.publisher,
            "image_url": self.image_url,
            "training_interactions": self.training_interactions,
        }


class BookCatalog:
    """Join book metadata to one trained model's encoded item catalog.

    The model mapping is part of the catalog because an ISBN present in
    ``Books.csv`` is not necessarily an item that the deployed model learned.
    Training counts allow title resolution and recommendation candidates to
    prefer editions with enough collaborative-filtering evidence.
    """

    def __init__(
        self,
        books: pd.DataFrame,
        item_to_index: Mapping[object, int],
        training_item_counts: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        missing_columns = {ITEM_COLUMN, TITLE_COLUMN} - set(books.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Book metadata is missing columns: {missing}")
        if not item_to_index:
            raise ValueError("item_to_index must not be empty")

        self._item_to_index = item_to_index
        seen_indices = bytearray(len(item_to_index))
        for item_index in item_to_index.values():
            if not isinstance(item_index, Integral):
                raise ValueError("Model item indices must be integers")
            item_index = int(item_index)
            if (
                not 0 <= item_index < len(item_to_index)
                or seen_indices[item_index]
            ):
                raise ValueError(
                    "Model item indices must be consecutive and unique"
                )
            seen_indices[item_index] = 1

        self._has_training_counts = training_item_counts is not None
        if training_item_counts is None:
            counts = torch.zeros(len(item_to_index), dtype=torch.long)
        else:
            counts = torch.as_tensor(training_item_counts).detach().cpu().flatten()
            if len(counts) != len(item_to_index):
                raise ValueError(
                    "training_item_counts must contain one value per model item"
                )
            if torch.any(counts < 0):
                raise ValueError("training_item_counts must be non-negative")
            counts = counts.to(torch.long)
        self._model_items: dict[str, tuple[object, int]] = {}
        for isbn, item_index in self._item_to_index.items():
            item_index = int(item_index)
            normalized_isbn = normalize_isbn(isbn)
            if not normalized_isbn:
                raise ValueError("Model item mapping contains an empty ISBN")
            existing = self._model_items.get(normalized_isbn)
            if existing is None or (
                int(counts[item_index]), -item_index
            ) > (
                int(counts[existing[1]]), -existing[1]
            ):
                self._model_items[normalized_isbn] = (isbn, item_index)

        metadata = books.reindex(columns=METADATA_COLUMNS).copy()
        metadata["_normalized_isbn"] = metadata[ITEM_COLUMN].map(normalize_isbn)
        metadata = metadata[
            metadata["_normalized_isbn"].isin(self._model_items)
        ]
        metadata = metadata.drop_duplicates("_normalized_isbn", keep="first")

        records: list[CatalogBook] = []
        for row in metadata.itertuples(index=False, name=None):
            (
                _,
                title_value,
                author,
                year,
                publisher,
                image_url,
                normalized_isbn,
            ) = row
            model_item = self._model_items[normalized_isbn]
            canonical_isbn, item_index = model_item
            title = _optional_text(title_value)
            if title is None:
                continue
            records.append(
                CatalogBook(
                    isbn=str(canonical_isbn),
                    title=title,
                    author=_optional_text(author),
                    year=_optional_text(year),
                    publisher=_optional_text(publisher),
                    image_url=_optional_text(image_url),
                    item_index=item_index,
                    training_interactions=int(counts[item_index]),
                )
            )

        self._records = tuple(records)
        self._records_by_item_index = {
            record.item_index: record for record in self._records
        }
        self._records_by_isbn = {
            normalize_isbn(record.isbn): record for record in self._records
        }
        records_by_title: dict[str, list[CatalogBook]] = {}
        records_with_titles = []
        for record in self._records:
            normalized_title = _normalize_title(record.title)
            records_with_titles.append((record, normalized_title))
            records_by_title.setdefault(normalized_title, []).append(record)
        self._records_with_titles = tuple(records_with_titles)
        self._records_by_title = {
            title: tuple(records) for title, records in records_by_title.items()
        }

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        item_to_index: Mapping[object, int],
        training_item_counts: Sequence[int] | torch.Tensor | None = None,
        **read_csv_kwargs,
    ) -> "BookCatalog":
        """Load metadata from a local CSV and construct a catalog.

        Delimiter inference handles both the comma- and semicolon-separated
        variants of the Book-Crossing files. Callers can override any pandas
        CSV option through ``read_csv_kwargs``.
        """
        options = {
            "sep": None,
            "engine": "python",
            "encoding": "latin-1",
            "dtype": str,
        }
        options.update(read_csv_kwargs)
        books = pd.read_csv(path, **options)
        return cls(books, item_to_index, training_item_counts)

    def matches_item_mapping(
        self,
        item_to_index: Mapping[object, int],
    ) -> bool:
        """Return whether this catalog was built for the supplied mapping."""
        return self._item_to_index == item_to_index

    @property
    def model_item_count(self) -> int:
        return len(self._item_to_index)

    @property
    def metadata_item_count(self) -> int:
        return len(self._records)

    def item_index(self, isbn: object) -> int:
        """Resolve an ISBN to its encoded model index."""
        model_item = self._model_items.get(normalize_isbn(isbn))
        if model_item is None:
            raise KeyError(f"ISBN is not supported by the model: {isbn}")
        return model_item[1]

    def get_by_isbn(self, isbn: object) -> CatalogBook:
        """Return model-supported display metadata for an ISBN."""
        record = self._records_by_isbn.get(normalize_isbn(isbn))
        if record is None:
            raise KeyError(f"ISBN has no model-supported metadata: {isbn}")
        return record

    def get_books(self, isbns: Sequence[object]) -> list[CatalogBook]:
        """Return metadata for ISBNs in the same order they were supplied."""
        if isinstance(isbns, (str, bytes)):
            raise TypeError("isbns must be a sequence, not a single string")
        requested_isbns = list(isbns)
        if not requested_isbns:
            raise ValueError("isbns must contain at least one ISBN")
        return [self.get_by_isbn(isbn) for isbn in requested_isbns]

    def get_by_item_index(self, item_index: int) -> CatalogBook:
        """Return display metadata for an encoded model item."""
        record = self._records_by_item_index.get(item_index)
        if record is None:
            raise KeyError(
                f"Model item index has no book metadata: {item_index}"
            )
        return record

    def eligible_candidate_indices(
        self,
        min_training_interactions: int = 5,
    ) -> torch.Tensor:
        """Return displayable model items meeting the support threshold."""
        self._validate_minimum_support(min_training_interactions)
        return torch.tensor(
            [
                record.item_index
                for record in self._records
                if record.training_interactions >= min_training_interactions
            ],
            dtype=torch.long,
        )

    def resolve_title(
        self,
        title: str,
        min_training_interactions: int = 20,
    ) -> CatalogBook:
        """Resolve a title to the most-supported learned ISBN edition.

        Exact case-insensitive title matches are preferred. If none meet the
        support threshold, variants beginning with the requested title are
        considered, matching the exploratory notebook's edition behavior.
        """
        normalized_title = _normalize_title(title)
        if not normalized_title:
            raise ValueError("title must not be empty")
        self._validate_minimum_support(min_training_interactions)

        exact_matches = [
            record
            for record in self._records_by_title.get(normalized_title, ())
            if record.training_interactions >= min_training_interactions
        ]
        if exact_matches:
            return self._most_supported(exact_matches)

        variant_matches = [
            record
            for record, record_title in self._records_with_titles
            if record.training_interactions >= min_training_interactions
            and record_title.startswith(normalized_title)
        ]
        if variant_matches:
            return self._most_supported(variant_matches)
        raise LookupError(
            f"No supported edition found for title: {title!r}"
        )

    def resolve_titles(
        self,
        titles: Sequence[str],
        min_training_interactions: int = 20,
    ) -> list[str]:
        """Resolve titles to ISBNs in the same order they were supplied."""
        if isinstance(titles, (str, bytes)):
            raise TypeError("titles must be a sequence, not a single string")
        requested_titles = list(titles)
        if not requested_titles:
            raise ValueError("titles must contain at least one title")
        return [
            self.resolve_title(
                title,
                min_training_interactions=min_training_interactions,
            ).isbn
            for title in requested_titles
        ]

    def _validate_minimum_support(self, minimum: int) -> None:
        if minimum < 0:
            raise ValueError("min_training_interactions must be non-negative")
        if minimum > 0 and not self._has_training_counts:
            raise ValueError(
                "Training item counts are required for a positive support "
                "threshold"
            )

    @staticmethod
    def _most_supported(records: list[CatalogBook]) -> CatalogBook:
        return min(
            records,
            key=lambda record: (
                -record.training_interactions,
                record.item_index,
            ),
        )
