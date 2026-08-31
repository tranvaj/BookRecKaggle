import unittest
from types import SimpleNamespace

from app.main import book_metadata, resolve_titles
from app.schemas import (
    BookMetadataRequest,
    BookResponse,
    ResolveTitlesRequest,
)


class _FakeBook:
    def __init__(self, isbn):
        self.isbn = isbn

    def to_dict(self):
        return {
            "isbn": self.isbn,
            "title": f"Book {self.isbn}",
            "author": "Author",
            "year": "2000",
            "publisher": "Publisher",
            "image_url": None,
            "training_interactions": 42,
        }


class _FakeCatalog:
    def __init__(self):
        self.received_titles = None
        self.received_minimum = None
        self.received_isbns = None

    def resolve_titles(self, titles, min_training_interactions):
        self.received_titles = titles
        self.received_minimum = min_training_interactions
        return ["query-isbn"]

    def get_books(self, isbns):
        self.received_isbns = isbns
        return [_FakeBook(isbn) for isbn in isbns]


def _request_for(catalog):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                recommender=SimpleNamespace(catalog=catalog)
            )
        )
    )


class ApiTests(unittest.TestCase):
    def test_resolve_endpoint_returns_only_isbns(self):
        catalog = _FakeCatalog()
        payload = ResolveTitlesRequest(titles=["Query Book"])

        result = resolve_titles(payload, _request_for(catalog))

        self.assertEqual(result, ["query-isbn"])
        self.assertEqual(catalog.received_titles, ["Query Book"])
        self.assertEqual(catalog.received_minimum, 20)

    def test_metadata_endpoint_returns_books_with_interaction_counts(self):
        catalog = _FakeCatalog()
        payload = BookMetadataRequest(isbns=["first", "second"])

        result = book_metadata(payload, _request_for(catalog))
        response = [BookResponse.model_validate(book) for book in result]

        self.assertEqual(catalog.received_isbns, ["first", "second"])
        self.assertEqual([book.isbn for book in response], ["first", "second"])
        self.assertEqual(response[0].training_interactions, 42)


if __name__ == "__main__":
    unittest.main()
