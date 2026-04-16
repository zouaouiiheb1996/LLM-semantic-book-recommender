import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.recommendation.recommender import retrieve_semantic_recommendations, format_authors


@pytest.fixture
def sample_books():
    return pd.DataFrame({
        "isbn13": [1, 2, 3],
        "title": ["Book A", "Book B", "Book C"],
        "authors": ["Author A", "Author B", "Author C"],
        "description": ["Desc A", "Desc B", "Desc C"],
        "large_thumbnail": ["img_a", "img_b", "img_c"],
        "simple_categories": ["Fiction", "Non-Fiction", "Fiction"],
        "joy": [0.9, 0.2, 0.5],
        "surprise": [0.1, 0.8, 0.3],
        "anger": [0.05, 0.1, 0.7],
        "fear": [0.02, 0.4, 0.6],
        "sadness": [0.01, 0.3, 0.2],
    })


@pytest.fixture
def mock_db(sample_books):
    db = MagicMock()
    with patch("src.recommendation.recommender.semantic_search", return_value=[1, 2, 3]):
        yield db


def test_retrieve_all_categories(mock_db, sample_books):
    result = retrieve_semantic_recommendations(mock_db, sample_books, "test query")
    assert len(result) <= 16


def test_retrieve_filtered_category(mock_db, sample_books):
    with patch("src.recommendation.recommender.semantic_search", return_value=[1, 2, 3]):
        result = retrieve_semantic_recommendations(
            mock_db, sample_books, "test", category="Fiction"
        )
    assert all(result["simple_categories"] == "Fiction")


def test_retrieve_sorted_by_tone(mock_db, sample_books):
    with patch("src.recommendation.recommender.semantic_search", return_value=[1, 2, 3]):
        result = retrieve_semantic_recommendations(
            mock_db, sample_books, "test", tone="Happy"
        )
    assert result["joy"].iloc[0] == result["joy"].max()


@pytest.mark.parametrize("raw,expected", [
    ("Author A", "Author A"),
    ("Author A;Author B", "Author A and Author B"),
    ("Author A;Author B;Author C", "Author A, Author B, and Author C"),
])
def test_format_authors(raw, expected):
    assert format_authors(raw) == expected
