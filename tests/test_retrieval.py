import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.rag import load_books, semantic_search


@pytest.fixture
def sample_books_csv(tmp_path):
    csv = tmp_path / "books_with_emotions.csv"
    csv.write_text(
        "isbn13,title,authors,description,thumbnail,simple_categories,joy,surprise,anger,fear,sadness\n"
        "9780000000001,Test Book,Author A,A great test book.,http://example.com/img,Fiction,0.8,0.1,0.05,0.03,0.02\n"
    )
    return str(csv)


def test_load_books_adds_large_thumbnail(sample_books_csv):
    df = load_books(sample_books_csv)
    assert "large_thumbnail" in df.columns
    assert df["large_thumbnail"].iloc[0].endswith("&fife=w800")


def test_load_books_fallback_thumbnail(sample_books_csv):
    df = load_books(sample_books_csv)
    df.loc[0, "thumbnail"] = None
    df["large_thumbnail"] = df["thumbnail"].apply(
        lambda x: "cover-not-found.jpg" if pd.isna(x) else x + "&fife=w800"
    )
    assert df["large_thumbnail"].iloc[0] == "cover-not-found.jpg"


def test_semantic_search_returns_isbn_list():
    mock_db = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = '"9780000000001 some description"'
    mock_db.similarity_search.return_value = [mock_doc]

    result = semantic_search(mock_db, "adventure story", k=1)
    assert result == [9780000000001]
    mock_db.similarity_search.assert_called_once_with("adventure story", k=1)
