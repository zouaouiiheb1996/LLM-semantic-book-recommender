import pandas as pd
from langchain_chroma import Chroma

from src.retrieval.rag import semantic_search

TONE_COLUMN_MAP = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}


def retrieve_semantic_recommendations(
    db: Chroma,
    books: pd.DataFrame,
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    isbn_list = semantic_search(db, query, k=initial_top_k)
    book_recs = books[books["isbn13"].isin(isbn_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    sort_col = TONE_COLUMN_MAP.get(tone)
    if sort_col:
        book_recs = book_recs.sort_values(by=sort_col, ascending=False)

    return book_recs


def format_authors(raw: str) -> str:
    parts = raw.split(";")
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    if len(parts) > 2:
        return f"{', '.join(parts[:-1])}, and {parts[-1]}"
    return raw
