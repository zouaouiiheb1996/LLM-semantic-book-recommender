import pandas as pd
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from src.config import DATA_DIR


def load_books(csv_path: str = None) -> pd.DataFrame:
    path = csv_path or DATA_DIR / "books_with_emotions.csv"
    books = pd.read_csv(path)
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )
    return books


def build_vector_store(txt_path: str = None) -> Chroma:
    path = txt_path or DATA_DIR / "tagged_description.txt"
    raw_documents = TextLoader(str(path), encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    return Chroma.from_documents(documents, OpenAIEmbeddings())


def semantic_search(db: Chroma, query: str, k: int = 50) -> list[int]:
    recs = db.similarity_search(query, k=k)
    return [int(rec.page_content.strip('"').split()[0]) for rec in recs]
