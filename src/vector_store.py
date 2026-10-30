from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Store the database locally inside the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Gemini converts text into numerical vectors
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

# Create or connect to the persistent Chroma database
vector_store = Chroma(
    collection_name="project_report",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)

import time

def add_chunks(chunks: list[dict], batch_size: int = 10) -> None:
    """Store chunks in small batches to reduce API rate-limit errors."""
    if not chunks:
        raise ValueError("No chunks provided.")

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        documents = [chunk["text"] for chunk in batch]
        metadatas = [
            {"page": chunk["page"], "source": chunk["source"]}
            for chunk in batch
        ]
        ids = [chunk["id"] for chunk in batch]

        vector_store.add_texts(
            texts=documents,
            metadatas=metadatas,
            ids=ids,
        )

        print(f"Indexed {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

        if start + batch_size < len(chunks):
            time.sleep(7)


def search_chunks(question: str, k: int = 4):
    """Retrieve relevant, diverse passages from the report."""
    return vector_store.max_marginal_relevance_search(
        question,
        k=k,
        fetch_k=20,
        lambda_mult=0.7,
    )
