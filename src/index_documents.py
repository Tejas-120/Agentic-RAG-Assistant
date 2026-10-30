
import json
import hashlib
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    CHROMA_DIR,
    JSONL_PATH,
    EMBEDDING_MODEL,
    MULTIMODAL_DIR,
)

COLLECTION_NAME = "project_report_multimodal"

BATCH_SIZE = 5
DELAY_BETWEEN_BATCHES = 15
MAX_RETRIES = 4


def load_records(jsonl_path=JSONL_PATH):
    """Load extracted text and table records from JSONL."""
    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Extracted records not found: {jsonl_path}\n"
            "Run: python -u src/multimodal_ingestion.py"
        )

    documents = []

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON on line {line_number}: {exc}")
                continue

            content_type = record.get("content_type")

            # Images remain separate until visual understanding is added.
            if content_type not in ("text", "table"):
                continue

            content = str(record.get("content", "")).strip()
            if not content:
                continue

            metadata = {
                "source": str(record.get("source", "unknown")),
                "page": int(record.get("page", 0)),
                "content_type": content_type,
                "record_id": str(record.get("id", "")),
            }

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

    return documents


def make_document_ids(documents):
    """Create stable IDs for safe reruns."""
    ids = []

    for document in documents:
        identity = (
            f"{document.metadata['source']}|"
            f"{document.metadata['page']}|"
            f"{document.metadata['content_type']}|"
            f"{document.metadata['record_id']}"
        )

        ids.append(
            hashlib.sha256(identity.encode("utf-8")).hexdigest()
        )

    return ids


def add_batch_with_retry(vector_store, documents, ids):
    """Retry temporary quota and service errors with backoff."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            vector_store.add_documents(
                documents=documents,
                ids=ids,
            )
            return True

        except Exception as exc:
            error_text = str(exc).lower()

            retryable = any(
                marker in error_text
                for marker in (
                    "429",
                    "resource_exhausted",
                    "rate limit",
                    "503",
                    "504",
                    "service unavailable",
                    "deadline exceeded",
                    "temporarily unavailable",
                )
            )

            if not retryable or attempt == MAX_RETRIES:
                print(f"\nBatch failed: {exc}")
                return False

            # Exponential backoff: 30, 60, 120, 240 seconds.
            wait_seconds = 30 * (2 ** attempt)

            print(
                f"\nTemporary API error. "
                f"Retrying in {wait_seconds} seconds "
                f"(attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(wait_seconds)

    return False


def main():
    MULTIMODAL_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_records()

    if not documents:
        print("No text or table records found.")
        return

    ids = make_document_ids(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    total = len(documents)
    indexed = 0

    print(f"Records to index: {total}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Starting indexing...\n")

    for start in range(0, total, BATCH_SIZE):
        batch_documents = documents[start:start + BATCH_SIZE]
        batch_ids = ids[start:start + BATCH_SIZE]

        success = add_batch_with_retry(
            vector_store,
            batch_documents,
            batch_ids,
        )

        if not success:
            print(
                f"\nIndexing stopped at approximately "
                f"{indexed}/{total} records."
            )
            print(
                "Your successfully indexed records are retained. "
                "Fix the API quota issue and rerun this script."
            )
            return

        indexed += len(batch_documents)
        print(f"Indexed {indexed}/{total} records")

        if indexed < total:
            time.sleep(DELAY_BETWEEN_BATCHES)

    print("\nIndexing completed successfully!")
    print(f"Total records processed: {indexed}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database directory: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
