from pathlib import Path
from pypdf import PdfReader


def extract_pages_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from each PDF page while preserving page numbers."""
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip(),
                "source": pdf_path.name,
            })

    return pages


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_document_chunks(
    pdf_path: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    """Create chunks with page numbers, source names, and unique IDs."""
    pages = extract_pages_from_pdf(pdf_path)
    all_chunks = []

    for page in pages:
        text_chunks = split_text_into_chunks(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, chunk_text in enumerate(text_chunks):
            all_chunks.append({
                "id": f"{pdf_path.stem}-p{page['page']}-c{chunk_index}",
                "text": chunk_text,
                "page": page["page"],
                "source": page["source"],
            })

    return all_chunks