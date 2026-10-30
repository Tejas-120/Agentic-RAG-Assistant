from pathlib import Path

from ingestion import create_document_chunks
from vector_store import add_chunks, search_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "documents" / "Project_Report.pdf"

print("Step 1: Extracting and chunking the PDF...")
chunks = create_document_chunks(PDF_PATH)
print(f"Created {len(chunks)} chunks.")

print("\nStep 2: Storing chunks in ChromaDB...")
add_chunks(chunks)
print("Indexing completed!")

print("\nStep 3: Searching the report...")
question = "Which deep learning models were used for brain tumor classification?"
results = search_chunks(question, k=3)

for index, document in enumerate(results, start=1):
    print(f"\nResult {index}")
    print("Page:", document.metadata.get("page"))
    print("Source:", document.metadata.get("source"))
    print("Text:", document.page_content[:400])

print("\nVector database test completed!")