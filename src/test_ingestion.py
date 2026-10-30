from pathlib import Path
from ingestion import create_document_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "documents" / "Project_Report.pdf"

chunks = create_document_chunks(PDF_PATH)

print("PDF:", PDF_PATH.name)
print("Total chunks:", len(chunks))
print("First chunk page:", chunks[0]["page"])
print("First chunk ID:", chunks[0]["id"])
print("First chunk source:", chunks[0]["source"])
print("\nFirst chunk preview:")
print(chunks[0]["text"][:300])