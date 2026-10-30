
from pathlib import Path
import fitz
import pdfplumber

project_root = Path(__file__).resolve().parent.parent
pdf_path = project_root / "documents" / "Project_Report.pdf"

print("Inspecting:", pdf_path.name)

# Inspect pages and embedded images using PyMuPDF.
document = fitz.open(pdf_path)

print("Total pages:", len(document))

for page_number, page in enumerate(document, start=1):
    text = page.get_text()
    images = page.get_images(full=True)

    print(
        f"Page {page_number}: "
        f"{len(text)} text characters, "
        f"{len(images)} embedded images"
    )

document.close()

# Inspect tables using pdfplumber.
print("\nTable inspection:")

with pdfplumber.open(pdf_path) as pdf:
    total_tables = 0

    for page_number, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables()
        total_tables += len(tables)

        if tables:
            print(f"Page {page_number}: {len(tables)} table(s)")

print("Total tables detected:", total_tables)
