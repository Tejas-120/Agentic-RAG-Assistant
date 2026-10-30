
from pathlib import Path
import pdfplumber

project_root = Path(__file__).resolve().parent.parent
pdf_path = project_root / "documents" / "Project_Report.pdf"

with pdfplumber.open(pdf_path) as pdf:
    # PDF page 9 corresponds to index 8 in Python.
    page = pdf.pages[8]

    tables = page.extract_tables()

    print("PDF page:", 9)
    print("Tables detected:", len(tables))

    for table_number, table in enumerate(tables, start=1):
        print(f"\nTable {table_number}:")

        for row in table[:10]:
            print(row)

        if len(table) > 10:
            print("... additional rows omitted")
