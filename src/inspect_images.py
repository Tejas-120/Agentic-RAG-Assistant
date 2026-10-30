
from pathlib import Path
import fitz

project_root = Path(__file__).resolve().parent.parent
pdf_path = project_root / "documents" / "Project_Report.pdf"
output_dir = project_root / "documents" / "extracted_images"
output_dir.mkdir(parents=True, exist_ok=True)

with fitz.open(pdf_path) as pdf:
    page = pdf[79]  # PDF page 80

    images = page.get_images(full=True)
    print("PDF page: 80")
    print("Embedded images detected:", len(images))

    for image_number, image_info in enumerate(images, start=1):
        xref = image_info[0]
        image_data = pdf.extract_image(xref)

        extension = image_data["ext"]
        image_bytes = image_data["image"]
        output_path = output_dir / f"page80_image{image_number}.{extension}"

        output_path.write_bytes(image_bytes)
        print(f"Saved: {output_path.name}")
