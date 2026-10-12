
from pathlib import Path
import pymupdf


def extract_pdfs(folder="data/raw"):
    pages = []
    metadata = []

    for pdf_path in Path(folder).glob("*.pdf"):
        with pymupdf.open(pdf_path) as document:
            # Preserve title-page information separately.
            first_page_text = (
                document[0].get_text("text")
                if len(document) > 0
                else ""
            )

            metadata.append({
                "source": pdf_path.name,
                "title_page": first_page_text,
            })

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if text:
                    pages.append({
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_number,
                    })

    return pages


def extract_title_pages(folder="data/raw"):
    records = []

    for pdf_path in Path(folder).glob("*.pdf"):
        with pymupdf.open(pdf_path) as document:
            if len(document) > 0:
                records.append({
                    "source": pdf_path.name,
                    "text": document[0].get_text("text"),
                    "page": 1,
                })

    return records
