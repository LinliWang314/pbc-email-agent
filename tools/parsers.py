"""
Document parsing tools — PDF, Excel, Image OCR.

These are deterministic tools (no LLM calls) that extract raw content
from attachments for the agent to reason about.
"""

import os
from typing import Any

# Global attachment directory — set by the ingestion layer
ATTACHMENTS_DIR: str = ""


def set_attachments_dir(path: str) -> None:
    global ATTACHMENTS_DIR
    ATTACHMENTS_DIR = path


def _resolve_path(filename: str) -> str:
    """Resolve a filename to its full path in the attachments directory."""
    path = os.path.join(ATTACHMENTS_DIR, filename)
    if not os.path.exists(path):
        # Try case-insensitive match
        for f in os.listdir(ATTACHMENTS_DIR):
            if f.lower() == filename.lower():
                return os.path.join(ATTACHMENTS_DIR, f)
        raise FileNotFoundError(f"Attachment not found: {filename}")
    return path


def parse_pdf(filename: str) -> dict[str, Any]:
    """
    Extract text from a PDF file with page numbers for citation.
    Returns structured content with page-level granularity.
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return {"error": "PyPDF2 not installed"}

    try:
        path = _resolve_path(filename)
        reader = PdfReader(path)
        pages = []
        full_text = ""

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({
                "page_number": i + 1,
                "text": text.strip(),
            })
            full_text += f"\n--- Page {i+1} ---\n{text}"

        return {
            "filename": filename,
            "total_pages": len(reader.pages),
            "pages": pages,
            "full_text": full_text.strip()[:5000],  # Cap for token control
        }
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to parse PDF {filename}: {str(e)}"}


def parse_excel(filename: str, sheet_name: str | None = None) -> dict[str, Any]:
    """
    Parse an Excel file into structured data with cell references.
    Returns sheet names, headers, and row data with cell coordinates.
    """
    try:
        from openpyxl import load_workbook
    except ImportError:
        return {"error": "openpyxl not installed"}

    try:
        path = _resolve_path(filename)
        wb = load_workbook(path, data_only=True)

        result = {
            "filename": filename,
            "sheet_names": wb.sheetnames,
            "sheets": {},
        }

        sheets_to_parse = [sheet_name] if sheet_name else wb.sheetnames

        for sname in sheets_to_parse:
            if sname not in wb.sheetnames:
                continue
            ws = wb[sname]
            rows = []
            headers = []

            for row_idx, row in enumerate(ws.iter_rows(values_only=False), start=1):
                row_data = {}
                for cell in row:
                    if cell.value is not None:
                        col_letter = cell.column_letter
                        cell_ref = f"{col_letter}{cell.row}"
                        if row_idx == 1:
                            headers.append({"ref": cell_ref, "value": str(cell.value)})
                        row_data[cell_ref] = str(cell.value)
                if row_data:
                    rows.append(row_data)

                # Cap rows for token control
                if row_idx > 100:
                    rows.append({"_truncated": f"Showing first 100 of {ws.max_row} rows"})
                    break

            result["sheets"][sname] = {
                "headers": headers,
                "rows": rows,
                "total_rows": ws.max_row,
                "total_cols": ws.max_column,
            }

        return result
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to parse Excel {filename}: {str(e)}"}


def ocr_image(filename: str) -> dict[str, Any]:
    """
    Run OCR on an image file. Returns extracted text with bounding boxes.
    Falls back gracefully if tesseract is not available.
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return {
            "error": "OCR dependencies not installed (Pillow/pytesseract)",
            "fallback": "Image attachment detected but cannot be parsed. Mark as requiring manual review.",
        }

    try:
        path = _resolve_path(filename)
        image = Image.open(path)

        # Get text with bounding box data
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            words = []
            for i, word in enumerate(data["text"]):
                if word.strip():
                    words.append({
                        "text": word,
                        "bbox": {
                            "left": data["left"][i],
                            "top": data["top"][i],
                            "width": data["width"][i],
                            "height": data["height"][i],
                        },
                        "confidence": data["conf"][i],
                    })

            full_text = " ".join(w["text"] for w in words)
            return {
                "filename": filename,
                "full_text": full_text,
                "words_with_bbox": words[:200],  # Cap for token control
                "image_size": {"width": image.width, "height": image.height},
                "ocr_confidence": sum(w["confidence"] for w in words) / max(len(words), 1),
            }
        except Exception:
            # Tesseract not available — return what we can
            return {
                "filename": filename,
                "full_text": "[OCR unavailable — image detected but text not extractable]",
                "image_size": {"width": image.width, "height": image.height},
                "note": "This appears to be a photograph/scan. Manual review recommended.",
            }
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to OCR {filename}: {str(e)}"}
