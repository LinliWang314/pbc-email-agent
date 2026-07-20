"""
Tool registry — defines all tools the agent can call and dispatches execution.

Each tool is defined with its Anthropic tool-use schema and an implementation function.
The agent sees these definitions and calls them via native tool-use.
"""

from typing import Any

from tools.parsers import parse_pdf, parse_excel, ocr_image, parse_zip
from tools.classification import classify_document
from tools.extraction import extract_fields
from tools.tracker_ops import update_item_status, get_item_status


TOOL_DEFINITIONS = [
    {
        "name": "parse_pdf",
        "description": "Extract text content from a PDF attachment. Returns the full text with page numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename of the PDF attachment to parse"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "parse_excel",
        "description": "Parse an Excel file and return structured data (sheets, rows, columns). Includes cell references for citations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename of the Excel attachment to parse"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "Optional: specific sheet to parse. If omitted, parses all sheets."
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "ocr_image",
        "description": "Run OCR on an image attachment (JPG, PNG). Returns extracted text with bounding box coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename of the image to OCR"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "parse_zip",
        "description": "Inspect a ZIP attachment: list contents and parse inner PDF/Excel/image files. Use for batched submissions (e.g. multiple confirmation letters in one archive).",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename of the ZIP attachment to inspect"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "classify_document",
        "description": "Classify a parsed document against the PBC list. Returns the most likely PBC item match(es) with confidence scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename being classified"
                },
                "content_summary": {
                    "type": "string",
                    "description": "A summary of the document content (first 500 chars or key fields)"
                },
                "document_type": {
                    "type": "string",
                    "enum": ["pdf", "excel", "image", "zip"],
                    "description": "The type of document"
                }
            },
            "required": ["filename", "content_summary", "document_type"]
        }
    },
    {
        "name": "extract_fields",
        "description": "Extract specific fields from a parsed document relevant to a PBC item. Returns structured data with citations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Source filename"
                },
                "pbc_item_id": {
                    "type": "string",
                    "description": "The PBC item this document is being matched against"
                },
                "content": {
                    "type": "string",
                    "description": "The parsed content to extract from"
                },
                "fields_to_extract": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of field names to extract (e.g. period_end, entity, total_amount)"
                }
            },
            "required": ["filename", "pbc_item_id", "content"]
        }
    },
    {
        "name": "update_item_status",
        "description": "Update the status of a PBC item in the tracker. Use after verifying a document matches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "The PBC item ID (e.g. PBC-01)"
                },
                "status": {
                    "type": "string",
                    "enum": ["Not started", "Received", "Under review", "Insufficient", "Complete"],
                    "description": "The new status"
                },
                "evidence_filename": {
                    "type": "string",
                    "description": "The filename that serves as evidence"
                },
                "source_email_id": {
                    "type": "string",
                    "description": "The email message ID where this was received"
                },
                "extracted_fields": {
                    "type": "object",
                    "description": "Key fields extracted from the document"
                },
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["page", "cell", "bbox"]},
                            "reference": {"type": "string"},
                            "text": {"type": "string"}
                        }
                    },
                    "description": "Citations for the extracted data"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this status was assigned"
                }
            },
            "required": ["item_id", "status", "reasoning"]
        }
    },
    {
        "name": "get_item_status",
        "description": "Get the current status and evidence for a PBC item.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "The PBC item ID to look up"
                }
            },
            "required": ["item_id"]
        }
    },
]


def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    run: Any,
    email: Any,
) -> Any:
    """
    Dispatch a tool call to its implementation.

    Parsed documents are cached on the run (run.parsed_documents) so that when the
    agent later cites a value, update_item_status can deterministically verify the
    citation against the actual parsed content — the core anti-hallucination guardrail.
    """
    # Ensure a per-run parsed-document cache exists
    if not hasattr(run, "parsed_documents"):
        run.parsed_documents = {}

    dispatch = {
        "parse_pdf": parse_pdf,
        "parse_excel": parse_excel,
        "ocr_image": ocr_image,
        "parse_zip": parse_zip,
        "classify_document": lambda **kwargs: classify_document(run=run, **kwargs),
        "extract_fields": extract_fields,
        "update_item_status": lambda **kwargs: update_item_status(run=run, email=email, **kwargs),
        "get_item_status": lambda **kwargs: get_item_status(run=run, **kwargs),
    }

    handler = dispatch.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        result = handler(**tool_input)
        # Cache parse outputs by filename for later citation verification
        if tool_name in ("parse_pdf", "parse_excel", "ocr_image", "parse_zip"):
            filename = tool_input.get("filename")
            if filename and isinstance(result, dict) and "error" not in result:
                lock = getattr(run, "lock", None)
                if lock is not None:
                    with lock:
                        run.parsed_documents[filename] = result
                else:
                    run.parsed_documents[filename] = result
        return result
    except Exception as e:
        return {"error": f"Tool {tool_name} failed: {str(e)}"}
