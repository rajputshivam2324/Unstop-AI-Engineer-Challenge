"""<MODULE PURPOSE>
DOCX extractor: python-docx paragraph walk in true document order (paragraphs +
tables interleaved), corrupted-container detection (message per E8), table
flattening (cells joined " | ", appended), heading-style runs kept as outline
text (spec §4.1), empty paragraphs collapsed, page=None everywhere (A3).

<PUBLIC API> extract(path, config) -> ProcessedDocument
<DEPENDS ON> python-docx, ingestion.base, text_processing, types, config.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from docx.document import Document as DocumentObject
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from integrity_engine.config import Config
from integrity_engine.ingestion.base import finish_document
from integrity_engine.text_processing import normalize_text
from integrity_engine.types import Page, ProcessedDocument

logger = logging.getLogger("integrity_engine.ingestion.docx")

_MAX_HEADING_SHORT = 60  # headings shorter than this count toward the outline run


def _iter_block_items(document: DocumentObject) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in true document order (python-docx recipe)."""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _is_heading(paragraph: Paragraph) -> bool:
    try:
        return bool(paragraph.style) and paragraph.style.name.startswith("Heading")
    except (AttributeError, KeyError):  # malformed style refs tolerate as body text
        return False


def extract(path: Path, config: Config) -> ProcessedDocument:
    """Extract a DOCX to a ProcessedDocument; never raises on file content."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        logger.warning("docx: unreadable file: %s (%s: %s)", path.name,
                       type(exc).__name__, exc)
        return finish_document(path, [], [f"unreadable docx container ({exc})"],
                               "docx", config, {"_ingestion_status": "error", })
    if size == 0:  # E1 zero-byte fast path
        logger.warning("file is empty (0 bytes); nothing to analyze: %s", path.name)
        return finish_document(
            path, [], ["file is empty (0 bytes); nothing to analyze"],
            "docx", config, {"_ingestion_status": "error"})

    warnings: list[str] = []
    try:
        document = Document(str(path))
    except PackageNotFoundError:
        # E8: legacy .doc or corrupt container renamed .docx
        logger.warning("docx: not a valid .docx container: %s", path.name)
        return finish_document(
            path, [], ["not a valid .docx container (may be legacy .doc or corrupted)"],
            "docx", config, {"_ingestion_status": "error"})
    except (BadZipFile, ValueError, OSError):
        logger.warning("docx: not a valid .docx container: %s", path.name)
        return finish_document(
            path, [], ["not a valid .docx container (may be legacy .doc or corrupted)"],
            "docx", config, {"_ingestion_status": "error"})

    texts: list[str] = []
    paragraphs_seen = 0
    try:
        for block in _iter_block_items(document):
            if isinstance(block, Table):  # tables flattened row-wise, A3
                rows = [" | ".join(cell.text.strip() for cell in row.cells)
                        for row in block.rows]
                texts.append(normalize_text("\n".join(rows)))
                continue
            paragraphs_seen += 1
            normalized = normalize_text(block.text or "")
            if not normalized.strip():
                continue  # empty paragraphs collapse (no double newlines)
            if _is_heading(block) and len(normalized) < _MAX_HEADING_SHORT:
                # Outline headings are KEPT (assignment strips only headers/footers).
                logger.debug("docx: outline heading kept: %s", normalized)
            texts.append(normalized)
    except (BadZipFile, ValueError, KeyError, AttributeError) as exc:
        summary = f"{type(exc).__name__}: {exc}"[:120]
        logger.warning("docx: unreadable docx container: %s (%s)", path.name, summary)
        return finish_document(path, [], [f"unreadable docx container ({summary})"],
                               "docx", config, {"_ingestion_status": "error"})

    parts = [p for p in texts if p.strip()]
    page = Page(page=None, text="\n".join(parts), char_start=0, char_end=0)
    if paragraphs_seen == 0 and not parts:
        warnings.append("document contains no usable text")

    return finish_document(path, [page], warnings, "docx", config,
                           {"paragraph_count": paragraphs_seen})
