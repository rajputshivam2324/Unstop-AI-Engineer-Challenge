"""<MODULE PURPOSE>
PDF extractor: pypdf page-by-page text extraction with per-page error tolerance,
zero-byte fast path, scanned-image (no text layer) detection, and a repeated
header/footer strip (any first/last nonempty line occurring on >=50% of pages
when page_count >= 2 is removed from every page and logged as a warning).

<PUBLIC API> extract(path, config) -> ProcessedDocument
<DEPENDS ON> pypdf, ingestion.base (finish_document), text_processing (normalize_text),
types, config.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from integrity_engine.config import Config
from integrity_engine.ingestion.base import EMPTY_TEXT_THRESHOLD_CH, finish_document
from integrity_engine.text_processing import normalize_text
from integrity_engine.types import Page, ProcessedDocument

logger = logging.getLogger("integrity_engine.ingestion.pdf")

MAX_PAGES = 500  # per-file guard (spec §4.1)


def _header_footer_lines(lines_by_page: list[list[str]]) -> set[str]:
    """Normalized lines occurring as first/last line on >=50% of pages (count >= 2 pages)."""
    if len(lines_by_page) < 2:
        return set()
    edge: Counter[str] = Counter()
    for lines in lines_by_page:
        if not lines:
            continue
        edge[" ".join(lines[0].split())] += 1
        edge[" ".join(lines[-1].split())] += 1
    half = len(lines_by_page) / 2
    return {line for line, count in edge.items() if line and count >= half}


def extract(path: Path, config: Config) -> ProcessedDocument:  # noqa: ARG001 - contract signature
    """Extract a PDF to a ProcessedDocument; never raises on file content."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        logger.warning("pdf: unreadable file: %s (%s: %s)", path.name,
                       type(exc).__name__, exc)
        return finish_document(path, [], [f"unreadable pdf container ({exc})"],
                               "pdf", config, {"_ingestion_status": "error"})
    if size == 0:  # E1 zero-byte fast path
        logger.warning("file is empty (0 bytes); nothing to analyze: %s", path.name)
        return finish_document(
            path, [], ["file is empty (0 bytes); nothing to analyze"],
            "pdf", config, {"_ingestion_status": "error", "page_count": 0})

    warnings: list[str] = []
    try:
        reader = PdfReader(str(path))
    except (PdfReadError, ValueError, OSError) as exc:  # corrupted container (E6)
        summary = f"{type(exc).__name__}: {exc}"[:120]
        logger.warning("pdf: unreadable pdf container: %s (%s)", path.name, summary)
        return finish_document(path, [], [f"unreadable pdf container ({summary})"],
                               "pdf", config, {"_ingestion_status": "error"})

    pages: list[Page] = []
    lines_by_page: list[list[str]] = []
    had_text_layer: list[bool] = []
    page_count = min(len(reader.pages), MAX_PAGES)
    if len(reader.pages) > MAX_PAGES:
        warnings.append(f"page count exceeds guard ({MAX_PAGES}); truncated")

    for index in range(page_count):
        try:
            raw = reader.pages[index].extract_text() or ""
            error = False
        except Exception as exc:  # noqa: BLE001 - per-page tolerance (spec §4.1)
            summary = f"{type(exc).__name__}: {exc}"[:100]
            logger.debug("pdf: page %d extraction failed: %s (%s)", index + 1,
                         path.name, summary)
            raw, error = "", True
            warnings.append(f"page {index + 1}: text extraction failed ({summary})")
        text = normalize_text(raw)
        had_text_layer.append(bool(text) and not error)
        lines_by_page.append([line for line in text.split("\n") if line.strip()])
        pages.append(Page(page=index + 1, text=text, char_start=0, char_end=0))

    # Header/footer strip: multiset of first/last nonempty lines on >=50% of pages.
    for repeated in sorted(_header_footer_lines(lines_by_page)):
        for page in pages:
            kept = [line for line in page.text.split("\n")
                    if line.strip() and " ".join(line.split()) != repeated]
            page.text = normalize_text("\n".join(kept))
        label = repeated[:40]
        warnings.append(f'removed repeated header/footer line: "{label}"')

    total_chars = sum(len(p.text.strip()) for p in pages)
    if total_chars < EMPTY_TEXT_THRESHOLD_CH:
        for index in range(page_count):  # E7: scanned-image attribution
            if not had_text_layer[index]:
                warn = f"page {index + 1}: no extractable text (likely scanned image)"
                if warn not in warnings:
                    warnings.append(warn)

    return finish_document(path, pages, warnings, "pdf", config,
                           {"page_count": page_count})
