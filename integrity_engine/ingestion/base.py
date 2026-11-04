"""<MODULE PURPOSE>
Shared extractor contract and the common tail every extractor runs through:
document identity (document_id = filename), sha256, size, joined full_text, and the
status decision {ok, empty} plus the <15-chars usable-text rule. error/silent_video
statuses are set by module-specific wrappers BEFORE calling finish_document.

<PUBLIC API>
Extractor (Protocol), finish_document, EMPTY_TEXT_THRESHOLD_CH, MIN_USABLE_CHARS

<DEPENDS ON>
types (Page, ProcessedDocument), hashlib sha256 (via common), config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from integrity_engine.common import sha256_file
from integrity_engine.config import Config
from integrity_engine.types import Page, ProcessedDocument

logger = logging.getLogger("integrity_engine.ingestion")

# CHUNK/FILE: zero-byte fast path precedes any parser (spec E1)
EMPTY_TEXT_THRESHOLD_CH = 15  # total normalized chars below this => status empty (E9)


class Extractor(Protocol):
    """Signature every modality extractor honors: never raises on file content."""

    def extract(self, path: Path, config: Config) -> ProcessedDocument: ...


def finish_document(path: Path, pages: list[Page], warnings: list[str],
                    modality: str, config: Config,
                    extra_metadata: dict | None = None,
                    separator: str = "\n\n") -> ProcessedDocument:
    """Shared tail for all extractors: identity, hash, status decision, full_text.

    Builds the joined full_text using `separator` between pages ("\\n\\n" for
    pdf/docx; a single space for mp4 transcripts per A5). Page.char_* are then
    exact offsets into the joined full_text. Status: error/silent_video come from
    the caller via extra_metadata["_ingestion_status"]; otherwise empty when the
    total normalized chars < 15 (E9), else ok.
    """
    separator_len = len(separator)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for page in pages:
        offsets.append((cursor, cursor + len(page.text)))
        cursor += len(page.text) + separator_len
    full_text = separator.join(page.text for page in pages)
    for page, (start, end) in zip(pages, offsets, strict=True):
        page.char_start = start
        page.char_end = end

    metadata = dict(extra_metadata or {})
    metadata.setdefault("warning_count", len(warnings))

    status = metadata.pop("_ingestion_status", "ok")  # set by caller for error/silent
    if status not in ("error", "silent_video"):
        total = sum(len(p.text.strip()) for p in pages)
        if total < EMPTY_TEXT_THRESHOLD_CH:
            status = "empty"
            if "document contains no usable text" not in warnings:
                warnings.append("document contains no usable text")

    try:
        size_bytes = path.stat().st_size
        digest = sha256_file(path) if size_bytes > 0 else ""
    except OSError as exc:  # defensive: identity should always survive
        logger.debug("stat/sha256 failed for %s: %s", path.name, exc)
        size_bytes, digest = 0, ""

    return ProcessedDocument(
        document_id=path.name,
        path=path,
        modality=modality,  # type: ignore[arg-type]  # callers pass a literal
        status=status,  # type: ignore[arg-type]
        warnings=warnings,
        pages=pages,
        full_text=full_text,
        size_bytes=size_bytes,
        sha256=digest,
        metadata=metadata,
    )
