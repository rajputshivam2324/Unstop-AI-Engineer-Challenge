"""<MODULE PURPOSE>
Extractor registry: file extension -> extractor callable. Public entry point for
the pipeline: extract_any(path, config) dispatches to the typed module extractor
and never raises on file-content problems (content failures degrade to
error/silent_video statuses inside the extractor).

<PUBLIC API> EXTENSION_TO_EXTRACTOR, extract_any
<DEPENDS ON> ingestion.{pdf,docx,mp4}, ingestion.base, types, config.
"""

from __future__ import annotations

from pathlib import Path

from integrity_engine.config import Config
from integrity_engine.ingestion import docx, mp4, pdf
from integrity_engine.types import ProcessedDocument

EXTENSION_TO_EXTRACTOR = {
    ".pdf": pdf.extract,
    ".docx": docx.extract,
    ".mp4": mp4.extract,
}


def extract_any(path: Path, config: Config) -> ProcessedDocument | None:
    """Dispatch by extension; None (skip) for unknown/unsupported extensions."""
    extractor = EXTENSION_TO_EXTRACTOR.get(path.suffix.lower())
    if extractor is None:
        return None
    return extractor(path, config)
