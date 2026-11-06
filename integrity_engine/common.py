"""<MODULE PURPOSE>
Small shared utilities used across modules: trimmed loggers with a per-run DEBUG
file sink, deterministic input-file listing, file hashing, UTC timestamps.

<PUBLIC API>
get_trimmed_logger, list_input_files, sha256_file, utc_now_iso, utc_now_compact

<DEPENDS ON>
logging, hashlib, pathlib, datetime.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".mp4")


def get_trimmed_logger(name: str, log_file: Path | None, level: str) -> logging.Logger:
    """Logger writing to stderr at the console level and full DEBUG to a file.

    Handlers are trimmed first so repeated calls (per-run init, tests) never
    duplicate output lines.
    """
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(logging.DEBUG)
    logger_obj.propagate = False
    for handler in list(logger_obj.handlers):
        logger_obj.removeHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger_obj.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger_obj.addHandler(file_handler)
    return logger_obj


def list_input_files(root: Path) -> list[Path]:
    """Non-recursive scan of `root`, sorted by filename, unsupported ext skipped+logged.

    Raises FileNotFoundError when the directory is missing or unreadable (the CLI
    maps this to exit code 1 per E14).
    """
    if not root.is_dir():
        raise FileNotFoundError(f"input_dir does not exist or is not readable: {root}")
    log = logging.getLogger("integrity_engine.common")
    entries = sorted(
        (p for p in root.iterdir() if p.is_file()),
        key=lambda p: p.name,
    )
    selected: list[Path] = []
    for path in entries:
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            selected.append(path)
        else:
            log.info("skipping %s: unsupported extension", path.name)
    return selected


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's raw bytes, streamed (dedup key A2)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now_iso() -> str:
    """UTC timestamp, ISO-8601 with seconds resolution (report field)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now_compact() -> str:
    """UTC timestamp, compact YYYYMMDD_HHMMSS (filename-safe, per-run suffix)."""
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
