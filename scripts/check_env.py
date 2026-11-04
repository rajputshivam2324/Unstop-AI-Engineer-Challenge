#!/usr/bin/env python
"""<PURPOSE>
Environment sanity printer + peak-RSS reporter (NFR N2/N9).
Runs inside the .venv and exits 0 only when every hard dependency imports.

<PUBLIC API> main(argv) -> int
<DEPENDS ON> importlib, resource, sys.
"""

from __future__ import annotations

import argparse
import importlib
import resource
import sys


REQUIRED_MODULES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("faiss", "faiss-cpu"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("sentence_transformers", "sentence-transformers"),
    ("faster_whisper", "faster-whisper"),
    ("pypdf", "pypdf"),
    ("docx", "python-docx"),
    ("fpdf", "fpdf2"),
    ("requests", "requests"),
    ("tomllib", "stdlib (3.11+)"),
]


def peak_rss_mb() -> float:
    """Peak resident set size of this process (self), in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print/import-verify the version matrix")
    parser.add_argument(
        "--rss", action="store_true",
        help="exit 1 if peak RSS exceeds the 2 GB NFR budget (measure your run with it)")
    args = parser.parse_args(argv)

    print(f"{'python':<24} {sys.version}")
    rows: list[str] = []
    for module_name, package in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "n/a")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            print(f"FAIL {package:<24} import error: {exc}", file=sys.stderr)
            return 1
        rows.append(f"{package:<24} {version}")
        print(f"{package:<24} {version}")
    print(f"{'ffmpeg_test':<24} see README (external binary, not a python module)")
    print("peak RSS: %.0f MiB" % peak_rss_mb())
    if args.rss:
        budget_mb = 2048.0
        status = "OK" if peak_rss_mb() <= budget_mb else "OVER BUDGET"
        print(f"NFR N2 budget check ({budget_mb:.0f} MiB): {status}")
        return 0 if status == "OK" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
