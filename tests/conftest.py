"""Shared fixtures. Models must already be cached (run --models-only once);
tests never touch the network (HF offline mode, web tier skipped)."""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SAMPLES = ROOT / "sample_submissions"
MANIFEST = SAMPLES / "dataset_manifest.json"


def run_cli(args: list[str]) -> tuple[int, str]:
    """Run the CLI in-process; returns (exit_code, stdout)."""
    from integrity_engine.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            code = main(args)
        except SystemExit as exc:  # argparse errors
            code = int(exc.code or 0)
    return code, buf.getvalue()


@pytest.fixture(scope="session")
def corpus_run(tmp_path_factory) -> dict:
    """One full pipeline run over the real 14-file sample corpus (the money fixture)."""
    out = tmp_path_factory.mktemp("reports")
    code, stdout = run_cli(["--input_dir", str(SAMPLES), "--output_dir", str(out),
                            "--skip-web", "--log-level", "WARNING"])
    reports = {}
    for path in sorted((out / "per_doc").glob("*.json")):
        report = json.loads(path.read_text())
        reports[report["document_id"]] = report
    return {"code": code, "stdout": stdout, "out": out, "reports": reports,
            "manifest": json.loads(MANIFEST.read_text())}


@pytest.fixture()
def no_classifier_config(tmp_path) -> Path:
    """Config that points the AI classifier at a non-existent model -> fallback path."""
    cfg = tmp_path / "fallback.toml"
    cfg.write_text((ROOT / "config" / "default.toml").read_text().replace(
        'classifier_model = "wasitaigeneratedcom/ai-text-detector-small"',
        'classifier_model = "/nonexistent/detector"'))
    return cfg
