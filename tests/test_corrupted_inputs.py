"""Corrupted / degraded file handling (Scenario 3 unit level) + CLI exit codes."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import run_cli
from integrity_engine.config import Config
from integrity_engine.ingestion import extract_any

CFG = Config()


@pytest.mark.parametrize("ext", [".pdf", ".docx", ".mp4"])
def test_zero_byte_file_is_error(tmp_path, ext):
    path = tmp_path / f"empty{ext}"
    path.touch()
    doc = extract_any(path, CFG)
    assert doc.status == "error"
    assert "0 bytes" in doc.warnings[0]


@pytest.mark.parametrize("ext", [".pdf", ".docx", ".mp4"])
def test_random_bytes_is_error(tmp_path, ext):
    path = tmp_path / f"garbage{ext}"
    path.write_bytes(os.urandom(4096))
    doc = extract_any(path, CFG)
    assert doc.status == "error"
    assert doc.warnings


def test_text_file_renamed_docx(tmp_path):
    path = tmp_path / "legacy.docx"
    path.write_text("this is not a zip container")
    doc = extract_any(path, CFG)
    assert doc.status == "error"
    assert "not a valid .docx" in doc.warnings[0]


def test_unsupported_extension_is_skipped(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    assert extract_any(path, CFG) is None


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def test_video_without_audio_stream(tmp_path):
    path = tmp_path / "noaudio.mp4"
    _ffmpeg(["-f", "lavfi", "-i", "color=c=black:s=64x64:r=5", "-t", "2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    doc = extract_any(path, CFG)
    assert doc.status == "silent_video"
    assert doc.warnings == ["no audio stream in MP4"]


def test_silent_audio_track(tmp_path):
    path = tmp_path / "silent.mp4"
    _ffmpeg(["-f", "lavfi", "-i", "color=c=black:s=64x64:r=5", "-f", "lavfi", "-i",
             "anullsrc=r=16000:cl=mono", "-t", "3", "-c:v", "libx264", "-pix_fmt",
             "yuv420p", "-c:a", "aac", str(path)])
    doc = extract_any(path, CFG)
    assert doc.status == "silent_video"


def test_pdf_without_text_layer_is_empty(tmp_path):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.rect(10, 10, 50, 50, style="F")  # drawing only, no text
    path = tmp_path / "scan.pdf"
    pdf.output(str(path))
    doc = extract_any(path, CFG)
    assert doc.status == "empty"
    assert any("no extractable text" in w for w in doc.warnings)


def test_cli_never_crashes_on_garbage_folder(tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    for i, ext in enumerate([".pdf", ".docx", ".mp4", ".pdf"]):
        (inp / f"bad_{i}{ext}").write_bytes(os.urandom(2048) if i else b"")
    (inp / "ignored.txt").write_text("not a submission")
    code, stdout = run_cli(["--input_dir", str(inp), "--output_dir", str(out),
                            "--skip-web", "--log-level", "ERROR"])
    assert code == 0
    assert json.loads(stdout.strip())["documents_evaluated"] == 4
    reports = list((out / "per_doc").glob("*.json"))
    assert len(reports) == 4
    for path in reports:
        r = json.loads(path.read_text())
        assert r["status"] == "error" and r["warnings"]
        assert r["total_integrity_risk"] == "LOW"


def test_cli_exit_codes(tmp_path):
    assert run_cli(["--input_dir", str(tmp_path / "missing"), "--output_dir",
                    str(tmp_path / "o")])[0] == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "a.txt").write_text("x")
    assert run_cli(["--input_dir", str(empty), "--output_dir", str(tmp_path / "o")])[0] == 1
    assert run_cli(["--output_dir", str(tmp_path / "o")])[0] == 2
    bad = tmp_path / "bad.toml"
    bad.write_text("[[[ not toml")
    assert run_cli(["--input_dir", str(empty), "--output_dir", str(tmp_path / "o"),
                    "--config", str(bad)])[0] == 1


def test_markdown_written_for_degraded(tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    (inp / "zero.pdf").touch()
    run_cli(["--input_dir", str(inp), "--output_dir", str(out), "--skip-web",
             "--log-level", "ERROR"])
    md = Path(out / "per_doc" / "zero.pdf.md").read_text()
    assert "file is empty (0 bytes)" in md



def test_header_footer_strip_only_removes_repeated_lines(tmp_path):
    from fpdf import FPDF

    pdf = FPDF()
    for body in ("First page body line one.\nFirst page closing line.",
                 "Second page opening line.\nSecond page closing line."):
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, "ACME Case Report\n" + body + "\nConfidential draft")
    path = tmp_path / "two_pages.pdf"
    pdf.output(str(path))
    doc = extract_any(path, CFG)
    assert "ACME Case Report" not in doc.full_text      # repeated header removed
    assert "Confidential draft" not in doc.full_text    # repeated footer removed
    assert "First page closing line." in doc.full_text  # unique body lines kept
    assert "Second page opening line." in doc.full_text
