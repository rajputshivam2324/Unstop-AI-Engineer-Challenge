"""N5: same input -> identical reports (ignoring timestamps). Uses the AI
heuristic-fallback config so the test is fast and also exercises that path end to end."""

from __future__ import annotations

import json
import shutil

from conftest import SAMPLES, run_cli

SUBSET = ["07_pitch_crossmodal.pdf", "08_pitch_crossmodal_video.mp4",
          "12_case_template_A.docx", "13_case_template_B.docx", "14_case_template_C.docx"]
VOLATILE = {"generated_at_utc"}


def _normalized(out_dir) -> dict:
    data = {}
    for path in sorted((out_dir / "per_doc").glob("*.json")):
        report = json.loads(path.read_text())
        data[path.name] = {k: v for k, v in report.items() if k not in VOLATILE}
    return data


def test_two_runs_are_identical(tmp_path, no_classifier_config):
    inp = tmp_path / "in"
    inp.mkdir()
    for name in SUBSET:
        shutil.copy(SAMPLES / name, inp / name)
    outs = []
    for run in ("a", "b"):
        out = tmp_path / run
        code, _ = run_cli(["--input_dir", str(inp), "--output_dir", str(out), "--skip-web",
                           "--config", str(no_classifier_config), "--log-level", "ERROR"])
        assert code == 0
        outs.append(_normalized(out))
    assert outs[0] == outs[1]
    r = outs[0]["12_case_template_A.docx.json"]
    assert r["ai_analysis"]["ai_detector_source"] == "heuristic_fallback"
    assert "heuristic" in r["explanations"]
    assert "07_pitch_crossmodal.pdf.json" in outs[0]
    assert outs[0]["07_pitch_crossmodal.pdf.json"]["multimodal_matches"]
