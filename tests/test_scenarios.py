"""One integration test per assignment scenario (S1-S5), run on the real corpus."""

from __future__ import annotations

import json
import re

TS = re.compile(r"^\d{2}:\d{2}:\d{2} - \d{2}:\d{2}:\d{2}$")


def test_s1_cross_modal_duplication(corpus_run):
    """MP4 transcript matched against the PDF with video timestamp AND PDF page."""
    pdf = corpus_run["reports"]["07_pitch_crossmodal.pdf"]
    mm = [m for m in pdf["multimodal_matches"]
          if m["matched_file"] == "08_pitch_crossmodal_video.mp4"]
    assert mm, "07 PDF must list the 08 MP4 under multimodal_matches"
    loc = mm[0]["location"]
    assert loc["query_page"] == 2  # PITCH_PASSAGE lives on page 2
    assert TS.match(loc["matched_timestamp"])
    assert mm[0]["overlap_kind"] == "cross_modal"
    assert mm[0]["similarity_score"] >= 0.80
    assert pdf["total_integrity_risk"] == "HIGH"
    video = corpus_run["reports"]["08_pitch_crossmodal_video.mp4"]
    vloc = video["multimodal_matches"][0]["location"]
    assert TS.match(vloc["query_timestamp"]) and vloc["matched_page"] == 2


def test_s2_paraphrase_semantic_not_lexical(corpus_run):
    """Embedding flags the reworded pair with cos > 0.80 while lexical overlap is low."""
    r03 = corpus_run["reports"]["03_paraphrased_strategy.docx"]
    para = [m for m in r03["peer_matches"]
            if m["matched_file"] == "04_original_strategy.docx"
            and m["overlap_kind"] == "paraphrase"]
    assert para
    assert any(m["similarity_score"] > 0.80 and m["lexical_jaccard"] < 0.35 for m in para)
    assert r03["scores"]["peer_similarity_percentage"] >= 60
    assert r03["total_integrity_risk"] == "HIGH"


def test_s3_corrupted_and_silent_inputs_do_not_crash(corpus_run):
    assert corpus_run["code"] == 0
    reports = corpus_run["reports"]
    assert reports["10_corrupted_empty.pdf"]["status"] == "error"
    assert reports["10_corrupted_empty.pdf"]["warnings"]
    assert reports["11_silent_video.mp4"]["status"] == "silent_video"
    assert reports["11_silent_video.mp4"]["warnings"]
    # every remaining document was still processed
    assert len(reports) == len(corpus_run["manifest"]["expected_integrity_risk"])


def test_s4_shared_case_prompt_is_discounted(corpus_run):
    for doc in corpus_run["manifest"]["template_group"]["files"]:
        r = corpus_run["reports"][doc]
        assert r["total_integrity_risk"] != "HIGH"
        assert r["scores"]["peer_similarity_percentage"] == 0.0
        common = [m for m in r["peer_matches"] if m["match_type"] == "common_source"]
        assert len(common) == 2  # the other two template docs, listed but discounted
        assert not [m for m in r["peer_matches"] if m["match_type"] == "overlap"]
        assert r["template_registry_hits"]


def test_s5_batch_outputs_and_summary_matrix(corpus_run):
    out = corpus_run["out"]
    n = len(corpus_run["manifest"]["expected_integrity_risk"])
    assert len(list((out / "per_doc").glob("*.json"))) == n
    assert len(list((out / "per_doc").glob("*.md"))) == n
    summary = json.loads((out / "summary" / "latest.json").read_text())
    assert summary["aggregate"]["n_docs"] == n
    assert len(summary["peer_similarity_matrix"]) == n
    assert (out / "summary" / "SUMMARY.md").exists()
    line = json.loads(corpus_run["stdout"].strip().splitlines()[-1])
    assert line["documents_evaluated"] == n


def test_originals_have_no_overlap_matches(corpus_run):
    for doc in corpus_run["manifest"]["original_false_positive_guard"]:
        r = corpus_run["reports"][doc]
        assert not [m for m in r["peer_matches"] + r["multimodal_matches"]
                    if m["match_type"] != "common_source"], doc


def test_report_schema_matches_assignment(corpus_run):
    head = ["document_id", "total_integrity_risk", "scores", "peer_matches",
            "multimodal_matches", "explanations"]
    for r in corpus_run["reports"].values():
        assert list(r)[:6] == head
        assert set(r["scores"]) == {"peer_similarity_percentage",
                                    "external_similarity_percentage",
                                    "ai_generated_probability"}
        for m in r["peer_matches"] + r["multimodal_matches"]:
            assert {"matched_file", "similarity_score", "query_passage", "matched_passage",
                    "location"} <= set(m)
            assert m["query_passage"] and m["matched_passage"]
    md = (corpus_run["out"] / "per_doc" / "07_pitch_crossmodal.pdf.md").read_text()
    assert "⏱" in md and "📄 p2" in md
