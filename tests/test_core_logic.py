"""Unit tests for the math that the rubric weighs most: self-match exclusion,
coverage aggregation, alignment monotonicity, template discount, AI fusion,
metrics. No models needed (fixture vectors)."""

from __future__ import annotations

import math

import numpy as np

from integrity_engine.ai_detection import (
    AIDetector,
    band_label,
    feature_heuristic_ai_score,
    fuse_probabilities,
)
from integrity_engine.alignment import TemplateRegistry, sentence_lcs_alignment
from integrity_engine.config import Config, PairwiseConfig
from integrity_engine.indexing import (
    SelfMatchExclusion,
    build_index,
    pairwise_doc_similarity,
    search_neighbors,
)
from integrity_engine.metrics import prf, roc_auc
from integrity_engine.scoring import verdict_from_signals
from integrity_engine.types import Chunk

CFG = Config()


def _chunk(cid: str, doc: str, text: str = "t") -> Chunk:
    return Chunk(chunk_id=cid, document_id=doc, text=text, start_char=0, end_char=len(text))


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_self_match_and_duplicate_exclusion():
    ex = SelfMatchExclusion(["a", "a", "b", "c"], duplicates={"c": "b"})
    assert ex.is_excluded(0, 0)            # identical chunk
    assert ex.is_excluded(0, 1)            # same document
    assert not ex.is_excluded(0, 2)
    assert ex.is_excluded(2, 3)            # byte-identical duplicate group
    assert ex.is_duplicate_pair("c", "b")


def test_faiss_search_never_returns_same_document():
    vecs = np.stack([_unit([1, 0, 0]), _unit([1, 0.01, 0]), _unit([0, 1, 0]),
                     _unit([0.9, 0.1, 0])])
    docs = ["a", "a", "b", "c"]
    index = build_index(vecs)
    ex = SelfMatchExclusion(docs)
    for i in range(len(docs)):
        for j, _ in search_neighbors(index, vecs[i:i + 1], 10, ex, i):
            assert docs[j] != docs[i]


def test_coverage_weighted_mean_max_golden_value():
    chunks = [_chunk("q0", "Q"), _chunk("q1", "Q"), _chunk("m0", "M")]
    sims = {0: [(2, 0.9)], 1: [(2, 0.3)], 2: [(0, 0.9), (1, 0.3)]}
    pairs = pairwise_doc_similarity(chunks, sims, PairwiseConfig(chunk_min_cos=0.55))
    qm = next(p for p in pairs if p.query_document_id == "Q")
    # S(Q,M) = (0.9 + 0 [0.3 < floor]) / 2 = 0.45 ; S(M,Q) = 0.9 / 1 ; score = max
    assert math.isclose(qm.score, 0.9)
    assert qm.directionality == "reverse"


def test_evidence_gate_blocks_topic_only_similarity():
    chunks = [_chunk("q0", "Q"), _chunk("m0", "M")]
    sims = {0: [(1, 0.8)], 1: [(0, 0.8)]}
    pairs = pairwise_doc_similarity(chunks, sims, PairwiseConfig(), verify=lambda i, j: False)
    assert all(p.score == 0.0 for p in pairs)


def test_alignment_keeps_all_rows_of_exact_copy_with_float_ties():
    sents = [f"sentence {i} alpha beta gamma" for i in range(6)]
    m = [[1.0 if i == j else 0.1 for j in range(6)] for i in range(6)]
    m[3][3], m[0][0] = 1.0000002, 0.9999999  # float noise that broke the old greedy
    assert [(q, k) for q, k, _ in sentence_lcs_alignment(sents, sents, m)] == \
        [(i, i) for i in range(6)]


def test_alignment_is_monotonic():
    s = ["a b c", "d e f", "g h i"]
    m = [[0.8, 0.1, 0.9], [0.1, 0.8, 0.1], [0.9, 0.1, 0.8]]
    out = sentence_lcs_alignment(s, s, m)
    q_idx, m_idx = [q for q, _, _ in out], [k for _, k, _ in out]
    assert q_idx == sorted(q_idx) and m_idx == sorted(m_idx)


def test_template_registry_needs_three_documents():
    block = ("Answer both questions using the attached data and show every calculation "
             "line by line in the appendix exhibit labelled with your candidate code. "
             "Do not consult external market reports because the case is self contained.")
    three = [_chunk(f"{d}:000", d, block) for d in ("A", "B", "C")]
    TemplateRegistry().register_all_chunks(three)
    assert all(c.is_template for c in three)
    two = [_chunk(f"{d}:000", d, block) for d in ("A", "B")]
    TemplateRegistry().register_all_chunks(two)
    assert not any(c.is_template for c in two)


def test_ensemble_math_and_bands():
    # 0.6*0.9 + 0.4*(0.75*0.2 + 0.25*0.4) = 0.54 + 0.1 = 0.64
    assert math.isclose(fuse_probabilities(0.9, 0.2, 0.4, CFG.ai), 0.64)
    assert band_label(34.9, CFG.ai) == "HUMAN"
    assert band_label(35.0, CFG.ai) == "UNSURE"
    assert band_label(65.0, CFG.ai) == "AI_LIKELY"


def test_ai_length_gate_and_fallback_never_blend(monkeypatch):
    det = AIDetector(CFG.ai, CFG.runtime.cache_dir)
    monkeypatch.setattr(det, "_run_gpt2", lambda texts: None)  # model unavailable
    text = " ".join(["The committee reviewed the quarterly plan in detail."] * 12)
    out = det.detect_many({"short": "too short", "long": text})
    assert out["short"].label == "INSUFFICIENT_TEXT" and out["short"].length_gate_triggered
    assert out["long"].ai_detector_source == "heuristic_fallback"
    assert out["long"].p_classifier is None and out["long"].p_perplexity is None
    prob, detail = feature_heuristic_ai_score(text)
    assert out["long"].ai_generated_probability == round(100 * prob, 1)
    assert "rule-based mock fallback" in detail["method"]


def test_risk_rule_peer_dominant():
    assert verdict_from_signals(70, 0, "HUMAN", 10, False) == "HIGH"
    assert verdict_from_signals(45, 31, "HUMAN", 10, False) == "HIGH"
    assert verdict_from_signals(45, 0, "HUMAN", 10, False) == "MEDIUM"
    assert verdict_from_signals(20, 0, "HUMAN", 10, False) == "MEDIUM"
    assert verdict_from_signals(19.9, 0, "HUMAN", 10, False) == "LOW"
    assert verdict_from_signals(0, 0, "AI_LIKELY", 80, False) == "MEDIUM"
    assert verdict_from_signals(0, 0, "HUMAN", 10, False) == "LOW"


def test_metrics():
    assert prf(2, 1, 1) == (2 / 3, 2 / 3, 2 / 3)
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.5, 0.5], [1, 0]) == 0.5
