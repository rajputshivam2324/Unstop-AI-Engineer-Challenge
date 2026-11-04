"""<MODULE PURPOSE>
Signal combination (§4.8): peer (dominant) -> external (secondary) -> AI
(supporting) with the exact threshold arithmetic, and the explanation templates
whose tokens are asserted by tests.

<PUBLIC API>
compute_integrity_signals, verdict_from_signals, build_explanation,
build_human_sentence

<DEPENDS ON>
types, config; peer/external/AI results passed in as plain data.
"""

from __future__ import annotations

import logging
from typing import Any

from integrity_engine.config import Config
from integrity_engine.types import AIDetectionResult, ExternalEvidence, Scores

logger = logging.getLogger("integrity_engine.scoring")

PEER_DOMINANCE_SENTENCE = (
    "Peer overlap is the strongest evidence because it confirms reuse within the "
    "evaluated cohort; external and AI scores are supporting signals.")

_AI_CAVEAT = ("Note: AI detectors are unreliable on non-native-English and polished "
              "human text (Stanford 2023: up to 61% false-positive rate on ESL "
              "essays); treat this as a lead for human review, not proof.")

_EXPLANATION_TEMPLATE = (
    "Peer similarity: {peer}% across {n} listed source(s) ({top_summary}). "
    "External: {ext}% ({tier}). AI: {label}{ai_pct_clause}. "
    "Overlap types: {cross_modal} cross-modal, {exact_copy} exact-copy, "
    "{paraphrase} paraphrase, {template} discounted common-source.")



def compute_integrity_signals(doc: Any, peer_result: dict[str, Any],
                              external: ExternalEvidence, ai_result: AIDetectionResult,
                              cfg: Config) -> tuple[Scores, str, str]:
    """Combine the three signals for one document.

    Returns (Scores, explanation_template_string, longer_plain_english_sentence).
    `peer_result` expects keys: score(float), matches(list), multimodal(list),
    discounted_noise(int), flagged_cross_modal(bool).
    """
    peer_pct = round(100.0 * float(peer_result.get("score", 0.0)), 1)
    peer_pct = min(max(peer_pct, 0.0), 100.0)
    external_pct = round(float(external.score_pct), 1)
    ai_prob = ai_result.ai_generated_probability

    risk = verdict_from_signals(
        peer_pct=peer_pct, external_pct=external_pct,
        ai_label=ai_result.label, ai_prob=ai_prob,
        flagged_cross_modal=bool(peer_result.get("flagged_cross_modal", False)))
    scores = Scores(peer_similarity_percentage=peer_pct,
                    external_similarity_percentage=external_pct,
                    ai_generated_probability=ai_prob, risk=risk)
    explanation = build_explanation(
        cfg=cfg, peer_pct=peer_pct, listed_sources=_listed_source_lines(peer_result),
        external=external, ai_result=ai_result, peer_result=peer_result)
    human = build_human_sentence(scores, peer_result, external, ai_result)
    return scores, explanation, human


def verdict_from_signals(peer_pct: float, external_pct: float, ai_label: str,
                         ai_prob: float | None, flagged_cross_modal: bool) -> str:
    """Exact threshold arithmetic (§4.8; peer dominant, external secondary, AI third)."""
    if peer_pct >= 70 or (peer_pct >= 40 and external_pct >= 30) or \
            (peer_pct >= 40 and ai_label == "AI_LIKELY" and flagged_cross_modal):
        return "HIGH"
    if peer_pct >= 40 or (peer_pct >= 25 and external_pct >= 25) or \
            (ai_label == "AI_LIKELY" and ai_prob is not None and ai_prob >= 65):
        return "MEDIUM"
    return "LOW"


def _listed_source_lines(peer_result: dict[str, Any]) -> list[str]:
    """peer_matches entries at or above the small-source floor (0.05)."""
    return peer_result.get("listed_sources", [])


def build_explanation(cfg: Config, peer_pct: float, listed_sources: list[str],
                      external: ExternalEvidence, ai_result: AIDetectionResult,
                      peer_result: dict[str, Any]) -> str:
    """The mandatory tokens string asserted by test_scoring.py (§4.8)."""
    matches = list(peer_result.get("matches", []))
    multimodal = list(peer_result.get("multimodal", []))
    if matches:
        top = matches[0]
        top_kind = ("common-source, discounted" if top.match_type == "common_source"
                    else ("cross-modal" if (top.matched_timestamp or top.query_timestamp)
                          else "overlap"))
        top_summary = f"top: {top.matched_file} @ {top.similarity_score:.2f}; {top_kind}"
    else:
        top_summary = "no substantive peer overlap"
    counts = count_match_types(matches, multimodal)
    ai_pct_clause = (""
                     if ai_result.ai_generated_probability is None
                     else f" at {ai_result.ai_generated_probability:.1f}%")
    caveat_clause = ""
    if ((ai_result.ai_generated_probability or 0.0) >= cfg.ai.band_unsure_max or
            ai_result.label == "UNSURE"):
        caveat_clause = "; " + _AI_CAVEAT
    source_count = len(listed_sources) if listed_sources else (1 if matches else 0)
    explanation = _EXPLANATION_TEMPLATE.format(
        peer=f"{peer_pct:.1f}", n=source_count, top_summary=top_summary,
        ext=f"{external.score_pct:.1f}", tier=external.tier_used,
        label=ai_result.label, ai_pct_clause=ai_pct_clause,
        caveat_clause=caveat_clause, **counts)
    if ai_result.ai_detector_source == "heuristic_fallback":
        explanation += (" AI detector ran as a documented rule-based heuristic "
                        "fallback (lower confidence).")
    return explanation


def count_match_types(matches: list[Any], multimodal: list[Any]) -> dict[str, int]:
    """Deterministic overlap-type counts for the explanation tail.

    exact-copy = verbatim passages (token-LCS ratio >= 0.90); paraphrase = the
    remaining same-modality overlaps; common-source = discounted template text.
    """
    multimodal_ids = {id(m) for m in multimodal}
    text_matches = [m for m in matches if id(m) not in multimodal_ids]
    exact = sum(1 for m in text_matches if m.overlap_kind == "verbatim")
    common = sum(1 for m in text_matches if m.match_type == "common_source")
    paraphrase = sum(1 for m in text_matches if m.overlap_kind == "paraphrase")
    cross_modal = len(multimodal)
    return {"cross_modal": cross_modal, "exact_copy": exact,
            "paraphrase": paraphrase, "template": common}


def build_human_sentence(scores: Scores, peer_result: dict[str, Any],
                         external: ExternalEvidence, ai_result: AIDetectionResult) \
        -> str:
    """Longer plain-English verdict paragraph for the Markdown report."""
    matches = list(peer_result.get("matches", []))
    top = matches[0] if matches else None
    parts: list[str] = []
    if top is not None:
        kind = ("shared institutional text (discounted)"
                if top.match_type == "common_source"
                else f"overlap with {top.matched_file}")
        parts.append(
            f"{scores.peer_similarity_percentage:.1f}% of peer similarity comes from "
            f"{kind} (chunk cosine {top.similarity_score:.2f}).")
    else:
        parts.append("No substantive peer overlap was found in the corpus.")
    tier_name = {"local_kb": "the local reference knowledge base", "web": "the web tier",
                 "mock": "an offline mock", "none": "no source"}.get(external.tier_used,
                                                                    external.tier_used)
    if external.hits:
        first = external.hits[0]
        parts.append(
            f"External/online similarity {scores.external_similarity_percentage:.1f}% "
            f"via {tier_name} (best match: {first.source_title}).")
    else:
        parts.append(f"External/online similarity {scores.external_similarity_percentage:.1f}% "
                     f"(no source above threshold).")
    ai_clause = AI_LABEL_SENTENCES.get(ai_result.label, "")
    if scores.ai_generated_probability is not None:
        parts.append(
            f"AI-generated estimate {scores.ai_generated_probability:.1f}% ({ai_clause}).")
    else:
        parts.append(f"AI-generated estimate unavailable ({ai_clause}).")
    ai_sentences = " ".join(parts)
    if ai_result.label in ("AI_LIKELY", "UNSURE"):
        ai_sentences += f" {_AI_CAVEAT}"
    if scores.risk == "HIGH":
        ai_sentences += f" {PEER_DOMINANCE_SENTENCE}"
    return ai_sentences


AI_LABEL_SENTENCES: dict[str, str] = {
    "HUMAN": "consistent with human-style variance",
    "UNSURE": "in the uncertain band; treat as a review lead",
    "AI_LIKELY": "flagged as likely machine-generated",
    "INSUFFICIENT_TEXT": "too little text for reliable detection",
}
