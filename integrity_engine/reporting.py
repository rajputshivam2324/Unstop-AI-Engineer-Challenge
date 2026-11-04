"""<MODULE PURPOSE>
Report writers (§4.9): per-document JSON (field names frozen to match the
assignment sample + allowed extras), per-document Markdown (verdict card +
side-by-side evidence with page/timestamp markers), and the aggregate summary.

<PUBLIC API>
write_json_report, write_markdown_report, write_summary_report, build_report_dict

<DEPENDS ON>
json, pathlib, types, config.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from integrity_engine.common import utc_now_compact, utc_now_iso
from integrity_engine.types import (AIDetectionResult, ExternalEvidence,
                                    MatchedPassage, ProcessedDocument, Scores)

logger = logging.getLogger("integrity_engine.reporting")

ENGINE_VERSION = "1.0.0"


def _location_for(match: MatchedPassage) -> dict[str, Any]:
    """Assignment-shape location: pages for text pairs, matched_timestamp for cross-modal."""
    if match.matched_timestamp is not None:
        return {"query_page": match.query_page, "matched_timestamp": match.matched_timestamp}
    return {"query_page": match.query_page, "matched_page": match.matched_page}


def match_to_json(match: MatchedPassage) -> dict[str, Any]:
    """One match entry in the assignment's JSON shape + allowed extras."""
    return {
        "matched_file": match.matched_file,
        "similarity_score": match.similarity_score,
        "query_passage": match.query_passage,
        "matched_passage": match.matched_passage,
        "location": _location_for(match),
        "match_type": match.match_type,
        "evidence_sentence_scores": match.evidence_sentence_scores,
        "note": match.note,
    }


def build_report_dict(doc: ProcessedDocument, scores: Scores,
                      peer_matches: list[MatchedPassage],
                      multimodal_matches: list[MatchedPassage],
                      external: ExternalEvidence,
                      ai_result: AIDetectionResult,
                      template_hits: list[dict[str, Any]],
                      warnings: list[str], explanations: str,
                      discounted_noise: int) -> dict[str, Any]:
    """Frozen top-level field order: assignment sample keys, then extras."""
    return {
        "document_id": doc.document_id,
        "total_integrity_risk": scores.risk,
        "scores": {
            "peer_similarity_percentage": scores.peer_similarity_percentage,
            "external_similarity_percentage": scores.external_similarity_percentage,
            "ai_generated_probability": scores.ai_generated_probability,
        },
        "peer_matches": [match_to_json(m) for m in peer_matches],
        "multimodal_matches": [match_to_json(m) for m in multimodal_matches],
        "external_matches": [
            {"source_title": hit.source_title, "source_url": hit.source_url,
             "similarity": round(hit.similarity, 4),
             "query_passage": hit.query_passage,
             "matched_passage": hit.matched_passage}
            for hit in external.hits
        ],
        "ai_analysis": {
            "ai_generated_probability": ai_result.ai_generated_probability,
            "label": ai_result.label,
            "p_perplexity": ai_result.p_perplexity,
            "p_burstiness": ai_result.p_burstiness,
            "p_classifier": ai_result.p_classifier,
            "length_gate_triggered": ai_result.length_gate_triggered,
            "ai_detector_source": ai_result.ai_detector_source,
            "caveat": ai_result.caveat,
        },
        "template_registry_hits": template_hits,
        "warnings": warnings,
        "status": doc.status,
        "discounted_noise_matches": discounted_noise,
        "explanations": explanations,
        "engine_version": ENGINE_VERSION,
        "generated_at_utc": utc_now_iso(),
    }


def _page_marker(page: int | None) -> str:
    return f"page {page}" if page is not None else "page *n/a*"


def write_json_report(vehicle: dict[str, Any], out_dir: Path) -> Path:
    """Dump one report JSON under per_doc/ (created on demand)."""
    per_doc = out_dir / "per_doc"
    per_doc.mkdir(parents=True, exist_ok=True)
    path = per_doc / f"{vehicle['document_id']}.json"
    path.write_text(json.dumps(vehicle, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def _evidence_bullets(matches: list[MatchedPassage]) -> list[str]:
    bullets: list[str] = []
    for index, match in enumerate(matches, start=1):
        query_side = (f"📄 this doc {_page_marker(match.query_page)}"
                      if match.matched_timestamp is None else "⏱ this doc (mp4)")
        if match.matched_timestamp is not None:
            source_side = (f"⏱ {match.matched_file} {match.matched_timestamp}")
        else:
            source_side = f"📄 {match.matched_file} {_page_marker(match.matched_page)}"
        bullets.append(
            f"{index}. **matched → {match.matched_file}** · chunk cosine "
            f"**{match.similarity_score:.2f}** · {query_side} → {source_side}\n"
            f"   > query:   \"{match.query_passage[:300]}{'…' if len(match.query_passage) > 300 else ''}\"\n"
            f"   > matched: \"{match.matched_passage[:300]}{'…' if len(match.matched_passage) > 300 else ''}\"\n"
            + (f"   *{match.note}*\n" if match.note else ""))
    return bullets


def write_markdown_report(vehicle: dict[str, Any], human_sentence: str,
                          out_dir: Path) -> Path:
    """Verdict-card Markdown targeted at the <30-second verify bar (§4.9)."""
    per_doc = out_dir / "per_doc"
    per_doc.mkdir(parents=True, exist_ok=True)
    mention_risk = vehicle["total_integrity_risk"]
    ring = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(mention_risk, "⚪")
    scores = vehicle["scores"]
    ai_analysis = vehicle["ai_analysis"]
    lines: list[str] = [
        f"# Submission Integrity Report: {vehicle['document_id']}",
        "",
        "| Risk | Peer% | External% | AI% | Status |",
        "|---|---|---|---|---|",
        (f"| {ring} {mention_risk} | {scores['peer_similarity_percentage']} | "
         f"{scores['external_similarity_percentage']} | "
         f"{scores['ai_generated_probability'] if scores['ai_generated_probability'] is not None else 'n/a'} "
         f"| {vehicle['status']} |"),
        "",
        f"> **Why:** {vehicle['explanations']}",
        "",
    ]
    peer = [match_to_json(m) for m in _decode_matches(vehicle, "peer_matches")]
    lines.append(f"## Peer Matches ({len(peer)})")
    lines.extend(_evidence_bullets(_decode_matches(vehicle, "peer_matches")))
    if not peer:
        lines.append("1. *none*")
    cross = _decode_matches(vehicle, "multimodal_matches")
    lines += ["", f"## Cross-Modal Matches ({len(cross)})"]
    lines.extend(_evidence_bullets(cross))
    if not cross:
        lines.append("1. *none*")
    ext = vehicle["external_matches"]
    lines += ["", f"## External Reference Matches ({len(ext)})"]
    for index, hit in enumerate(ext, start=1):
        lines.append(
            f"{index}. **{hit['source_title']}** · similarity "
            f"**{hit['similarity']:.2f}** · {hit['source_url'] or 'local KB'}\n"
            f"   > query:   \"{hit['query_passage'][:300]}\"\n"
            f"   > matched: \"{hit['matched_passage'][:300]}\"")
    if not ext:
        lines.append("1. *none*")
    lines += ["", "## AI Analysis"]
    ai_line = (
        f"- Label: **{ai_analysis['label']}** "
        f"({ai_analysis['ai_generated_probability'] if ai_analysis['ai_generated_probability'] is not None else 'n/a'}/100)"
        f" · perplexity signal {ai_analysis['p_perplexity']}"
        f" · burstiness signal {ai_analysis['p_burstiness']}"
        f" · classifier {ai_analysis['p_classifier']}"
        f" · detector source: `{ai_analysis['ai_detector_source']}`"
        + (" · length gate triggered" if ai_analysis["length_gate_triggered"] else ""))
    lines.append(ai_line)
    lines.append(f"- Caveat: {ai_analysis['caveat'] or '—'}")
    lines += ["", "## Warnings"]
    warnings = vehicle["warnings"]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- *none*")
    if vehicle["discounted_noise_matches"]:
        lines.append(
            f"- ℹ️ {vehicle['discounted_noise_matches']} tiny match(es) below the "
            f"{60}-char noise floor were counted but not reported.")
    if vehicle["template_registry_hits"]:
        lines += ["", "## Discounted Common-Source Blocks"]
        for hit in vehicle["template_registry_hits"]:
            lines.append(
                f"- `{hit['template_cluster_id']}` chunks {hit['chunk_ids']}: "
                f"{hit['note']}")
    lines += ["", f"*(explanation paragraph)* {human_sentence}", ""]
    path = out_dir / "per_doc" / f"{vehicle['document_id']}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _decode_matches(vehicle: dict[str, Any], key: str) -> list[MatchedPassage]:
    out: list[MatchedPassage] = []
    for entry in vehicle[key]:
        location = entry["location"]
        out.append(MatchedPassage(
            matched_file=entry["matched_file"],
            similarity_score=entry["similarity_score"],
            query_passage=entry["query_passage"],
            matched_passage=entry["matched_passage"],
            query_page=location.get("query_page"),
            matched_page=location.get("matched_page"),
            matched_timestamp=location.get("matched_timestamp"),
            match_type=entry.get("match_type", "overlap"),
            evidence_sentence_scores=entry.get("evidence_sentence_scores", []),
            note=entry.get("note"),
        ))
    return out


def write_summary_report(entries: list[dict[str, Any]], out_dir: Path,
                         elapsed_s: float) -> Path:
    """Aggregate matrix + one-line verdicts (Scenario 5). summary/report_{ts}.json
    plus a stable copy at summary/latest.json (FN5)."""
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    counts = {"high": 0, "medium": 0, "low": 0, "degraded": 0}
    for entry in entries:
        counts[str(entry["risk"]).lower()] = counts.get(str(entry["risk"]).lower(), 0) + 1
        if entry.get("status") != "ok":
            counts["degraded"] += 1
    aggregate = {"n_docs": len(entries), **counts}
    payload = {
        "generated_at_utc": utc_now_iso(),
        "elapsed_s": round(elapsed_s, 1),
        "documents": entries,
        "aggregate": aggregate,
    }
    stamp = utc_now_compact()
    path = summary_dir / f"report_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = summary_dir / "latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("summary written: %s", path)
    return path
