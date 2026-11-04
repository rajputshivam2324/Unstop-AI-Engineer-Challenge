"""<MODULE PURPOSE>
Report writers (§4.9): per-document JSON (top-level field names frozen to the
assignment sample, followed by documented extension fields), per-document
Markdown (verdict card + side-by-side evidence with page / timestamp markers,
built for a <30-second human check), and the aggregate run summary (JSON +
Markdown) with the document x document peer-similarity matrix (Scenario 5).

<PUBLIC API>
build_report_dict, match_to_json, write_json_report, write_markdown_report,
write_summary_report, write_summary_markdown, ENGINE_VERSION

<DEPENDS ON>
json, pathlib, types, common.
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
_MD_PASSAGE_CHARS = 400  # Markdown preview length; JSON always holds the full passage
_RISK_ICON = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}


def _location_for(match: MatchedPassage) -> dict[str, Any]:
    """Assignment-shape location: each side carries a page OR a video timestamp.

    text->text: {query_page, matched_page}; text->mp4: {query_page,
    matched_timestamp} (the assignment sample); mp4->text: {query_timestamp,
    matched_page}; mp4->mp4: {query_timestamp, matched_timestamp}.
    """
    location: dict[str, Any] = {}
    if match.query_timestamp is not None:
        location["query_timestamp"] = match.query_timestamp
    else:
        location["query_page"] = match.query_page
    if match.matched_timestamp is not None:
        location["matched_timestamp"] = match.matched_timestamp
    else:
        location["matched_page"] = match.matched_page
    return location


def match_to_json(match: MatchedPassage) -> dict[str, Any]:
    """One match entry in the assignment's JSON shape + allowed extras."""
    return {
        "matched_file": match.matched_file,
        "similarity_score": match.similarity_score,
        "query_passage": match.query_passage,
        "matched_passage": match.matched_passage,
        "location": _location_for(match),
        "match_type": match.match_type,
        "overlap_kind": match.overlap_kind,
        "evidence_sentence_scores": match.evidence_sentence_scores,
        "lexical_jaccard": match.lexical_jaccard,
        "token_lcs_ratio": match.token_lcs_ratio,
        "note": match.note,
    }


def build_report_dict(doc: ProcessedDocument, scores: Scores,
                      peer_matches: list[MatchedPassage],
                      multimodal_matches: list[MatchedPassage],
                      external: ExternalEvidence, ai_result: AIDetectionResult,
                      template_hits: list[dict[str, Any]], warnings: list[str],
                      explanations: str, discounted_noise: int,
                      minor_overlaps: list[dict[str, Any]] | None = None,
                      pair_scores: dict[str, float] | None = None) -> dict[str, Any]:
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
        "explanations": explanations,
        # ---- extension fields (beyond the assignment sample) ----
        "status": doc.status,
        "modality": doc.modality,
        "warnings": warnings,
        "external_matches": [
            {"source_title": hit.source_title, "source_url": hit.source_url,
             "similarity": round(hit.similarity, 4), "source": hit.source,
             "query_passage": hit.query_passage, "matched_passage": hit.matched_passage}
            for hit in external.hits],
        "external_tier_used": external.tier_used,
        "ai_analysis": {
            "ai_generated_probability": ai_result.ai_generated_probability,
            "label": ai_result.label,
            "p_classifier": ai_result.p_classifier,
            "p_perplexity": ai_result.p_perplexity,
            "p_burstiness": ai_result.p_burstiness,
            "perplexity": ai_result.detail.get("perplexity"),
            "burstiness": ai_result.detail.get("burstiness"),
            "length_gate_triggered": ai_result.length_gate_triggered,
            "ai_detector_source": ai_result.ai_detector_source,
            "caveat": ai_result.caveat,
        },
        "template_registry_hits": template_hits,
        "minor_peer_overlaps": minor_overlaps or [],
        "doc_pair_scores": pair_scores or {},
        "discounted_noise_matches": discounted_noise,
        "provenance": {
            "size_bytes": doc.size_bytes, "sha256": doc.sha256,
            "page_count": doc.metadata.get("page_count"),
            "duration_s": doc.metadata.get("duration_s"),
            "asr_model": doc.metadata.get("asr_model"),
            "asr_language": doc.metadata.get("asr_language"),
            "chunk_count": doc.metadata.get("chunk_count", 0),
            "duplicate_of": doc.metadata.get("duplicate_of"),
        },
        "engine_version": ENGINE_VERSION,
        "generated_at_utc": utc_now_iso(),
    }


def write_json_report(report: dict[str, Any], out_dir: Path) -> Path:
    """per_doc/<document_id>.json (created on demand)."""
    per_doc = out_dir / "per_doc"
    per_doc.mkdir(parents=True, exist_ok=True)
    path = per_doc / f"{report['document_id']}.json"  # full name: a.pdf/a.docx never collide
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _side(page: int | None, timestamp: str | None, modality_hint: str) -> str:
    if timestamp is not None:
        return f"⏱ {timestamp}"
    if page is not None:
        return f"📄 p{page}"
    return f"*page n/a ({modality_hint})*"


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MD_PASSAGE_CHARS else text[:_MD_PASSAGE_CHARS] + "…"


def _evidence_bullets(entries: list[dict[str, Any]], doc_id: str) -> list[str]:
    bullets: list[str] = []
    for index, entry in enumerate(entries, start=1):
        loc = entry["location"]
        query_side = _side(loc.get("query_page"), loc.get("query_timestamp"),
                           Path(doc_id).suffix.lstrip("."))
        source_side = _side(loc.get("matched_page"), loc.get("matched_timestamp"),
                            Path(entry["matched_file"]).suffix.lstrip("."))
        sentence_scores = entry.get("evidence_sentence_scores") or []
        sentence_note = (f" · aligned sentences: {len(sentence_scores)} "
                         f"(min cos {min(sentence_scores):.2f})" if sentence_scores else "")
        bullets.append(
            f"{index}. **→ {entry['matched_file']}** · chunk cosine "
            f"**{entry['similarity_score']:.2f}** · this doc {query_side} → "
            f"source {source_side}{sentence_note}")
        bullets.append(f"   - **this doc:** \"{_clip(entry['query_passage'])}\"")
        bullets.append(f"   - **source:** \"{_clip(entry['matched_passage'])}\"")
        if entry.get("note"):
            bullets.append(f"   - *{entry['note']}*")
    return bullets


def write_markdown_report(report: dict[str, Any], human_sentence: str,
                          out_dir: Path) -> Path:
    """Verdict-card Markdown targeted at the <30-second verify bar (§4.9)."""
    per_doc = out_dir / "per_doc"
    per_doc.mkdir(parents=True, exist_ok=True)
    risk = report["total_integrity_risk"]
    scores = report["scores"]
    ai = report["ai_analysis"]
    ai_pct = scores["ai_generated_probability"]
    doc_id = report["document_id"]
    lines: list[str] = [
        f"# Submission Integrity Report: `{doc_id}`",
        "",
        "| Risk | Peer % | External % | AI % (label) | Status |",
        "|---|---|---|---|---|",
        (f"| {_RISK_ICON.get(risk, '⚪')} **{risk}** | "
         f"{scores['peer_similarity_percentage']} | "
         f"{scores['external_similarity_percentage']} | "
         f"{ai_pct if ai_pct is not None else 'n/a'} ({ai['label']}) | "
         f"{report['status']} |"),
        "",
        f"> **Verdict in one line:** {human_sentence}",
        "",
        f"<sub>Machine explanation: {report['explanations']}</sub>",
        "",
    ]
    overlaps = [m for m in report["peer_matches"] if m["match_type"] == "overlap"]
    common = [m for m in report["peer_matches"] if m["match_type"] == "common_source"]
    lines.append(f"## Peer matches ({len(overlaps)})")
    lines.extend(_evidence_bullets(overlaps, doc_id) or ["*none*"])
    lines += ["", f"## Cross-modal matches (video ↔ document) "
                  f"({len(report['multimodal_matches'])})"]
    lines.extend(_evidence_bullets(report["multimodal_matches"], doc_id) or ["*none*"])
    if common:
        lines += ["", f"## Shared template / boilerplate (discounted, {len(common)})",
                  "These blocks recur in ≥3 submissions (e.g. the case prompt). They are "
                  "shown for transparency but excluded from the peer %."]
        lines.extend(_evidence_bullets(common, doc_id))
    ext = report["external_matches"]
    lines += ["", f"## External reference matches ({len(ext)}) — tier: "
                  f"`{report['external_tier_used']}`"]
    for index, hit in enumerate(ext, start=1):
        lines.append(f"{index}. **{hit['source_title']}** "
                     f"({hit['source_url'] or 'local reference KB'}) · cosine "
                     f"**{hit['similarity']:.2f}**")
        lines.append(f"   - **this doc:** \"{_clip(hit['query_passage'])}\"")
        lines.append(f"   - **reference:** \"{_clip(hit['matched_passage'])}\"")
    if not ext:
        lines.append("*none*")
    if report["minor_peer_overlaps"]:
        lines += ["", "## Weak topical overlaps (listed only, not evidence)"]
        for entry in report["minor_peer_overlaps"]:
            lines.append(f"- {entry['matched_file']}: doc-pair score "
                         f"{entry['pair_score']:.2f}")
    lines += ["", "## AI-generated text analysis",
              f"- Label **{ai['label']}** · fused score "
              f"{ai['ai_generated_probability'] if ai['ai_generated_probability'] is not None else 'n/a'}"
              f"/100 · detector source `{ai['ai_detector_source']}`",
              f"- Components: classifier {ai['p_classifier']} · perplexity signal "
              f"{ai['p_perplexity']} (raw ppl {ai['perplexity']}) · burstiness signal "
              f"{ai['p_burstiness']} (raw {ai['burstiness']})"]
    if ai["length_gate_triggered"]:
        lines.append("- Length gate triggered: too little text for a reliable estimate.")
    if ai["caveat"]:
        lines.append(f"- ⚠️ {ai['caveat']}")
    lines += ["", "## Warnings"]
    lines.extend([f"- {w}" for w in report["warnings"]] or ["- *none*"])
    if report["discounted_noise_matches"]:
        lines.append(f"- {report['discounted_noise_matches']} tiny match(es) below the "
                     "evidence-length noise floor were counted but not listed.")
    prov = report["provenance"]
    lines += ["", "## File provenance",
              f"- size {prov['size_bytes']} bytes · sha256 `{prov['sha256'][:16] or 'n/a'}…`"
              f" · chunks {prov['chunk_count']}"
              + (f" · pages {prov['page_count']}" if prov.get("page_count") else "")
              + (f" · duration {prov['duration_s']:.1f}s · ASR whisper-{prov['asr_model']}"
                 if prov.get("duration_s") and prov.get("asr_model") else ""),
              ""]
    path = per_doc / f"{doc_id}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_summary_report(entries: list[dict[str, Any]], out_dir: Path, elapsed_s: float,
                         pair_matrix: dict[str, dict[str, float]] | None = None,
                         runtime: dict[str, Any] | None = None) -> Path:
    """summary/report_{ts}.json + summary/latest.json (FN5)."""
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    counts = {"high": 0, "medium": 0, "low": 0, "degraded": 0}
    for entry in entries:
        counts[str(entry["risk"]).lower()] += 1
        if entry.get("status") != "ok":
            counts["degraded"] += 1
    payload = {
        "generated_at_utc": utc_now_iso(),
        "elapsed_s": round(elapsed_s, 1),
        "engine_version": ENGINE_VERSION,
        "documents": entries,
        "aggregate": {"n_docs": len(entries), **counts},
        "peer_similarity_matrix": pair_matrix or {},
        "runtime": runtime or {},
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    path = summary_dir / f"report_{utc_now_compact()}.json"
    path.write_text(text, encoding="utf-8")
    (summary_dir / "latest.json").write_text(text, encoding="utf-8")
    logger.info("summary written: %s", path)
    return path


def write_summary_markdown(entries: list[dict[str, Any]],
                           pair_matrix: dict[str, dict[str, float]],
                           runtime: dict[str, Any], out_dir: Path) -> Path:
    """Human-readable batch overview: verdict table + doc x doc similarity matrix."""
    lines = ["# Batch integrity summary", "",
             f"{len(entries)} documents · elapsed {runtime.get('elapsed_s')} s · "
             f"peak RSS {runtime.get('peak_rss_mb')} MiB", "",
             "| # | Document | Status | Risk | Peer % | External % | AI % (label) | Top match |",
             "|---|---|---|---|---|---|---|---|"]
    for i, e in enumerate(entries, start=1):
        top = e.get("top_match") or {}
        top_txt = (f"{top['matched_file']} ({top['pair_score']:.2f})" if top else "—")
        ai = e["ai"] if e["ai"] is not None else "n/a"
        lines.append(f"| {i} | `{e['document_id']}` | {e['status']} | "
                     f"{_RISK_ICON.get(e['risk'], '')} {e['risk']} | {e['peer']} | "
                     f"{e['external']} | {ai} ({e['ai_label']}) | {top_txt} |")
    ids = [e["document_id"] for e in entries]
    lines += ["", "## Peer similarity matrix (doc-pair score, row = query document)", "",
              "Columns use the two-digit file prefix. Template chunks are excluded; "
              "blank = no chunk pair above the cosine floor.", "",
              "| query \\ source | " + " | ".join(i[:2] for i in ids) + " |",
              "|---" * (len(ids) + 1) + "|"]
    for q in ids:
        cells = []
        for m in ids:
            if m == q:
                cells.append("—")
                continue
            value = pair_matrix.get(q, {}).get(m, 0.0)
            cells.append(f"**{value:.2f}**" if value >= 0.70 else
                         (f"{value:.2f}" if value > 0 else ""))
        lines.append(f"| `{q[:2]}` {q[3:24]} | " + " | ".join(cells) + " |")
    lines += ["", "## Stage timings (s)", "",
              "| " + " | ".join(runtime.get("stage_seconds", {})) + " |",
              "|---" * len(runtime.get("stage_seconds", {})) + "|",
              "| " + " | ".join(str(v) for v in runtime.get("stage_seconds", {}).values())
              + " |", ""]
    path = out_dir / "summary" / "SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
