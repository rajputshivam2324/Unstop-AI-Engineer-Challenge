"""<MODULE PURPOSE>
Command-line entry point and pipeline orchestration (spec §3, §6):

  S0 args/config/logging -> S1 ingest every file (never aborts the batch)
  -> S2 chunk -> S3 template registry + embeddings + FAISS neighbours
  -> S4 doc-pair aggregation -> S5 evidence alignment, external KB, AI detection
  -> S6 scoring -> S7 JSON/MD reports + aggregate summary -> S8 optional --eval.

Exit codes: 0 run completed; 1 fatal (bad input dir, no supported files, bad
config, core embedding model unavailable); 2 argparse usage error.
stdout carries exactly one JSON line (the run summary); narrative goes to stderr.

<PUBLIC API> main(argv) -> int, run_pipeline(...)
<DEPENDS ON> every engine module; only this module wires them together.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "cache" / "hf"
# Model caches must be pinned BEFORE transformers / sentence-transformers import.
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "cache"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from integrity_engine.common import (
    get_trimmed_logger,
    list_input_files,
    utc_now_compact,
)
from integrity_engine.config import Config, load_config

logger = logging.getLogger("integrity_engine.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_integrity_check.py",
        description="Multimodal submission integrity & plagiarism detection engine "
                    "(PDF / DOCX / MP4 -> peer, external and AI-generated signals with "
                    "explainable JSON + Markdown audit reports).")
    parser.add_argument("--input_dir", required=True,
                        help="folder of candidate files (.pdf/.docx/.mp4; non-recursive)")
    parser.add_argument("--output_dir", required=True, help="where reports are written")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "default.toml"),
                        help="TOML threshold overrides (default: config/default.toml)")
    parser.add_argument("--eval", metavar="MANIFEST", default=None,
                        help="after the run, score reports against this ground-truth "
                             "manifest and write EVALUATION.md")
    parser.add_argument("--eval-out", default=str(REPO_ROOT / "EVALUATION.md"),
                        help="EVALUATION.md destination (default: repo root)")
    parser.add_argument("--models-only", action="store_true",
                        help="download/pre-cache all models, print sizes, exit")
    parser.add_argument("--generate-dataset", action="store_true",
                        help="regenerate the 14-file synthetic corpus + manifest into "
                             "--input_dir, then exit")
    parser.add_argument("--skip-web", action="store_true",
                        help="disable the optional DuckDuckGo tier (local KB still runs)")
    parser.add_argument("--matching-mode", default=None,
                        choices=["baseline", "boundary", "semantic_sts",
                                 "semantic_nli", "semantic_hybrid", "evidence_v2"],
                        help="override matching mode; original baseline remains available")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--seed", type=int, default=42,
                        help="dataset-generation seed (the engine itself has no RNG)")
    return parser


def peak_rss_mb() -> float:
    """Peak RSS of this process in MiB (Linux ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# --------------------------------------------------------------------------- S1
def ingest_all(paths: list[Path], cfg: Config) -> list[Any]:
    """Extract every file; any unexpected exception becomes status=error (N4)."""
    from integrity_engine.ingestion import extract_any
    from integrity_engine.ingestion.base import finish_document

    docs = []
    for path in paths:
        started = time.monotonic()
        try:
            doc = extract_any(path, cfg)
        except Exception as exc:  # noqa: BLE001 - never abort the batch
            logger.warning("ingestion: unexpected failure on %s (%s: %s)", path.name,
                           type(exc).__name__, exc)
            logger.debug("traceback:\n%s", traceback.format_exc())
            doc = finish_document(
                path, [], [f"unexpected extraction failure ({type(exc).__name__})"],
                path.suffix.lower().lstrip("."), cfg, {"_ingestion_status": "error"})
        if doc is None:
            continue
        doc.metadata["ingest_seconds"] = round(time.monotonic() - started, 2)
        logger.info("ingested %-34s status=%-12s chars=%5d (%.1fs)", doc.document_id,
                    doc.status, len(doc.full_text), doc.metadata["ingest_seconds"])
        docs.append(doc)
    return docs


def find_duplicates(docs: list[Any]) -> dict[str, str]:
    """Byte-identical files (sha256): later document_id -> first (canonical) id (A2)."""
    first_by_hash: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    for doc in docs:
        if doc.status != "ok" or not doc.sha256:
            continue
        canonical = first_by_hash.setdefault(doc.sha256, doc.document_id)
        if canonical != doc.document_id:
            duplicates[doc.document_id] = canonical
            doc.metadata["duplicate_of"] = canonical
            logger.info("dedup: %s is byte-identical to %s", doc.document_id, canonical)
    return duplicates


# ------------------------------------------------------------------ main pipeline
def run_pipeline(input_dir: Path, output_dir: Path, cfg: Config,
                 skip_web: bool = False) -> dict[str, Any]:
    """Run S1-S7 and return the run summary dict (also written to disk)."""
    from integrity_engine.ai_detection import AIDetector
    from integrity_engine.alignment import (
        DocumentAligner,
        EvidenceGate,
        TemplateRegistry,
        align_pair,
        evidence_pair_scores,
    )
    from integrity_engine.embeddings import build_embedder
    from integrity_engine.external_check import LocalReferenceKB
    from integrity_engine.indexing import (
        SelfMatchExclusion,
        build_index,
        pairwise_doc_similarity,
        search_neighbors,
    )
    from integrity_engine.ingestion.mp4 import release_asr
    from integrity_engine.reporting import (
        build_report_dict,
        write_json_report,
        write_markdown_report,
        write_summary_markdown,
        write_summary_report,
    )
    from integrity_engine.scoring import compute_integrity_signals
    from integrity_engine.text_processing import build_chunks
    from integrity_engine.types import AIDetectionResult, ExternalEvidence

    run_started = time.monotonic()
    stage_s: dict[str, float] = {}

    def mark(stage: str, since: float) -> float:
        now = time.monotonic()
        stage_s[stage] = round(now - since, 2)
        return now

    paths = list_input_files(input_dir)
    if not paths:
        raise FileNotFoundError(
            f"no supported files (.pdf/.docx/.mp4) found in {input_dir}")
    t = time.monotonic()
    docs = ingest_all(paths, cfg)
    release_asr()  # free whisper before the AI models load (RAM discipline §6)
    t = mark("ingestion", t)
    if len([d for d in docs if d.status == "ok"]) <= 1:
        logger.info("only one document in corpus; peer comparisons are vacuous")
    duplicates = find_duplicates(docs)
    by_id = {d.document_id: d for d in docs}

    # S2 chunking (only analyzable docs)
    all_chunks = []
    for doc in docs:
        if doc.status != "ok":
            continue
        chunks = build_chunks(doc, cfg)
        for chunk in chunks:
            chunk.is_duplicate = doc.document_id in duplicates
        all_chunks.extend(chunks)
        doc.metadata["chunk_count"] = len(chunks)
    chunks_by_id = {c.chunk_id: c for c in all_chunks}
    logger.info("chunking: %d chunks from %d analyzable documents", len(all_chunks),
                sum(1 for d in docs if d.status == "ok"))

    # S3 template registry (pure text pass) + embeddings + FAISS
    registry = TemplateRegistry(
        min_docs=cfg.alignment.template_min_docs,
        hamming_max=cfg.alignment.template_hamming_max,
        min_fingerprint_tokens=cfg.alignment.template_min_tokens,
        verbatim_template_min_tokens=cfg.alignment.verbatim_template_min_tokens)
    registry.register_all_chunks(all_chunks)
    t = mark("chunking_and_templates", t)

    try:
        embedder = build_embedder(cfg.embedding, str(CACHE_DIR))
    except Exception as exc:
        logger.error("core model unavailable (offline and not cached): %s (%s)",
                     cfg.embedding.model_name, exc)
        raise RuntimeError("embedding model unavailable") from exc
    matrix = embedder.encode_corpus([c.text for c in all_chunks])
    for chunk, vec in zip(all_chunks, matrix, strict=True):
        chunk.embedding = vec
    index = build_index(matrix)
    exclude = SelfMatchExclusion([c.document_id for c in all_chunks], duplicates)
    chunk_sims: dict[int, list[tuple[int, float]]] = {}
    for i in range(len(all_chunks)):
        chunk_sims[i] = search_neighbors(index, matrix[i:i + 1],
                                         cfg.pairwise.chunk_query_topk, exclude, i)
    t = mark("embedding_and_search", t)

    # S4 doc-pair aggregation
    gate = EvidenceGate(all_chunks, embedder, cfg)
    chunk_pairs = pairwise_doc_similarity(all_chunks, chunk_sims, cfg.pairwise, duplicates,
                                          verify=gate)
    # S4b sentence-level evidence: candidate pairs from the chunk sieve + a sentence
    # FAISS index, scored by aligned-evidence coverage (A1)
    if cfg.matching.mode == "baseline":
        aligner = DocumentAligner(embedder, registry, cfg, all_chunks)
    elif cfg.matching.mode == "evidence_v2":
        from integrity_engine.evidence_v2 import EvidenceV2Aligner

        aligner = EvidenceV2Aligner(embedder, registry, cfg, all_chunks)
    else:
        from integrity_engine.enhanced_alignment import EnhancedDocumentAligner

        aligner = EnhancedDocumentAligner(embedder, registry, cfg, all_chunks)
    pairs = evidence_pair_scores([d for d in docs if d.status == "ok"], aligner,
                                 chunk_pairs, duplicates, cfg.pairwise.chunk_query_topk)
    pairs_by_query: dict[str, list[Any]] = {}
    for pair in pairs:
        pairs_by_query.setdefault(pair.query_document_id, []).append(pair)
    t = mark("pair_scoring", t)

    # S5a evidence passages per document
    peer_results: dict[str, dict[str, Any]] = {}
    for doc in docs:
        peer_results[doc.document_id] = _peer_evidence(
            doc, pairs_by_query.get(doc.document_id, []), all_chunks, chunks_by_id,
            by_id, embedder, registry, cfg, align_pair, aligner)
    t = mark("alignment", t)
    if hasattr(aligner, "release"):
        aligner.release()  # verifier must not remain resident during AI detection

    # S5b external check (local KB always; web tier optional)
    kb = LocalReferenceKB(embedder, cfg, skip_web=skip_web)
    kb.build()
    externals: dict[str, ExternalEvidence] = {}
    for doc in docs:
        if doc.status == "ok":
            externals[doc.document_id] = kb.search_doc(doc)
        else:
            externals[doc.document_id] = ExternalEvidence(score_pct=0.0)
    t = mark("external_check", t)
    embedder.release()

    # S5c AI detection (batch; GPT-2 then classifier, each released after use)
    detector = AIDetector(cfg.ai, CACHE_DIR)
    ai_results = detector.detect_many(
        {d.document_id: d.full_text for d in docs if d.status == "ok"})
    for doc in docs:
        if doc.document_id not in ai_results:
            ai_results[doc.document_id] = AIDetectionResult(
                ai_generated_probability=None, label="INSUFFICIENT_TEXT",
                length_gate_triggered=True, ai_detector_source="model",
                detail={"reason": f"status={doc.status}; no analyzable text"})
    t = mark("ai_detection", t)

    # S6 + S7 scoring and reports
    per_doc_dir_entries: list[dict[str, Any]] = []
    for doc in docs:
        peer = peer_results[doc.document_id]
        ai_result = ai_results[doc.document_id]
        scores, explanation, human = compute_integrity_signals(
            doc, peer, externals[doc.document_id], ai_result, cfg)
        warnings = list(doc.warnings)
        if doc.status != "ok":
            explanation = (f"Document could not be analyzed (status={doc.status}): "
                           f"{'; '.join(doc.warnings) or 'no usable content'}. "
                           "It is reported for reviewer follow-up, not scored as "
                           "plagiarism. " + explanation)
        report = build_report_dict(
            doc, scores, peer["peer_matches"], peer["multimodal"],
            externals[doc.document_id], ai_result, peer["template_hits"], warnings,
            explanation, peer["discounted_noise"], peer["minor"], peer["pair_scores"])
        write_json_report(report, output_dir)
        write_markdown_report(report, human, output_dir)
        per_doc_dir_entries.append({
            "document_id": doc.document_id, "modality": doc.modality,
            "status": doc.status, "risk": scores.risk,
            "peer": scores.peer_similarity_percentage,
            "external": scores.external_similarity_percentage,
            "ai_label": ai_result.label, "ai": scores.ai_generated_probability,
            "ai_detector_source": ai_result.ai_detector_source,
            "top_match": peer["top_match"], "warnings": len(warnings)})
    t = mark("scoring_and_reports", t)

    elapsed = time.monotonic() - run_started
    doc_ids = [d.document_id for d in docs]
    pair_matrix = {q: {m: 0.0 for m in doc_ids if m != q} for q in doc_ids}
    for pair in pairs:
        pair_matrix[pair.query_document_id][pair.match_document_id] = round(pair.score, 4)
    runtime = {"elapsed_s": round(elapsed, 2), "stage_seconds": stage_s,
               "peak_rss_mb": round(peak_rss_mb(), 1),
               "n_chunks": len(all_chunks), "cpu_count": os.cpu_count(),
               "ai_detector_missing_models": detector.missing}
    summary_path = write_summary_report(per_doc_dir_entries, output_dir, elapsed,
                                        pair_matrix, runtime)
    write_summary_markdown(per_doc_dir_entries, pair_matrix, runtime, output_dir)
    return {"summary_path": str(summary_path), "documents_evaluated": len(docs),
            "warnings": sum(e["warnings"] for e in per_doc_dir_entries),
            "elapsed_s": round(elapsed, 1), "peak_rss_mb": runtime["peak_rss_mb"]}


def _peer_evidence(doc, pairs, all_chunks, chunks_by_id, by_id, embedder, registry,
                   cfg: Config, align_pair, aligner) -> dict[str, Any]:
    """Peer %, evidence passages, common-source (template) matches for one document."""
    result: dict[str, Any] = {
        "score": 0.0, "peer_matches": [], "multimodal": [], "matches": [],
        "template_hits": [], "discounted_noise": 0, "minor": [], "pair_scores": {},
        "flagged_cross_modal": False, "listed_sources": [], "top_match": None}
    if doc.status != "ok":
        return result
    floor = cfg.pairwise.min_pair_score_floor
    report_floor = floor  # every pair with accepted sentence evidence is reported
    scored = [p for p in pairs if p.score >= floor]
    result["pair_scores"] = {p.match_document_id: round(p.score, 4) for p in pairs}
    if scored:
        best = max(scored, key=lambda p: (p.score, p.match_document_id))
        result["score"] = best.score
        result["flagged_cross_modal"] = best.is_cross_modal
        result["top_match"] = {"matched_file": best.match_document_id,
                               "pair_score": round(best.score, 4)}
    overlaps, multimodal = [], []
    for pair in sorted(scored, key=lambda p: (-p.score, p.match_document_id)):
        if pair.score < report_floor:
            result["minor"].append({"matched_file": pair.match_document_id,
                                    "pair_score": round(pair.score, 4),
                                    "note": "below evidence-reporting floor "
                                            f"({report_floor:.2f}); listed only"})
            continue
        passages, noise = aligner.align(doc, by_id[pair.match_document_id], pair.score)
        result["discounted_noise"] += noise
        result["listed_sources"].append(pair.match_document_id)
        (multimodal if pair.is_cross_modal else overlaps).extend(passages)

    # Common-source (template) evidence: best template-chunk partner per other doc.
    own_templates = [c for c in all_chunks if c.document_id == doc.document_id
                     and c.is_template]
    common = []
    if own_templates:
        clusters: dict[str, list[str]] = {}
        for chunk in own_templates:
            clusters.setdefault(chunk.template_cluster_id or "tmpl", []).append(
                chunk.chunk_id)
        result["template_hits"] = [
            {"template_cluster_id": cid, "chunk_ids": ids,
             "note": f"repeated block in >={cfg.alignment.template_min_docs} documents "
                     "(discounted)"} for cid, ids in sorted(clusters.items())]
        for chunk in own_templates:
            partners: dict[str, Any] = {}
            for other in all_chunks:
                if (other.document_id == doc.document_id or not other.is_template
                        or other.template_cluster_id != chunk.template_cluster_id):
                    continue
                sim = float(chunk.embedding @ other.embedding)
                if other.document_id not in partners or sim > partners[other.document_id][1]:
                    partners[other.document_id] = (other, sim)
            for other_doc in sorted(partners):
                other, sim = partners[other_doc]
                passages, _ = align_pair(doc, by_id[other_doc],
                                         [(chunk.chunk_id, other.chunk_id, sim)],
                                         chunks_by_id, embedder, registry, cfg)
                common.extend(passages)
    result["peer_matches"] = overlaps + common
    result["multimodal"] = multimodal
    result["matches"] = sorted(overlaps + multimodal,
                               key=lambda m: -m.similarity_score) + common
    if common and not result["listed_sources"]:
        result["listed_sources"] = sorted({m.matched_file for m in common})
    return result


# ----------------------------------------------------------------------- models
def precache_models(cfg: Config) -> int:
    """--models-only: download every model into cache/hf and print sizes (stderr)."""
    from huggingface_hub import snapshot_download

    from integrity_engine.ingestion.mp4 import WHISPER_MODEL_SIZE

    repos = [
        (cfg.embedding.model_name, None),
        (cfg.ai.gpt2_model, ["config.json", "generation_config.json", "merges.txt",
                             "model.safetensors", "tokenizer.json",
                             "tokenizer_config.json", "vocab.json"]),
        (cfg.ai.classifier_model, None),
        (f"Systran/faster-whisper-{WHISPER_MODEL_SIZE}", None),
    ]
    status = 0
    for repo, allow in repos:
        try:
            path = Path(snapshot_download(repo, cache_dir=str(CACHE_DIR),
                                          allow_patterns=allow))
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            logger.info("model %-48s %8.1f MB  %s", repo, size / 1e6, path)
        except Exception as exc:  # noqa: BLE001
            logger.error("model %s could not be cached: %s", repo, exc)
            status = 1
    if cfg.matching.mode.startswith("semantic_"):
        from integrity_engine.semantic import build_verifier

        verifier = build_verifier(cfg.matching.mode.removeprefix("semantic_"), str(CACHE_DIR))
        try:
            verifier.load()
        except RuntimeError as exc:
            logger.error("%s", exc)
            status = 1
        finally:
            verifier.release()
    return status


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stamp = utc_now_compact()
    log_dir = REPO_ROOT / "logs"
    root_logger = get_trimmed_logger("integrity_engine", log_dir / f"run_{stamp}.log",
                                     args.log_level)
    current = logging.FileHandler(log_dir / "current_run.log", mode="w", encoding="utf-8")
    current.setLevel(logging.DEBUG)
    current.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger.addHandler(current)

    try:
        cfg = load_config(Path(args.config) if args.config else None)
        if args.matching_mode:
            from dataclasses import replace

            cfg = replace(cfg, matching=replace(cfg.matching, mode=args.matching_mode))
        if cfg.matching.mode not in ("baseline", "boundary", "semantic_sts",
                                     "semantic_nli", "semantic_hybrid", "evidence_v2"):
            raise ValueError(f"unknown matching mode: {cfg.matching.mode}")
    except Exception as exc:  # noqa: BLE001 - parse error => exit 1
        logger.error("config could not be loaded: %s", exc)
        return 1

    if args.models_only:
        return precache_models(cfg)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if args.generate_dataset:
        from integrity_engine.dataset_generator import generate_dataset

        generate_dataset(input_dir, seed=args.seed, cfg=cfg)
        return 0

    if not input_dir.is_dir():
        logger.error("input_dir does not exist or is not readable: %s", input_dir)
        return 1
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("output_dir not creatable: %s (%s)", output_dir, exc)
        return 1

    try:
        summary = run_pipeline(input_dir, output_dir, cfg, skip_web=args.skip_web)
    except FileNotFoundError as exc:
        logger.error("ERROR: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("fatal: %s", exc)
        return 1

    if args.eval:
        from integrity_engine.evaluation import run_evaluation

        try:
            eval_path = run_evaluation(output_dir, Path(args.eval), Path(args.eval_out), cfg)
            summary["evaluation_path"] = str(eval_path)
        except Exception as exc:  # noqa: BLE001 - eval never fails the run
            logger.error("eval failed: %s", exc)
            logger.debug("traceback:\n%s", traceback.format_exc())
    print(json.dumps(summary), file=sys.stdout, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
