"""<MODULE PURPOSE>
Single source of truth for every data structure passed between pipeline stages:
ingestion pages/segments, chunks with provenance, pair candidates, matched-passage
evidence, external-KB hits, AI-detection results, integrity scores and evaluation
summaries. All types are light dataclasses; no module logic lives here.

<PUBLIC API>
Page, ProcessedDocument, Chunk, PairCandidate, MatchedPassage, Sentence,
ExternalHit, ExternalEvidence, AIDetectionResult, Scores, EvaluationSummary

<DEPENDS ON>
typing, dataclasses, pathlib, numpy (runtime-optional embedding field only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass
class Page:
    """A provenance unit: one PDF/docx page, or one faster-whisper ASR segment (mp4)."""

    page: int | None  # 1-based for pdf; None for docx/mp4
    text: str
    char_start: int  # offset into the document's joined full_text
    char_end: int


@dataclass
class ProcessedDocument:
    """Output of an extractor for one input file."""

    document_id: str  # exact filename incl. extension, e.g. "06_llm_generated_summary.pdf"
    path: Path
    modality: Literal["pdf", "docx", "mp4"]
    status: Literal["ok", "empty", "error", "silent_video"]
    warnings: list[str] = field(default_factory=list)
    pages: list[Page] = field(default_factory=list)  # always ordered; page None allowed
    full_text: str = ""  # joined via "\n\n"; Page.char_* refer to this string
    size_bytes: int = 0
    sha256: str = ""  # dedup key (A2)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrieval unit with full location provenance."""

    chunk_id: str  # "{doc_idx:02d}:{chunk_idx:03d}"
    document_id: str
    text: str
    start_char: int
    end_char: int
    page_start: int | None = None  # inclusive; None for docx/mp4 paths
    page_end: int | None = None
    timestamp_start: float | None = None  # seconds, mp4 only
    timestamp_end: float | None = None
    sentence_ids: list[int] = field(default_factory=list)
    token_count: int = 0
    is_template: bool = False
    template_cluster_id: str | None = None
    is_duplicate: bool = False  # byte-identical duplicate of another file (A2)
    embedding: np.ndarray | None = None  # excluded from json; runtime only


@dataclass
class Sentence:
    """Sentence with char span into the owning document's joined full_text."""

    text: str
    char_start: int
    char_end: int
    page: int | None = None  # owning page for pdf; None for docx/mp4


@dataclass
class PairCandidate:
    """A scored document pair emerging from the chunk-level sieve."""

    query_document_id: str
    match_document_id: str
    score: float  # doc-level coverage-weighted mean-max cosine (peer % basis)
    shared_chunk_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    is_cross_modal: bool = False
    directionality: Literal["forward", "reverse", "both"] = "forward"


@dataclass
class MatchedPassage:
    """Human-verifiable evidence for one overlap region."""

    matched_file: str
    similarity_score: float
    query_passage: str
    matched_passage: str
    query_page: int | None = None
    matched_page: int | None = None
    matched_timestamp: str | None = None  # "00:00:24 - 00:00:36"
    match_type: Literal["overlap", "common_source"] = "overlap"
    evidence_sentence_scores: list[float] = field(default_factory=list)
    note: str | None = None


@dataclass
class ExternalHit:
    """One evidenced external-source overlap (local KB or web tier)."""

    source_title: str
    source_url: str | None  # None for local KB
    similarity: float  # cosine (local KB) or token-Jaccard (web tier)
    query_passage: str
    matched_passage: str
    source: Literal["local_kb", "web", "mock"]
    aligned_tokens: int  # query tokens covered by this hit (feeds external %)


@dataclass
class ExternalEvidence:
    """Aggregate external-check result for one document."""

    score_pct: float
    hits: list[ExternalHit] = field(default_factory=list)
    tier_used: Literal["local_kb", "web", "mock", "none"] = "none"


@dataclass
class AIDetectionResult:
    """AI-detection verdict; model and heuristic-fallback sources are never merged."""

    ai_generated_probability: float | None  # 0-100, None when length-gated or failed
    label: Literal["HUMAN", "UNSURE", "AI_LIKELY", "INSUFFICIENT_TEXT"]
    p_perplexity: float | None = None
    p_burstiness: float | None = None
    p_classifier: float | None = None
    length_gate_triggered: bool = False
    ai_detector_source: str = "model"  # "model" | "heuristic_fallback" (never blended)
    caveat: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scores:
    """The three integrity signals plus the combined verdict."""

    peer_similarity_percentage: float  # 0-100 after template/noise discount
    external_similarity_percentage: float
    ai_generated_probability: float | None
    risk: Literal["LOW", "MEDIUM", "HIGH"]


@dataclass
class EvaluationSummary:
    """Aggregated self-evaluation numbers (fed to EVALUATION.md)."""

    relation_prf: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    fpr_originals: float = 0.0
    ai_detector_fallback_count: int = 0
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    mislabeled: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    elapsed_cold_s: float | None = None
    elapsed_warm_s: float | None = None
    accuracy: float | None = None
    balanced_accuracy: float | None = None
    fallback_rows: list[dict[str, Any]] = field(default_factory=list)
    model_rows: list[dict[str, Any]] = field(default_factory=list)
