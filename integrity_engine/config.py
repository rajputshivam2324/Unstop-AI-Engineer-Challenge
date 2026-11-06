"""<MODULE PURPOSE>
Central configuration: frozen dataclasses holding every threshold and constant used
anywhere in the engine, plus TOML loading/merging from config/default.toml (or a
user-supplied override). No other module may hardcode a tunable constant.

<PUBLIC API>
ChunkingConfig, EmbeddingConfig, PairwiseConfig, AlignmentConfig, AIDetectionConfig,
ExternalCheckConfig, ScoringConfig, RuntimeConfig, Config, load_config

<DEPENDS ON>
dataclasses, pathlib, tomllib (stdlib), logging.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

logger = logging.getLogger("integrity_engine.config")


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunk-size and overlap rules (§4.2)."""

    max_chunk_tokens: int = 200
    min_chunk_tokens: int = 40
    overlap_sentences: int = 2
    mp4_target_tokens: int = 120
    mp4_max_seconds: int = 45


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model selection (the one hard-required model, A8)."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32


@dataclass(frozen=True)
class PairwiseConfig:
    """Chunk-sieve and doc-pair thresholds (§4.4)."""

    chunk_query_topk: int = 10
    chunk_min_cos: float = 0.55
    doc_flag_threshold: float = 0.70
    doc_report_floor: float = 0.40
    min_pair_score_floor: float = 0.05
    duplicate_similarity: float = 1.0


@dataclass(frozen=True)
class AlignmentConfig:
    """Template registry + sentence evidence rules (§4.5)."""

    template_min_docs: int = 3
    template_hamming_max: int = 3
    template_min_tokens: int = 40
    verbatim_template_min_tokens: int = 20
    min_evidence_chars: int = 60
    sentence_align_cos_floor: float = 0.65
    sentence_lcs_token_ratio: float = 0.60
    min_run_sentences: int = 2  # evidence = >= 2 consecutive aligned sentences ...
    single_sentence_cos: float = 0.85  # ... or one near-verbatim sentence
    single_sentence_min_tokens: int = 12
    mp4_template_voice_a: str = "en+f2"  # dataset-only voices live here too (§4.10)
    mp4_template_voice_b: str = "en+M2"


@dataclass(frozen=True)
class MatchingConfig:
    """Opt-in alternatives; baseline preserves the original engine exactly."""

    mode: str = "baseline"
    asr_max_gap_s: float = 1.5
    boundary_max_units: int = 3
    window_max_tokens: int = 120
    retrieval_min_cos: float = 0.35  # candidate generation ONLY, never acceptance
    retrieval_topk: int = 5
    verifier_topk: int = 3
    max_verifications_per_pair: int = 48
    nli_entailment_min: float = 0.70
    nli_contradiction_max: float = 0.10
    sts_min: float = 0.50  # development-only STS comparator calibration; not cosine

    def __post_init__(self) -> None:
        if self.mode not in ("baseline", "boundary", "semantic_sts",
                              "semantic_nli", "semantic_hybrid"):
            raise ValueError(f"unknown matching mode: {self.mode}")
        for name in ("boundary_max_units", "window_max_tokens", "retrieval_topk",
                     "verifier_topk", "max_verifications_per_pair"):
            if getattr(self, name) < 1:
                raise ValueError(f"matching.{name} must be positive")
        for name in ("retrieval_min_cos", "nli_entailment_min",
                     "nli_contradiction_max", "sts_min"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"matching.{name} must be in [0, 1]")
        if self.asr_max_gap_s < 0:
            raise ValueError("matching.asr_max_gap_s must be nonnegative")


@dataclass(frozen=True)
class AIDetectionConfig:
    """2-component ensemble detector constants (§4.6, de-scope rule R3)."""

    classifier_model: str = "wasitaigeneratedcom/ai-text-detector-small"
    gpt2_model: str = "openai-community/gpt2"
    # Fitted on the HC3 calibration split (benchmarks/bench_ai_detection.py);
    # spec a-priori values were 22.0/9.5 and 0.55/0.14.
    perplexity_mu: float = 13.69  # p_perp = 1/(1+exp((perplexity - mu)/k))
    perplexity_k: float = 0.86
    burst_mu: float = 0.2174  # p_burst = 1/(1+exp((burstiness - mu)/k))
    burst_k: float = 0.2486
    p_text_perp_weight: float = 0.75  # p_text = w*p_perp + (1-w)*p_burst
    p_final_model_weight: float = 0.60  # p_ai = w*p_cls + (1-w)*p_text
    band_human_max: float = 35.0  # < 35 HUMAN
    band_unsure_max: float = 65.0  # 35..<65 UNSURE, >=65 AI_LIKELY
    min_length_tokens: int = 50
    max_chars: int = 20000
    max_tokens_per_window: int = 1024


@dataclass(frozen=True)
class ExternalCheckConfig:
    """Local KB + optional DDG web tier (§4.7)."""

    local_kb_min_cos: float = 0.70
    web_budget_per_doc: int = 2
    query_topk: int = 3
    request_timeout_s: int = 6
    allow_external_web: bool = True
    web_sleep_s: float = 0.6
    web_min_query_words: int = 12
    web_jaccard_min: float = 0.35
    ddg_backend: str = "duckduckgo_fallback"


@dataclass(frozen=True)
class ScoringConfig:
    """Risk-rule thresholds in percent (§4.8). peer_medium_pct deviates from the spec
    (40 -> 20), justified by benchmarks/bench_retrieval.py (README Assumptions)."""

    peer_high_pct: float = 70.0
    peer_high_with_external_pct: float = 40.0
    external_high_pct: float = 30.0
    peer_medium_pct: float = 20.0
    peer_medium_with_external_pct: float = 10.0
    external_medium_pct: float = 25.0
    ai_medium_pct: float = 65.0


@dataclass(frozen=True)
class RuntimeConfig:
    """Process-level knobs (paths, logging, seed)."""

    log_level: str = "INFO"
    hf_home: Path | None = None
    cache_dir: Path = Path("cache")
    seed: int = 42
    max_ingest_seconds_per_file: int = 120


@dataclass(frozen=True)
class Config:
    """Aggregate config object passed through all modules."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    pairwise: PairwiseConfig = field(default_factory=PairwiseConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    ai: AIDetectionConfig = field(default_factory=AIDetectionConfig)
    external: ExternalCheckConfig = field(default_factory=ExternalCheckConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


_SECTIONS: dict[str, type] = {
    "chunking": ChunkingConfig,
    "embedding": EmbeddingConfig,
    "pairwise": PairwiseConfig,
    "alignment": AlignmentConfig,
    "matching": MatchingConfig,
    "ai": AIDetectionConfig,
    "external": ExternalCheckConfig,
    "scoring": ScoringConfig,
    "runtime": RuntimeConfig,
}

_TOML_TO_DATACLASS: dict[str, str] = {  # accepted TOML aliases (kebab-case)
    "min_length_tokens": "min_length_tokens",
    "request_timeout_s": "request_timeout_s",
    "web_budget_per_doc": "web_budget_per_doc",
}


def _override_section(section_name: str, section_obj: object, table: dict[str, object],
                      unknown_keys: list[str]) -> object:
    """Return a copy of the frozen section proto with valid TOML keys overlaid."""
    valid = {f.name for f in fields(type(section_obj))}  # type: ignore[arg-type]
    merged: dict[str, object] = {}
    for key, value in table.items():
        key_norm = _TOML_TO_DATACLASS.get(key, key)
        if key_norm in valid:
            current = getattr(section_obj, key_norm)
            if isinstance(current, Path) and isinstance(value, str):
                value = Path(value)
            merged[key_norm] = value
        else:
            unknown_keys.append(f"{section_name}.{key}")
    return replace(section_obj, **merged)


def load_config(path: Path | None) -> Config:
    """Load defaults, then overlay a TOML file section-by-section.

    Unknown TOML keys are logged as warnings and ignored (forward compatibility);
    a parse error raises and the CLI turns it into exit code 1.
    """
    cfg = Config()
    if path is None:
        return cfg
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    unknown_keys: list[str] = []
    for section_name, table in data.items():
        if not isinstance(table, dict):
            unknown_keys.append(section_name)
            continue
        if section_name not in _SECTIONS:
            unknown_keys.append(section_name)
            continue
        overridden = _override_section(
            section_name, getattr(cfg, section_name), table, unknown_keys)
        cfg = replace(cfg, **{section_name: overridden})
    if unknown_keys:
        logger.warning("config: unknown keys ignored: %s", ", ".join(unknown_keys))
    return cfg
