"""<MODULE PURPOSE>
Central configuration: frozen dataclasses holding every threshold and constant used
anywhere in the engine, plus TOML loading/merging from config/default.toml (or a
user-supplied override). No other module may hardcode a tunable constant.

<PUBLIC API>
ChunkingConfig, EmbeddingConfig, PairwiseConfig, AlignmentConfig, AIDetectionConfig,
ExternalCheckConfig, RuntimeConfig, Config, load_config

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
    mp4_template_voice_a: str = "en+f2"  # dataset-only voices live here too (§4.10)
    mp4_template_voice_b: str = "en+M2"


@dataclass(frozen=True)
class AIDetectionConfig:
    """2-component ensemble detector constants (§4.6, de-scope rule R3)."""

    classifier_model: str = "wasitaigeneratedcom/ai-text-detector-small"
    gpt2_model: str = "openai-community/gpt2"
    perplexity_mu: float = 22.0  # p_perp = 1/(1+exp((perplexity - mu)/k))
    perplexity_k: float = 9.5
    burst_mu: float = 0.55  # p_burst = 1/(1+exp((burstiness - mu)/k))
    burst_k: float = 0.14
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
    ai: AIDetectionConfig = field(default_factory=AIDetectionConfig)
    external: ExternalCheckConfig = field(default_factory=ExternalCheckConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


_SECTIONS: dict[str, type] = {
    "chunking": ChunkingConfig,
    "embedding": EmbeddingConfig,
    "pairwise": PairwiseConfig,
    "alignment": AlignmentConfig,
    "ai": AIDetectionConfig,
    "external": ExternalCheckConfig,
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
