"""<MODULE PURPOSE>
Text normalization, deterministic sentence segmentation with abbreviation guards,
and provenance-preserving chunking for all modalities: text paths accumulate
sentences to a 200-token budget with 2-sentence overlap; MP4 path accumulates ASR
segments to ~120 tokens / 45 s with 1-segment replay overlap.

<PUBLIC API>
normalize_text, de_hyphenate_linebreaks, collapse_ws, split_sentences,
sentence_char_spans_by_pages, build_chunks, token_count, doc_clean_warnings

<DEPENDS ON>
types (Page, ProcessedDocument, Chunk, Sentence), config.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import zlib

from integrity_engine.config import Config
from integrity_engine.types import Chunk, Page, ProcessedDocument, Sentence

logger = logging.getLogger("integrity_engine.text_processing")

# Built once: NFKC-compatible ligature/quote/dash mapping (§4.2).
UNICODE_MAP: dict[str, str] = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u2022": "-", "\t": " ",
}

_SENT_BOUNDARY_RE = re.compile(r"([.!?])(\s+)(?=[A-Z0-9\"'])")
_ABBREV_TOKEN_RE = re.compile(
    r"(?:Mr|Mrs|Ms|Dr|Prof|vs|etc|e\.g|i\.e|Fig|No|approx|Inc|Ltd|Jr|Sr|St)\.$",
    re.IGNORECASE)  # case-insensitive on purpose: "Approx. 40% ..." is not a boundary
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w+)-\n\s*(\w+)")
_WORD_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

_MAX_SENTENCE_CHARS = 800


def _map_unicode(raw: str) -> str:
    for src, dst in UNICODE_MAP.items():
        raw = raw.replace(src, dst)
    return unicodedata.normalize("NFKC", raw)


def de_hyphenate_linebreaks(text: str) -> str:
    """Join words broken across a line wrap: "inter-\\n national" -> "international"."""
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


def collapse_ws(text: str) -> str:
    """Collapse runs of inline whitespace to one space; trim lines; keep paragraphs.

    Newlines survive (PDF header/footer + page-join logic needs line structure);
    blank-line runs collapse to a single blank line ("\\n\\n").
    """
    lines = [re.sub(r"[ ]+", " ", line).strip() for line in text.split("\n")]
    out: list[str] = []
    prev_blank = False
    for line in lines:
        if line:
            out.append(line)
        prev_blank = not line
        if not line:
            out.append("")  # candidate paragraph break
            continue
    text_out = "\n".join(out)
    text_out = _MULTI_BLANK_RE.sub("\n\n", text_out)
    while text_out.startswith("\n"):
        text_out = text_out[1:]
    while text_out.endswith("\n"):
        text_out = text_out[:-1]
    return text_out


def normalize_text(raw: str) -> str:
    """Full normalizer: unicode mapping, hyphen-join, whitespace/paragraph collapse."""
    text = _map_unicode(raw or "")
    text = de_hyphenate_linebreaks(text)
    return collapse_ws(text)


def token_count(text: str) -> int:
    """The ONLY token definition in the project: regex \\w+ tokens (digits included)."""
    return len(_WORD_TOKEN_RE.findall(text))


def split_sentences(text: str) -> list[Sentence]:
    """Deterministic, model-free sentence split with char spans into `text`.

    Boundary candidates: [.!?] + whitespace + capital/digit/quote. A candidate is
    rejected when the token ending at the punctuation is a whitelisted
    abbreviation, or when the sentence produced so far is < 20 chars and itself
    ends in such an abbreviation (merge back). Sentences > 800 chars split at the
    nearest ";" / "," midpoint (logged at DEBUG).
    """
    if not text.strip():
        return []
    boundaries = [m.end(1) for m in _SENT_BOUNDARY_RE.finditer(text)]

    pieces: list[tuple[int, int]] = []
    cursor = 0
    for boundary in boundaries:
        piece = text[cursor:boundary]
        last_token = re.search(r"(\S+)\s*$", piece)
        tail = last_token.group(1) if last_token else ""
        if _ABBREV_TOKEN_RE.match(tail):
            # Abbreviation guard (covers both spec rules): a token like "Mr." /
            # "e.g." at the punctuation never ends a sentence; a too-short
            # produced sentence that ends in one is merged back into the next.
            continue
        pieces.append((cursor, boundary))
        cursor = boundary
    pieces.append((cursor, len(text)))

    sentences: list[Sentence] = []
    for start, end in inside_trim(pieces, text):
        body = text[start:end]
        for sub_start, sub_end in _split_long(body, start):
            sentences.append(Sentence(text=text[sub_start:sub_end],
                                      char_start=sub_start, char_end=sub_end))
    return sentences


def inside_trim(pieces: list[tuple[int, int]], text: str
                ) -> list[tuple[int, int]]:
    """Trim each piece span to its first/last non-space char (span == sentence)."""
    out: list[tuple[int, int]] = []
    for start, end in pieces:
        body = text[start:end]
        trimmed_start = start + (len(body) - len(body.lstrip()))
        trimmed_end = end - (len(body) - len(body.rstrip()))
        if trimmed_end > trimmed_start:
            out.append((trimmed_start, trimmed_end))
    return out


def _split_long(body: str, offset: int) -> list[tuple[int, int]]:
    """Split an over-long sentence at the nearest ';' / ',' midpoint (deterministic).

    `offset` is the absolute start of `body` within the owning text.
    """
    span = (offset, offset + len(body))
    if len(body) <= _MAX_SENTENCE_CHARS:
        return [span]
    mid_abs = offset + len(body) // 2
    window = 200
    cut = -1
    absolute_lo = max(offset + 1, mid_abs - window)
    absolute_hi = min(offset + len(body) - 1, mid_abs + window)
    for punct in (";", ","):
        best = max(
            (i for i in range(absolute_lo, absolute_hi) if body[i - offset] == punct),
            default=-1)
        if best >= 0 and (cut < 0 or abs(best - mid_abs) < abs(cut - mid_abs)):
            cut = best
    if cut < 0:  # no punctuation nearby: hard word-boundary split
        tokens = list(_WORD_TOKEN_RE.finditer(body))
        if len(tokens) < 2:
            return [span]
        cut = offset + tokens[len(tokens) // 2].start()
    left, right = body[:cut - offset], body[cut - offset:]
    logger.debug("sentence over %d chars split at midpoint", _MAX_SENTENCE_CHARS)
    return [*_split_long(left, offset), *_split_long(right, cut)]


def sentence_char_spans_by_pages(pages: list[Page]) -> list[Sentence]:
    """Sentences of pages joined "\\n\\n", keeping page attribution per sentence.

    The join math mirrors ingestion.base.finish_document so char offsets land in
    the document's joined full_text.
    """
    sentences: list[Sentence] = []
    cursor = 0
    sep_len = len("\n\n")
    for page in pages:
        page_start = cursor
        for sentence in split_sentences(page.text):
            sentences.append(Sentence(
                text=sentence.text,
                char_start=page_start + sentence.char_start,
                char_end=page_start + sentence.char_end,
                page=page.page,
            ))
        cursor += len(page.text) + sep_len
    return sentences


def build_chunks(doc: ProcessedDocument, cfg: Config) -> list[Chunk]:
    """Build provenance-complete chunks; MP4 path groups ASR segments (A5)."""
    if doc.modality == "mp4":
        return _build_chunks_mp4(doc, cfg)
    return _build_chunks_text(doc, cfg)


def _expand_overlong(sentences: list[Sentence], max_tokens: int) -> list[Sentence]:
    """Split any single sentence > max_tokens at word boundaries into <=max pieces."""
    expanded: list[Sentence] = []
    for sentence in sentences:
        tokens = list(_WORD_TOKEN_RE.finditer(sentence.text))
        if len(tokens) <= max_tokens:
            expanded.append(sentence)
            continue
        logger.debug("single sentence > %d tokens; hard-splitting", max_tokens)
        piece_count = (len(tokens) + max_tokens - 1) // max_tokens
        size = (len(tokens) + piece_count - 1) // piece_count
        for piece_idx in range(piece_count):
            piece_tokens = tokens[piece_idx * size:(piece_idx + 1) * size]
            start = sentence.char_start + piece_tokens[0].start()
            end = sentence.char_start + piece_tokens[-1].end()
            expanded.append(Sentence(text=sentence.text[piece_tokens[0].start():
                                                     piece_tokens[-1].end()],
                                     char_start=start, char_end=end,
                                     page=sentence.page))
    return expanded


def _build_chunks_text(doc: ProcessedDocument, cfg: Config) -> list[Chunk]:
    """Text/pdf/docx path: sentence accumulation to the token budget (§4.2)."""
    cfg_c = cfg.chunking
    sentences = _expand_overlong(sentence_char_spans_by_pages(doc.pages),
                                 cfg_c.max_chunk_tokens)
    tokens = [token_count(s.text) for s in sentences]

    # Phase 1: accumulate sentences until the 200-token budget.
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx, count in enumerate(tokens):
        if current and current_tokens + count > cfg_c.max_chunk_tokens:
            groups.append(current)
            overlap = current[-cfg_c.overlap_sentences:]
            if sum(tokens[i] for i in overlap) > 0.25 * cfg_c.max_chunk_tokens:
                overlap = []  # overlap must stay <= 25% of the budget; else start clean
            current = list(overlap)
            current_tokens = sum(tokens[i] for i in current)
        current.append(idx)
        current_tokens += count
    if current:
        groups.append(current)

    # Phase 2: enforce the min-token floor — fold a starved tail into the previous
    # group; starved FIRST group folds forward into the next (may exceed 200 in that
    # merge case only, per spec). A single whole-doc below-floor group is kept.
    while len(groups) > 1 and sum(tokens[i] for i in groups[0]) < cfg_c.min_chunk_tokens:
        groups[1] = [*groups[0], *groups[1]]
        groups.pop(0)
    merged: list[list[int]] = []
    for group in groups[:-1] if len(groups) > 1 else groups:
        if merged and sum(tokens[i] for i in group) < cfg_c.min_chunk_tokens:
            merged[-1] = [*merged[-1], *group]
            continue
        merged.append(group)
    if len(groups) > 1:
        tail = groups[-1]
        if sum(tokens[i] for i in tail) < cfg_c.min_chunk_tokens and merged:
            merged[-1] = [*merged[-1], *tail]  # tail merge: over-200 explicitly allowed
        else:
            merged.append(tail)

    chunks: list[Chunk] = []
    doc_idx = _doc_index(doc.document_id)
    for chunk_idx, group in enumerate(merged):
        inside = sorted(set(group))
        start_char = sentences[inside[0]].char_start
        end_char = sentences[inside[-1]].char_end
        pages_ = [sentences[i].page for i in inside]
        chunks.append(Chunk(
            chunk_id=f"{doc_idx:02d}:{chunk_idx:03d}",
            document_id=doc.document_id,
            text=doc.full_text[start_char:end_char],
            start_char=start_char,
            end_char=end_char,
            page_start=min((p for p in pages_ if p is not None), default=None),
            page_end=max((p for p in pages_ if p is not None), default=None),
            timestamp_start=None,
            timestamp_end=None,
            sentence_ids=inside,
            token_count=sum(token_count(sentences[i].text) for i in inside),
        ))
    return chunks


def _build_chunks_mp4(doc: ProcessedDocument, cfg: Config) -> list[Chunk]:
    """MP4: accumulate back-to-back ASR segments; close at >=120 tokens or >=45 s.

    One Page == one ASR segment (A5); timestamps come from metadata["segments"]
    aligned with the pages list.
    """
    segments: list[tuple[float, float]] = doc.metadata.get("segments", [])
    target_tokens = cfg.chunking.mp4_target_tokens
    max_seconds = cfg.chunking.mp4_max_seconds
    doc_idx = _doc_index(doc.document_id)

    groups: list[tuple[int, int]] = []  # (first_idx, last_idx)
    current: list[int] = []
    group_tokens = 0
    for idx, page in enumerate(doc.pages):
        current.append(idx)
        group_tokens += token_count(page.text)
        t0 = segments[current[0]][0] if current[0] < len(segments) else None
        t1 = segments[current[-1]][1] if current[-1] < len(segments) else None
        duration = (float(t1) - float(t0)) if (t0 is not None and t1 is not None) else 0.0
        if group_tokens >= target_tokens or (duration >= max_seconds
                                             and len(current) > 1):
            groups.append((current[0], current[-1]))
            overlap = current[-1]  # replay previous 1 segment
            current = [overlap]
            group_tokens = token_count(doc.pages[overlap].text)
    if current:
        groups.append((current[0], current[-1]))

    chunks: list[Chunk] = []
    for chunk_idx, (first_idx, last_idx) in enumerate(groups):
        first, last = doc.pages[first_idx], doc.pages[last_idx]
        ts0 = segments[first_idx][0] if first_idx < len(segments) else None
        ts1 = segments[last_idx][1] if last_idx < len(segments) else None
        chunks.append(Chunk(
            chunk_id=f"{doc_idx:02d}:{chunk_idx:03d}",
            document_id=doc.document_id,
            text=doc.full_text[first.char_start:last.char_end],
            start_char=first.char_start,
            end_char=last.char_end,
            page_start=None,
            page_end=None,
            timestamp_start=ts0,
            timestamp_end=ts1,
            sentence_ids=[],
            token_count=sum(token_count(doc.pages[i].text)
                            for i in range(first_idx, last_idx + 1)),
        ))
    if len(chunks) == 1 and chunks[0].token_count < cfg.chunking.min_chunk_tokens:
        logger.debug("mp4: single below-min chunk kept for %s", doc.document_id)
    return chunks


def _doc_index(document_id: str) -> int:
    """Stable ordering key: leading NN_ prefix when present, else crc32 (deterministic
    across runs -- never hash(), whose str seeding is process-randomized)."""
    prefix = document_id.split("_", 1)[0]
    if prefix.isdigit() and len(prefix) <= 3:
        return int(prefix)
    return zlib.crc32(document_id.encode("utf-8")) % 10000


def doc_clean_warnings(doc: ProcessedDocument, cfg: Config | None = None) -> list[str]:
    """Warnings for oversized documents (e.g. AI-analysis truncation)."""
    limit = cfg.ai.max_chars if cfg is not None else 20000
    warnings: list[str] = []
    if len(doc.full_text) > limit:
        warnings.append(f">{limit:,} chars truncated for AI analysis")
        logger.debug("doc truncated for AI analysis: %s", doc.document_id)
    return warnings
