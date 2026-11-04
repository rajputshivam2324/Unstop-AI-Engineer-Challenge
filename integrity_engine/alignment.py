"""<MODULE PURPOSE>
Evidence extraction + boilerplate discount (§4.5): simhash64 TemplateRegistry
(chunk fingerprints recurring in >= 3 distinct documents + verbatim-text rule
=> institutional template), sentence-level alignment with the cosine/LCS
inclusion rule, and MatchedPassage construction with page/timestamp provenance.

<PUBLIC API>
simhash64, hamming, TemplateRegistry, align_pair, sentence_lcs_alignment,
format_timestamp

<DEPENDS ON>
types, config, text_processing (split_sentences/token_count/normalize_text),
embeddings.
"""

from __future__ import annotations

import hashlib
import logging
import re

from integrity_engine.config import Config
from integrity_engine.embeddings import Embedder
from integrity_engine.text_processing import (normalize_text, split_sentences,
                                              token_count)
from integrity_engine.types import Chunk, MatchedPassage, ProcessedDocument

logger = logging.getLogger("integrity_engine.alignment")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_TS_FORMAT = "{h:02d}:{m:02d}:{s:02d}"
_PARAPHRASE_BAND_LABEL = "paraphrase-band (>0.80)"


def format_timestamp(seconds: float | None) -> str:
    """Seconds -> "hh:mm:ss" (report-facing format)."""
    if seconds is None:
        return "n/a"
    total = max(0, int(round(seconds)))
    return _TS_FORMAT.format(h=total // 3600, m=(total % 3600) // 60, s=total % 60)


def simhash64(text: str) -> int:
    """Standard 64-bit simhash over token 5-gram hashes (deterministic blake2b)."""
    tokens = _TOKEN_RE.findall(normalize_text(text).lower())
    vectors = [0] * 64
    ngrams: list[str] = []
    if len(tokens) < 5:
        ngrams = [" ".join(tokens)] if tokens else []
    else:
        for i in range(len(tokens) - 4):
            ngrams.append(" ".join(tokens[i:i + 5]))
    for ngram in ngrams:
        digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            if (value >> bit) & 1:
                vectors[bit] += 1
            else:
                vectors[bit] -= 1
    fingerprint = 0
    for bit, count in enumerate(vectors):
        if count > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit fingerprints."""
    return bin(a ^ b).count("1")


class TemplateRegistry:
    """Chunks whose simhash fingerprint recurs in >= min_docs distinct documents
    (or whose exact normalized text repeats across >= 3 docs, 20-token floor)
    are institutional boilerplate (winnowing's special-ID scheme, D6)."""

    def __init__(self, min_docs: int = 3, hamming_max: int = 3,
                 min_fingerprint_tokens: int = 40,
                 verbatim_template_min_tokens: int = 20) -> None:
        self.min_docs = min_docs
        self.hamming_max = hamming_max
        self.min_fingerprint_tokens = min_fingerprint_tokens
        self.verbatim_min_tokens = verbatim_template_min_tokens
        self._clusters: list[dict[str, object]] = []  # {fp, members: list[(chunk, fp)]}
        self._chunks: list[Chunk] = []
        self._template_ids: set[str] = set()

    def register_all_chunks(self, chunks: list[Chunk]) -> None:
        """Pass 1 of the pipeline: pure text pass BEFORE embeddings (research D6)."""
        self._chunks = chunks
        self._clusters = []
        self._template_ids = set()
        fingerprints: dict[str, int] = {}
        verbatim_docs: dict[str, set[str]] = {}
        for chunk in chunks:
            tokens = token_count(chunk.text)
            if tokens >= self.min_fingerprint_tokens:
                fingerprints[chunk.chunk_id] = simhash64(chunk.text)
            if tokens >= self.verbatim_min_tokens:
                verbatim_hash = hashlib.sha256(
                    normalize_text(chunk.text).encode("utf-8")).hexdigest()
                verbatim_docs.setdefault(verbatim_hash, set()).add(chunk.document_id)

        for chunk in chunks:
            fp = fingerprints.get(chunk.chunk_id)
            if fp is None:
                continue
            joined = False
            for cluster in self._clusters:
                anchor = int(cluster["anchor"])  # type: ignore[arg-type]
                if hamming(anchor, fp) <= self.hamming_max:
                    cluster["members"].append((chunk, fp))  # type: ignore[union-attr]
                    joined = True
                    break
            if not joined:
                self._clusters.append({"anchor": fp, "members": [(chunk, fp)]})

        template_clusters: list[list[Chunk]] = []
        for cluster in self._clusters:
            members = cluster["members"]  # type: ignore[union-attr]
            docs = {chunk.document_id for chunk, _ in members}
            if len(docs) >= self.min_docs:
                template_clusters.append([chunk for chunk, _ in members])
        # verbatim exact-text rule (BT fallback): identical normalized chunk text
        verbatim_clusters: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            if token_count(chunk.text) < self.verbatim_min_tokens:
                continue
            digest = hashlib.sha256(
                normalize_text(chunk.text).encode("utf-8")).hexdigest()
            if len(verbatim_docs.get(digest, set())) >= self.min_docs:
                verbatim_clusters.setdefault(digest, []).append(chunk)

        all_template_chunks: list[Chunk] = []
        ordered = template_clusters + list(verbatim_clusters.values())
        ordered.sort(key=lambda members: members[0].chunk_id)
        for cluster_index, members in enumerate(ordered):
            cluster_id = f"tmpl_{cluster_index:02d}"
            for chunk in members:
                if chunk.is_template:  # simhash and verbatim may both fire
                    continue
                chunk.is_template = True
                chunk.template_cluster_id = cluster_id
                all_template_chunks.append(chunk)
                self._template_ids.add(chunk.chunk_id)
        if all_template_chunks:
            logger.info("template registry: %d boilerplate chunk(s) in %d cluster(s)",
                        len(all_template_chunks), len(ordered))

    def is_template(self, chunk: Chunk) -> bool:
        """True when the chunk was assigned to a template cluster."""
        return chunk.is_template

    def template_fingerprints(self) -> list[int]:
        """Anchors of all registered (not only template) clusters — debug/EVAL use."""
        return [int(cluster["anchor"]) for cluster in self._clusters]  # type: ignore[arg-type]


def sentence_lcs_alignment(q_sents: list[str], m_sents: list[str],
                           score_matrix: list[list[float]] | None = None,
                           cos_floor: float = 0.65,
                           lcs_ratio_floor: float = 0.60) \
        -> list[tuple[int, int, float]]:
    """Inclusion rule: cos >= 0.65 OR (0.45 <= cos < 0.65 AND LCS-token ratio >= 0.60).

    Mapping is monotonic greedy: candidate pairs sorted by score descending, a
    pair is accepted only when both indices stay increasing (order-preserving).
    """
    if not q_sents or not m_sents:
        return []
    if score_matrix is None:
        score_matrix = [[0.0 for _ in m_sents] for _ in q_sents]
    candidates: list[tuple[float, int, int]] = []
    for qi in range(len(q_sents)):
        for mi in range(len(m_sents)):
            cos = score_matrix[qi][mi]
            q_tokens, m_tokens = token_count(q_sents[qi]), token_count(m_sents[mi])
            ratio = _lcs_token_ratio(q_sents[qi], m_sents[mi])
            included = cos >= cos_floor or (0.45 <= cos < cos_floor and
                                            ratio >= lcs_ratio_floor)
            if included:
                candidates.append((cos, qi, mi))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_q: set[int] = set()
    used_m: set[int] = set()
    order_q: list[int] = []
    order_m: list[int] = []
    pairs: list[tuple[int, int, float]] = []
    for cos, qi, mi in candidates:
        if qi in used_q or mi in used_m:
            continue
        if order_q and (qi <= order_q[-1] or mi <= order_m[-1]):
            continue  # monotonic: preserve relative order on both sides
        used_q.add(qi)
        used_m.add(mi)
        order_q.append(qi)
        order_m.append(mi)
        pairs.append((qi, mi, cos))
    pairs.sort(key=lambda t: t[0])
    return pairs


def _lcs_token_ratio(a_text: str, b_text: str) -> float:
    """Length-normalized token-LCS ratio: LCS / max(len(a), len(b))."""
    a_tokens = _TOKEN_RE.findall(a_text.lower())
    b_tokens = _TOKEN_RE.findall(b_text.lower())
    if not a_tokens or not b_tokens:
        return 0.0
    prev_row = [0] * (len(b_tokens) + 1)
    for a_token in a_tokens:
        row = [0]
        for j, b_token in enumerate(b_tokens, start=1):
            if a_token == b_token:
                row.append(prev_row[j - 1] + 1)
            else:
                row.append(max(row[j - 1], prev_row[j]))
        prev_row = row
    return prev_row[-1] / max(len(a_tokens), len(b_tokens))


def align_pair(query_doc: ProcessedDocument, match_doc: ProcessedDocument,
               chunk_pairs: list[tuple[str, str, float]],
               chunks_by_id: dict[str, Chunk], embedder: Embedder,
               registry: TemplateRegistry, cfg: Config) \
        -> tuple[list[MatchedPassage], int]:
    """Build sentence-level MatchedPassage evidence for one doc pair.

    Returns (passages, discounted_noise_count); passages below
    cfg.alignment.min_evidence_chars combined aligned chars are counted
    (discounted_noise_matches) but never silently dropped (§4.5).
    """
    matches: list[MatchedPassage] = []
    discounted = 0
    for query_chunk_id, match_chunk_id, chunk_cos in chunk_pairs:
        q_chunk = chunks_by_id.get(query_chunk_id)
        m_chunk = chunks_by_id.get(match_chunk_id)
        if q_chunk is None or m_chunk is None:
            continue
        match_type = ("common_source" if q_chunk.is_template and m_chunk.is_template
                      else "overlap")
        q_sents = split_sentences(q_chunk.text)
        m_sents = split_sentences(m_chunk.text)
        q_texts = [s.text for s in q_sents]
        m_texts = [s.text for s in m_sents]
        matrix = _cosine_matrix(embedder, q_texts, m_texts)
        aligned = sentence_lcs_alignment(
            q_texts, m_texts, matrix,
            cos_floor=cfg.alignment.sentence_align_cos_floor,
            lcs_ratio_floor=cfg.alignment.sentence_lcs_token_ratio)
        if aligned:
            q_span = (q_sents[aligned[0][0]].char_start,
                      q_sents[aligned[-1][0]].char_end)
            m_span = (m_sents[aligned[0][1]].char_start,
                      m_sents[aligned[-1][1]].char_end)
            q_passage = q_chunk.text[q_span[0]:q_span[1]]
            m_passage = m_chunk.text[m_span[0]:m_span[1]]
            scores = [cos for _, _, cos in aligned]
            evidence_chars = len(q_passage) + len(m_passage)
        else:  # passage-level fallback: whole chunks stay fully auditable
            q_passage, m_passage = q_chunk.text, m_chunk.text
            scores = []
            evidence_chars = len(q_passage) + len(m_passage)
        if evidence_chars < cfg.alignment.min_evidence_chars:
            discounted += 1
            logger.debug("alignment: evidence below %d chars discounted (%s vs %s)",
                         cfg.alignment.min_evidence_chars, q_chunk.chunk_id,
                         m_chunk.chunk_id)
            continue
        note = _passage_note(q_chunk, m_chunk, chunk_cos)
        matches.append(MatchedPassage(
            matched_file=match_chunk.document_id,
            similarity_score=round(chunk_cos, 4),
            query_passage=q_passage,
            matched_passage=m_passage,
            query_page=q_chunk.page_start,
            matched_page=m_chunk.page_start,
            matched_timestamp=_timestamp_pair(q_chunk, m_chunk),
            match_type=match_type,  # type: ignore[arg-type]
            evidence_sentence_scores=[round(s, 4) for s in scores],
            note=note,
        ))
    return matches, discounted


def _passage_note(q_chunk: Chunk, m_chunk: Chunk, chunk_cos: float) -> str | None:
    duplicate_note = (f"byte-identical duplicate of "
                      f"{_canonical_hint(q_chunk, m_chunk)}"
                      if q_chunk.is_duplicate or m_chunk.is_duplicate else None)
    if duplicate_note:
        return duplicate_note
    if q_chunk.is_template and m_chunk.is_template:
        return "shared institutional text (discounted)"
    if chunk_cos >= 0.80:
        return _PARAPHRASE_BAND_LABEL
    return None


def _canonical_hint(q_chunk: Chunk, m_chunk: Chunk) -> str:
    doc = m_chunk.document_id if m_chunk.is_duplicate else q_chunk.document_id
    return doc


def _timestamp_pair(q_chunk: Chunk, m_chunk: Chunk) -> str | None:
    """The MP4 side's "hh:mm:ss - hh:mm:ss" span (either direction)."""
    for chunk in (m_chunk, q_chunk):  # matched (mp4) side first per assignment shape
        if chunk.timestamp_start is not None and chunk.timestamp_end is not None:
            return (f"{format_timestamp(chunk.timestamp_start)} - "
                    f"{format_timestamp(chunk.timestamp_end)}")
    return None


def _cosine_matrix(embedder: Embedder, q_texts: list[str],
                   m_texts: list[str]) -> list[list[float]]:
    """Sentence-level cosine matrix between aligned sentence lists."""
    if not q_texts or not m_texts:
        return []
    q_vecs = embedder.encode_corpus(q_texts)
    m_vecs = embedder.encode_corpus(m_texts)
    matrix: list[list[float]] = []
    for q_vec in q_vecs:
        matrix.append([_cos(q_vec, m_vec) for m_vec in m_vecs])
    return matrix


def _cos(a, b) -> float:
    """Cosine of two numpy vectors with zero-norm safety."""
    norm = (float((a * a).sum()) ** 0.5) * (float((b * b).sum()) ** 0.5)
    return float((a * b).sum()) / norm if norm > 0 else 0.0
