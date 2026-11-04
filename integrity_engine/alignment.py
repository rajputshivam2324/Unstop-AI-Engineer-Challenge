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

import bisect
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


_MIN_TEMPLATE_SENTENCE_TOKENS = 6


def _sentence_key(text: str) -> str:
    """Normalized lowercase token string; '' when too short to be distinctive."""
    tokens = _TOKEN_RE.findall(text.lower())
    return " ".join(tokens) if len(tokens) >= _MIN_TEMPLATE_SENTENCE_TOKENS else ""


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
        self.template_sentences: set[str] = set()

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
                        len(all_template_chunks), len({c.template_cluster_id
                                                       for c in all_template_chunks}))
        self._mark_template_sentences(chunks)

    def _mark_template_sentences(self, chunks: list[Chunk]) -> None:
        """Sentence-level boilerplate rule (spec R1 shingle fallback, sentence grain).

        A normalized sentence of >= 6 tokens that occurs verbatim in >= min_docs
        distinct documents is template text. Non-template chunks then have leading
        / trailing template sentences trimmed (chunk overlap otherwise carries the
        tail of a case prompt into the first answer chunk), and a chunk whose
        tokens are >= 50% template sentences is itself marked template.
        """
        docs_by_sentence: dict[str, set[str]] = {}
        split_cache: dict[str, list] = {}
        for chunk in chunks:
            sentences = split_sentences(chunk.text)
            split_cache[chunk.chunk_id] = sentences
            for sentence in sentences:
                key = _sentence_key(sentence.text)
                if key:
                    docs_by_sentence.setdefault(key, set()).add(chunk.document_id)
        self.template_sentences = {k for k, docs in docs_by_sentence.items()
                                   if len(docs) >= self.min_docs}
        if not self.template_sentences:
            return
        for chunk in chunks:
            if chunk.is_template:
                continue
            sentences = split_cache[chunk.chunk_id]
            flags = [_sentence_key(s.text) in self.template_sentences for s in sentences]
            if not any(flags):
                continue
            total = sum(token_count(s.text) for s in sentences) or 1
            templ = sum(token_count(s.text) for s, f in zip(sentences, flags) if f)
            if templ / total >= 0.5:
                chunk.is_template = True
                chunk.template_cluster_id = "tmpl_sentences"
                self._template_ids.add(chunk.chunk_id)
                continue
            lo, hi = 0, len(sentences)
            while lo < hi and flags[lo]:
                lo += 1
            while hi > lo and flags[hi - 1]:
                hi -= 1
            if lo == 0 and hi == len(sentences):
                continue  # template sentence only in the middle: keep, alignment shows it
            new_start = sentences[lo].char_start
            new_end = sentences[hi - 1].char_end
            logger.debug("template registry: trimmed %d template sentence(s) from %s",
                         len(sentences) - (hi - lo), chunk.chunk_id)
            chunk.text = chunk.text[new_start:new_end]
            chunk.start_char += new_start
            chunk.end_char = chunk.start_char + len(chunk.text)
            chunk.token_count = token_count(chunk.text)

    def is_template(self, chunk: Chunk) -> bool:
        """True when the chunk was assigned to a template cluster."""
        return chunk.is_template

    def template_fingerprints(self) -> list[int]:
        """Anchors of all registered (not only template) clusters — debug/EVAL use."""
        return [int(cluster["anchor"]) for cluster in self._clusters]  # type: ignore[arg-type]


class EvidenceGate:
    """Callable (i, j) -> bool: do chunks i and j share >= 1 aligned sentence pair?

    Applied inside doc-pair aggregation so peer % only counts chunk pairs a
    reviewer could verify sentence-by-sentence; topic-level chunk similarity
    without any alignable sentence is not evidence of copying. Sentence
    embeddings are computed once per chunk and cached.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder, cfg: Config) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.cfg = cfg
        self._sent_cache: dict[int, tuple[list[str], object]] = {}
        self._result_cache: dict[tuple[int, int], bool] = {}

    def _sentences(self, i: int) -> tuple[list[str], object]:
        if i not in self._sent_cache:
            texts = [s.text for s in split_sentences(self.chunks[i].text)]
            self._sent_cache[i] = (texts, self.embedder.encode_corpus(texts))
        return self._sent_cache[i]

    def __call__(self, i: int, j: int) -> bool:
        key = (i, j)
        if key not in self._result_cache:
            q_texts, q_vecs = self._sentences(i)
            m_texts, m_vecs = self._sentences(j)
            if not q_texts or not m_texts:
                self._result_cache[key] = False
            else:
                matrix = (q_vecs @ m_vecs.T).tolist()  # type: ignore[union-attr]
                aligned = sentence_lcs_alignment(
                    q_texts, m_texts, matrix,
                    cos_floor=self.cfg.alignment.sentence_align_cos_floor,
                    lcs_ratio_floor=self.cfg.alignment.sentence_lcs_token_ratio)
                self._result_cache[key] = bool(aligned)
        return self._result_cache[key]


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
    candidates.sort(key=lambda t: (-round(t[0], 4), t[1], t[2]))
    used_q: set[int] = set()
    used_m: set[int] = set()
    accepted: list[tuple[int, int]] = []  # kept sorted by query index
    pairs: list[tuple[int, int, float]] = []
    for cos, qi, mi in candidates:
        if qi in used_q or mi in used_m:
            continue
        # Monotonic (order-preserving) mapping: the new pair must not CROSS any
        # accepted pair, i.e. its neighbours by query index must bracket it on the
        # match side too. (Checking only the last accepted pair is wrong: with
        # near-tied cosines it rejects most of an exact copy.)
        pos = bisect.bisect_left(accepted, (qi, mi))
        if pos > 0 and accepted[pos - 1][1] >= mi:
            continue
        if pos < len(accepted) and accepted[pos][1] <= mi:
            continue
        accepted.insert(pos, (qi, mi))
        used_q.add(qi)
        used_m.add(mi)
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
    seen: set[tuple[str, str]] = set()
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
            scores = [cos for _, _, cos in aligned]
        else:  # passage-level fallback: whole chunks stay fully auditable
            q_span = (0, len(q_chunk.text))
            m_span = (0, len(m_chunk.text))
            scores = []
        q_passage = q_chunk.text[q_span[0]:q_span[1]]
        m_passage = m_chunk.text[m_span[0]:m_span[1]]
        evidence_chars = len(q_passage) + len(m_passage)
        if evidence_chars < cfg.alignment.min_evidence_chars:
            discounted += 1
            logger.debug("alignment: evidence below %d chars discounted (%s vs %s)",
                         cfg.alignment.min_evidence_chars, q_chunk.chunk_id,
                         m_chunk.chunk_id)
            continue
        if (q_passage, m_passage) in seen:  # chunk overlap can repeat evidence
            continue
        seen.add((q_passage, m_passage))
        # Provenance of the ALIGNED span (not just the chunk start): absolute
        # char offsets -> owning PDF page, or covered ASR segment time range.
        q_abs = (q_chunk.start_char + q_span[0], q_chunk.start_char + q_span[1])
        m_abs = (m_chunk.start_char + m_span[0], m_chunk.start_char + m_span[1])
        jaccard = round(token_jaccard(q_passage, m_passage), 4)
        lcs_ratio = round(_lcs_token_ratio(q_passage, m_passage), 4)
        kind = _overlap_kind(query_doc, match_doc, match_type, lcs_ratio)
        note = _passage_note(q_chunk, m_chunk, chunk_cos, kind, jaccard, lcs_ratio)
        matches.append(MatchedPassage(
            matched_file=m_chunk.document_id,
            similarity_score=round(chunk_cos, 4),
            query_passage=q_passage,
            matched_passage=m_passage,
            query_page=page_at(query_doc, q_abs[0]),
            matched_page=page_at(match_doc, m_abs[0]),
            matched_timestamp=time_span(match_doc, *m_abs),
            query_timestamp=time_span(query_doc, *q_abs),
            match_type=match_type,  # type: ignore[arg-type]
            evidence_sentence_scores=[round(s, 4) for s in scores],
            note=note,
            overlap_kind=kind,
            lexical_jaccard=jaccard,
            token_lcs_ratio=lcs_ratio,
        ))
    return matches, discounted


def page_at(doc: ProcessedDocument, abs_char: int) -> int | None:
    """1-based PDF page that owns an absolute char offset (None for docx/mp4)."""
    if doc.modality != "pdf":
        return None
    for page in doc.pages:
        if page.char_start <= abs_char <= page.char_end:
            return page.page
    return doc.pages[-1].page if doc.pages else None


def time_span(doc: ProcessedDocument, abs_start: int, abs_end: int) -> str | None:
    """"hh:mm:ss - hh:mm:ss" of the ASR segments overlapping [abs_start, abs_end)."""
    if doc.modality != "mp4":
        return None
    segments: list[tuple[float, float]] = doc.metadata.get("segments", [])
    covered = [segments[i] for i, page in enumerate(doc.pages)
               if i < len(segments) and page.char_start < abs_end
               and page.char_end > abs_start]
    if not covered:
        return None
    return (f"{format_timestamp(covered[0][0])} - "
            f"{format_timestamp(covered[-1][1])}")


VERBATIM_LCS_RATIO = 0.90  # token-LCS ratio at/above which a passage is a verbatim copy


def token_jaccard(a_text: str, b_text: str) -> float:
    """Token-set Jaccard (the naive lexical metric Scenario 2 contrasts against)."""
    a = set(_TOKEN_RE.findall(a_text.lower()))
    b = set(_TOKEN_RE.findall(b_text.lower()))
    return len(a & b) / len(a | b) if a and b else 0.0


def _overlap_kind(query_doc: ProcessedDocument, match_doc: ProcessedDocument,
                  match_type: str, lcs_ratio: float) -> str:
    """common_source | cross_modal | verbatim | paraphrase (report + eval label)."""
    if match_type == "common_source":
        return "common_source"
    if (query_doc.modality == "mp4") != (match_doc.modality == "mp4"):
        return "cross_modal"
    if lcs_ratio >= VERBATIM_LCS_RATIO:
        return "verbatim"
    return "paraphrase"


def _passage_note(q_chunk: Chunk, m_chunk: Chunk, chunk_cos: float, kind: str,
                  jaccard: float, lcs_ratio: float) -> str | None:
    """One human-readable line telling the reviewer what kind of overlap this is."""
    if q_chunk.is_duplicate or m_chunk.is_duplicate:
        return f"byte-identical duplicate of {_canonical_hint(q_chunk, m_chunk)}"
    if kind == "common_source":
        return "shared institutional text (discounted)"
    lexical = f"lexical Jaccard {jaccard:.2f}, token-LCS {lcs_ratio:.2f}"
    if kind == "verbatim":
        return f"verbatim copy ({lexical})"
    if kind == "cross_modal":
        return f"spoken/written cross-modal match ({lexical}; ASR noise expected)"
    band = f" {_PARAPHRASE_BAND_LABEL}" if chunk_cos >= 0.80 else ""
    return f"semantic paraphrase{band}: meaning matches while wording differs ({lexical})"


def _canonical_hint(q_chunk: Chunk, m_chunk: Chunk) -> str:
    doc = m_chunk.document_id if m_chunk.is_duplicate else q_chunk.document_id
    return doc


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



# --------------------------------------------------------------------------
# Document-level evidence: sentence alignment across the WHOLE pair of documents
# --------------------------------------------------------------------------
_RUN_MAX_GAP = 2  # aligned sentences within 2 positions on both sides form one passage


class DocumentAligner:
    """Sentence-level evidence extraction for a flagged document pair.

    Chunk retrieval decides WHICH document pairs are suspicious; this class
    decides WHAT a reviewer sees. Every non-template sentence of the query is
    aligned against every non-template sentence of the source (cosine matrix +
    the same monotonic cosine/LCS inclusion rule), and consecutive aligned
    sentences are grouped into passages. Chunk boundaries therefore never hide
    evidence (e.g. a copied sentence that sits in a chunk dominated by other text).
    Sentence vectors are cached per document.
    """

    def __init__(self, embedder: Embedder, registry: TemplateRegistry, cfg: Config,
                 chunks: list[Chunk]) -> None:
        self.embedder = embedder
        self.registry = registry
        self.cfg = cfg
        self._template_spans: dict[str, list[tuple[int, int]]] = {}
        for chunk in chunks:
            if chunk.is_template:
                self._template_spans.setdefault(chunk.document_id, []).append(
                    (chunk.start_char, chunk.end_char))
        self._cache: dict[str, tuple[list, object]] = {}

    def _doc_sentences(self, doc: ProcessedDocument) -> tuple[list, object]:
        if doc.document_id not in self._cache:
            from integrity_engine.text_processing import sentence_char_spans_by_pages

            spans = self._template_spans.get(doc.document_id, [])
            kept = []
            for sentence in sentence_char_spans_by_pages(doc.pages):
                if _sentence_key(sentence.text) in self.registry.template_sentences:
                    continue
                if any(a <= sentence.char_start and sentence.char_end <= b
                       for a, b in spans):
                    continue
                kept.append(sentence)
            vecs = self.embedder.encode_corpus([s.text for s in kept])
            self._cache[doc.document_id] = (kept, vecs)
        return self._cache[doc.document_id]

    def align(self, query_doc: ProcessedDocument, match_doc: ProcessedDocument,
              chunk_cosine: float) -> tuple[list[MatchedPassage], int]:
        """Return (passages, discounted_noise_count) for one document pair."""
        q_sents, q_vecs = self._doc_sentences(query_doc)
        m_sents, m_vecs = self._doc_sentences(match_doc)
        if not q_sents or not m_sents:
            return [], 0
        matrix = (q_vecs @ m_vecs.T).tolist()  # type: ignore[union-attr]
        aligned = sentence_lcs_alignment(
            [s.text for s in q_sents], [s.text for s in m_sents], matrix,
            cos_floor=self.cfg.alignment.sentence_align_cos_floor,
            lcs_ratio_floor=self.cfg.alignment.sentence_lcs_token_ratio)
        runs: list[list[tuple[int, int, float]]] = []
        for triple in aligned:
            if runs:
                prev = runs[-1][-1]
                same_pages = (q_sents[prev[0]].page == q_sents[triple[0]].page and
                              m_sents[prev[1]].page == m_sents[triple[1]].page)
                if (same_pages and triple[0] - prev[0] <= _RUN_MAX_GAP
                        and 0 < triple[1] - prev[1] <= _RUN_MAX_GAP):
                    runs[-1].append(triple)
                    continue
            runs.append([triple])  # new passage: gap, or a PDF page boundary
        passages: list[MatchedPassage] = []
        discounted = 0
        for run in runs:
            q_first, q_last = q_sents[run[0][0]], q_sents[run[-1][0]]
            m_first, m_last = m_sents[run[0][1]], m_sents[run[-1][1]]
            q_passage = query_doc.full_text[q_first.char_start:q_last.char_end]
            m_passage = match_doc.full_text[m_first.char_start:m_last.char_end]
            if len(q_passage) + len(m_passage) < self.cfg.alignment.min_evidence_chars:
                discounted += 1
                continue
            pv = self.embedder.encode_corpus([q_passage, m_passage])
            passage_cos = float(pv[0] @ pv[1])
            jaccard = round(token_jaccard(q_passage, m_passage), 4)
            lcs_ratio = round(_lcs_token_ratio(q_passage, m_passage), 4)
            kind = _overlap_kind(query_doc, match_doc, "overlap", lcs_ratio)
            note = _doc_passage_note(kind, passage_cos, jaccard, lcs_ratio,
                                     query_doc, match_doc)
            passages.append(MatchedPassage(
                matched_file=match_doc.document_id,
                similarity_score=round(passage_cos, 4),
                query_passage=q_passage,
                matched_passage=m_passage,
                query_page=page_at(query_doc, q_first.char_start),
                matched_page=page_at(match_doc, m_first.char_start),
                matched_timestamp=time_span(match_doc, m_first.char_start,
                                            m_last.char_end),
                query_timestamp=time_span(query_doc, q_first.char_start,
                                          q_last.char_end),
                match_type="overlap",
                evidence_sentence_scores=[round(c, 4) for _, _, c in run],
                note=note,
                overlap_kind=kind,
                lexical_jaccard=jaccard,
                token_lcs_ratio=lcs_ratio,
            ))
        passages.sort(key=lambda m: (-m.similarity_score, m.query_passage))
        _ = chunk_cosine  # retained in the caller's doc_pair_scores for transparency
        return passages, discounted


def _doc_passage_note(kind: str, passage_cos: float, jaccard: float, lcs_ratio: float,
                      query_doc: ProcessedDocument, match_doc: ProcessedDocument) -> str:
    if query_doc.metadata.get("duplicate_of") == match_doc.document_id or \
            match_doc.metadata.get("duplicate_of") == query_doc.document_id:
        return "byte-identical duplicate file"
    lexical = f"lexical Jaccard {jaccard:.2f}, token-LCS {lcs_ratio:.2f}"
    if kind == "verbatim":
        return f"verbatim copy ({lexical})"
    if kind == "cross_modal":
        return f"spoken/written cross-modal match ({lexical}; ASR noise expected)"
    band = f" {_PARAPHRASE_BAND_LABEL}" if passage_cos >= 0.80 else ""
    return f"semantic paraphrase{band}: meaning matches while wording differs ({lexical})"
