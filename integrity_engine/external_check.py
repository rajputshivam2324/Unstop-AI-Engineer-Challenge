"""<MODULE PURPOSE>
Two-tier external-plagiarism engine (§4.7). Tier 1, always-on: a local reference
KB (6 public-domain-style passages) embedded once with the shared MiniLM model
and searched per query sentence (cos >= local_kb_min_cos). Tier 2, optional:
no-key DuckDuckGo search with a per-doc call budget and a documented mock
fallback (the assignment explicitly allows a documented mock when offline;
DuckDuckGo is non-commercial, respects robots, and returns snippets).
External % = coverage-style share: tokens of matched query sentences over all
document tokens, with overlapping spans merged so they count once (A6 honesty).

<PUBLIC API>
LocalReferenceKB, pick_query_strings, pick_query_strings_index, ddg_query, MockDDG

<DEPENDS ON>
faiss, numpy (via Embedder), config (ExternalCheckConfig), types, reference_kb.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter

from integrity_engine.config import Config
from integrity_engine.embeddings import Embedder
from integrity_engine.indexing import build_index
from integrity_engine.types import (ExternalEvidence, ExternalHit,
                                    ProcessedDocument)
from reference_kb import kb_texts

logger = logging.getLogger("integrity_engine.external_check")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Noun phrases: Capitalized or >=4-letter lowercase words chained 2-3 deep —
# cheap, dependency-free "distinctive keyword" queries for the web tier.
_NOUN_PHRASE_RE = re.compile(
    r"(?:\b[A-Z][a-z]+\b|\b[a-z]{4,}\b)(?:\s+(?:\b[A-Z][a-z]+\b|\b[a-z]{4,}\b)){1,2}")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _token_jaccard(a: str, b: str) -> float:
    ta = {t.lower() for t in _tokens(a)}
    tb = {t.lower() for t in _tokens(b)}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def doc_sentences(doc: ProcessedDocument) -> list:
    """Sentences with char spans, decoupled via text_processing (module isolation)."""
    from integrity_engine.text_processing import sentence_char_spans_by_pages

    return sentence_char_spans_by_pages(doc.pages)


def doc_token_count(doc: ProcessedDocument) -> int:
    return len(_tokens(doc.full_text))


def pick_query_strings(sentences: list[str], k: int, min_words: int) -> list[str]:
    """Top-k tf-idf-grade sentence queries in the [min_words, 60]-word band.

    When fewer than k sentences qualify, distinctive noun phrases (2-3 word
    collocations) extend the query list. Deterministic: ties break by order.
    """
    chosen = [sentences[i] for i in pick_query_strings_index(sentences, k, min_words)]
    if len(chosen) < k:
        for phrase in _noun_phrases(sentences):
            if len(chosen) >= k:
                break
            if phrase not in chosen:
                chosen.append(phrase)
    return chosen


def pick_query_strings_index(sentences: list[str], k: int, min_words: int = 12) \
        -> list[int]:
    """Top-k sentence indexes by tf-idf-style weight (long-sentence soft penalty)."""
    word_counts = [len(_tokens(s)) for s in sentences]
    doc_freq: Counter[str] = Counter()
    sentence_counters: list[Counter[str]] = []
    for sentence in sentences:
        lower_tokens = [t.lower() for t in _tokens(sentence)]
        sentence_counters.append(Counter(lower_tokens))
        doc_freq.update(set(lower_tokens))
    total = max(1, len(sentences))
    scored: list[tuple[float, int]] = []
    for idx, counter in enumerate(sentence_counters):
        if not min_words <= word_counts[idx] <= 60:
            continue  # too short is junk; too long drifts off-topic for a single query
        weight = sum(counter[t] * (total / (1 + doc_freq[t])) for t in counter)
        if word_counts[idx] > 35:
            weight *= 0.5  # soft penalty: very long sentences make poor web queries
        scored.append((weight, idx))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in scored][:k]


def _noun_phrases(sentences: list[str]) -> list[str]:
    """First 3 unique noun phrases, in document order (web-tier extenders)."""
    out: list[str] = []
    for sentence in sentences:
        for match in _NOUN_PHRASE_RE.finditer(sentence):
            phrase = match.group(0)
            if phrase not in out:
                out.append(phrase)
    return out[:3]


def _merge_span_tokens(doc: ProcessedDocument, spans: list[tuple[int, int]]) -> int:
    """Union of overlapping char spans -> covered token count (A6: count once)."""
    if not spans:
        return 0
    ordered = sorted(spans)
    merged: list[list[int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    covered = 0
    for start, end in merged:
        text = doc.full_text[max(0, start):min(len(doc.full_text), end)]
        covered += len(_tokens(text))
    return covered


def _query_spans(sentences, queries: list[str]) -> list[tuple[int, int]]:
    """Char span for each query: exact sentence match first, then substring."""
    exact = {sentence.text: (sentence.char_start, sentence.char_end)
             for sentence in sentences}
    spans: list[tuple[int, int]] = []
    for query in queries:
        span = exact.get(query)
        if span is None:
            for sentence in sentences:
                if query and query in sentence.text:
                    span = (sentence.char_start, sentence.char_end)
                    break
        if span is not None:
            spans.append(span)
    return spans


class LocalReferenceKB:
    """Tier 1: always-on offline knowledge base; Tier 2: budgeted web + mock."""

    def __init__(self, embedder: Embedder, cfg: Config, skip_web: bool = False) -> None:
        self.embedder = embedder
        self.cfg = cfg
        self.skip_web = skip_web
        self._passages = kb_texts.all_passages()  # [(kb_id, text), ...]
        self._index = None
        self.mock_engaged = False  # last search_doc disclosed the mock fallback

    def build(self) -> None:
        """Embed KB passages once at startup; reused by every query document."""
        texts = [text for _, text in self._passages]
        self._matrix = self.embedder.encode_corpus(texts)
        self._index = build_index(self._matrix)
        logger.info("external: local KB built (%d passages, dim=%d)",
                    len(texts), self.embedder.dim)

    def search_doc(self, doc: ProcessedDocument) -> ExternalEvidence:
        """Search one document; evidence = best KB passage per query sentence."""
        if self._index is None:
            self.build()
        sentences = doc_sentences(doc)
        top_indexes = pick_query_strings_index(
            [s.text for s in sentences],
            k=self.cfg.external.query_topk, min_words=12)
        covered_spans: list[tuple[int, int]] = []  # spans of sentences that hit
        hits: list[ExternalHit] = []
        kb_floor = self.cfg.external.local_kb_min_cos
        for i in top_indexes:
            sentence = sentences[i]
            q_vec = self.embedder.encode_query(sentence.text)
            sims, ids = self._index.search(q_vec, 2)
            for sim, passage_idx in zip(sims[0], ids[0], strict=True):
                if int(passage_idx) == -1 or float(sim) < kb_floor:
                    continue
                kb_id, passage = self._passages[int(passage_idx)]
                aligned = min(len(_tokens(sentence.text)), len(_tokens(passage)))
                hits.append(ExternalHit(
                    source_title=kb_id, source_url=None, similarity=float(sim),
                    query_passage=sentence.text, matched_passage=passage,
                    source="local_kb", aligned_tokens=aligned))
                covered_spans.append((sentence.char_start, sentence.char_end))
                break  # one best KB passage per query sentence

        queries = pick_query_strings([s.text for s in sentences],
                                     k=self.cfg.external.query_topk,
                                     min_words=self.cfg.external.web_min_query_words)
        web_hits: list[ExternalHit] = []
        mock_engaged = False
        if not self.skip_web and self.cfg.external.allow_external_web and queries:
            web_hits, mock_engaged = self._web_tier(queries)
            if web_hits:
                covered_spans.extend(_query_spans(sentences,
                                                  [h.query_passage for h in web_hits]))
        hits.extend(web_hits)
        self.mock_engaged = mock_engaged
        if mock_engaged:
            logger.info("external: web tier unavailable; documented mock engaged (%s)",
                        doc.document_id)

        score_pct = round(100.0 * _merge_span_tokens(doc, covered_spans)
                          / max(1, doc_token_count(doc)), 1)
        tier: str = ("web" if any(h.source == "web" for h in hits)
                     else ("local_kb" if any(h.source == "local_kb" for h in hits)
                           else ("mock" if mock_engaged else "none")))
        return ExternalEvidence(score_pct=float(min(max(score_pct, 0.0), 100.0)),
                                hits=hits, tier_used=tier)  # type: ignore[arg-type]

    def _web_tier(self, queries: list[str]) -> tuple[list[ExternalHit], bool]:
        """Budgeted DDG calls; every failure path is the documented mock (A8).

        Returns (hits, mock_engaged). Mock engaged = a call degraded to the
        documented mock with zero usable results (returned None).
        """
        hits: list[ExternalHit] = []
        mock_engaged = False
        budget = self.cfg.external.web_budget_per_doc
        for attempt, query in enumerate(queries):
            if attempt >= budget:
                break
            results = ddg_query(query, self.cfg.external.request_timeout_s)
            if results is None:
                mock_engaged = True  # network/parse failure -> disclosed mock
                continue
            time.sleep(self.cfg.external.web_sleep_s)  # rate-limit real calls only
            for result in results[:3]:
                snippet = result.get("body") or result.get("snippet") or ""
                if not snippet:
                    continue
                jaccard = _token_jaccard(query, snippet)
                if jaccard < self.cfg.external.web_jaccard_min:
                    continue
                hits.append(ExternalHit(
                    source_title=(result.get("title") or "")[:120],
                    source_url=result.get("href") or result.get("url"),
                    similarity=jaccard, query_passage=query,
                    matched_passage=snippet, source="web",
                    aligned_tokens=len(_tokens(query))))
        return hits, mock_engaged


class MockDDG:
    """Assignment-sanctioned documented mock wrapper (empty result set).

    Never fabricates findings: offline it yields no results, so external %
    stays honest (KB tier only) and `tier_used="mock"` discloses the attempt.
    """

    @staticmethod
    def text(_query: str, **_kwargs) -> list[dict]:
        return []


def ddg_query(query: str, request_timeout_s: int) -> list[dict] | None:
    """Tier-2 transport: ddgs library, then DDG instant-answer API, else None.

    None = both transports unavailable -> caller engages the documented mock.
    Never raises (A8); every network/parse error is a logged debug line.
    """
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=5, region="wt-wt",
                              safesearch="moderate", timelimit=None,
                              backend="auto")
        return [{"title": r.get("title", ""), "href": r.get("href"),
                 "body": r.get("body", "")} for r in results]
    except Exception as exc:  # noqa: BLE001 - network/parse errors never raise (A8)
        logger.debug("web tier: ddgs unavailable/failing (%s); "
                     "trying instant-answer API", type(exc).__name__)
    try:
        import requests

        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json"},
            timeout=request_timeout_s)
        payload = response.json()
        abstract = payload.get("AbstractText") or ""
        if abstract:
            return [{"title": payload.get("Heading") or query,
                     "href": payload.get("AbstractURL"), "body": abstract}]
        return []
    except Exception as exc:  # noqa: BLE001 - A8: never raise
        logger.debug("web tier: instant-answer API also unavailable (%s); "
                     "mock tier active", type(exc).__name__)
        return None
