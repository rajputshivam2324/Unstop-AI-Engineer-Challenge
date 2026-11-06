"""evidence_v2 matching mode: three measured fixes to the baseline sentence aligner.

Found on the realistic benchmark DEV split (benchmarks/realistic), never on test.
Every acceptance threshold (cosine 0.65 / 0.45+LCS, run rules, single-sentence
rule) and the coverage definition are inherited unchanged from DocumentAligner.

1. Template masking at sentence grain. Baseline drops every sentence inside a
   chunk that the registry marked template. A chunk is up to 200 tokens, so a
   student paragraph that shares a chunk with the assignment brief vanished with
   it (a fully copied paragraph scored 0.10). Here a sentence is template text
   only when the sentence itself recurs: its normalized text is in >= min_docs
   documents (unchanged rule), or >= template_shingle_share of its word 5-grams
   occur in >= min_docs documents (catches lightly edited boilerplate). Sentences
   too short for a 5-gram keep the baseline chunk rule.
2. Long-sentence splitting. Text without full stops (ASR, some LLM/tool output)
   produced 60+ word "sentences" whose embeddings never align with normally
   punctuated text. Units longer than max_unit_chars are split at the nearest
   ';' / ',' (else a word boundary), exactly as baseline does above 800 chars.
3. Exact-run evidence. A query sentence of >= single_sentence_min_tokens tokens
   that shares an exact run of >= 10 words (lexical_min_shared distinct 8-grams)
   with a source sentence, outside template text, is evidence even when it stands
   alone or when several of its sentences were merged into one (GPT-3 output with
   dropped full stops). This is the MOSS/Turnitin idea of exact-run matching.
   Caveat: the benchmark forbids accidental shared 8-grams between unrelated
   documents, so it cannot measure false positives from common 10-word phrases.
"""

from __future__ import annotations

import re

from integrity_engine.alignment import DocumentAligner, _sentence_key
from integrity_engine.text_processing import sentence_char_spans_by_pages, token_count

_WORD = re.compile(r"\w+")
_N = 5
_LEX_N = 8
_RUN_GAP = 2  # same grouping gap as the baseline evidence runs


def _shingles(text: str, n: int = _N) -> set[tuple[str, ...]]:
    w = _WORD.findall(text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


class EvidenceV2Aligner(DocumentAligner):
    def __init__(self, embedder, registry, cfg, chunks) -> None:
        super().__init__(embedder, registry, cfg, chunks)
        opts = cfg.matching
        self.max_unit_chars = opts.max_unit_chars
        self.shingle_share = opts.template_shingle_share
        owners: dict[tuple[str, ...], set[str]] = {}
        for chunk in chunks:  # chunk overlap is harmless: distinct documents are counted
            for g in _shingles(chunk.text):
                owners.setdefault(g, set()).add(chunk.document_id)
        self.template_shingles = {
            g for g, d in owners.items() if len(d) >= cfg.alignment.template_min_docs
        }
        self.lexical_min_shared = opts.lexical_min_shared
        self._lex_cache: dict[str, list[set[tuple[str, ...]]]] = {}

    def is_template_sentence(self, doc_id: str, text: str, start: int, end: int) -> bool:
        if _sentence_key(text) in self.registry.template_sentences:
            return True
        grams = _shingles(text)
        if grams:
            return len(grams & self.template_shingles) / len(grams) >= self.shingle_share
        return any(a <= start and end <= b for a, b in self._template_spans.get(doc_id, []))

    def _doc_sentences(self, doc):
        if doc.document_id not in self._cache:
            kept = [
                s
                for s in sentence_char_spans_by_pages(doc.pages, self.max_unit_chars)
                if not self.is_template_sentence(doc.document_id, s.text, s.char_start, s.char_end)
            ]
            self._cache[doc.document_id] = (
                kept,
                self.embedder.encode_corpus([s.text for s in kept]),
            )
        return self._cache[doc.document_id]

    def _lex_grams(self, doc) -> list[set[tuple[str, ...]]]:
        if doc.document_id not in self._lex_cache:
            sents, _ = self._doc_sentences(doc)
            self._lex_cache[doc.document_id] = [_shingles(s.text, _LEX_N) for s in sents]
        return self._lex_cache[doc.document_id]

    def accepted_runs(self, query_doc, match_doc):
        """Baseline runs, plus exact-run sentence pairs not already covered."""
        key = (query_doc.document_id, match_doc.document_id)
        if key in self._runs_cache:
            return self._runs_cache[key]
        runs, rejected = super().accepted_runs(query_doc, match_doc)
        if self.lexical_min_shared <= 0:
            return runs, rejected
        q_sents, q_vecs = self._doc_sentences(query_doc)
        m_sents, m_vecs = self._doc_sentences(match_doc)
        covered = {qi for run in runs for qi, _, _ in run}
        q_grams, m_grams = self._lex_grams(query_doc), self._lex_grams(match_doc)
        min_tokens = self.cfg.alignment.single_sentence_min_tokens
        extra: list[tuple[int, int, float]] = []
        for qi, qg in enumerate(q_grams):
            if qi in covered or not qg or token_count(q_sents[qi].text) < min_tokens:
                continue
            best_mi, best_n = -1, 0
            for mi, mg in enumerate(m_grams):
                n = len(qg & mg)
                if n > best_n:
                    best_mi, best_n = mi, n
            if best_n >= self.lexical_min_shared:
                extra.append((qi, best_mi, float(q_vecs[qi] @ m_vecs[best_mi])))
        if not extra:
            return runs, rejected
        grouped: list[list[tuple[int, int, float]]] = []
        for t in sorted(extra):
            if grouped:
                pq, pm, _ = grouped[-1][-1]
                same_pages = (
                    q_sents[pq].page == q_sents[t[0]].page
                    and m_sents[pm].page == m_sents[t[1]].page
                )
                if same_pages and 0 < t[0] - pq <= _RUN_GAP and 0 <= t[1] - pm <= _RUN_GAP:
                    grouped[-1].append(t)
                    continue
            grouped.append([t])
        merged = sorted(runs + grouped, key=lambda r: r[0][0])
        self._runs_cache[key] = (merged, rejected)
        return self._runs_cache[key]
