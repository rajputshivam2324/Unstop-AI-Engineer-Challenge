"""Opt-in boundary reconstruction and retrieve-then-verify alternatives.

The original DocumentAligner remains the baseline. MiniLM still determines
reported cosine; verifier scores and method appear in the existing note field.
"""

from __future__ import annotations

import numpy as np

from integrity_engine.alignment import (
    DocumentAligner,
    _lcs_token_ratio,
    _overlap_kind,
    _sentence_key,
    page_at,
    time_span,
    token_jaccard,
)
from integrity_engine.boundaries import adjacent_windows, reconstructed_sentences
from integrity_engine.semantic import build_verifier
from integrity_engine.text_processing import token_count
from integrity_engine.types import MatchedPassage


class EnhancedDocumentAligner(DocumentAligner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = self.cfg.matching
        self.verifier = None
        if self.options.mode.startswith("semantic_"):
            kind = self.options.mode.removeprefix("semantic_")
            self.verifier = build_verifier(kind, self.embedder.cache_dir)
        self._window_cache = {}
        self._evidence_cache = {}
        self._evidence_spans = {}

    def _doc_sentences(self, doc):
        if doc.document_id not in self._cache:
            spans = self._template_spans.get(doc.document_id, [])
            kept = [s for s in reconstructed_sentences(
                doc, max_gap_s=self.options.asr_max_gap_s,
                max_tokens=self.options.window_max_tokens)
                if _sentence_key(s.text) not in self.registry.template_sentences
                and not any(a <= s.char_start and s.char_end <= b for a, b in spans)]
            self._cache[doc.document_id] = (
                kept, self.embedder.encode_corpus([s.text for s in kept]))
        return self._cache[doc.document_id]

    def _windows(self, doc):
        if doc.document_id not in self._window_cache:
            sents, _ = self._doc_sentences(doc)
            windows = adjacent_windows(doc, sents, self.options.boundary_max_units,
                                       self.options.window_max_tokens,
                                       self.options.asr_max_gap_s)
            # Include a whole short page/paragraph: NLI needs argumentative context
            # that a single sentence may omit. Never span PDF pages/templates.
            for page in dict.fromkeys(s.page for s in sents):
                group = [s for s in sents if s.page == page]
                if (doc.modality != "mp4" and group
                        and token_count(doc.full_text[group[0].char_start:group[-1].char_end])
                        <= self.options.window_max_tokens):
                    # Require continuity through canonical units (no excluded template).
                    text = doc.full_text[group[0].char_start:group[-1].char_end]
                    if token_count(text) == sum(token_count(s.text) for s in group):
                        from integrity_engine.types import Sentence
                        windows.append(Sentence(text, group[0].char_start,
                                                group[-1].char_end, page))
            minimum = self.cfg.alignment.single_sentence_min_tokens
            windows = list({(w.char_start, w.char_end): w for w in windows
                            if token_count(w.text) >= minimum}.values())
            # Do not bridge an excluded internal template span in sliding windows.
            spans = self._template_spans.get(doc.document_id, [])
            windows = [w for w in windows
                       if not any(a < w.char_end and b > w.char_start for a, b in spans)
                       and not any(_sentence_key(s.text) in self.registry.template_sentences
                                   for s in reconstructed_sentences(doc)
                                   if s.char_start >= w.char_start and s.char_end <= w.char_end)]
            self._window_cache[doc.document_id] = (
                windows, self.embedder.encode_corpus([w.text for w in windows]))
        return self._window_cache[doc.document_id]

    def candidate_pairs(self, docs, top_k):
        """Broad candidate generation is separate from unchanged acceptance gates."""
        import faiss

        best = super().candidate_pairs(docs, top_k)
        owners, blocks = [], []
        for doc in docs:
            windows, vecs = self._windows(doc)
            if windows:
                owners.extend([doc.document_id] * len(windows))
                blocks.append(vecs)
        if not blocks:
            return best
        matrix = np.ascontiguousarray(np.vstack(blocks), dtype=np.float32)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        # Adaptive overfetch excludes arbitrarily many units of one's own file.
        for i, vector in enumerate(matrix):
            k = min(len(owners), self.options.retrieval_topk + 1)
            while True:
                sims, ids = index.search(vector[None, :], k)
                valid = [(int(j), float(s)) for j, s in zip(ids[0], sims[0])
                         if j >= 0 and owners[int(j)] != owners[i]]
                if len(valid) >= self.options.retrieval_topk or k == len(owners):
                    break
                k = min(len(owners), k * 2)
            for j, score in valid[:self.options.retrieval_topk]:
                if score >= self.options.retrieval_min_cos:
                    key = frozenset((owners[i], owners[j]))
                    best[key] = max(best.get(key, 0.0), score)
        return best

    def _evidence(self, query, match):
        key = (query.document_id, match.document_id)
        if key in self._evidence_cache:
            return self._evidence_cache[key]
        original, rejected = super().align(query, match)
        # Exact lexical evidence stays available. A verifier is not a plagiarism
        # oracle: nearly identical text is reuse even if a fact was altered.
        evidence = [p for p in original if self.verifier is None or
                    (p.token_lcs_ratio or 0) >= 0.90]
        # Map baseline passages back via their accepted runs, including repeated
        # identical sentences on different pages. Never guess using str.find().
        sents, _ = self._doc_sentences(query)
        runs, _ = self.accepted_runs(query, match)
        for p in evidence:
            for run in runs:
                start, end = sents[run[0][0]].char_start, sents[run[-1][0]].char_end
                if (query.full_text[start:end] == p.query_passage
                        and page_at(query, start) == p.query_page
                        and time_span(query, start, end) == p.query_timestamp):
                    self._evidence_spans[id(p)] = (start, end)
                    break
        qw, qv = self._windows(query)
        mw, mv = self._windows(match)
        if qw and mw:
            sims = qv @ mv.T
            candidates = set()
            for i, row in enumerate(sims):
                candidates.update((i, int(j)) for j in np.argsort(-row)[:self.options.verifier_topk]
                                  if row[j] >= self.options.retrieval_min_cos)
            for j, col in enumerate(sims.T):
                candidates.update((int(i), j) for i in np.argsort(-col)[:self.options.verifier_topk]
                                  if col[i] >= self.options.retrieval_min_cos)
            candidates = sorted(candidates, key=lambda ij: (-float(sims[ij]), ij))[
                :self.options.max_verifications_per_pair]
            if self.verifier:
                verified = self.verifier.score_pairs([(qw[i].text, mw[j].text)
                                                       for i, j in candidates])
            else:
                verified = [None] * len(candidates)
            for (i, j), result in zip(candidates, verified):
                a, b, cosine = qw[i], mw[j], float(sims[i, j])
                if len(a.text) + len(b.text) < self.cfg.alignment.min_evidence_chars:
                    continue
                lcs = _lcs_token_ratio(a.text, b.text)
                if self.verifier:
                    threshold = (self.options.nli_entailment_min if self.verifier.kind == "nli"
                                 else self.options.sts_min)
                    strong = result.similarity >= threshold
                    if self.verifier.kind == "hybrid":
                        strong |= (result.entailment or 0.0) >= self.options.nli_entailment_min
                    accept = (not result.truncated and strong
                              and result.contradiction <= self.options.nli_contradiction_max)
                    note = (f"{self.verifier.kind} verified candidate: "
                            f"score={result.similarity:.4f}, "
                            f"contradiction={result.contradiction:.4f}; "
                            + (f"bidirectional_entailment={result.entailment:.4f}; "
                               if result.entailment is not None else "") +
                            "reported similarity is MiniLM cosine; human review required")
                else:
                    # Adjacent ASR boundaries may contain premature punctuation;
                    # demand strong lexical AND semantic support, never lower floors.
                    accept = ((query.modality == "mp4" or match.modality == "mp4")
                              and lcs >= 0.80
                              and cosine >= self.cfg.alignment.sentence_align_cos_floor)
                    note = "boundary-tolerant spoken/written overlap; adjacent ASR spans retained"
                if not accept:
                    continue
                kind = _overlap_kind(query, match, "overlap", lcs)
                passage = MatchedPassage(
                    matched_file=match.document_id, similarity_score=round(cosine, 4),
                    query_passage=a.text, matched_passage=b.text,
                    query_page=page_at(query, a.char_start),
                    matched_page=page_at(match, b.char_start),
                    query_timestamp=time_span(query, a.char_start, a.char_end),
                    matched_timestamp=time_span(match, b.char_start, b.char_end),
                    # This is a verified WINDOW, not a sentence-to-sentence
                    # alignment. Do not fabricate per-sentence cosine values.
                    match_type="overlap", evidence_sentence_scores=[],
                    overlap_kind=kind, lexical_jaccard=round(token_jaccard(a.text, b.text), 4),
                    token_lcs_ratio=round(lcs, 4), note=note)
                evidence.append(passage)
                self._evidence_spans[id(passage)] = (a.char_start, a.char_end)
        # Prefer complete evidence, suppress contained window duplicates. Distinct
        # partial overlaps remain visible, but their coverage is never double-counted.
        kept = []
        for p in sorted(evidence, key=lambda p: (-(len(p.query_passage) + len(p.matched_passage)),
                                                -p.similarity_score)):
            if not any(p.query_passage in q.query_passage and
                       p.matched_passage in q.matched_passage for q in kept):
                kept.append(p)
        self._evidence_cache[key] = (kept, rejected)
        return self._evidence_cache[key]

    def align(self, query_doc, match_doc, chunk_cosine=0.0):
        return self._evidence(query_doc, match_doc)

    def coverage(self, query_doc, match_doc):
        sentences, _ = self._doc_sentences(query_doc)
        total = sum(token_count(s.text) for s in sentences)
        if not total:
            return 0.0
        passages, _ = self._evidence(query_doc, match_doc)
        spans = []
        for p in passages:
            if id(p) in self._evidence_spans:
                spans.append(self._evidence_spans[id(p)])
        merged = []
        for a, b in sorted(set(spans)):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        covered = sum(token_count(query_doc.full_text[a:b]) for a, b in merged)
        return min(1.0, covered / total)

    def release(self):
        if self.verifier:
            self.verifier.release()
        self._window_cache.clear()
