"""<MODULE PURPOSE>
FAISS IndexFlatIP search over L2-normalized chunk embeddings, self-match +
byte-duplicate exclusion (D7/A2), and the coverage-weighted mean-max doc-pair
aggregation that produces PairCandidates (the peer% basis: max of the
query-side and source-side coverage scores).

<PUBLIC API>
build_index, SelfMatchExclusion, search_neighbors, pairwise_doc_similarity

<DEPENDS ON>
faiss, numpy, types (Chunk, PairCandidate), config (PairwiseConfig).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import faiss
import numpy as np

from integrity_engine.config import PairwiseConfig
from integrity_engine.types import Chunk, PairCandidate

logger = logging.getLogger("integrity_engine.indexing")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Exact-cosine flat index; embeddings must already be L2-normalized (D4)."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        dim = embeddings.shape[1] if embeddings.ndim == 2 else 384
    else:
        dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    if embeddings.shape[0] > 0:
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return index


class SelfMatchExclusion:
    """Decides which (query_chunk_i, candidate_chunk_j) pairs must be dropped.

    Rules (D7 + A2, exact):
    1. i == j (self).
    2. same document_id ("identical self-matches").
    3. byte-identical duplicates resolve to one canonical file: pairs across the
       duplicate group are excluded from aggregation math; the duplicate relation
       is instead reported at similarity 1.0 by `pairwise_doc_similarity`.
    """

    def __init__(self, chunk_docs: list[str], duplicates: dict[str, str] | None = None) -> None:
        self.chunk_docs = chunk_docs
        self.duplicates: dict[str, str] = dict(duplicates or {})
        self._canonical: dict[str, str] = {}
        for dup, canonical in self.duplicates.items():
            self._canonical[dup] = canonical

    def canonical_of(self, doc_id: str) -> str:
        """Duplicate doc -> its canonical file's document_id (identity otherwise)."""
        return self._canonical.get(doc_id, doc_id)

    def is_excluded(self, i: int, j: int) -> bool:
        if i == j:
            return True
        if self.chunk_docs[i] == self.chunk_docs[j]:
            return True
        return self.canonical_of(self.chunk_docs[i]) == self.canonical_of(self.chunk_docs[j])

    def is_duplicate_pair(self, doc_a: str, doc_b: str) -> bool:
        """True when one doc is a byte-duplicate of the other's canonical file."""
        return (doc_a != doc_b and
                self.canonical_of(doc_a) == self.canonical_of(doc_b))


def search_neighbors(index: faiss.IndexFlatIP, query_vec: np.ndarray, top_k: int,
                     exclude: SelfMatchExclusion, query_chunk_idx: int) \
        -> list[tuple[int, float]]:
    """Top-k neighbor chunk ids with exclusions applied (over-fetch compensates)."""
    if index.ntotal == 0:
        return []
    fetch_k = int(min(index.ntotal, max(top_k * 3, top_k + 10)))
    sims, ids = index.search(np.ascontiguousarray(query_vec, dtype=np.float32), fetch_k)
    results: list[tuple[int, float]] = []
    for sim, chunk_id in zip(sims[0], ids[0], strict=True):
        if chunk_id == -1:
            continue
        j = int(chunk_id)
        if exclude.is_excluded(query_chunk_idx, j):
            continue
        results.append((j, float(sim)))
        if len(results) >= top_k:
            break
    return results


def pairwise_doc_similarity(all_chunks: list[Chunk],
                            chunk_sims: dict[int, list[tuple[int, float]]],
                            cfg: PairwiseConfig,
                            duplicates: dict[str, str] | None = None,
                            verify: Callable[[int, int], bool] | None = None) \
        -> list[PairCandidate]:
    """Coverage-weighted mean-max doc-pair scores (exact aggregation math §4.4).

    S(Q,M) = (Σ_{c ∈ Q, non-template} max_{m ∈ M, non-template} cos(c, m)) / |Q|,
    where a Q chunk without any ≥ chunk_min_cos neighbor in M contributes 0.
    The reverse score S(M,Q) is computed symmetrically and doc_pair_score uses
    max(S(Q,M), S(M,Q)). Template chunks are removed on both sides (Scenario-4
    discount). Byte-identical duplicate relations are excluded from the math and
    appended as synthetic 1.0 candidates (reported, never double-counted, A2).
    """
    duplicates = dict(duplicates or {})
    exclude = SelfMatchExclusion([c.document_id for c in all_chunks], duplicates)
    docs = sorted({c.document_id for c in all_chunks})
    chunks_of: dict[str, list[int]] = {}
    for i, chunk in enumerate(all_chunks):
        if chunk.is_template:
            continue
        chunks_of.setdefault(chunk.document_id, []).append(i)

    grouped: dict[tuple[str, str], PairCandidate] = {}

    def add_pair(query: str, match: str, score: float, shared: list[tuple[str, str, float]],
                 forward: float, reverse: float) -> None:
        key = (query, match)
        existing = grouped.get(key)
        direction = ("both" if abs(forward - reverse) < 1e-9
                     else ("forward" if forward > reverse else "reverse"))
        cross = query.lower().endswith(".mp4") or match.lower().endswith(".mp4")
        if existing is None or score > existing.score:
            grouped[key] = PairCandidate(
                query_document_id=query, match_document_id=match, score=score,
                shared_chunk_pairs=sorted(shared, key=lambda t: t[0]),
                is_cross_modal=cross, directionality=direction)  # type: ignore[arg-type]

    for query_doc in docs:
        for match_doc in docs:
            if query_doc == match_doc:
                continue
            if exclude.is_duplicate_pair(query_doc, match_doc):
                continue  # synthetic 1.0 candidate is appended below
            q_idx = chunks_of.get(query_doc, [])
            m_idx = chunks_of.get(match_doc, [])
            if not q_idx or not m_idx:
                continue
            forward, f_shared = _coverage(all_chunks, q_idx, m_idx, match_doc,
                                          chunk_sims, exclude, cfg, verify)
            reverse, r_shared = _coverage(
                all_chunks, m_idx, q_idx, query_doc, chunk_sims, exclude, cfg,
                (lambda i, j: verify(j, i)) if verify is not None else None)
            # Evidence = forward pairs + reverse pairs re-oriented to (query, match)
            # so a small source fully contained in a long query still yields
            # evidence on the query's side.
            shared = list(f_shared)
            known = {(q, m) for q, m, _ in shared}
            for m_id, q_id, sim in r_shared:
                if (q_id, m_id) not in known:
                    shared.append((q_id, m_id, sim))
                    known.add((q_id, m_id))
            score = max(forward, reverse)
            add_pair(query_doc, match_doc, score, shared, forward, reverse)

    for dup, canonical in sorted(duplicates.items()):
        dup_chunks = [c.chunk_id for c in all_chunks if c.document_id == dup]
        can_chunks = [c.chunk_id for c in all_chunks if c.document_id == canonical]
        sim = cfg.duplicate_similarity
        add_pair(dup, canonical, sim,
                 [(a, b, sim) for a, b in zip(dup_chunks, can_chunks)], 1.0, 1.0)
        add_pair(canonical, dup, sim,
                 [(b, a, sim) for a, b in zip(dup_chunks, can_chunks)], 1.0, 1.0)

    pairs = list(grouped.values())
    pairs.sort(key=lambda p: (-p.score, p.query_document_id, p.match_document_id))
    return pairs


def _coverage(all_chunks: list[Chunk], q_idx: list[int], m_idx: list[int],
              match_doc: str, chunk_sims: dict[int, list[tuple[int, float]]],
              exclude: SelfMatchExclusion, cfg: PairwiseConfig,
              verify: Callable[[int, int], bool] | None = None) \
        -> tuple[float, list[tuple[str, str, float]]]:
    """S(Q,M): mean over Q chunks of their best >=floor, evidence-verified cosine into M.

    `verify(i, j)` (evidence gate) returns True only when the two chunks share at
    least one aligned sentence pair; a high chunk cosine produced by topic
    similarity alone (no alignable sentence) contributes 0 to coverage.
    """
    m_set = set(m_idx)
    numerator = 0.0
    shared: list[tuple[str, str, float]] = []
    for qi in q_idx:
        best_sim, best_j = 0.0, -1
        for j, sim in sorted(chunk_sims.get(qi, []), key=lambda t: -t[1]):
            if j not in m_set or exclude.is_excluded(qi, j) or sim < cfg.chunk_min_cos:
                continue
            if verify is not None and not verify(qi, j):
                continue
            best_sim, best_j = sim, j
            break
        if best_j >= 0:
            numerator += best_sim
            shared.append((all_chunks[qi].chunk_id, all_chunks[best_j].chunk_id,
                           best_sim))
    coverage = numerator / len(q_idx) if q_idx else 0.0
    return coverage, shared
