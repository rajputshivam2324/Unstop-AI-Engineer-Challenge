#!/usr/bin/env python
"""Corpus-scale plagiarism-retrieval benchmark (the engine's peer path at N=400).

The 14-file sample corpus has only 3 positive pairs, so its precision/recall have
huge variance. This benchmark builds a larger, REAL-TEXT corpus with known
ground truth and runs the engine's exact peer pipeline on it:

  host documents : 400 docs, each = 3 human answers from one HC3 subset (finance /
                   open_qa / wiki_csai). Same-subset docs share vocabulary and topic,
                   which is the realistic false-positive pressure.
  exact_copy     : 40 disjoint pairs (A, B): B's middle answer is replaced by A's
                   middle answer verbatim (~1/3 of each doc copied).
  paraphrase     : 40 disjoint pairs: both docs' middle answers are replaced by a
                   4-sentence passage built from MRPC *human-labelled paraphrase* pairs
                   (A gets sentence1s, B gets sentence2s) -> real human paraphrases.
  template       : 12 further docs get the same 110-word instruction block prepended
                   (Scenario 4 at scale) -> those pairs must NOT be retrieved.

Pipeline (identical to cli.run_pipeline S2-S4): build_chunks -> TemplateRegistry ->
MiniLM embeddings -> FAISS IndexFlatIP top-k -> EvidenceGate ->
pairwise_doc_similarity. A pair's score is max over both directions.
Usage: .venv/bin/python benchmarks/bench_retrieval.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "cache" / "hf"))

from integrity_engine.alignment import (DocumentAligner, EvidenceGate,  # noqa: E402
                                        TemplateRegistry, evidence_pair_scores)
from integrity_engine.config import load_config  # noqa: E402
from integrity_engine.embeddings import Embedder  # noqa: E402
from integrity_engine.indexing import (SelfMatchExclusion, build_index,  # noqa: E402
                                       pairwise_doc_similarity, search_neighbors)
from integrity_engine.metrics import prf, roc_auc  # noqa: E402
from integrity_engine.text_processing import build_chunks, token_count  # noqa: E402
from integrity_engine.types import Page, ProcessedDocument  # noqa: E402

DATA = ROOT / "benchmarks" / "data"
RESULTS = ROOT / "benchmarks" / "results"
N_DOCS, N_COPY, N_PARA, N_TEMPLATE = 400, 40, 40, 12
TEMPLATE_BLOCK = (
    "Submission instructions for the regional finance case challenge. Read the "
    "attached brief carefully before you begin. Answer every question in your own "
    "words and cite each figure you use from the data appendix. Keep the main memo "
    "under two pages and place supporting calculations in a separate exhibit. Do not "
    "include your name or university anywhere in the document; use only your "
    "candidate code. Late submissions are not accepted, and the panel may request a "
    "short oral defence of any figure in the memo. Plagiarism of any kind leads to "
    "disqualification from the competition and notification of your institution.")


def clean(text: str) -> str:
    return " ".join(re.sub(r"([.!?])([A-Z])", r"\1 \2", text).split())


def human_answers() -> dict[str, list[str]]:
    """First human answer per question, 50-220 tokens, DE-DUPLICATED: HC3 reuses
    some answers / Wikipedia sentences across questions (37 duplicate answers, 201
    repeated sentences in this pool), which would be unlabelled true copies. Any
    answer sharing a >= 6-word sentence with an earlier answer is dropped."""
    out: dict[str, list[str]] = {}
    seen: set[str] = set()
    for subset in ("finance", "open_qa", "wiki_csai"):
        answers = []
        for line in (DATA / f"hc3_{subset}.jsonl").read_text().splitlines():
            row = json.loads(line)
            for ans in row.get("human_answers", [])[:1]:
                ans = clean(ans)
                if not 50 <= token_count(ans) <= 220:
                    continue
                keys = {" ".join(re.findall(r"\w+", x.lower()))
                        for x in re.split(r"(?<=[.!?])\s+", ans) if len(x.split()) >= 6}
                if keys & seen:
                    continue
                seen |= keys
                answers.append(ans)
        out[subset] = answers
    return out


def mrpc_passages(n: int, rng: random.Random) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in (DATA / "mrpc_test.jsonl").read_text().splitlines()]
    pos = [(" ".join(r["sentence1"].split()), " ".join(r["sentence2"].split()))
           for r in rows if int(r["label"]) == 1]
    rng.shuffle(pos)
    passages = []
    for i in range(n):
        group = pos[4 * i:4 * i + 4]
        passages.append((" ".join(a for a, _ in group), " ".join(b for _, b in group)))
    return passages


def make_doc(doc_id: str, text: str) -> ProcessedDocument:
    return ProcessedDocument(document_id=doc_id, path=Path(doc_id), modality="docx",
                             status="ok", pages=[Page(None, text, 0, len(text))],
                             full_text=text, sha256=doc_id, size_bytes=len(text))


def main() -> int:
    rng = random.Random(42)
    cfg = load_config(ROOT / "config" / "default.toml")
    answers = human_answers()
    subsets = sorted(answers)
    for s in subsets:
        rng.shuffle(answers[s])
    segments: list[list[str]] = []
    cursor = {s: 0 for s in subsets}
    i = 0
    while len(segments) < N_DOCS:  # round-robin over subsets that still have 3 answers
        s = subsets[i % len(subsets)]
        i += 1
        if cursor[s] + 3 <= len(answers[s]):
            segments.append(answers[s][cursor[s]:cursor[s] + 3])
            cursor[s] += 3
        elif all(cursor[x] + 3 > len(answers[x]) for x in subsets):
            raise SystemExit("not enough HC3 answers for the requested corpus size")
    order = list(range(N_DOCS))
    rng.shuffle(order)
    truth: dict[frozenset, str] = {}
    pos = 0
    for _ in range(N_COPY):
        a, b = order[pos], order[pos + 1]
        pos += 2
        segments[b][1] = segments[a][1]
        truth[frozenset((a, b))] = "exact_copy"
    for p1, p2 in mrpc_passages(N_PARA, rng):
        a, b = order[pos], order[pos + 1]
        pos += 2
        segments[a][1], segments[b][1] = p1, p2
        truth[frozenset((a, b))] = "paraphrase"
    template_docs = order[pos:pos + N_TEMPLATE]
    for t in template_docs:
        segments[t] = [TEMPLATE_BLOCK] + segments[t]

    docs = [make_doc(f"{i:03d}_doc", "\n\n".join(seg)) for i, seg in enumerate(segments)]
    t0 = time.time()
    chunks = []
    for doc in docs:
        chunks.extend(build_chunks(doc, cfg))
    registry = TemplateRegistry(cfg.alignment.template_min_docs,
                                cfg.alignment.template_hamming_max,
                                cfg.alignment.template_min_tokens,
                                cfg.alignment.verbatim_template_min_tokens)
    registry.register_all_chunks(chunks)
    t1 = time.time()
    embedder = Embedder(cfg.embedding.model_name, str(ROOT / "cache" / "hf"))
    t2 = time.time()
    matrix = embedder.encode_corpus([c.text for c in chunks])
    t3 = time.time()
    index = build_index(matrix)
    exclude = SelfMatchExclusion([c.document_id for c in chunks])
    sims = {i: search_neighbors(index, matrix[i:i + 1], cfg.pairwise.chunk_query_topk,
                                exclude, i) for i in range(len(chunks))}
    t4 = time.time()
    gate = EvidenceGate(chunks, embedder, cfg)
    chunk_pairs = pairwise_doc_similarity(chunks, sims, cfg.pairwise, {}, verify=gate)
    t5 = time.time()
    aligner = DocumentAligner(embedder, registry, cfg, chunks)
    pairs = evidence_pair_scores(docs, aligner, chunk_pairs, {}, cfg.pairwise.chunk_query_topk)
    t6 = time.time()
    chunk_score: dict[frozenset, float] = {}
    for p in chunk_pairs:
        k = frozenset((int(p.query_document_id[:3]), int(p.match_document_id[:3])))
        chunk_score[k] = max(chunk_score.get(k, 0.0), p.score)

    score: dict[frozenset, float] = {}
    for p in pairs:
        key = frozenset((int(p.query_document_id[:3]), int(p.match_document_id[:3])))
        score[key] = max(score.get(key, 0.0), p.score)
    all_keys = [frozenset((i, j)) for i in range(N_DOCS) for j in range(i + 1, N_DOCS)]
    y = [1 if k in truth else 0 for k in all_keys]
    s = [score.get(k, 0.0) for k in all_keys]
    sweep = []
    for th in (0.05, 0.1, 0.2, 0.3, 0.4, 0.7):
        pred = {k for k, v in score.items() if v >= th}
        neg_fp = sum(1 for k in pred if k not in truth)
        tp = sum(1 for k in pred if k in truth)
        p_, r_, f_ = prf(tp, neg_fp, len(truth) - tp)
        sweep.append({"threshold": th, "relation": "all", "precision": p_, "recall": r_,
                      "f1": f_, "fp": neg_fp})
        for rel in ("exact_copy", "paraphrase"):
            rel_keys = {k for k, v in truth.items() if v == rel}
            rtp = sum(1 for k in pred if k in rel_keys)
            p_, r_, f_ = prf(rtp, neg_fp, len(rel_keys) - rtp)
            sweep.append({"threshold": th, "relation": rel, "precision": p_, "recall": r_,
                          "f1": f_, "fp": neg_fp})
    tmpl_pairs = [score.get(frozenset((a, b)), 0.0) for i, a in enumerate(template_docs)
                  for b in template_docs[i + 1:]]
    rel_scores = {rel: sorted(round(score.get(k, 0.0), 3) for k, v in truth.items()
                              if v == rel) for rel in ("exact_copy", "paraphrase")}
    negatives = sorted((v for k, v in score.items() if k not in truth), reverse=True)
    result = {
        "corpus": f"{N_DOCS} docs from HC3 human answers + MRPC paraphrases "
                  f"({N_COPY} exact-copy pairs, {N_PARA} paraphrase pairs, {N_TEMPLATE} "
                  "template docs)",
        "n_chunks": len(chunks), "n_candidate_pairs": len(all_keys),
        "template_chunks": sum(c.is_template for c in chunks),
        "auc_all_pairs": roc_auc(s, y),
        "auc_all_pairs_chunk_level_only": roc_auc([chunk_score.get(k, 0.0) for k in all_keys], y),
        "n_sentences_indexed": aligner.n_sentences_indexed,
        "sweep": sweep,
        "truth_pair_scores": rel_scores,
        "top_negative_scores": [round(v, 3) for v in negatives[:10]],
        "template": {"n_docs": N_TEMPLATE, "max_pair_score": max(tmpl_pairs or [0.0])},
        "timing": {"chunk+template_s": round(t1 - t0, 2), "model_load_s": round(t2 - t1, 2),
                   "embed_s": round(t3 - t2, 2),
                   "chunks_per_s": round(len(chunks) / max(t3 - t2, 1e-9), 1),
                   "faiss_search_s": round(t4 - t3, 2),
                   "chunk_evidence_gate_s": round(t5 - t4, 2),
                   "sentence_index+alignment_s": round(t6 - t5, 2)},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=1))
    for row in sweep:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
