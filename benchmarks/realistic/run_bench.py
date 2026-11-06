#!/usr/bin/env python
"""Score every document pair of one benchmark split with one method.

One method per process, so peak RSS and wall time belong to that method alone.
Writes benchmarks/realistic/results/<split>_<method>.json with the score of every
pair that scored above 0 (all other pairs score 0).

Methods
  shingle_moss    word 5-gram containment, max over both directions; shingles found
                  in >= 3 documents are ignored (MOSS-style frequent-passage filter)
  tfidf_cosine    TF-IDF (1-2 grams, sublinear tf, English stop words) document cosine
  embed_doc       MiniLM: cosine of mean chunk embeddings (whole-document vector)
  embed_maxchunk  MiniLM: best chunk-to-chunk cosine (naive vector-search detector)
  engine_baseline engine peer path, default config (sentence-evidence coverage)
  engine_hybrid   engine peer path, improvements/configs/proposed.toml (semantic_hybrid)

The test split refuses to run unless benchmarks/realistic/freeze.json matches the
current data, benchmark code, engine code and configs.

Usage: .venv/bin/python benchmarks/realistic/run_bench.py --split dev --method tfidf_cosine
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

METHODS = (
    "shingle_moss",
    "tfidf_cosine",
    "embed_doc",
    "embed_maxchunk",
    "engine_baseline",
    "engine_hybrid",
    "engine_v2",
)
FROZEN = [
    "benchmarks/realistic/data/test.json",
    "benchmarks/realistic/run_bench.py",
    "benchmarks/realistic/evaluate.py",
    "config/default.toml",
    "improvements/configs/proposed.toml",
    "config/evidence_v2.toml",
]
ENGINE_CONFIGS = {
    "engine_baseline": "config/default.toml",
    "engine_hybrid": "improvements/configs/proposed.toml",
    "engine_v2": "config/evidence_v2.toml",
}
FROZEN_GLOBS = ["integrity_engine/*.py"]


def frozen_files() -> list[str]:
    out = list(FROZEN)
    for g in FROZEN_GLOBS:
        out += sorted(str(p.relative_to(ROOT)) for p in ROOT.glob(g))
    return out


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def check_freeze() -> None:
    freeze = json.loads((HERE / "freeze.json").read_text())
    current = {p: sha(p) for p in frozen_files()}
    if current != freeze["sha256"]:
        changed = sorted(set(current.items()) ^ set(freeze["sha256"].items()))
        raise SystemExit(f"frozen inputs changed; refusing to score test: {changed[:4]}")


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def shingle_moss(texts: list[str]) -> dict[tuple[int, int], float]:
    sh = [{tuple(w[i : i + 5]) for i in range(len(w) - 4)} for w in map(words, texts)]
    owners: dict[tuple, list[int]] = defaultdict(list)
    for d, s in enumerate(sh):
        for g in s:
            owners[g].append(d)
    shared: dict[tuple[int, int], int] = defaultdict(int)
    for docs in owners.values():
        if len(docs) == 2:  # >= 3 documents -> frequent passage, ignored
            shared[(docs[0], docs[1])] += 1
    keep = [sum(1 for g in s if len(owners[g]) < 3) for s in sh]
    return {k: max(v / max(keep[k[0]], 1), v / max(keep[k[1]], 1)) for k, v in shared.items()}


def tfidf_cosine(texts: list[str]) -> dict[tuple[int, int], float]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    m = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english").fit_transform(
        texts
    )
    sim = (m @ m.T).toarray()
    return {(i, j): float(sim[i, j]) for i, j in itertools.combinations(range(len(texts)), 2)}


def engine_inputs(texts: list[str], ids: list[str], cfg):
    from integrity_engine.text_processing import build_chunks
    from integrity_engine.types import Page, ProcessedDocument

    docs = [
        ProcessedDocument(
            document_id=f"{i}.docx",
            path=Path(f"{i}.docx"),
            modality="docx",
            status="ok",
            pages=[Page(None, t, 0, len(t))],
            full_text=t,
            sha256=hashlib.sha256(t.encode()).hexdigest(),
            size_bytes=len(t),
        )
        for i, t in zip(ids, texts)
    ]
    chunks = [c for d in docs for c in build_chunks(d, cfg)]
    return docs, chunks


def embed_scores(texts: list[str], ids: list[str], whole_doc: bool) -> dict[tuple[int, int], float]:
    import numpy as np

    from integrity_engine.config import load_config
    from integrity_engine.embeddings import build_embedder

    cfg = load_config(ROOT / "config" / "default.toml")
    _, chunks = engine_inputs(texts, ids, cfg)
    emb = build_embedder(cfg.embedding, str(ROOT / "cache" / "hf"))
    mat = emb.encode_corpus([c.text for c in chunks])
    pos = {d: n for n, d in enumerate(f"{i}.docx" for i in ids)}
    owner = np.array([pos[c.document_id] for c in chunks])
    n = len(texts)
    if whole_doc:
        doc = np.stack([mat[owner == d].mean(0) for d in range(n)])
        doc /= np.linalg.norm(doc, axis=1, keepdims=True)
        sim = doc @ doc.T
    else:
        cs = mat @ mat.T
        sim = np.full((n, n), -1.0)
        for d in range(n):
            rows = cs[owner == d]
            for e in range(d + 1, n):
                sim[d, e] = rows[:, owner == e].max()
    return {(i, j): float(sim[i, j]) for i, j in itertools.combinations(range(n), 2)}


def engine_scores(texts: list[str], ids: list[str], config: str) -> tuple[dict, dict]:
    """The CLI peer path (cli.run_pipeline S2-S4b) on in-memory documents."""
    from integrity_engine.alignment import (
        DocumentAligner,
        EvidenceGate,
        TemplateRegistry,
        evidence_pair_scores,
    )
    from integrity_engine.config import load_config
    from integrity_engine.embeddings import build_embedder
    from integrity_engine.indexing import (
        SelfMatchExclusion,
        build_index,
        pairwise_doc_similarity,
        search_neighbors,
    )

    cfg = load_config(ROOT / config)
    docs, chunks = engine_inputs(texts, ids, cfg)
    registry = TemplateRegistry(
        min_docs=cfg.alignment.template_min_docs,
        hamming_max=cfg.alignment.template_hamming_max,
        min_fingerprint_tokens=cfg.alignment.template_min_tokens,
        verbatim_template_min_tokens=cfg.alignment.verbatim_template_min_tokens,
    )
    registry.register_all_chunks(chunks)
    emb = build_embedder(cfg.embedding, str(ROOT / "cache" / "hf"))
    mat = emb.encode_corpus([c.text for c in chunks])
    for c, v in zip(chunks, mat, strict=True):
        c.embedding = v
    index = build_index(mat)
    exclude = SelfMatchExclusion([c.document_id for c in chunks], {})
    sims = {
        i: search_neighbors(index, mat[i : i + 1], cfg.pairwise.chunk_query_topk, exclude, i)
        for i in range(len(chunks))
    }
    chunk_pairs = pairwise_doc_similarity(
        chunks, sims, cfg.pairwise, {}, verify=EvidenceGate(chunks, emb, cfg)
    )
    if cfg.matching.mode == "baseline":
        aligner = DocumentAligner(emb, registry, cfg, chunks)
    elif cfg.matching.mode == "evidence_v2":
        from integrity_engine.evidence_v2 import EvidenceV2Aligner

        aligner = EvidenceV2Aligner(emb, registry, cfg, chunks)
    else:
        from integrity_engine.enhanced_alignment import EnhancedDocumentAligner

        aligner = EnhancedDocumentAligner(emb, registry, cfg, chunks)
    pairs = evidence_pair_scores(docs, aligner, chunk_pairs, {}, cfg.pairwise.chunk_query_topk)
    pos = {f"{i}.docx": n for n, i in enumerate(ids)}
    out: dict[tuple[int, int], float] = {}
    for p in pairs:
        k = tuple(sorted((pos[p.query_document_id], pos[p.match_document_id])))
        out[k] = max(out.get(k, 0.0), p.score)
    extra = {"mode": cfg.matching.mode, "template_chunks": len(registry._template_ids)}
    v = getattr(aligner, "verifier", None)
    if v is not None:
        extra |= {
            "verifier_pairs_scored": getattr(v, "pairs_scored", None),
            "truncated_pairs": getattr(v, "truncated_pairs", None),
        }
    return out, extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument(
        "--config", default=None, help="dev only: alternative engine config (for tuning sweeps)"
    )
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    a = ap.parse_args()
    out = Path(a.out_dir) / f"{a.split}_{a.method}{a.tag}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    if a.split == "test":
        if a.config or a.tag:
            raise SystemExit("--config/--tag are dev-only")
        check_freeze()
    data = json.loads((HERE / "data" / f"{a.split}.json").read_text())
    texts = [d["text"] for d in data["docs"]]
    ids = [d["id"] for d in data["docs"]]
    t0 = time.monotonic()
    extra: dict = {}
    if a.method == "shingle_moss":
        scores = shingle_moss(texts)
    elif a.method == "tfidf_cosine":
        scores = tfidf_cosine(texts)
    elif a.method.startswith("embed_"):
        scores = embed_scores(texts, ids, whole_doc=a.method == "embed_doc")
    else:
        scores, extra = engine_scores(texts, ids, a.config or ENGINE_CONFIGS[a.method])
        extra["config"] = a.config or ENGINE_CONFIGS[a.method]
    wall = time.monotonic() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": a.split,
        "method": a.method,
        "n_docs": len(texts),
        "wall_s": round(wall, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "data_sha256": sha(f"benchmarks/realistic/data/{a.split}.json"),
        **extra,
        "scores": [[i, j, round(s, 6)] for (i, j), s in sorted(scores.items()) if s > 0],
    }
    out.write_text(json.dumps(payload))
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k != "scores"}
            | {"nonzero": len(payload["scores"])}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
