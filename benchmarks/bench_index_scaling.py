#!/usr/bin/env python
"""Vector-index scaling benchmark: FAISS Flat vs IVF vs HNSW on REAL MiniLM vectors.

Feeds the README's 10k-submission brief with measured numbers instead of folklore.
Vectors: MiniLM-L6 embeddings of HC3 answer sentences (human + ChatGPT, 3 subsets).
For N in {10k, all available}: build each index, run 1,000 held-out queries,
measure build time, single-query latency, batch throughput, recall@10 against the
exact Flat result, and serialized index size.

A 10,000-submission challenge at ~3 chunks per document is ~30k chunk vectors,
so the measured N range brackets the production case.
Usage: .venv/bin/python benchmarks/bench_index_scaling.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "cache" / "hf"))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from integrity_engine.config import load_config  # noqa: E402
from integrity_engine.embeddings import Embedder  # noqa: E402

DATA = ROOT / "benchmarks" / "data"
RESULTS = ROOT / "benchmarks" / "results"
CACHE = RESULTS / "hc3_sentence_vectors.npy"  # gitignored (large)
MAX_VECTORS = 60_000
N_QUERIES = 1000


def sentences() -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for subset in ("finance", "open_qa", "wiki_csai"):
        for line in (DATA / f"hc3_{subset}.jsonl").read_text().splitlines():
            row = json.loads(line)
            for ans in row.get("human_answers", []) + row.get("chatgpt_answers", []):
                text = re.sub(r"([.!?])([A-Z])", r"\1 \2", ans)
                for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
                    if len(s.split()) >= 6 and s not in seen:
                        seen.add(s)
                        out.append(s)
    return out[:MAX_VECTORS + N_QUERIES]


def vectors() -> tuple[np.ndarray, float]:
    if CACHE.exists():
        return np.load(CACHE), float("nan")
    cfg = load_config(ROOT / "config" / "default.toml")
    emb = Embedder(cfg.embedding.model_name, str(ROOT / "cache" / "hf"), batch_size=128)
    texts = sentences()
    t0 = time.time()
    mat = emb.encode_corpus(texts)
    rate = len(texts) / (time.time() - t0)
    np.save(CACHE, mat)
    return mat, rate


def measure(name: str, index, base: np.ndarray, queries: np.ndarray,
            truth: np.ndarray | None, train: bool) -> dict:
    t0 = time.time()
    if train:
        index.train(base)
    index.add(base)
    build = time.time() - t0
    t0 = time.time()
    for q in queries[:200]:
        index.search(q[None, :], 10)
    single_ms = (time.time() - t0) / 200 * 1000
    t0 = time.time()
    _, ids = index.search(queries, 10)
    qps = len(queries) / (time.time() - t0)
    recall = 1.0 if truth is None else float(np.mean(
        [len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)]))
    size_mb = faiss.serialize_index(index).nbytes / 1e6
    return {"index": name, "n": base.shape[0], "build_s": build, "single_query_ms": single_ms,
            "batch_qps": qps, "recall_at_10": recall, "memory_mb": size_mb, "_ids": ids}


def main() -> int:
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    mat, rate = vectors()
    queries, pool = mat[:N_QUERIES], mat[N_QUERIES:]
    dim = mat.shape[1]
    rows = []
    for n in sorted({10_000, pool.shape[0]}):
        base = np.ascontiguousarray(pool[:n])
        flat = measure("Flat (exact IP)", faiss.IndexFlatIP(dim), base, queries, None, False)
        truth = flat.pop("_ids")
        rows.append(flat)
        nlist = int(4 * math.sqrt(n))
        for nprobe in (8, 32):
            ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist,
                                     faiss.METRIC_INNER_PRODUCT)
            ivf.nprobe = nprobe
            r = measure(f"IVFFlat nlist={nlist} nprobe={nprobe}", ivf, base, queries, truth,
                        True)
            r.pop("_ids")
            rows.append(r)
        for ef in (32, 128):
            hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            hnsw.hnsw.efConstruction = 80
            hnsw.hnsw.efSearch = ef
            r = measure(f"HNSWFlat M=32 efSearch={ef}", hnsw, base, queries, truth, False)
            r.pop("_ids")
            rows.append(r)
    result = {"vectors": f"{pool.shape[0]:,} MiniLM-L6 (384-d) HC3 sentence embeddings, "
                         f"{N_QUERIES} held-out queries, top-10, "
                         f"{faiss.omp_get_max_threads()} threads",
              "encode_sentences_per_s": rate, "rows": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "index_scaling.json").write_text(json.dumps(result, indent=2))
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("encode rate", rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
