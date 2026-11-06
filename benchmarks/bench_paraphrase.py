#!/usr/bin/env python
"""Paraphrase-threshold benchmark (Scenario 2) on MRPC and PAWS.

Question answered: "is MiniLM cosine a better paraphrase signal than the naive
lexical metrics the assignment says fail (Jaccard / edit-style overlap), and what
do the engine's thresholds (0.65 sentence-alignment floor, 0.80 assignment band)
buy in precision / recall?"

  * MRPC test (1,725 human-labelled news sentence pairs, 66% paraphrases).
  * PAWS labeled_final test, first 2,000 pairs: adversarial word-scrambles with
    near-identical vocabulary but often different meaning -> the known failure
    mode of bag-of-meaning embeddings.

Metrics: ROC-AUC for cosine / token-Jaccard / token-LCS ratio, plus P/R/F1/FPR at
the engine thresholds and the best-F1 threshold.
Usage: .venv/bin/python benchmarks/bench_paraphrase.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "cache" / "hf"))

from integrity_engine.alignment import _lcs_token_ratio, token_jaccard
from integrity_engine.config import load_config
from integrity_engine.embeddings import Embedder
from integrity_engine.metrics import best_f1_threshold, roc_auc, threshold_metrics

DATA = ROOT / "benchmarks" / "data"
RESULTS = ROOT / "benchmarks" / "results"


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line]


def bench(rows: list[dict], embedder: Embedder, cfg) -> dict:
    s1 = [r["sentence1"] for r in rows]
    s2 = [r["sentence2"] for r in rows]
    y = [int(r["label"]) for r in rows]
    t0 = time.time()
    a, b = embedder.encode_corpus(s1), embedder.encode_corpus(s2)
    encode_s = time.time() - t0
    cos = [float(x @ z) for x, z in zip(a, b)]
    jac = [token_jaccard(x, z) for x, z in zip(s1, s2)]
    lcs = [_lcs_token_ratio(x, z) for x, z in zip(s1, s2)]
    floor = cfg.alignment.sentence_align_cos_floor
    return {
        "n_pairs": len(rows), "positive_rate": sum(y) / len(y),
        "encode_seconds": round(encode_s, 2),
        "sentences_per_second": round(2 * len(rows) / encode_s, 1),
        "auc": {"minilm_cosine": roc_auc(cos, y), "token_jaccard": roc_auc(jac, y),
                "token_lcs_ratio": roc_auc(lcs, y)},
        "cosine_at_0.80_assignment_band": threshold_metrics(cos, y, 0.80),
        f"cosine_at_{floor}_engine_alignment_floor": threshold_metrics(cos, y, floor),
        "cosine_best_f1": best_f1_threshold(cos, y),
        "jaccard_best_f1": best_f1_threshold(jac, y),
        "scenario2_cases": {
            "desc": "true paraphrases with LOW lexical overlap (Jaccard < 0.35)",
            "n": sum(1 for j, t in zip(jac, y) if t and j < 0.35),
            "caught_at_cos_0.65": sum(1 for c, j, t in zip(cos, jac, y)
                                      if t and j < 0.35 and c >= floor),
            "caught_at_cos_0.80": sum(1 for c, j, t in zip(cos, jac, y)
                                      if t and j < 0.35 and c >= 0.80),
            "non_paraphrases_with_jaccard_lt_0.35": sum(1 for j, t in zip(jac, y)
                                                        if not t and j < 0.35),
            "false_alarms_at_cos_0.65": sum(1 for c, j, t in zip(cos, jac, y)
                                            if not t and j < 0.35 and c >= floor),
            "false_alarms_at_cos_0.80": sum(1 for c, j, t in zip(cos, jac, y)
                                            if not t and j < 0.35 and c >= 0.80),
        },
    }


def main() -> int:
    cfg = load_config(ROOT / "config" / "default.toml")
    RESULTS.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(cfg.embedding.model_name, str(ROOT / "cache" / "hf"))
    result = {"model": cfg.embedding.model_name,
              "mrpc_test": bench(load("mrpc_test.jsonl"), embedder, cfg),
              "paws_test_first2000": bench(load("paws_test.jsonl"), embedder, cfg)}
    (RESULTS / "paraphrase.json").write_text(json.dumps(result, indent=2))
    for name in ("mrpc_test", "paws_test_first2000"):
        r = result[name]
        print(name, {k: round(v, 3) for k, v in r["auc"].items()},
              "cos>=0.80:", {k: round(r["cosine_at_0.80_assignment_band"][k], 3)
                             for k in ("precision", "recall", "fpr")},
              "best:", {k: round(r["cosine_best_f1"][k], 3)
                        for k in ("threshold", "precision", "recall", "f1")},
              r["scenario2_cases"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
