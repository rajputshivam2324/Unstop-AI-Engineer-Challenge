#!/usr/bin/env python
"""AI-generated-text detection benchmark on HC3 (human vs ChatGPT answers).

Why HC3: the 13 "synthetic original" files in sample_submissions/ were drafted in
an AI-coding-assistant session, so they cannot serve as human ground truth. HC3
pairs a human answer and a ChatGPT answer to the SAME question, which controls for
topic. Pre-processing: HC3's ChatGPT answers lost the space after sentence
punctuation ("time.Small-cap"), an artefact a detector could exploit; we re-insert
it and collapse whitespace for BOTH classes before scoring.

Protocol (deterministic, seed 42):
  1. Keep questions whose first human AND first ChatGPT answer have 60..400 tokens.
  2. Sample N pairs per subset (finance, open_qa, wiki_csai).
  3. Split by QUESTION: 30% calibration, 70% test (both answers of a question stay
     in the same split -> no topic leakage).
  4. Score every text with the engine's own components (DeBERTa classifier,
     GPT-2 perplexity, GPT-2 burstiness) - cached to results/ai_components.json.
  5. Fit the two logistic curves (perplexity, burstiness) on calibration only;
     report everything on the held-out test split.

Usage: .venv/bin/python benchmarks/bench_ai_detection.py [--pairs-per-subset 50]
"""

from __future__ import annotations

import argparse
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from integrity_engine.ai_detection import (
    AIDetector,
    _logistic_low_is_ai,
    band_label,
    fuse_probabilities,
)
from integrity_engine.config import load_config
from integrity_engine.metrics import (
    best_f1_threshold,
    confusion_matrix,
    fit_logistic_low_is_ai,
    roc_auc,
    threshold_metrics,
)
from integrity_engine.text_processing import token_count

DATA = ROOT / "benchmarks" / "data"
RESULTS = ROOT / "benchmarks" / "results"
SUBSETS = ("finance", "open_qa", "wiki_csai")


def clean(text: str) -> str:
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)  # HC3 ChatGPT artefact
    return " ".join(text.split())


def load_pairs(per_subset: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pairs: list[dict] = []
    for subset in SUBSETS:
        rows = [json.loads(line) for line in
                (DATA / f"hc3_{subset}.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        usable = []
        for i, row in enumerate(rows):
            if not row.get("human_answers") or not row.get("chatgpt_answers"):
                continue
            human, ai = clean(row["human_answers"][0]), clean(row["chatgpt_answers"][0])
            if 60 <= token_count(human) <= 400 and 60 <= token_count(ai) <= 400:
                usable.append({"qid": f"{subset}:{i}", "subset": subset,
                               "human": human, "ai": ai})
        rng.shuffle(usable)
        pairs.extend(usable[:per_subset])
    return pairs


def split(pairs: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed + 1)
    order = list(pairs)
    rng.shuffle(order)
    cut = int(round(0.3 * len(order)))
    return order[:cut], order[cut:]


def score_texts(texts: dict[str, str], cfg, cache_path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    todo = {k: v for k, v in texts.items() if k not in cache}
    if todo:
        det = AIDetector(cfg.ai, ROOT / "cache" / "hf")
        t0 = time.time()
        gpt2 = det._run_gpt2(todo)
        t1 = time.time()
        cls = det._run_classifier(todo)
        t2 = time.time()
        if gpt2 is None or cls is None:
            raise SystemExit(f"models unavailable: {det.missing}")
        for k in todo:
            ppl, burst, n_tok = gpt2[k]
            cache[k] = {"p_cls": cls[k], "ppl": ppl, "burst": burst, "gpt2_tokens": n_tok}
        cache["_timing"] = {"n_texts": len(todo), "gpt2_s": round(t1 - t0, 1),
                            "classifier_s": round(t2 - t1, 1)}
        cache_path.write_text(json.dumps(cache, indent=1))
    return cache


def evaluate(rows: list[dict], cfg, mu_p, k_p, mu_b, k_b) -> dict:
    y = [r["y"] for r in rows]
    p_cls = [r["p_cls"] for r in rows]
    p_perp = [_logistic_low_is_ai(r["ppl"], mu_p, k_p) for r in rows]
    p_burst = [_logistic_low_is_ai(r["burst"], mu_b, k_b) for r in rows]
    fused = [100 * fuse_probabilities(c, p, b, cfg.ai)
             for c, p, b in zip(p_cls, p_perp, p_burst)]
    p_text = [cfg.ai.p_text_perp_weight * p + (1 - cfg.ai.p_text_perp_weight) * b
              for p, b in zip(p_perp, p_burst)]
    labels = [band_label(s, cfg.ai) for s in fused]
    truth = ["AI" if v else "HUMAN" for v in y]
    cm = confusion_matrix(truth, labels, ["HUMAN", "AI"])
    band_counts = {t: {b: sum(1 for tt, ll in zip(truth, labels) if tt == t and ll == b)
                       for b in ("HUMAN", "UNSURE", "AI_LIKELY")} for t in ("HUMAN", "AI")}
    _ = cm
    return {
        "n": len(rows), "n_ai": sum(y), "n_human": len(y) - sum(y),
        "auc": {"classifier": roc_auc(p_cls, y),
                "perplexity_component": roc_auc(p_perp, y),
                "burstiness_component": roc_auc(p_burst, y),
                "text_statistics_component": roc_auc(p_text, y),
                "fused": roc_auc(fused, y)},
        "fused_at_ai_likely_band": threshold_metrics(fused, y, cfg.ai.band_unsure_max),
        "classifier_at_0.5": threshold_metrics(p_cls, y, 0.5),
        "fused_best_f1": best_f1_threshold(fused, y),
        "band_counts": band_counts,
        "per_subset_fused_auc": {
            s: roc_auc([f for f, r in zip(fused, rows) if r["subset"] == s],
                       [r["y"] for r in rows if r["subset"] == s]) for s in SUBSETS},
        "median": {
            "ppl_human": _median([r["ppl"] for r in rows if not r["y"]]),
            "ppl_ai": _median([r["ppl"] for r in rows if r["y"]]),
            "burst_human": _median([r["burst"] for r in rows if not r["y"]]),
            "burst_ai": _median([r["burst"] for r in rows if r["y"]]),
            "p_cls_human": _median([r["p_cls"] for r in rows if not r["y"]]),
            "p_cls_ai": _median([r["p_cls"] for r in rows if r["y"]]),
        },
    }


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2) if n else float("nan")


def to_rows(pairs: list[dict], scores: dict[str, dict]) -> list[dict]:
    rows = []
    for p in pairs:
        for kind, y in (("human", 0), ("ai", 1)):
            s = scores[f"{p['qid']}:{kind}"]
            rows.append({"y": y, "subset": p["subset"], **s})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-per-subset", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(ROOT / "config" / "default.toml")
    RESULTS.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.pairs_per_subset, args.seed)
    calib, test = split(pairs, args.seed)
    texts = {f"{p['qid']}:{k}": p[k] for p in pairs for k in ("human", "ai")}
    scores = score_texts(texts, cfg, RESULTS / "ai_components.json")
    calib_rows, test_rows = to_rows(calib, scores), to_rows(test, scores)

    mu_p, k_p, ll_p = fit_logistic_low_is_ai([r["ppl"] for r in calib_rows],
                                             [r["y"] for r in calib_rows])
    mu_b, k_b, ll_b = fit_logistic_low_is_ai([r["burst"] for r in calib_rows],
                                             [r["y"] for r in calib_rows])
    spec = (22.0, 9.5, 0.55, 0.14)  # spec §4.6 a-priori constants
    shipped = (cfg.ai.perplexity_mu, cfg.ai.perplexity_k, cfg.ai.burst_mu, cfg.ai.burst_k)
    result = {
        "dataset": "HC3 (Hello-SimpleAI/HC3, CC BY-SA 4.0) subsets " + ", ".join(SUBSETS),
        "protocol": {"pairs_total": len(pairs), "calibration_pairs": len(calib),
                     "test_pairs": len(test), "token_range": [60, 400], "seed": args.seed},
        "timing": scores.get("_timing"),
        "calibrated_constants": {"perplexity_mu": round(mu_p, 2), "perplexity_k": round(k_p, 2),
                                 "burst_mu": round(mu_b, 4), "burst_k": round(k_b, 4),
                                 "calibration_mean_loglik": [round(ll_p, 4), round(ll_b, 4)]},
        "test_with_spec_constants": evaluate(test_rows, cfg, *spec),
        "test_with_calibrated_constants": evaluate(test_rows, cfg, mu_p, k_p, mu_b, k_b),
        "test_with_shipped_config": evaluate(test_rows, cfg, *shipped),
        "shipped_config": dict(zip(("perplexity_mu", "perplexity_k", "burst_mu", "burst_k"),
                                   shipped)),
    }
    (RESULTS / "ai_detection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("protocol", "calibrated_constants", "timing")},
                     indent=1))
    for name in ("test_with_spec_constants", "test_with_calibrated_constants",
                 "test_with_shipped_config"):
        r = result[name]
        print(name, {k: round(v, 3) for k, v in r["auc"].items()},
              "AI_LIKELY band:", {k: round(r["fused_at_ai_likely_band"][k], 3)
                                  for k in ("precision", "recall", "fpr")},
              r["band_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
