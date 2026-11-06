#!/usr/bin/env python
# ruff: noqa: E501  (long provenance strings and table format lines)
"""Turn per-method pair scores into the benchmark tables.

Protocol (identical for every method):
  1. On dev, pick the decision threshold that maximises F1 (ties -> higher threshold).
  2. Apply that threshold unchanged to test. Test is scored once, after freeze.
  3. Report precision / recall / F1, per-kind recall, false positives by negative
     type, average precision (threshold-free), and 95 % bootstrap intervals
     (2,000 stratified resamples of positive pairs and of negative pairs; paired
     across methods, so the F1 difference interval is valid).
The engine is also reported at its shipped flag threshold (peer >= 20 % = MEDIUM),
which was not tuned on this benchmark.

Usage: .venv/bin/python benchmarks/realistic/evaluate.py [--split-for-report test]
Writes benchmarks/realistic/results/summary_<split>.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
METHODS = (
    "shingle_moss",
    "tfidf_cosine",
    "embed_doc",
    "embed_maxchunk",
    "engine_baseline",
    "engine_hybrid",
    "engine_v2",
)
SHIPPED = 0.20  # config/default.toml [scoring] peer_medium_pct = 20
POS_KINDS = (
    "verbatim",
    "partial_copy",
    "tool_paraphrase",
    "llm_paraphrase",
    "human_paraphrase",
    "human_rewrite",
)
B = 2000


def load_split(split: str):
    data = json.loads((HERE / "data" / f"{split}.json").read_text())
    n = data["n_docs"]
    keys = list(itertools.combinations(range(n), 2))
    kind = {(p["a"], p["b"]): p["kind"] for p in data["planted_pairs"]}
    tmpl = set(data["template_docs"])
    y = np.array([1 if kind.get(k) in POS_KINDS else 0 for k in keys])
    neg_type = np.array(
        [
            ""
            if y[i]
            else (
                "same_topic"
                if kind.get(k) == "same_topic"
                else "template"
                if (k[0] in tmpl and k[1] in tmpl)
                else "other"
            )
            for i, k in enumerate(keys)
        ]
    )
    pos_kind = np.array([kind.get(k, "") if y[i] else "" for i, k in enumerate(keys)])
    return keys, y, neg_type, pos_kind


def scores_for(split: str, method: str, keys) -> tuple[np.ndarray, dict] | None:
    path = RES / f"{split}_{method}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    m = {(i, j): s for i, j, s in r["scores"]}
    return np.array([m.get(k, 0.0) for k in keys]), {k: v for k, v in r.items() if k != "scores"}


def prf(pred: np.ndarray, y: np.ndarray) -> tuple[int, int, int, float, float, float]:
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, p, r, (2 * p * r / (p + r) if p + r else 0.0)


def best_threshold(s: np.ndarray, y: np.ndarray) -> float:
    cands = np.unique(s[s > 0])
    best = (-1.0, 0.0)
    for t in cands:
        f = prf(s >= t, y)[5]
        if f > best[0] or (f == best[0] and t > best[1]):
            best = (f, float(t))
    return best[1]


def average_precision(s: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    hits = np.cumsum(ys)
    prec = hits / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / max(ys.sum(), 1))


def report(split: str) -> dict:
    keys, y, neg_type, pos_kind = load_split(split)
    dkeys, dy, _, _ = load_split("dev")
    pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(0)
    samples = [
        np.concatenate([rng.choice(pos_idx, len(pos_idx)), rng.choice(neg_idx, len(neg_idx))])
        for _ in range(B)
    ]
    out: dict = {
        "split": split,
        "n_pairs": len(keys),
        "n_positive": int(y.sum()),
        "n_same_topic": int((neg_type == "same_topic").sum()),
        "n_template_pairs": int((neg_type == "template").sum()),
        "methods": {},
    }
    boot_f1: dict[str, np.ndarray] = {}
    for m in METHODS:
        dev = scores_for("dev", m, dkeys)
        cur = scores_for(split, m, keys)
        if dev is None or cur is None:
            continue
        th = best_threshold(dev[0], dy)
        s, meta = cur
        rows = {}
        for label, t in [("dev_tuned", th)] + (
            [("shipped", SHIPPED)] if m.startswith("engine") else []
        ):
            pred = s >= t
            tp, fp, fn, p, r, f = prf(pred, y)
            bf = np.array([prf(pred[ix], y[ix])[5] for ix in samples])
            if label == "dev_tuned":
                boot_f1[m] = bf
            rows[label] = {
                "threshold": round(t, 6),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": p,
                "recall": r,
                "f1": f,
                "f1_ci95": [float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))],
                "recall_by_kind": {k: float(pred[pos_kind == k].mean()) for k in POS_KINDS},
                "fp_by_negative_type": {
                    t_: int((pred & (neg_type == t_)).sum())
                    for t_ in ("same_topic", "template", "other")
                },
            }
        out["methods"][m] = {
            "average_precision": average_precision(s, y),
            "wall_s": meta["wall_s"],
            "peak_rss_mb": meta["peak_rss_mb"],
            **rows,
        }
    if any(e in boot_f1 for e in ("engine_baseline", "engine_hybrid", "engine_v2")):
        comps = {
            m: out["methods"][m]["dev_tuned"]["f1"] for m in boot_f1 if not m.startswith("engine")
        }
        if comps:
            best_comp = max(comps, key=comps.get)
            out["best_competitor"] = best_comp
            for e in ("engine_baseline", "engine_hybrid", "engine_v2"):
                if e in boot_f1:
                    d = boot_f1[e] - boot_f1[best_comp]
                    out[f"{e}_minus_best_competitor_f1"] = {
                        "point": out["methods"][e]["dev_tuned"]["f1"] - comps[best_comp],
                        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                        "share_of_resamples_engine_better": float((d > 0).mean()),
                    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-for-report", choices=("dev", "test"), default="dev")
    a = ap.parse_args()
    out = report(a.split_for_report)
    (RES / f"summary_{a.split_for_report}.json").write_text(json.dumps(out, indent=1))
    print(
        f"{'method':16} {'thr':>7} {'P':>6} {'R':>6} {'F1':>6} {'F1 95% CI':>15} {'AP':>6}  FP st/tm/ot  recall verb/part/tool/llm/human"
    )
    for m, v in out["methods"].items():
        for lab in ("dev_tuned", "shipped"):
            if lab in v:
                r = v[lab]
                print(
                    f"{m + ('*' if lab == 'shipped' else ''):16} {r['threshold']:7.3f} {r['precision']:6.3f} {r['recall']:6.3f} "
                    f"{r['f1']:6.3f} [{r['f1_ci95'][0]:.3f},{r['f1_ci95'][1]:.3f}] {v['average_precision']:6.3f}  "
                    f"{'/'.join(str(x) for x in r['fp_by_negative_type'].values()):11}  "
                    f"{'/'.join(f'{x:.2f}' for x in r['recall_by_kind'].values())}"
                )
    for k in (
        "engine_baseline_minus_best_competitor_f1",
        "engine_hybrid_minus_best_competitor_f1",
        "engine_v2_minus_best_competitor_f1",
    ):
        if k in out:
            print(k, out["best_competitor"], out[k])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
