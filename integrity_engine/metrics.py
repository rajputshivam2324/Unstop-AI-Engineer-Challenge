"""<MODULE PURPOSE>
Dependency-free evaluation metrics shared by the in-corpus evaluation
(evaluation.py) and the public-dataset benchmarks (benchmarks/): precision /
recall / F1, ROC-AUC (Mann-Whitney), confusion matrices, threshold sweeps and a
1-D logistic fit used to calibrate the AI-detector's statistical components.

<PUBLIC API>
prf, roc_auc, confusion_matrix, threshold_metrics, best_f1_threshold,
fit_logistic_low_is_ai

<DEPENDS ON> math only (no sklearn, spec §9).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1; 0.0 where undefined."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (ties count 0.5)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    ranked = sorted((s, y) for s, y in zip(scores, labels))
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(1 for k in range(i, j) if ranked[k][1] == 1)
        i = j
    u = rank_sum - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str],
                     labels: Sequence[str]) -> dict[tuple[str, str], int]:
    """(truth, predicted) -> count over the given label set."""
    matrix = {(t, p): 0 for t in labels for p in labels}
    for t, p in zip(y_true, y_pred):
        matrix[(t, p)] = matrix.get((t, p), 0) + 1
    return matrix


def threshold_metrics(scores: Sequence[float], labels: Sequence[int],
                      threshold: float) -> dict[str, float]:
    """Binary metrics when predicting positive at score >= threshold."""
    tp = sum(1 for s, y in zip(scores, labels) if s >= threshold and y == 1)
    fp = sum(1 for s, y in zip(scores, labels) if s >= threshold and y == 0)
    fn = sum(1 for s, y in zip(scores, labels) if s < threshold and y == 1)
    tn = sum(1 for s, y in zip(scores, labels) if s < threshold and y == 0)
    precision, recall, f1 = prf(tp, fp, fn)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "accuracy": (tp + tn) / max(1, tp + fp + fn + tn)}


def best_f1_threshold(scores: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    """Threshold (among observed scores) maximizing F1; ties -> higher threshold."""
    best: dict[str, float] | None = None
    for threshold in sorted(set(scores)):
        m = threshold_metrics(scores, labels, threshold)
        if best is None or m["f1"] >= best["f1"]:
            best = m
    return best or threshold_metrics(scores, labels, 0.5)


def _sigmoid_low(x: float, mu: float, k: float) -> float:
    z = max(-50.0, min(50.0, (x - mu) / k))
    return 1.0 / (1.0 + math.exp(z))


def fit_logistic_low_is_ai(values: Sequence[float], labels: Sequence[int]) \
        -> tuple[float, float, float]:
    """Fit p(ai | x) = 1/(1+exp((x-mu)/k)) by grid-search maximum likelihood.

    Deterministic (no RNG). Returns (mu, k, mean log-likelihood). Used only on a
    held-out CALIBRATION split of an external labelled dataset; the fitted
    constants are then frozen in config/default.toml.
    """
    xs = sorted(values)
    lo, hi = xs[int(0.02 * (len(xs) - 1))], xs[int(0.98 * (len(xs) - 1))]
    span = max(hi - lo, 1e-6)
    best = (lo, span, -math.inf)
    for i in range(81):
        mu = lo + span * i / 80
        for j in range(1, 61):
            k = span * j / 120
            ll = 0.0
            for x, y in zip(values, labels):
                p = min(max(_sigmoid_low(x, mu, k), 1e-6), 1 - 1e-6)
                ll += math.log(p) if y == 1 else math.log(1 - p)
            ll /= max(1, len(values))
            if ll > best[2]:
                best = (mu, k, ll)
    return best
