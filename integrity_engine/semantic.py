"""Pinned CPU cross-encoders for second-stage verification, never retrieval.

NLI uses BOTH entailment directions: one-way entailment is insufficient to
establish paraphrase. Scores are model signals, not misconduct probabilities.
"""

from __future__ import annotations

import gc
from collections import OrderedDict
from dataclasses import dataclass

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
STS_MODEL = "cross-encoder/stsb-distilroberta-base"
STS_REVISION = "6b71347df6e2b34246b53e06d6bce70ef67de368"


@dataclass(frozen=True)
class SemanticScore:
    similarity: float
    contradiction: float = 0.0
    truncated: bool = False
    entailment: float | None = None


class SemanticVerifier:
    """Lazy optional models; failure is explicit, never silently accepted."""

    def __init__(self, kind: str, cache_dir: str, batch_size: int = 8) -> None:
        if kind not in ("nli", "nli_base", "sts"):
            raise ValueError(f"unknown verifier: {kind}")
        self.kind, self.cache_dir, self.batch_size = kind, cache_dir, batch_size
        self.model = self.tokenizer = None
        self.cache: OrderedDict[tuple[str, str], SemanticScore] = OrderedDict()
        self.pairs_scored = 0
        self.truncated_pairs = 0

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name, rev = ((NLI_MODEL, NLI_REVISION) if self.kind == "nli"
                     else ("cross-encoder/nli-deberta-v3-base",
                           "6c749ce3425cd33b46d187e45b92bbf96ee12ec7")
                     if self.kind == "nli_base" else (STS_MODEL, STS_REVISION))
        torch.set_num_threads(max(1, min(6, (torch.get_num_threads()))))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                name, revision=rev, cache_dir=self.cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                name, revision=rev, cache_dir=self.cache_dir).eval()
        except Exception as exc:
            raise RuntimeError(f"semantic verifier unavailable: {name}@{rev}: {exc}") from exc
        if self.kind.startswith("nli"):
            self.labels = {str(v).lower(): int(k)
                           for k, v in self.model.config.id2label.items()}
            if not {"entailment", "contradiction", "neutral"} <= self.labels.keys():
                raise RuntimeError("NLI label mapping is missing; refusing to guess")

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[SemanticScore]:
        if not pairs:
            return []
        self.load()
        import torch

        # Symmetric tasks: canonical keys prevent inconsistent reverse results.
        keys = [tuple(sorted(pair)) for pair in pairs]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        computed = {}
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            inputs = batch + [(b, a) for a, b in batch] if self.kind.startswith("nli") else batch
            lengths = self.tokenizer([a for a, _ in inputs], [b for _, b in inputs],
                                     truncation=False, padding=False)["input_ids"]
            too_long = [len(v) > 512 for v in lengths]
            encoded = self.tokenizer([a for a, _ in inputs], [b for _, b in inputs],
                                     padding=True, truncation=True, max_length=512,
                                     return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                probs = (torch.softmax(logits, -1) if self.kind.startswith("nli")
                         else torch.sigmoid(logits))
            for i, key in enumerate(batch):
                truncated = too_long[i]
                if self.kind.startswith("nli"):
                    j = i + len(batch)
                    truncated |= too_long[j]
                    similarity = min(float(probs[i, self.labels["entailment"]]),
                                     float(probs[j, self.labels["entailment"]]))
                    contradiction = max(float(probs[i, self.labels["contradiction"]]),
                                        float(probs[j, self.labels["contradiction"]]))
                else:
                    similarity, contradiction = float(probs[i, 0]), 0.0
                value = SemanticScore(similarity, contradiction, truncated)
                computed[key] = value
                self.cache[key] = value
                self.pairs_scored += 1
                self.truncated_pairs += int(truncated)
        # Keep this call's values even if a large batch exceeds the cache budget.
        result = [computed[k] if k in computed else self.cache[k] for k in keys]
        while len(self.cache) > 4096:
            self.cache.popitem(last=False)
        return result

    def release(self) -> None:
        self.model = self.tokenizer = None
        self.cache.clear()
        gc.collect()


class HybridVerifier:
    """STS similarity OR strict bidirectional entailment, with an NLI veto.

    Similarity is the STS model's score. Entailment/contradiction remain separate
    signals; the caller applies explicitly configured operating points.
    """

    kind = "hybrid"

    def __init__(self, cache_dir: str) -> None:
        self.sts = SemanticVerifier("sts", cache_dir)
        self.nli = SemanticVerifier("nli", cache_dir)

    def load(self) -> None:
        self.sts.load()
        self.nli.load()

    @property
    def pairs_scored(self):
        return self.sts.pairs_scored + self.nli.pairs_scored

    @property
    def truncated_pairs(self):
        return self.sts.truncated_pairs + self.nli.truncated_pairs

    def score_pairs(self, pairs):
        sts, nli = self.sts.score_pairs(pairs), self.nli.score_pairs(pairs)
        return [SemanticScore(s.similarity, n.contradiction,
                              s.truncated or n.truncated, n.similarity)
                for s, n in zip(sts, nli)]

    def release(self):
        self.sts.release()
        self.nli.release()


def build_verifier(kind: str, cache_dir: str):
    return HybridVerifier(cache_dir) if kind == "hybrid" else SemanticVerifier(kind, cache_dir)
