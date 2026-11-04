"""<MODULE PURPOSE>
AI-generated-text estimation (§4.6, de-scope rule R3): exactly two top-level
components fused as

    p_ai   = 0.6 * p_cls + 0.4 * p_text
    p_text = 0.75 * p_perp + 0.25 * p_burst

p_cls  : open-weights DeBERTa-v3 detector "wasitaigeneratedcom/ai-text-detector-small"
         (tropa-mini; ai_score = 1 - P(human), length-weighted over <=512-token windows).
p_perp : logistic of GPT-2 perplexity  (low perplexity => machine-like).
p_burst: logistic of GPT-2 burstiness  (coefficient of variation of per-sentence
         mean NLL; uniform sentences => machine-like).

If EITHER model cannot be loaded, the whole analysis switches to a deterministic,
documented rule-based fallback and every result is marked
ai_detector_source="heuristic_fallback". Model output and heuristic output are
never blended.

Models are loaded one at a time for the whole batch and released afterwards
(GPT-2 ~0.5 GB, DeBERTa-v3-large ~1.7 GB fp32) so peak RSS stays bounded.

<PUBLIC API>
AIDetector, feature_heuristic_ai_score, fuse_probabilities, band_label, AI_CAVEAT

<DEPENDS ON>
torch, transformers (lazy), config, text_processing, types.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Any

from integrity_engine.config import AIDetectionConfig
from integrity_engine.text_processing import split_sentences, token_count
from integrity_engine.types import AIDetectionResult

logger = logging.getLogger("integrity_engine.ai_detection")

AI_CAVEAT = ("Note: AI detectors are unreliable on non-native-English and polished "
             "human text (Stanford 2023: up to 61% false-positive rate on ESL "
             "essays); treat this as a lead for human review, not proof.")
FALLBACK_NOTE = ("rule-based mock fallback, lower confidence, per the assignment's "
                 "allowance for a documented reproducible mock wrapper")
_CLS_WINDOW_TOKENS = 512  # DeBERTa max_position_embeddings
_MIN_SENTENCE_TOKENS = 4  # sentences shorter than this are skipped for burstiness


def _logistic_low_is_ai(value: float, mu: float, k: float) -> float:
    """1/(1+exp((value-mu)/k)): values below mu push the probability above 0.5."""
    z = (value - mu) / k
    if z > 50:
        return 0.0
    if z < -50:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def fuse_probabilities(p_cls: float, p_perp: float, p_burst: float,
                       cfg: AIDetectionConfig) -> float:
    """p_ai = w*p_cls + (1-w)*(v*p_perp + (1-v)*p_burst), w=0.6, v=0.75 (R3)."""
    p_text = cfg.p_text_perp_weight * p_perp + (1 - cfg.p_text_perp_weight) * p_burst
    return cfg.p_final_model_weight * p_cls + (1 - cfg.p_final_model_weight) * p_text


def band_label(score_pct: float, cfg: AIDetectionConfig) -> str:
    """<35 HUMAN, 35..<65 UNSURE (explicit uncertain band), >=65 AI_LIKELY."""
    if score_pct < cfg.band_human_max:
        return "HUMAN"
    if score_pct < cfg.band_unsure_max:
        return "UNSURE"
    return "AI_LIKELY"


def feature_heuristic_ai_score(text: str) -> tuple[float, dict[str, Any]]:
    """Deterministic no-model fallback (spec §4.6).

    p = 0.75 if (avg_sentence_len > 11 and type_token_ratio < 0.55 and
    sentence-length CV < 0.35) else 0.25.
    """
    sentences = [s.text for s in split_sentences(text)]
    lengths = [token_count(s) for s in sentences if token_count(s) > 0]
    words = [w.lower() for w in text.split() if w.strip()]
    avg_len = statistics.fmean(lengths) if lengths else 0.0
    ttr = len(set(words)) / len(words) if words else 1.0
    cv = (statistics.pstdev(lengths) / avg_len) if lengths and avg_len else 1.0
    flags = {"avg_sentence_len_gt_11": avg_len > 11, "type_token_ratio_lt_0_55": ttr < 0.55,
             "low_burstiness_indicator": cv < 0.35}
    prob = 0.75 if all(flags.values()) else 0.25
    detail = {**flags, "avg_sentence_len": round(avg_len, 2),
              "type_token_ratio": round(ttr, 3), "sentence_len_cv": round(cv, 3),
              "method": FALLBACK_NOTE}
    return prob, detail


class AIDetector:
    """Batch AI-content estimator; loads each model once, then frees it."""

    def __init__(self, cfg: AIDetectionConfig, cache_dir: Path) -> None:
        self.cfg = cfg
        self.cache_dir = str(cache_dir)
        self.missing: list[str] = []

    # ------------------------------------------------------------------ public
    def detect_many(self, texts: dict[str, str]) -> dict[str, AIDetectionResult]:
        """Analyze {document_id: text}. Never raises; degrades to the fallback."""
        results: dict[str, AIDetectionResult] = {}
        gated: dict[str, str] = {}
        for doc_id, raw in texts.items():
            text = raw[: self.cfg.max_chars]
            if len(raw) > self.cfg.max_chars:
                logger.info("ai: %s truncated to %d chars", doc_id, self.cfg.max_chars)
            n_tokens = token_count(text)
            if n_tokens < self.cfg.min_length_tokens:
                results[doc_id] = AIDetectionResult(
                    ai_generated_probability=None, label="INSUFFICIENT_TEXT",
                    length_gate_triggered=True, ai_detector_source="model",
                    caveat="", detail={"token_count": n_tokens})
            else:
                gated[doc_id] = text
        if not gated:
            return results

        gpt2_stats = self._run_gpt2(gated)
        cls_scores = self._run_classifier(gated) if gpt2_stats is not None else None
        if gpt2_stats is None or cls_scores is None:
            for doc_id, text in gated.items():
                results[doc_id] = self._fallback_result(doc_id, text)
            return results

        for doc_id, text in gated.items():
            ppl, burst, n_tok = gpt2_stats[doc_id]
            p_perp = _logistic_low_is_ai(ppl, self.cfg.perplexity_mu, self.cfg.perplexity_k)
            p_burst = _logistic_low_is_ai(burst, self.cfg.burst_mu, self.cfg.burst_k)
            p_cls = cls_scores[doc_id]
            score = round(100.0 * fuse_probabilities(p_cls, p_perp, p_burst, self.cfg), 1)
            label = band_label(score, self.cfg)
            results[doc_id] = AIDetectionResult(
                ai_generated_probability=score, label=label,  # type: ignore[arg-type]
                p_perplexity=round(p_perp, 4), p_burstiness=round(p_burst, 4),
                p_classifier=round(p_cls, 4), length_gate_triggered=False,
                ai_detector_source="model",
                caveat=AI_CAVEAT if label in ("UNSURE", "AI_LIKELY") else "",
                detail={"perplexity": round(ppl, 3), "burstiness": round(burst, 4),
                        "gpt2_tokens": n_tok, "token_count": token_count(text)})
        return results

    # --------------------------------------------------------------- fallback
    def _fallback_result(self, doc_id: str, text: str) -> AIDetectionResult:
        logger.warning("ai: model unavailable (%s); report %s marked heuristic_fallback",
                       ", ".join(self.missing) or "unknown", doc_id)
        prob, detail = feature_heuristic_ai_score(text)
        score = round(100.0 * prob, 1)
        label = band_label(score, self.cfg)
        return AIDetectionResult(
            ai_generated_probability=score, label=label,  # type: ignore[arg-type]
            length_gate_triggered=False, ai_detector_source="heuristic_fallback",
            caveat=AI_CAVEAT if label in ("UNSURE", "AI_LIKELY") else "",
            detail={**detail, "missing_models": list(self.missing)})

    # ------------------------------------------------------------------ gpt2
    def _run_gpt2(self, texts: dict[str, str]) -> dict[str, tuple[float, float, int]] | None:
        """Perplexity + burstiness per document; None if GPT-2 cannot be loaded."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch.set_num_threads(max(1, os.cpu_count() or 1))
            tok = AutoTokenizer.from_pretrained(self.cfg.gpt2_model, cache_dir=self.cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.gpt2_model, cache_dir=self.cache_dir).eval()
        except Exception as exc:  # noqa: BLE001 - offline / missing weights
            logger.warning("ai: gpt2 unavailable (%s: %s)", type(exc).__name__,
                           str(exc)[:160])
            self.missing.append(self.cfg.gpt2_model)
            return None
        out: dict[str, tuple[float, float, int]] = {}
        try:
            for doc_id, text in texts.items():
                out[doc_id] = self._gpt2_stats(text, tok, model, torch)
                logger.debug("ai: gpt2 %s ppl=%.2f burst=%.3f", doc_id, *out[doc_id][:2])
        finally:
            del model
            gc.collect()
        return out

    def _gpt2_stats(self, text: str, tok, model, torch) -> tuple[float, float, int]:
        """Token NLLs over <=1024-token windows; sentence means via offset mapping."""
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids: list[int] = enc["input_ids"]
        offsets: list[tuple[int, int]] = enc["offset_mapping"]
        window = self.cfg.max_tokens_per_window
        nll: list[float | None] = [None] * len(ids)  # first token of a window has none
        with torch.inference_mode():
            for start in range(0, len(ids), window):
                chunk = ids[start:start + window]
                if len(chunk) < 2:
                    continue
                input_ids = torch.tensor([chunk])
                logits = model(input_ids).logits[0, :-1]
                logp = torch.log_softmax(logits.float(), dim=-1)
                target = input_ids[0, 1:]
                token_nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
                for i, value in enumerate(token_nll.tolist(), start=start + 1):
                    nll[i] = value
        scored = [v for v in nll if v is not None]
        ppl = math.exp(statistics.fmean(scored)) if scored else float("inf")

        sentence_means: list[float] = []
        for sentence in split_sentences(text):
            if token_count(sentence.text) < _MIN_SENTENCE_TOKENS:
                continue
            vals = [nll[i] for i, (a, b) in enumerate(offsets)
                    if nll[i] is not None and a >= sentence.char_start
                    and b <= sentence.char_end]
            if vals:
                sentence_means.append(statistics.fmean(vals))
        if len(sentence_means) >= 2 and statistics.fmean(sentence_means) > 0:
            burst = statistics.pstdev(sentence_means) / statistics.fmean(sentence_means)
        else:
            burst = self.cfg.burst_mu  # undefined => neutral (p_burst = 0.5)
        return ppl, burst + 1e-6, len(ids)

    # ------------------------------------------------------------ classifier
    def _run_classifier(self, texts: dict[str, str]) -> dict[str, float] | None:
        """tropa-mini ai_score = 1 - P(human), length-weighted over windows."""
        try:
            import torch
            from torch import nn
            from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

            class _DetectorModel(PreTrainedModel):  # architecture from the model card
                config_class = AutoConfig
                _tied_weights_keys: list[str] = []

                @property
                def all_tied_weights_keys(self):  # transformers>=5 compatibility
                    return {}

                def __init__(self, config):
                    super().__init__(config)
                    self.model = AutoModel.from_config(config)
                    self.classifier = nn.Linear(
                        config.hidden_size, getattr(config, "detector_num_labels", 1))

                def forward(self, input_ids, attention_mask=None, **_kwargs):
                    hidden = self.model(input_ids, attention_mask=attention_mask)[0]
                    mask = attention_mask.unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                    return self.classifier(pooled)

            tok = AutoTokenizer.from_pretrained(self.cfg.classifier_model,
                                                cache_dir=self.cache_dir)
            model = _DetectorModel.from_pretrained(self.cfg.classifier_model,
                                                   cache_dir=self.cache_dir).eval()
            human_index = int(model.config.label2id.get("human", 0))
        except Exception as exc:  # noqa: BLE001 - offline / missing weights
            logger.warning("ai: classifier unavailable (%s: %s)", type(exc).__name__,
                           str(exc)[:160])
            self.missing.append(self.cfg.classifier_model)
            return None
        out: dict[str, float] = {}
        try:
            with torch.inference_mode():
                for doc_id, text in texts.items():
                    ids = tok(text, add_special_tokens=False)["input_ids"]
                    body = _CLS_WINDOW_TOKENS - 2
                    weighted, total = 0.0, 0
                    for start in range(0, len(ids), body):
                        piece = ids[start:start + body]
                        if not piece:
                            continue
                        input_ids = torch.tensor(
                            [[tok.cls_token_id, *piece, tok.sep_token_id]])
                        mask = torch.ones_like(input_ids)
                        probs = torch.softmax(model(input_ids, attention_mask=mask), -1)[0]
                        weighted += (1.0 - float(probs[human_index])) * len(piece)
                        total += len(piece)
                    out[doc_id] = weighted / total if total else 0.0
                    logger.debug("ai: classifier %s ai_score=%.4f", doc_id, out[doc_id])
        finally:
            del model
            gc.collect()
        return out
