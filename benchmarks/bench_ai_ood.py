#!/usr/bin/env python
"""Out-of-distribution AI-detection check (complements the HC3 benchmark).

HC3 is public since 2023, so the DeBERTa detector may have seen it in training;
its perfect HC3 score is an upper bound. This check uses texts the detector is
unlikely to have been tuned on, in our own business-report register:

  human : intro sections of 30 English Wikipedia business/economics articles AS OF
          their last revision before 2019-01-01 (pre-ChatGPT -> human-written),
          fetched via the MediaWiki API, plain text, first ~300 tokens.
  AI    : every analyzable TEXT document of sample_submissions/ (PDF/DOCX). All of
          them are machine-written: 06 by GLM in a single pass, the rest drafted
          inside an AI coding-assistant session (disclosed in README).

Reports per-text classifier, perplexity and fused scores with the shipped config,
and FPR / recall at the AI_LIKELY band.
Usage: .venv/bin/python benchmarks/bench_ai_ood.py
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "cache" / "hf"))

from integrity_engine.ai_detection import (
    AIDetector,
    _logistic_low_is_ai,
    band_label,
    fuse_probabilities,
)
from integrity_engine.config import load_config
from integrity_engine.ingestion import extract_any
from integrity_engine.metrics import roc_auc, threshold_metrics

DATA = ROOT / "benchmarks" / "data"
RESULTS = ROOT / "benchmarks" / "results"
API = "https://en.wikipedia.org/w/api.php"
TITLES = [
    "SWOT analysis", "Break-even (economics)", "Supply chain management", "Gross margin",
    "Inventory turnover", "Cash conversion cycle", "Contribution margin", "Economies of scale",
    "Vertical integration", "Market segmentation", "Franchising", "Logistics",
    "Just-in-time manufacturing", "Price elasticity of demand", "Working capital",
    "Supermarket", "Retail", "Solar power in India", "Electronic invoicing",
    "Porter's generic strategies", "Business model", "Customer acquisition cost",
    "Last mile (transportation)", "Freight transport", "Operating margin", "Fixed cost",
    "Shrinkage (accounting)", "Distribution (marketing)", "Strategic management",
    "Competitive advantage"]
UA = {"User-Agent": "integrity-engine-benchmark/1.0 (assignment research; offline cache)"}


def _api(params: dict) -> dict:
    """Polite MediaWiki call: 1.5 s spacing, backoff on HTTP 429."""
    url = f"{API}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
    for attempt in range(6):
        time.sleep(1.5)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_wikipedia() -> list[dict]:
    cache = DATA / "wikipedia_2018_intros.jsonl"
    if cache.exists():
        return [json.loads(line) for line in cache.read_text().splitlines() if line]
    rows = []
    for title in TITLES:
        try:
            q = _api({"action": "query", "prop": "revisions", "titles": title,
                      "rvlimit": 1, "rvstart": "2019-01-01T00:00:00Z", "rvdir": "older",
                      "rvprop": "ids|timestamp", "redirects": 1})
            page = next(iter(q["query"]["pages"].values()))
            rev = page["revisions"][0]
            parsed = _api({"action": "parse", "oldid": rev["revid"], "prop": "text"})
            raw = parsed["parse"]["text"]["*"]
        except Exception as exc:  # noqa: BLE001
            print(f"skip {title}: {exc}", file=sys.stderr)
            continue
        paras = re.findall(r"<p>(.*?)</p>", raw, flags=re.DOTALL)
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", "", p)) for p in paras)
        text = re.sub(r"\[\d+\]|\[citation needed\]|\[note \d+\]", "", text)
        words = " ".join(text.split()).split(" ")
        text = " ".join(words[:300])
        if len(words) >= 120:
            rows.append({"title": title, "revid": rev["revid"], "timestamp": rev["timestamp"],
                         "text": text})
    DATA.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def main() -> int:
    cfg = load_config(ROOT / "config" / "default.toml")
    wiki = fetch_wikipedia()
    texts = {f"wiki:{r['title']}": r["text"] for r in wiki}
    for path in sorted((ROOT / "sample_submissions").iterdir()):
        if path.suffix.lower() in (".pdf", ".docx"):
            doc = extract_any(path, cfg)
            if doc is not None and doc.status == "ok":
                texts[f"corpus:{path.name}"] = doc.full_text[: cfg.ai.max_chars]
    det = AIDetector(cfg.ai, ROOT / "cache" / "hf")
    gpt2 = det._run_gpt2(texts)
    cls = det._run_classifier(texts)
    rows = []
    for key, text in texts.items():
        ppl, burst, _ = gpt2[key]
        p_perp = _logistic_low_is_ai(ppl, cfg.ai.perplexity_mu, cfg.ai.perplexity_k)
        p_burst = _logistic_low_is_ai(burst, cfg.ai.burst_mu, cfg.ai.burst_k)
        fused = 100 * fuse_probabilities(cls[key], p_perp, p_burst, cfg.ai)
        rows.append({"id": key, "y": 0 if key.startswith("wiki:") else 1,
                     "p_cls": cls[key], "ppl": ppl, "burst": burst, "fused": fused,
                     "label": band_label(fused, cfg.ai), "words": len(text.split())})
    y = [r["y"] for r in rows]
    result = {
        "n_human_wikipedia_2018": sum(1 for v in y if v == 0),
        "n_ai_corpus_docs": sum(y),
        "auc": {"classifier": roc_auc([r["p_cls"] for r in rows], y),
                "perplexity(neg)": roc_auc([-r["ppl"] for r in rows], y),
                "fused": roc_auc([r["fused"] for r in rows], y)},
        "fused_at_ai_likely_band": threshold_metrics([r["fused"] for r in rows], y,
                                                     cfg.ai.band_unsure_max),
        "human_band_counts": {b: sum(1 for r in rows if r["y"] == 0 and r["label"] == b)
                              for b in ("HUMAN", "UNSURE", "AI_LIKELY")},
        "rows": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ai_ood.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    for r in sorted(rows, key=lambda r: r["fused"]):
        print(f"{r['id'][:48]:48s} y={r['y']} cls={r['p_cls']:.4f} ppl={r['ppl']:7.1f} "
              f"fused={r['fused']:5.1f} {r['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
