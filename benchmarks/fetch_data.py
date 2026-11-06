#!/usr/bin/env python
"""Download the public benchmark datasets into benchmarks/data/ (gitignored).

Datasets (all public, fetched from the Hugging Face Hub, no API key):
  * HC3 (Hello-SimpleAI/HC3, CC BY-SA 4.0): paired human vs ChatGPT answers to the
    same questions -> AI-detection benchmark. Subsets: finance, open_qa, wiki_csai.
  * MRPC (GLUE, nyu-mll/glue config "mrpc", test split, 1,725 pairs): human-labelled
    sentence paraphrases -> embedding-threshold benchmark (Scenario 2).
  * PAWS (google-research-datasets/paws, labeled_final test, first 2,000 pairs):
    adversarial pairs with high word overlap but different meaning -> stress test
    for the "cosine says paraphrase" assumption.

Usage:  .venv/bin/python benchmarks/fetch_data.py
Idempotent: existing files are kept.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/{name}.jsonl"
HC3_SUBSETS = ("finance", "open_qa", "wiki_csai")
ROWS_API = "https://datasets-server.huggingface.co/rows"


def _get(url: str, timeout: int = 120) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1} after {type(exc).__name__}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_rows(dataset: str, config: str, split: str, limit: int, out: Path) -> None:
    """Page through the datasets-server rows API (100 rows per request)."""
    if out.exists():
        print(f"exists: {out.name}")
        return
    rows: list[dict] = []
    offset = 0
    while offset < limit:
        query = urllib.parse.urlencode({"dataset": dataset, "config": config,
                                        "split": split, "offset": offset,
                                        "length": min(100, limit - offset)})
        payload = json.loads(_get(f"{ROWS_API}?{query}"))
        batch = [r["row"] for r in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {out.name}: {len(rows)} rows")


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    for name in HC3_SUBSETS:
        out = DATA / f"hc3_{name}.jsonl"
        if out.exists():
            print(f"exists: {out.name}")
            continue
        out.write_bytes(_get(HC3_URL.format(name=name), timeout=600))
        print(f"wrote {out.name}")
    fetch_rows("nyu-mll/glue", "mrpc", "test", 1725, DATA / "mrpc_test.jsonl")
    fetch_rows("google-research-datasets/paws", "labeled_final", "test", 2000,
               DATA / "paws_test.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
