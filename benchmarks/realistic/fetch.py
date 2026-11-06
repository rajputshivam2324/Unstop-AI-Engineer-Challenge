#!/usr/bin/env python
"""Download the three public sources of the realistic benchmark (pinned, hash-checked).

Writes to benchmarks/realistic/work/raw/ (gitignored, ~370 MB). Skips files that
already exist with the right SHA-256.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "work" / "raw"
SOURCES = {
    "machine_paraphrase_train.tsv": (
        "https://huggingface.co/datasets/jpwahle/machine-paraphrase-dataset/resolve/"
        "5b97d8f7a59d0414050b58e9cdb2c48fc78ec1a1/train.tsv",
        "aafff4f099451e4d2d990c0cb607592ac8f303aefa971a80cd773f222a11fc45"),
    "autoregressive_paraphrase.tsv": (
        "https://huggingface.co/datasets/jpwahle/autoregressive-paraphrase-dataset/resolve/"
        "2ba342d0d668e896b3a805691ab3bcba5f8cc9d3/dataset.tsv",
        "338730b032fd9acabdeff522f3a137581be934e3703ef3b4906aecef50d3d5b6"),
    "Webis-CPC-11.zip": (
        "https://zenodo.org/api/records/3251771/files/Webis-CPC-11.zip/content",
        "38e2aa8a99fb88f3e198e5ffdc7dc5086f4ad8476b4167ba5a15826bb872f261"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        path = RAW / name
        if not (path.exists() and sha256(path) == expected):
            print(f"downloading {name} ...", file=sys.stderr)
            urllib.request.urlretrieve(url, path)
        if sha256(path) != expected:
            raise SystemExit(f"hash mismatch for {name}")
        print(f"ok {name}")
    cpc = RAW / "Webis-CPC-11"
    if not cpc.exists():
        with zipfile.ZipFile(RAW / "Webis-CPC-11.zip") as z:
            z.extractall(RAW)
    print(f"ok {cpc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
