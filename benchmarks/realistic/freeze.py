#!/usr/bin/env python
"""Record SHA-256 of test data, benchmark code, engine code and configs.

run_bench.py --split test refuses to run if any of these files changed later.
Refuses to overwrite an existing freeze.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bench import frozen_files, sha  # noqa: E402


def main() -> int:
    out = HERE / "freeze.json"
    if out.exists():
        raise SystemExit("freeze.json exists; refusing to overwrite")
    payload = {
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "rule": "Methods, thresholds and configs were chosen on dev only. Test is scored "
        "once per method after this freeze; no change after seeing test results.",
        "sha256": {p: sha(p) for p in frozen_files()},
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"froze {len(payload['sha256'])} files at {payload['utc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
