#!/usr/bin/env python
"""Recover (original, paraphrase) paragraph pairs from the two Wahle et al. datasets.

The Hugging Face releases of
  * jpwahle/machine-paraphrase-dataset   (SpinBot, a commercial paraphrasing tool)
  * jpwahle/autoregressive-paraphrase-dataset (GPT-3 and T5 paraphrases)
store originals and paraphrases as shuffled, unlinked rows. Both start from human
Wikipedia paragraphs. This script links each paraphrase to its original using only
names and numbers (capitalised non-initial words and digits), which paraphrasers
mostly keep. A link is accepted only when it is unambiguous:

  shared >= 4 tokens, Jaccard >= 0.5, runner-up Jaccard <= 0.6 x best,
  word-length ratio in [0.6, 1.6].

Selection uses names/numbers only, never the wording the methods compare, but it
does favour paraphrases that kept their names. That makes the task slightly easier
for every method and slightly favours lexical baselines. It is disclosed in the
README. Output: benchmarks/realistic/work/pairs_{spinbot,gpt3}.json (gitignored).

Usage: .venv/bin/python benchmarks/realistic/pair_sources.py <machine.tsv> <autoregressive.tsv>
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10**9)
HERE = Path(__file__).resolve().parent
WORK = HERE / "work"


def fix_text(s: str) -> str:
    """Undo UTF-8-read-as-Latin-1 mojibake and PTB-style token spacing."""
    try:  # noqa: SIM105 - explicit is clearer for a mojibake round-trip
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    s = s.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    s = re.sub(r"\s+([,.;:!?%)\]])", r"\1", s)
    s = re.sub(r"([(\[$])\s+", r"\1", s)
    s = re.sub(r"\s+-\s+", "-", s)
    s = re.sub(r"\s+'s\b", "'s", s)
    if s.count('"') % 2 == 0 and re.search(r'"\s+\S[^"]*\s+"', s):
        parts = s.split('"')  # odd parts are quoted spans: trim inner padding only
        s = "".join((f'"{p.strip()}"' if k % 2 else p) for k, p in enumerate(parts))
    return " ".join(s.split())


def signature(s: str) -> set[str]:
    out = set()
    for sent in re.split(r"(?<=[.!?])\s+", s):
        words = re.findall(r"[A-Za-z][A-Za-z'-]+|\d[\d,.]*\d|\d", sent)
        for i, w in enumerate(words):
            if w[0].isdigit() or (i > 0 and w[0].isupper()):
                out.add(w.lower().rstrip(".,"))
    return out


def link(originals: list[str], paraphrases: list[str], tag: str) -> list[dict]:
    sigs = [signature(o) for o in originals]
    df = Counter(t for s in sigs for t in s)
    inv: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sigs):
        for t in s:
            if df[t] <= 200:
                inv[t].append(i)
    used: set[int] = set()
    out = []
    for p_idx, para in enumerate(paraphrases):
        ps = signature(para)
        hits = Counter(i for t in sorted(ps) if df.get(t, 0) <= 200 for i in inv.get(t, ()))
        scored = []
        for i, shared in hits.most_common(20):
            scored.append((shared / len(ps | sigs[i]), shared, i))
        scored.sort(reverse=True)
        if not scored:
            continue
        best, shared, i = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        lo, lp = len(originals[i].split()), len(para.split())
        if (
            shared >= 4
            and best >= 0.5
            and second <= 0.6 * best
            and 0.6 <= lp / lo <= 1.6
            and i not in used
        ):
            used.add(i)
            out.append(
                {
                    "source": tag,
                    "orig_idx": i,
                    "para_idx": p_idx,
                    "jaccard": round(best, 3),
                    "original": originals[i],
                    "paraphrase": para,
                }
            )
    return out


def main() -> int:
    WORK.mkdir(exist_ok=True)
    machine, auto = Path(sys.argv[1]), Path(sys.argv[2])
    rows = list(csv.reader(machine.open(newline=""), delimiter="\t"))[1:]
    orig = [fix_text(r[0]) for r in rows if r[1] == "0"]
    spin = [fix_text(r[0]) for r in rows if r[1] == "1" and r[3] == "spinbot"]
    spin_pairs = link(orig, spin[:20000], "spinbot")
    (WORK / "pairs_spinbot.json").write_text(json.dumps(spin_pairs))
    (WORK / "originals_machine.json").write_text(json.dumps(orig))
    rows = list(csv.reader(auto.open(newline=""), delimiter="\t"))[1:]
    orig2 = [fix_text(r[0]) for r in rows if r[2] == "original"]
    gpt3 = [fix_text(r[0]) for r in rows if r[2] == "gpt3"]
    gpt_pairs = link(orig2, gpt3[:20000], "gpt3")
    (WORK / "pairs_gpt3.json").write_text(json.dumps(gpt_pairs))
    print(
        json.dumps(
            {
                "spinbot_linked": len(spin_pairs),
                "spinbot_tried": min(20000, len(spin)),
                "gpt3_linked": len(gpt_pairs),
                "gpt3_tried": min(20000, len(gpt3)),
                "originals_machine": len(orig),
                "originals_auto": len(orig2),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
