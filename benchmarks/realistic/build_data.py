#!/usr/bin/env python
# ruff: noqa: E501  (long provenance strings and table format lines)
"""Build the realistic document-level plagiarism benchmark (dev + test splits).

Each split is one simulated challenge batch of 300 submissions. A submission is
three paragraphs of human-written Wikipedia text (~400 words). Every document
pair in the batch is scored, so a method sees 44,850 pairs per split.

Planted reuse (positive pairs, 16 of each kind per split; one paragraph of the
source document A is reused in document B, in a random position):
  verbatim        paragraph copied unchanged
  partial_copy    only 2 consecutive sentences copied into one of B's paragraphs
  tool_paraphrase paragraph rewritten by SpinBot (commercial paraphrasing tool)
  llm_paraphrase  paragraph rewritten by GPT-3
  human_paraphrase 4 human-written paraphrase sentence pairs (MRPC, label 1)
  human_rewrite   a book passage rewritten by a paid crowd worker who was asked to
                  paraphrase it (Webis-CPC-11, annotated "Paraphrase: Yes")

Host text is mostly Wikipedia; 15 % of host paragraphs are other Webis-CPC-11 book
passages, so 19th-century book prose alone does not reveal a planted pair.

Hard negatives (label 0, like every other unplanted pair):
  same_topic      40 pairs: each doc holds a different paragraph that shares >= 4
                  rare names/numbers with the other (same subject, no reuse)
  template        15 docs start with the same assignment instructions (sanctioned
                  shared text; must not be flagged)

Leakage guards: no 8-word sequence appears in two different planted units unless
the label says so; dev and test share no paragraph and no 8-word sequence; MRPC
dev comes from the train split and test from the validation split (the earlier
benchmarks used MRPC test). Test is built from data never used for tuning.

Usage:
  .venv/bin/python benchmarks/realistic/build_data.py
Needs benchmarks/realistic/work/{pairs_spinbot,pairs_gpt3,originals_machine}.json
from pair_sources.py. Writes benchmarks/realistic/data/{dev,test}.json.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.fetch_data import fetch_rows  # noqa: E402

WORK, OUT = HERE / "work", HERE / "data"
SEED = 20261106
N_DOCS, N_PER_KIND, N_SAME_TOPIC, N_TEMPLATE = 300, 16, 40, 15
KINDS = (
    "verbatim",
    "partial_copy",
    "tool_paraphrase",
    "llm_paraphrase",
    "human_paraphrase",
    "human_rewrite",
)
CPC_DIR = HERE / "work" / "raw" / "Webis-CPC-11"
BOOK_HOST_SHARE = 0.15
TEMPLATE_BLOCK = (
    "Assignment brief. This report is submitted for the history and society essay "
    "round. Write between three and five paragraphs on the topic you were given and "
    "support every claim with a source listed in your bibliography. Quotations must "
    "be marked and attributed. Use the official cover sheet, keep the font at eleven "
    "points, and do not write your name on any page. Reports that exceed the word "
    "limit will be marked down, and copying from another participant will lead to "
    "disqualification. By submitting you confirm that the work is your own."
)

WORD = re.compile(r"[a-z0-9]+")


def words(s: str) -> list[str]:
    return WORD.findall(s.lower())


def grams8(s: str) -> set[tuple[str, ...]]:
    w = words(s)
    return {tuple(w[i : i + 8]) for i in range(len(w) - 7)}


def sentences(s: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z\"(])", s) if x.strip()]


class Ledger:
    """Global record of every 8-word sequence placed in any document."""

    def __init__(self) -> None:
        self.used: set[tuple[str, ...]] = set()

    def free(self, *texts: str) -> bool:
        return all(not (grams8(t) & self.used) for t in texts)

    def take(self, *texts: str) -> None:
        for t in texts:
            self.used |= grams8(t)


def ok_len(s: str, lo: int = 60, hi: int = 200) -> bool:
    return lo <= len(s.split()) <= hi


def rare_names(s: str) -> set[str]:
    out = set()
    for sent in sentences(s):
        toks = re.findall(r"[A-Za-z][A-Za-z'-]+|\d{3,4}", sent)
        out |= {t.lower() for i, t in enumerate(toks) if t[0].isdigit() or (i and t[0].isupper())}
    return out


def cpc_cases() -> tuple[list[tuple[str, str]], list[str]]:
    """Webis-CPC-11: (original, paraphrase) for 'Yes' cases; all originals as book hosts.

    Cases are taken in case-id order and shuffled later; no case is selected by how
    much wording it shares, so easy and hard rewrites keep their natural mix."""
    def flat(p: Path) -> str:
        return " ".join(p.read_text(encoding="utf-8", errors="replace").replace("¶", " ").split())
    yes, books = [], []
    for meta in sorted(CPC_DIR.glob("*-metadata.txt"), key=lambda p: int(p.name.split("-")[0])):
        cid = meta.name.split("-")[0]
        o, p = flat(CPC_DIR / f"{cid}-original.txt"), flat(CPC_DIR / f"{cid}-paraphrase.txt")
        if "Paraphrase: Yes" in meta.read_text(errors="replace"):
            if ok_len(o) and ok_len(p, 40, 260):
                yes.append((o, p))
        elif ok_len(o):
            books.append(o)
    return yes, books


def mrpc_passages(split: str) -> list[tuple[str, str]]:
    path = ROOT / "benchmarks" / "data" / f"mrpc_{split}.jsonl"
    fetch_rows("nyu-mll/glue", "mrpc", split, 4000, path)
    rows = [json.loads(x) for x in path.read_text().splitlines() if x]
    pos = [
        (" ".join(r["sentence1"].split()), " ".join(r["sentence2"].split()))
        for r in rows
        if int(r["label"]) == 1
    ]
    random.Random(SEED).shuffle(pos)
    return [
        (" ".join(a for a, _ in pos[i : i + 4]), " ".join(b for _, b in pos[i : i + 4]))
        for i in range(0, len(pos) - 3, 4)
    ]


def same_topic_pairs(originals: list[str], rng: random.Random) -> list[tuple[str, str]]:
    idx = [i for i in range(len(originals)) if ok_len(originals[i])]
    rng.shuffle(idx)
    idx = idx[:40000]
    sig = {i: rare_names(originals[i]) for i in idx}
    df = Counter(t for s in sig.values() for t in s)
    inv: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        for t in sorted(sig[i]):
            if 2 <= df[t] <= 30:
                inv[t].append(i)
    out, seen = [], set()
    for i in idx:
        if i in seen:
            continue
        hits = Counter(
            j for t in sorted(sig[i]) if 2 <= df[t] <= 30 for j in inv[t] if j != i and j not in seen
        )
        for j, shared in hits.most_common(3):
            if shared >= 4 and not (grams8(originals[i]) & grams8(originals[j])):
                out.append((originals[i], originals[j]))
                seen |= {i, j}
                break
    return out


def build_split(name: str, pools: dict, ledger: Ledger, rng: random.Random) -> dict:
    def draw(pool_name: str, n: int, check) -> list:
        got, pool = [], pools[pool_name]
        while len(got) < n:
            if not pool:
                raise SystemExit(f"pool {pool_name} exhausted for split {name}")
            item = pool.pop()
            texts = item if isinstance(item, tuple) else (item,)
            if check(item) and ledger.free(*texts):
                ledger.take(*texts)
                got.append(item)
        return got

    plain = lambda x: True  # noqa: E731
    units: list[tuple[str, list, list]] = []  # (kind, reused_in_A, reused_in_B)
    for kind in KINDS:
        if kind in ("verbatim", "partial_copy"):
            for p in draw("hosts", N_PER_KIND, lambda x: len(sentences(x)) >= 3):
                if kind == "verbatim":
                    units.append((kind, p, p))
                else:
                    s = sentences(p)
                    k = rng.randrange(0, len(s) - 1)
                    units.append((kind, p, " ".join(s[k : k + 2])))
        else:
            pool = {
                "tool_paraphrase": "spinbot",
                "llm_paraphrase": "gpt3",
                "human_paraphrase": f"mrpc_{name}",
                "human_rewrite": "cpc",
            }[kind]
            for a, b in draw(pool, N_PER_KIND, plain):
                units.append((kind, a, b))
    same = draw("same_topic", N_SAME_TOPIC, plain)
    n_hosts = 2 * 2 * len(units) + 2 * 2 * len(same)
    n_plain = N_DOCS - 2 * len(units) - 2 * len(same)
    n_total = n_hosts + 3 * n_plain + len(units)
    n_book = round(BOOK_HOST_SHARE * n_total)
    hosts = draw("hosts", n_total - n_book, plain) + draw("books", n_book, plain)
    rng.shuffle(hosts)

    docs: list[list[str]] = []
    pairs: list[dict] = []

    def place(fill: list[str], para: str) -> list[str]:
        out = list(fill)
        out.insert(rng.randrange(0, 3), para)
        return out

    for kind, a_text, b_text in units:
        a = place([hosts.pop(), hosts.pop()], a_text)
        if kind == "partial_copy":  # two copied sentences inside one of B's own paragraphs
            host = sentences(hosts.pop())
            cut = rng.randrange(1, len(host)) if len(host) > 1 else 1
            b = place([hosts.pop(), hosts.pop()], " ".join(host[:cut] + [b_text] + host[cut:]))
        else:
            b = place([hosts.pop(), hosts.pop()], b_text)
        docs += [a, b]
        pairs.append({"a": len(docs) - 2, "b": len(docs) - 1, "kind": kind, "label": 1})
    for p, q in same:
        docs += [place([hosts.pop(), hosts.pop()], p), place([hosts.pop(), hosts.pop()], q)]
        pairs.append({"a": len(docs) - 2, "b": len(docs) - 1, "kind": "same_topic", "label": 0})
    for _ in range(n_plain):
        docs.append([hosts.pop(), hosts.pop(), hosts.pop()])

    order = list(range(len(docs)))
    rng.shuffle(order)  # document ids carry no information about pair membership
    new_id = {old: new for new, old in enumerate(order)}
    docs = [docs[old] for old in order]
    for p in pairs:
        p["a"], p["b"] = sorted((new_id[p["a"]], new_id[p["b"]]))
    template_docs = sorted(rng.sample(range(len(docs)), N_TEMPLATE))
    for t in template_docs:
        docs[t] = [TEMPLATE_BLOCK] + docs[t]
    assert len(docs) == N_DOCS
    return {
        "split": name,
        "n_docs": len(docs),
        "docs": [{"id": f"{name}_{i:03d}", "text": "\n\n".join(d)} for i, d in enumerate(docs)],
        "planted_pairs": sorted(pairs, key=lambda p: (p["a"], p["b"])),
        "template_docs": template_docs,
        "template_block": TEMPLATE_BLOCK,
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    if (OUT / "test.json").exists():
        raise SystemExit("test split already built; refusing to rebuild it")
    rng = random.Random(SEED)
    originals = json.loads((WORK / "originals_machine.json").read_text())
    spin = [
        (p["original"], p["paraphrase"])
        for p in json.loads((WORK / "pairs_spinbot.json").read_text())
    ]
    gpt3 = [
        (p["original"], p["paraphrase"]) for p in json.loads((WORK / "pairs_gpt3.json").read_text())
    ]
    spin = [x for x in spin if ok_len(x[0]) and ok_len(x[1])]
    gpt3 = [x for x in gpt3 if ok_len(x[0]) and ok_len(x[1], 40)]
    same = same_topic_pairs(originals, rng)
    # paraphrase sources must never double as host text: exclude their originals from hosts
    reserved = set().union(*(grams8(a) for a, _ in spin + gpt3))
    hosts = [o for o in originals if ok_len(o) and not (grams8(o) & reserved)]
    cpc_yes, books = cpc_cases()
    pools = {
        "hosts": hosts,
        "spinbot": spin,
        "gpt3": gpt3,
        "same_topic": same,
        "cpc": cpc_yes,
        "books": books,
        "mrpc_dev": mrpc_passages("train"),
        "mrpc_test": mrpc_passages("validation"),
    }
    for v in pools.values():
        rng.shuffle(v)
    ledger = Ledger()
    stats = {k: len(v) for k, v in pools.items()}
    for split in ("dev", "test"):
        data = build_split(split, pools, ledger, rng)
        data["provenance"] = {
            "host_text": "human-written English Wikipedia paragraphs (originals in jpwahle/machine-paraphrase-dataset)",
            "tool_paraphrase": "SpinBot outputs, jpwahle/machine-paraphrase-dataset (Wahle et al., iConference 2022)",
            "llm_paraphrase": "GPT-3 outputs, jpwahle/autoregressive-paraphrase-dataset (Wahle et al., EMNLP 2022)",
            "human_paraphrase": "MRPC "
            + ("train" if split == "dev" else "validation")
            + " label-1 pairs (GLUE)",
            "human_rewrite": "Webis-CPC-11 'Paraphrase: Yes' cases (Burrows, Potthast, Stein 2013; CC BY 4.0, zenodo 3251771)",
            "book_hosts": "Webis-CPC-11 original passages (Project Gutenberg books)",
            "seed": SEED,
            "pool_sizes": stats,
        }
        path = OUT / f"{split}.json"
        path.write_text(json.dumps(data, indent=1))
        print(
            split,
            path,
            hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            Counter(p["kind"] for p in data["planted_pairs"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
