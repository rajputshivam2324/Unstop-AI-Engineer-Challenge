# ruff: noqa: E501, B007  (long chart labels; unused loop label)
"""Render the realistic-batch benchmark image. Every number is read from result JSON.

Sources
  benchmarks/realistic/results/summary_test.json            held-out test, all methods
  benchmarks/realistic/results/test_*.json                  per-method runtime
  benchmarks/realistic/results/dev_engine_hybrid_incomplete.json  hybrid CPU stop record
  improvements/runs/heldout_{baseline,semantic_hybrid}_final.json earlier PAWS test
  improvements/comparison.json                              earlier 13-file regression test
Usage: .venv/bin/python output/benchmark/create_production_benchmark.py
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RES = ROOT / "benchmarks/realistic/results"
S = json.loads((RES / "summary_test.json").read_text())
M = S["methods"]
HYB = json.loads((RES / "dev_engine_hybrid_incomplete.json").read_text())
PAWS = {m: json.loads((ROOT / f"improvements/runs/heldout_{m}_final.json").read_text())["metrics"]
        for m in ("baseline", "semantic_hybrid")}
REG = json.loads((ROOT / "improvements/comparison.json").read_text())["pipeline_metrics"]
WALL = {m: json.loads((RES / f"test_{m}.json").read_text())["wall_s"] for m in M}

for p in ["/usr/share/fonts/open-sans/OpenSans-Regular.ttf",
          "/usr/share/fonts/open-sans/OpenSans-Semibold.ttf",
          "/usr/share/fonts/julietaula-montserrat-fonts/Montserrat-Bold.otf"]:
    if Path(p).exists():
        font_manager.fontManager.addfont(p)
plt.rcParams.update({"font.family": ["Open Sans", "DejaVu Sans"], "font.size": 11,
                     "axes.titlesize": 15, "axes.titleweight": "semibold",
                     "text.color": "#1B2D45", "axes.labelcolor": "#637287",
                     "xtick.color": "#637287", "ytick.color": "#1B2D45",
                     "xtick.labelsize": 9, "ytick.labelsize": 10.5,
                     "axes.edgecolor": "#DCE3EB", "savefig.facecolor": "#F7F9FC"})
NAVY, TEAL, MUTED, GRID = "#354C70", "#007F80", "#637287", "#E1E7EF"
ORANGE, RED, GREY = "#DE7640", "#C45460", "#A8B6CA"
NAME = {"shingle_moss": "Shingle match (MOSS-style)", "tfidf_cosine": "TF-IDF cosine",
        "embed_doc": "Embedding: whole doc", "embed_maxchunk": "Embedding: best chunk",
        "engine_baseline": "Engine baseline", "engine_v2": "Engine evidence_v2"}
COLOR = {"shingle_moss": "#6F9CA7", "tfidf_cosine": GREY, "embed_doc": GREY,
         "embed_maxchunk": GREY, "engine_baseline": NAVY, "engine_v2": TEAL}

def pct(v):
    """Round half up, so 87.5 -> 88 and 62.5 -> 63 (Python round() is half-even)."""
    return int(v * 100 + 0.5)


fig = plt.figure(figsize=(16, 12.4), facecolor="#F7F9FC")
fig.text(.055, .962, "MULTIMODAL INTEGRITY ENGINE", fontfamily="Montserrat", fontsize=27,
         fontweight="bold")
fig.text(.055, .930, "BENCHMARK  /  300-submission batch, real human text, held-out test",
         fontsize=13, color=MUTED)
fig.text(.945, .960, "TEST SPLIT FROZEN BEFORE SCORING", ha="right", fontsize=10, color=TEAL,
         weight="semibold")
fig.text(.945, .936, "CPU only  ·  Ryzen 5 4600H  ·  06 NOV 2026", ha="right", fontsize=9,
         color=MUTED)
fig.add_artist(plt.Line2D([.055, .945], [.912, .912], transform=fig.transFigure, color=GRID))

# KPI row: engine_v2 at its dev-tuned threshold, compared with engine baseline.
v2, base = M["engine_v2"]["dev_tuned"], M["engine_baseline"]["dev_tuned"]
kpis = [("F1", v2["f1"], base["f1"]), ("RECALL", v2["recall"], base["recall"]),
        ("PRECISION", v2["precision"], base["precision"]),
        ("RANKING  (AVG PRECISION)", M["engine_v2"]["average_precision"],
         M["engine_baseline"]["average_precision"])]
fig.text(.055, .884, "ENGINE evidence_v2  ·  HELD-OUT TEST", fontsize=10, color=MUTED,
         weight="semibold")
for k, (label, val, old) in enumerate(kpis):
    x = .055 + k * .160
    fig.text(x, .862, label, fontsize=9, color=MUTED)
    fig.text(x, .826, f"{100 * val:.1f}%", fontsize=28, color=TEAL, weight="semibold")
    d = 100 * (val - old)
    fig.text(x, .806, f"Baseline {100 * old:.1f}%   ({'+' if d >= 0 else '−'}{abs(d):.1f} pts)",
             fontsize=9.5, color=MUTED)
fig.text(.715, .884, "WHAT WAS SCORED", fontsize=10, color=MUTED, weight="semibold")
fig.text(.715, .838, f"{S['n_pairs']:,}", fontsize=28, weight="semibold")
fig.text(.715, .806, f"document pairs  ·  {S['n_positive']} planted copies  ·  "
         f"{S['n_same_topic']} same-topic traps", fontsize=9.5, color=MUTED)


def axes(rect, title, subtitle, x0=0.0):
    ax = fig.add_axes(rect, facecolor="none")
    ax.set_title(title, loc="left", x=x0, pad=32)
    ax.text(x0, 1.075, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, pad=6)
    return ax


# 1. F1 with 95% bootstrap intervals; hybrid shown as not finished.
order = sorted(M, key=lambda m: M[m]["dev_tuned"]["f1"])
ax = axes([.200, .47, .205, .245], "F1 by method", "Threshold picked on dev, applied unchanged "
          "to test  ·  line = 95% CI", x0=-.71)
y = np.arange(len(order)) + 1
for yi, m in zip(y, order):
    r = M[m]["dev_tuned"]
    ax.barh(yi, 100 * r["f1"], height=.6, color=COLOR[m])
    lo, hi = (100 * v for v in r["f1_ci95"])
    ax.plot([lo, hi], [yi, yi], color="#1B2D45", lw=1.2)
    ax.text(103, yi, f"{100 * r['f1']:.1f}%", va="center", fontsize=10, weight="semibold")
ax.barh(0, 100, height=.6, color="none", edgecolor=GREY, hatch="///", lw=0)
ax.text(50, 0, f"not finished: stopped after {HYB['elapsed_s_when_stopped'] / 3600:.1f} h on dev",
        ha="center", va="center", fontsize=9, color=MUTED,
        bbox=dict(fc="#F7F9FC", ec="none", pad=1.5))
ax.set_yticks([0, *y], ["Engine semantic_hybrid", *[NAME[m] for m in order]])
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
ax.xaxis.grid(True, color=GRID, lw=.7)
ax.set_axisbelow(True)

# 2. Recall per reuse type (heatmap).
kinds = ["verbatim", "partial_copy", "tool_paraphrase", "llm_paraphrase", "human_paraphrase",
         "human_rewrite"]
klabel = ["Verbatim\ncopy", "2 copied\nsentences", "SpinBot\nparaphrase", "GPT-3\nparaphrase",
          "Human\nparaphrase", "Human\nrewrite"]
rows = ["engine_v2", "engine_baseline", "shingle_moss", "tfidf_cosine", "embed_maxchunk"]
grid = np.array([[M[m]["dev_tuned"]["recall_by_kind"][k] for k in kinds] for m in rows])
ax = fig.add_axes([.615, .47, .33, .225], facecolor="none")
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.tick_params(axis="both", length=0)
fig.text(.475, .772, "Copies caught, by reuse type", fontsize=15, weight="semibold")
fig.text(.475, .752, "Recall on 16 planted pairs per type  ·  paraphrases from SpinBot, GPT-3, "
         "MRPC and Webis-CPC-11 crowd workers", fontsize=9, color=MUTED)
cmap = LinearSegmentedColormap.from_list("t", ["#F3D9CF", "#FFFFFF", "#9FD3CF", TEAL])
ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(len(rows)):
    for j in range(len(kinds)):
        v = grid[i, j]
        ax.text(j, i, f"{pct(v)}%", ha="center", va="center", fontsize=10.5,
                weight="semibold", color="white" if v > .8 else "#1B2D45")
ax.set_xticks(range(len(kinds)), klabel, fontsize=9)
ax.xaxis.tick_top()
ax.set_yticks(range(len(rows)), [NAME[m] for m in rows])
ax.tick_params(axis="x", pad=4)

# 3. False alarms by negative type at the dev-tuned threshold.
fa = ["engine_v2", "engine_baseline", "shingle_moss", "tfidf_cosine", "embed_maxchunk",
      "embed_doc"]
ax = axes([.175, .115, .17, .21], "False alarms", f"Wrong flags among {S['n_pairs'] - S['n_positive']:,} "
          "innocent pairs", x0=-.70)
left = np.zeros(len(fa))
for key, col, lab in [("same_topic", ORANGE, "Same topic"), ("template", RED, "Shared brief"),
                      ("other", GREY, "Unrelated")]:
    vals = np.array([M[m]["dev_tuned"]["fp_by_negative_type"][key] for m in fa])
    ax.barh(range(len(fa)), vals, left=left, color=col, height=.6, label=lab)
    left += vals
for i, t in enumerate(left):
    ax.text(t + 1, i, f"{int(t)}", va="center", fontsize=10, weight="semibold")
ax.set_yticks(range(len(fa)), [NAME[m] for m in fa], fontsize=9.5)
ax.invert_yaxis()
ax.set_xlim(0, max(left) * 1.18)
ax.xaxis.grid(True, color=GRID, lw=.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5, loc="center right", handlelength=.8)
ship = M["engine_v2"]["shipped"]
ax.text(-.70, -.22, f"At the shipped 20% flag level: {ship['fp']} false alarms, but only\n"
        f"{pct(ship['recall_by_kind']['partial_copy'])}% of 2-sentence copies are flagged",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="top", linespacing=1.4)

# 4. Runtime for the 300-document batch, log scale; hybrid is a lower bound.
rt = ["shingle_moss", "tfidf_cosine", "embed_maxchunk", "engine_baseline", "engine_v2"]
ax = axes([.505, .115, .17, .21], "Runtime, 300 documents", "Wall-clock seconds, one CPU "
          "process  ·  log scale", x0=-.62)
vals = [WALL[m] for m in rt]
ax.barh(range(len(rt)), vals, color=[COLOR[m] for m in rt], height=.6)
for i, v in enumerate(vals):
    ax.text(v * 1.25, i, f"{v:,.1f} s" if v < 10 else f"{v:,.0f} s", va="center", fontsize=10,
            weight="semibold")
hy = HYB["elapsed_s_when_stopped"]
ax.barh(len(rt), hy, color="none", edgecolor=ORANGE, hatch="///", height=.6)
ax.text(hy * 1.25, len(rt), f"> {hy:,} s", va="center", fontsize=10, weight="semibold",
        color=ORANGE)
ax.set_xscale("log")
ax.set_xlim(.1, 3e5)
ax.minorticks_off()
ax.set_xticks([1, 10, 100, 1e3, 1e4], ["1 s", "10", "100", "1k", "10k"])
ax.set_yticks(range(len(rt) + 1), [NAME[m].replace("Shingle match (MOSS-style)", "Shingle match")
                                   for m in rt] + ["Semantic hybrid (dev)"], fontsize=9.5)
ax.invert_yaxis()
ax.xaxis.grid(True, color=GRID, lw=.7)
ax.set_axisbelow(True)
ax.text(-.62, -.22, "Hybrid runs two cross-encoder models per candidate\npair; stopped unfinished on this CPU",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="top", linespacing=1.4)

# 5. Where semantic_hybrid was measured: two earlier, smaller tests.
ax = axes([.735, .115, .21, .21], "Semantic hybrid: earlier gains",
          "Earlier tests  ·  F1  ·  needs GPU to scale", x0=0)
groups = [("Word-order traps\n100 PAWS pairs", PAWS["baseline"]["f1"], PAWS["semantic_hybrid"]["f1"]),
          ("Known misses\n13 files, 78 pairs", REG["baseline"]["f1"], REG["semantic_hybrid"]["f1"])]
for i, (lab, b_, h_) in enumerate(groups):
    ax.bar(i - .19, 100 * b_, .36, color=NAVY, label="Baseline" if i == 0 else None)
    ax.bar(i + .19, 100 * h_, .36, color=TEAL, label="Hybrid" if i == 0 else None)
    ax.text(i - .19, 100 * b_ + 2, f"{100 * b_:.1f}", ha="center", fontsize=10, weight="semibold")
    ax.text(i + .19, 100 * h_ + 2, f"{100 * h_:.1f}", ha="center", fontsize=10, weight="semibold",
            color=TEAL)
ax.set_xticks([0, 1], [g[0] for g in groups], fontsize=9.5)
ax.set_ylim(0, 115)
ax.set_yticks([0, 50, 100], ["0", "50", "100%"])
ax.yaxis.grid(True, color=GRID, lw=.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
ax.text(0, -.22, "Not completed on the 300-doc batch (CPU limit);\nnot comparable with the panels on the left",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="top", linespacing=1.4)

# Footer: honest reading of the comparison.
c = S["engine_v2_minus_best_competitor_f1"]
sh = M["shingle_moss"]["dev_tuned"]
fig.add_artist(plt.Line2D([.055, .945], [.030, .030], transform=fig.transFigure,
                          color=GRID))
fig.add_artist(Rectangle((.055, .385), .89, .052, transform=fig.transFigure, fc="#EAF4F4",
                         ec="none"))
fig.text(.068, .418, "READING", fontsize=9, weight="semibold", color=TEAL)
fig.text(.135, .418, f"Engine v2 and shingle matching tie on F1 (difference {100 * c['point']:+.1f} "
         f"pts, 95% CI {100 * c['ci95'][0]:+.1f} to {100 * c['ci95'][1]:+.1f}). The engine catches "
         f"more human rewrites ({round(16 * v2['recall_by_kind']['human_rewrite'])}/16 vs "
         f"{round(16 * sh['recall_by_kind']['human_rewrite'])}/16) and ranks better; shingles raise "
         f"no false alarms.", fontsize=10)
fig.text(.135, .396, "Pure embedding search, the common shortcut, catches under half the copies. "
         "TF-IDF flags the shared assignment brief; the engine's template filter does not.",
         fontsize=10, color=MUTED)
fig.text(.055, .014, "SOURCE  benchmarks/realistic/results/summary_test.json  +  "
         "improvements/runs/heldout_*_final.json  +  improvements/comparison.json  |  "
         "Every number is read from these files.", fontsize=8.5, color=MUTED)

png = OUT / "production_benchmark.png"
fig.savefig(png, dpi=240, facecolor=fig.get_facecolor())
plt.close(fig)
srcs = [RES / "summary_test.json", RES / "dev_engine_hybrid_incomplete.json",
        ROOT / "improvements/runs/heldout_baseline_final.json",
        ROOT / "improvements/runs/heldout_semantic_hybrid_final.json",
        ROOT / "improvements/comparison.json", *[RES / f"test_{m}.json" for m in M]]
(OUT / "production_benchmark_provenance.json").write_text(json.dumps({
    "image": png.name,
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in srcs},
    "matplotlib": matplotlib.__version__}, indent=2))
print(png)
