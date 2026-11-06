"""<MODULE PURPOSE>
Self-evaluation against the ground-truth manifest (spec §4.11) plus roll-up of
the public-dataset benchmarks (benchmarks/results/*.json).

Computes: document-pair retrieval P/R/F1 (overall and per relation kind, where a
predicted pair's relation is the engine's own overlap_kind label), false-positive
rate on the declared originals, risk-level confusion matrix (as shipped, and with
the AI signal removed so plagiarism logic is judged on its own), the AI-detection
table (model rows and heuristic-fallback rows kept separate), manifest flag checks,
per-scenario pass/fail with the evidence file, and runtime.

EVALUATION.md is updated in place: only the regions between
`<!-- AUTO:BEGIN name -->` and `<!-- AUTO:END name -->` are regenerated, so the
hand-written analysis (failure studies, trade-offs) survives every --eval run.

<PUBLIC API> run_evaluation, evaluate, update_auto_blocks
<DEPENDS ON> metrics, scoring (verdict rule), config, json/pathlib.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from integrity_engine.config import Config
from integrity_engine.metrics import confusion_matrix, prf
from integrity_engine.scoring import verdict_from_signals

logger = logging.getLogger("integrity_engine.evaluation")

LABELS = ("LOW", "MEDIUM", "HIGH")
KIND_TO_RELATION = {"verbatim": "exact_copy", "paraphrase": "paraphrase",
                    "cross_modal": "cross_modal"}
RELATIONS = ("exact_copy", "paraphrase", "cross_modal")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_reports(output_dir: Path) -> dict[str, dict[str, Any]]:
    reports = {}
    for path in sorted((output_dir / "per_doc").glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[report["document_id"]] = report
    return reports


def _predicted_pairs(reports: dict[str, dict]) -> dict[frozenset, str]:
    """Unordered file pair -> engine relation (from the strongest passage's kind)."""
    best: dict[frozenset, tuple[float, str]] = {}
    for doc_id, report in reports.items():
        for match in report["peer_matches"] + report["multimodal_matches"]:
            if match["match_type"] != "overlap":
                continue
            key = frozenset((doc_id, match["matched_file"]))
            relation = KIND_TO_RELATION.get(match.get("overlap_kind"), "paraphrase")
            score = float(match["similarity_score"])
            if key not in best or score > best[key][0]:
                best[key] = (score, relation)
    return {k: v[1] for k, v in best.items()}


def evaluate(output_dir: Path, manifest: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """All metrics as a JSON-serializable dict (also written to summary_metrics.json)."""
    reports = _load_reports(output_dir)
    summary_path = output_dir / "summary" / "latest.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    truth_pairs = {frozenset((p["query"], p["source"])): p["relation"]
                   for p in manifest["ground_truth_pairs"]}
    missing = [d for d in manifest["expected_integrity_risk"] if d not in reports]
    for doc in missing:
        logger.warning("eval: manifest entry %s missing from reports; counted as FAIL "
                       "for recall", doc)
    predicted = _predicted_pairs(reports)

    # --- pair retrieval ---------------------------------------------------------
    tp = sum(1 for k in predicted if k in truth_pairs)
    fp_pairs = sorted(sorted(k) for k in predicted if k not in truth_pairs)
    fn_pairs = sorted(sorted(k) for k in truth_pairs if k not in predicted)
    overall = prf(tp, len(fp_pairs), len(fn_pairs))
    per_relation = {}
    for rel in RELATIONS:
        r_tp = sum(1 for k, v in predicted.items() if v == rel and truth_pairs.get(k) == rel)
        r_fp = sum(1 for k, v in predicted.items() if v == rel and truth_pairs.get(k) != rel)
        r_fn = sum(1 for k, v in truth_pairs.items() if v == rel and predicted.get(k) != rel)
        p, r, f = prf(r_tp, r_fp, r_fn)
        per_relation[rel] = {"tp": r_tp, "fp": r_fp, "fn": r_fn, "precision": p,
                             "recall": r, "f1": f}
    pair_scores = []
    for key, rel in truth_pairs.items():
        a, b = sorted(key)
        score = max(reports.get(a, {}).get("doc_pair_scores", {}).get(b, 0.0),
                    reports.get(b, {}).get("doc_pair_scores", {}).get(a, 0.0))
        expected = next(p["expected_min_peer_similarity"] for p in manifest[
            "ground_truth_pairs"] if frozenset((p["query"], p["source"])) == key)
        pair_scores.append({"pair": [a, b], "relation": rel, "doc_pair_score": score,
                            "expected_min": expected, "pass": score >= expected,
                            "engine_relation": predicted.get(key)})

    # --- originals FPR ----------------------------------------------------------
    guard = manifest.get("original_false_positive_guard", [])
    guard_hits = {d: [m["matched_file"] for m in reports.get(d, {}).get("peer_matches", [])
                      + reports.get(d, {}).get("multimodal_matches", [])
                      if m["match_type"] != "common_source"] for d in guard}
    fpr_originals = sum(1 for v in guard_hits.values() if v) / max(1, len(guard))

    # --- risk confusion matrices --------------------------------------------------
    truth_risk = manifest["expected_integrity_risk"]
    docs = list(truth_risk)
    y_true = [truth_risk[d] for d in docs]
    y_pred = [reports.get(d, {}).get("total_integrity_risk", "MISSING") for d in docs]
    y_plag = []
    for d in docs:
        r = reports.get(d)
        if r is None:
            y_plag.append("MISSING")
            continue
        s = r["scores"]
        cross = bool(r["multimodal_matches"])
        y_plag.append(verdict_from_signals(s["peer_similarity_percentage"],
                                           s["external_similarity_percentage"],
                                           "HUMAN", None, cross))

    def cm_block(pred: list[str]) -> dict[str, Any]:
        cm = confusion_matrix(y_true, pred, LABELS)
        correct = sum(1 for t, p in zip(y_true, pred) if t == p)
        recalls = [sum(1 for t, p in zip(y_true, pred) if t == lab and p == lab) /
                   max(1, sum(1 for t in y_true if t == lab))
                   for lab in LABELS if any(t == lab for t in y_true)]
        wrong = [{"document_id": d, "truth": t, "predicted": p}
                 for d, t, p in zip(docs, y_true, pred) if t != p]
        return {"matrix": {f"{t}->{p}": cm[(t, p)] for t in LABELS for p in LABELS},
                "accuracy": correct / max(1, len(docs)),
                "balanced_accuracy": sum(recalls) / max(1, len(recalls)),
                "mislabeled": wrong}

    # --- AI detection table -----------------------------------------------------
    ai_flags = {d: f["ai_expected_min_probability"]
                for d, f in manifest.get("ground_truth_flags", {}).items()
                if "ai_expected_min_probability" in f}
    ai_rows = []
    for d, r in reports.items():
        a = r["ai_analysis"]
        if d in ai_flags or a["label"] in ("UNSURE", "AI_LIKELY"):
            ai_rows.append({"document_id": d, "manifest_ai_flag": d in ai_flags,
                            "p_classifier": a["p_classifier"], "perplexity": a["perplexity"],
                            "p_perp": a["p_perplexity"], "burstiness": a["burstiness"],
                            "p_burst": a["p_burstiness"],
                            "fused": a["ai_generated_probability"], "label": a["label"],
                            "ai_detector_source": a["ai_detector_source"]})
    fallback_count = sum(1 for r in reports.values()
                         if r["ai_analysis"]["ai_detector_source"] == "heuristic_fallback")

    # --- manifest flag checks -----------------------------------------------------
    flag_checks = []
    for d, flags in manifest.get("ground_truth_flags", {}).items():
        r = reports.get(d)
        if r is None:
            continue
        if "external_expected_min" in flags:
            v = r["scores"]["external_similarity_percentage"]
            flag_checks.append({"document_id": d, "check": "external >= "
                                f"{flags['external_expected_min']}", "value": v,
                                "pass": v >= flags["external_expected_min"]})
        if "ai_expected_min_probability" in flags:
            v = r["scores"]["ai_generated_probability"]
            flag_checks.append({"document_id": d, "check": "ai >= "
                                f"{flags['ai_expected_min_probability']}", "value": v,
                                "pass": v is not None and v >= flags[
                                    "ai_expected_min_probability"]})
        if "expected_peer_matches" in flags:
            n = sum(1 for m in r["peer_matches"] + r["multimodal_matches"]
                    if m["match_type"] == "overlap")
            flag_checks.append({"document_id": d, "check": "overlap peer matches == "
                                f"{flags['expected_peer_matches']}", "value": n,
                                "pass": n == flags["expected_peer_matches"]})
    for entry in manifest.get("degraded", []):
        r = reports.get(entry["file"])
        status = r["status"] if r else "MISSING"
        flag_checks.append({"document_id": entry["file"], "check": "status == "
                            f"{entry['expected_status']}", "value": status,
                            "pass": status == entry["expected_status"]})

    scenarios = _scenario_checks(reports, manifest, summary)
    return {
        "n_reports": len(reports), "missing_reports": missing,
        "pair_retrieval": {"overall": {"tp": tp, "fp": len(fp_pairs), "fn": len(fn_pairs),
                                       "precision": overall[0], "recall": overall[1],
                                       "f1": overall[2]},
                           "per_relation": per_relation, "false_positive_pairs": fp_pairs,
                           "false_negative_pairs": fn_pairs, "truth_pair_scores": pair_scores},
        "fpr_originals": fpr_originals, "originals_overlap_matches": guard_hits,
        "risk_confusion_as_shipped": cm_block(y_pred),
        "risk_confusion_plagiarism_signals_only": cm_block(y_plag),
        "ai_rows": ai_rows, "ai_detector_fallback_count": fallback_count,
        "flag_checks": flag_checks, "scenarios": scenarios,
        "runtime": summary.get("runtime", {}),
        "thresholds": {"chunk_min_cos": cfg.pairwise.chunk_min_cos,
                       "doc_report_floor": cfg.pairwise.doc_report_floor,
                       "doc_flag_threshold(HIGH peer%)": 70,
                       "sentence_align_cos_floor": cfg.alignment.sentence_align_cos_floor,
                       "sentence_lcs_token_ratio": cfg.alignment.sentence_lcs_token_ratio,
                       "template_min_docs": cfg.alignment.template_min_docs,
                       "local_kb_min_cos": cfg.external.local_kb_min_cos,
                       "ai_bands": [cfg.ai.band_human_max, cfg.ai.band_unsure_max],
                       "ai_perplexity_mu_k": [cfg.ai.perplexity_mu, cfg.ai.perplexity_k],
                       "ai_burst_mu_k": [cfg.ai.burst_mu, cfg.ai.burst_k]},
    }


def _scenario_checks(reports: dict[str, dict], manifest: dict, summary: dict) -> list[dict]:
    def rep(d: str) -> dict:
        return reports.get(d, {})

    out = []
    r07 = rep("07_pitch_crossmodal.pdf")
    mm = r07.get("multimodal_matches", [])
    ok = bool(mm) and all(m["location"].get("matched_timestamp") and
                          m["location"].get("query_page") for m in mm)
    out.append({"scenario": "S1 cross-modal duplication (07 PDF <-> 08 MP4)", "pass": ok,
                "evidence": f"per_doc/07_pitch_crossmodal.pdf.json multimodal_matches[0]: "
                            + (f"page {mm[0]['location']['query_page']} <-> "
                               f"{mm[0]['location']['matched_timestamp']}, cos "
                               f"{mm[0]['similarity_score']}" if mm else "none")})
    r03 = rep("03_paraphrased_strategy.docx")
    para = [m for m in r03.get("peer_matches", []) if m.get("overlap_kind") == "paraphrase"
            and m["matched_file"] == "04_original_strategy.docx"]
    high_low = [m for m in para if m["similarity_score"] > 0.80
                and (m.get("lexical_jaccard") or 1) < 0.35]
    out.append({"scenario": "S2 paraphrase (03 <-> 04): cosine > 0.80 with Jaccard < 0.35",
                "pass": bool(high_low),
                "evidence": f"{len(para)} paraphrase passages; {len(high_low)} with cos>0.80 "
                            "& Jaccard<0.35; best: " + (
                                f"cos {max(m['similarity_score'] for m in para):.3f}"
                                if para else "none")})
    st10, st11 = rep("10_corrupted_empty.pdf").get("status"), \
        rep("11_silent_video.mp4").get("status")
    out.append({"scenario": "S3 corrupted / empty (10 zero-byte PDF, 11 silent MP4)",
                "pass": st10 == "error" and st11 == "silent_video" and
                        len(reports) == len(manifest["expected_integrity_risk"]),
                "evidence": f"10 status={st10}, 11 status={st11}; all "
                            f"{len(reports)} reports written, exit 0"})
    trio = manifest["template_group"]["files"]
    risks = {d: rep(d).get("total_integrity_risk") for d in trio}
    common = {d: sum(1 for m in rep(d).get("peer_matches", [])
                     if m["match_type"] == "common_source") for d in trio}
    peer = {d: rep(d).get("scores", {}).get("peer_similarity_percentage") for d in trio}
    out.append({"scenario": "S4 shared case prompt (12/13/14) not flagged",
                "pass": all(v != "HIGH" for v in risks.values()) and
                        all(v == 0.0 for v in peer.values()) and all(common.values()),
                "evidence": f"risk {risks}; peer% {peer}; common_source matches {common}"})
    n_docs = summary.get("aggregate", {}).get("n_docs")
    out.append({"scenario": "S5 batch run: per-doc JSON+MD + aggregate matrix",
                "pass": n_docs == len(manifest["expected_integrity_risk"]) and
                        bool(summary.get("peer_similarity_matrix")),
                "evidence": f"summary/latest.json n_docs={n_docs}, "
                            f"{len(summary.get('peer_similarity_matrix', {}))}-row matrix, "
                            "summary/SUMMARY.md"})
    return out


# ------------------------------------------------------------------ markdown
def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _cm_table(block: dict) -> list[str]:
    lines = ["| truth \\ predicted | LOW | MEDIUM | HIGH |", "|---|---|---|---|"]
    for t in LABELS:
        lines.append(f"| **{t}** | " + " | ".join(str(block["matrix"][f"{t}->{p}"])
                                                    for p in LABELS) + " |")
    lines.append("")
    lines.append(f"Accuracy {_pct(block['accuracy'])} · balanced accuracy "
                 f"{_pct(block['balanced_accuracy'])}.")
    if block["mislabeled"]:
        lines.append("Mislabeled: " + "; ".join(
            f"`{m['document_id']}` ({m['truth']} → {m['predicted']})"
            for m in block["mislabeled"]) + ".")
    return lines


def render_corpus_blocks(m: dict[str, Any]) -> dict[str, str]:
    blocks: dict[str, str] = {}
    pr = m["pair_retrieval"]
    o = pr["overall"]
    lines = ["| relation | TP | FP | FN | precision | recall | F1 |", "|---|---|---|---|---|---|---|",
             f"| **any (pair-level)** | {o['tp']} | {o['fp']} | {o['fn']} | "
             f"{o['precision']:.2f} | {o['recall']:.2f} | {o['f1']:.2f} |"]
    for rel, r in pr["per_relation"].items():
        lines.append(f"| {rel} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.2f} | "
                     f"{r['recall']:.2f} | {r['f1']:.2f} |")
    lines += ["", "Ground-truth pairs and their document-pair scores (coverage-weighted "
              "mean-max cosine):", "", "| pair | relation | doc-pair score | expected ≥ | "
              "engine relation label | pass |", "|---|---|---|---|---|---|"]
    for p in pr["truth_pair_scores"]:
        lines.append(f"| `{p['pair'][0]}` ↔ `{p['pair'][1]}` | {p['relation']} | "
                     f"{p['doc_pair_score']:.3f} | {p['expected_min']} | "
                     f"{p['engine_relation']} | {'✅' if p['pass'] else '❌'} |")
    if pr["false_positive_pairs"]:
        lines.append("")
        lines.append("False-positive pairs: " + "; ".join(
            f"`{a}` ↔ `{b}`" for a, b in pr["false_positive_pairs"]))
    blocks["pairs"] = "\n".join(lines)

    hits = m["originals_overlap_matches"]
    blocks["fpr"] = "\n".join(
        [f"FPR on the {len(hits)} declared originals = **{_pct(m['fpr_originals'])}** "
         "(an original counts as a false positive if it has ≥1 non-template peer or "
         "cross-modal match).", "",
         "| original | overlap matches |", "|---|---|"] +
        [f"| `{d}` | {', '.join(v) or 'none'} |" for d, v in hits.items()])

    ai_lines = [f"`ai_detector_fallback_count` = **{m['ai_detector_fallback_count']}** "
                "(reports whose AI verdict came from the heuristic fallback; such rows are "
                "listed separately and never averaged with model rows).", "",
                "| document | manifest AI flag | p_classifier | perplexity | p_perp | "
                "burstiness | p_burst | fused | label | source |",
                "|---|---|---|---|---|---|---|---|---|---|"]
    for source in ("model", "heuristic_fallback"):
        for r in [x for x in m["ai_rows"] if x["ai_detector_source"] == source]:
            ai_lines.append(
                f"| `{r['document_id']}` | {'yes' if r['manifest_ai_flag'] else 'no'} | "
                f"{_fmt(r['p_classifier'])} | {_fmt(r['perplexity'])} | {_fmt(r['p_perp'])} | "
                f"{_fmt(r['burstiness'])} | {_fmt(r['p_burst'])} | {_fmt(r['fused'])} | "
                f"{r['label']} | `{source}` |")
    blocks["ai"] = "\n".join(ai_lines)

    blocks["confusion"] = "\n".join(
        ["**As shipped** (all three signals; AI-likely alone ⇒ MEDIUM):", ""] +
        _cm_table(m["risk_confusion_as_shipped"]) +
        ["", "**Plagiarism signals only** (same rule with the AI label forced to HUMAN; "
         "isolates peer + external logic):", ""] +
        _cm_table(m["risk_confusion_plagiarism_signals_only"]))

    fl = ["| document | check | value | pass |", "|---|---|---|---|"]
    fl += [f"| `{c['document_id']}` | {c['check']} | {_fmt(c['value'])} | "
           f"{'✅' if c['pass'] else '❌'} |" for c in m["flag_checks"]]
    blocks["flags"] = "\n".join(fl)

    sc = ["| scenario | result | evidence |", "|---|---|---|"]
    sc += [f"| {s['scenario']} | {'✅ pass' if s['pass'] else '❌ fail'} | {s['evidence']} |"
           for s in m["scenarios"]]
    blocks["scenarios"] = "\n".join(sc)

    rt = m.get("runtime", {})
    stages = rt.get("stage_seconds", {})
    blocks["runtime"] = "\n".join(
        [f"Last run: **{rt.get('elapsed_s')} s** wall-clock for {m['n_reports']} files, "
         f"peak RSS **{rt.get('peak_rss_mb')} MiB**, {rt.get('n_chunks')} chunks, "
         f"{rt.get('cpu_count')} logical CPUs, CPU only.", "",
         "| " + " | ".join(stages) + " |", "|---" * len(stages) + "|",
         "| " + " | ".join(str(v) for v in stages.values()) + " |"])
    th = m["thresholds"]
    blocks["thresholds"] = "\n".join(["| parameter | value |", "|---|---|"] +
                                     [f"| `{k}` | {v} |" for k, v in th.items()])
    return blocks


def render_benchmark_block(results_dir: Path) -> str:
    """Roll-up of benchmarks/results/*.json (if the benchmark scripts were run)."""
    lines: list[str] = []
    ai = results_dir / "ai_detection.json"
    if ai.exists():
        d = json.loads(ai.read_text())
        p = d["protocol"]
        lines += [f"**AI detection — {d['dataset']}.** {p['test_pairs']} held-out questions "
                  f"({2 * p['test_pairs']} texts, balanced human/ChatGPT); logistic constants "
                  f"fitted on {p['calibration_pairs']} separate calibration questions.", "",
                  "| constants | AUC classifier | AUC perplexity | AUC burstiness | AUC fused "
                  "| AI_LIKELY precision | AI_LIKELY recall | human FPR | human→UNSURE |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, label in (("test_with_spec_constants", "spec a-priori (22/9.5, 0.55/0.14)"),
                            ("test_with_shipped_config", "shipped config")):
            r = d[name]
            a, t = r["auc"], r["fused_at_ai_likely_band"]
            lines.append(f"| {label} | {a['classifier']:.3f} | {a['perplexity_component']:.3f} "
                         f"| {a['burstiness_component']:.3f} | {a['fused']:.3f} | "
                         f"{t['precision']:.3f} | {t['recall']:.3f} | {t['fpr']:.3f} | "
                         f"{r['band_counts']['HUMAN']['UNSURE']}/{r['n_human']} |")
        med = d["test_with_shipped_config"]["median"]
        lines += ["", f"Medians (test): perplexity human {med['ppl_human']:.1f} vs ChatGPT "
                  f"{med['ppl_ai']:.1f}; burstiness human {med['burst_human']:.3f} vs ChatGPT "
                  f"{med['burst_ai']:.3f}; classifier human {med['p_cls_human']:.3f} vs ChatGPT "
                  f"{med['p_cls_ai']:.3f}. Timing: {d.get('timing')}.", ""]
    ood = results_dir / "ai_ood.json"
    if ood.exists():
        d = json.loads(ood.read_text())
        t = d["fused_at_ai_likely_band"]
        rows = d["rows"]
        hum_cls_hi = sum(1 for r in rows if r["y"] == 0 and r["p_cls"] >= 0.5)
        lines += [f"**AI detection, out-of-distribution check.** {d['n_human_wikipedia_2018']} "
                  "human texts (Wikipedia business/economics intros, last revision before "
                  f"2019-01-01) vs {d['n_ai_corpus_docs']} machine-written corpus documents. "
                  f"AUC classifier {d['auc']['classifier']:.3f}, perplexity "
                  f"{d['auc']['perplexity(neg)']:.3f}, fused {d['auc']['fused']:.3f}. "
                  f"At the AI_LIKELY band: precision {t['precision']:.2f}, recall "
                  f"{t['recall']:.2f}, human FPR {t['fpr']:.2f}. Human texts by band: "
                  f"{d['human_band_counts']}. Classifier alone scores ≥ 0.5 on "
                  f"{hum_cls_hi}/{d['n_human_wikipedia_2018']} human Wikipedia texts.", "",
                  "| text | truth | p_classifier | GPT-2 perplexity | fused | label |",
                  "|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: -r["fused"])[:16]:
            lines.append(f"| {r['id']} | {'AI' if r['y'] else 'human'} | {r['p_cls']:.4f} | "
                         f"{r['ppl']:.1f} | {r['fused']:.1f} | {r['label']} |")
        lines += ["", "(16 highest fused scores shown; full list in "
                  "`benchmarks/results/ai_ood.json`.)", ""]
    para = results_dir / "paraphrase.json"
    if para.exists():
        d = json.loads(para.read_text())
        lines += [f"**Paraphrase signal — {d['model']}.**", "",
                  "| dataset | pairs | AUC cosine | AUC Jaccard | AUC token-LCS | cos≥0.80 P / R / FPR "
                  "| cos≥0.65 P / R / FPR | best-F1 cos threshold (F1) |",
                  "|---|---|---|---|---|---|---|---|"]
        for name in ("mrpc_test", "paws_test_first2000"):
            r = d[name]
            a = r["auc"]
            t8 = r["cosine_at_0.80_assignment_band"]
            t6 = next(v for k, v in r.items() if k.startswith("cosine_at_0.6"))
            b = r["cosine_best_f1"]
            lines.append(f"| {name} | {r['n_pairs']} | {a['minilm_cosine']:.3f} | "
                         f"{a['token_jaccard']:.3f} | {a['token_lcs_ratio']:.3f} | "
                         f"{t8['precision']:.2f} / {t8['recall']:.2f} / {t8['fpr']:.2f} | "
                         f"{t6['precision']:.2f} / {t6['recall']:.2f} / {t6['fpr']:.2f} | "
                         f"{b['threshold']:.2f} ({b['f1']:.3f}) |")
        s2 = d["mrpc_test"]["scenario2_cases"]
        lines += ["", f"Scenario-2 slice of MRPC (true paraphrases with token Jaccard < 0.35): "
                  f"{s2['n']} pairs; cosine ≥ 0.65 catches {s2['caught_at_cos_0.65']}, "
                  f"cosine ≥ 0.80 catches {s2['caught_at_cos_0.80']}.", ""]
    ret = results_dir / "retrieval.json"
    if ret.exists():
        d = json.loads(ret.read_text())
        lines += [f"**Corpus-scale plagiarism retrieval — {d['corpus']}.**", "",
                  "| doc-pair score threshold | relation | precision | recall | F1 |",
                  "|---|---|---|---|---|"]
        for row in d["sweep"]:
            lines.append(f"| {row['threshold']} | {row['relation']} | {row['precision']:.3f} | "
                         f"{row['recall']:.3f} | {row['f1']:.3f} |")
        lines += ["", f"Pair-score AUC (all {d['n_candidate_pairs']} doc pairs): "
                  f"{d['auc_all_pairs']:.4f}. Template docs carrying the shared block: "
                  f"{d['template']['n_docs']}, max template-pair score "
                  f"{d['template']['max_pair_score']:.3f}. Throughput: {d['timing']}.", ""]
    idx = results_dir / "index_scaling.json"
    if idx.exists():
        d = json.loads(idx.read_text())
        lines += [f"**Vector index scaling — {d['vectors']}.**", "",
                  "| N vectors | index | build s | query ms (1 query) | batch q/s | recall@10 vs Flat | "
                  "memory MB |", "|---|---|---|---|---|---|---|"]
        for row in d["rows"]:
            lines.append(f"| {row['n']:,} | {row['index']} | {row['build_s']:.2f} | "
                         f"{row['single_query_ms']:.3f} | {row['batch_qps']:,.0f} | "
                         f"{row['recall_at_10']:.4f} | {row['memory_mb']:.1f} |")
        lines.append("")
    asr = results_dir / "asr.json"
    if asr.exists():
        d = json.loads(asr.read_text())
        lines += ["**ASR on the TTS pitch clips (word error rate vs the script).**", "",
                  "| clip | TTS engine | audio s | whisper model | WER | ASR s | real-time factor |",
                  "|---|---|---|---|---|---|---|"]
        for row in d["rows"]:
            lines.append(f"| {row['clip']} | {row['tts']} | {row['audio_s']:.1f} | "
                         f"{row['model']} | {row['wer']:.3f} | {row['asr_s']:.1f} | "
                         f"{row['rtf']:.3f} |")
        lines.append("")
    return "\n".join(lines) if lines else "*Benchmarks not run yet (see benchmarks/README).* "


SKELETON = """# EVALUATION

<!-- AUTO:BEGIN scenarios -->
<!-- AUTO:END scenarios -->

<!-- AUTO:BEGIN pairs -->
<!-- AUTO:END pairs -->

<!-- AUTO:BEGIN fpr -->
<!-- AUTO:END fpr -->

<!-- AUTO:BEGIN ai -->
<!-- AUTO:END ai -->

<!-- AUTO:BEGIN confusion -->
<!-- AUTO:END confusion -->

<!-- AUTO:BEGIN flags -->
<!-- AUTO:END flags -->

<!-- AUTO:BEGIN runtime -->
<!-- AUTO:END runtime -->

<!-- AUTO:BEGIN thresholds -->
<!-- AUTO:END thresholds -->

<!-- AUTO:BEGIN benchmarks -->
<!-- AUTO:END benchmarks -->
"""


def update_auto_blocks(path: Path, blocks: dict[str, str]) -> None:
    """Replace each AUTO region's body; append missing regions at the end."""
    text = path.read_text(encoding="utf-8") if path.exists() else SKELETON
    for name, body in blocks.items():
        pattern = re.compile(rf"(<!-- AUTO:BEGIN {name} -->)(.*?)(<!-- AUTO:END {name} -->)",
                             re.DOTALL)
        replacement = f"<!-- AUTO:BEGIN {name} -->\n{body.strip()}\n<!-- AUTO:END {name} -->"
        if pattern.search(text):
            text = pattern.sub(lambda _m: replacement, text)
        else:
            text += f"\n{replacement}\n"
    path.write_text(text, encoding="utf-8")


def run_evaluation(output_dir: Path, manifest_path: Path, eval_out: Path,
                   cfg: Config) -> Path:
    """--eval entry point: metrics JSON + EVALUATION.md auto-block refresh."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = evaluate(output_dir, manifest, cfg)
    (output_dir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2,
                                                                default=str))
    blocks = render_corpus_blocks(metrics)
    blocks["benchmarks"] = render_benchmark_block(REPO_ROOT / "benchmarks" / "results")
    update_auto_blocks(eval_out, blocks)
    o = metrics["pair_retrieval"]["overall"]
    logger.info("eval: pairs P=%.2f R=%.2f F1=%.2f; originals FPR=%.2f; risk accuracy=%.2f;"
                " scenarios passed %d/%d", o["precision"], o["recall"], o["f1"],
                metrics["fpr_originals"], metrics["risk_confusion_as_shipped"]["accuracy"],
                sum(s["pass"] for s in metrics["scenarios"]), len(metrics["scenarios"]))
    return eval_out
