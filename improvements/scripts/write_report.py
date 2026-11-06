"""Generate a new report only; never overwrite earlier evaluation documents."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];I=R/'improvements'
comparison=json.loads((I/'comparison.json').read_text())
methods=['baseline','boundary','semantic_sts','semantic_nli','semantic_hybrid']
read=lambda split,mode:json.loads((I/f'runs/{split}_{mode}_final.json').read_text())
dev={m:read('development',m) for m in methods};blind={m:read('heldout',m) for m in methods}

def table(rows):
 out=['| Mode | TP / FP / FN / TN | Precision | Recall | F1 | Case total s | Median / p95 s | Peak RSS MiB |','|---|---|---:|---:|---:|---:|---:|---:|']
 for mode,x in rows.items():
  v=x['metrics'];t=x['timing'];out.append(f"| {mode} | {v['tp']} / {v['fp']} / {v['fn']} / {v['tn']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {t['cases_total_s']:.2f} | {t['median_case_s']:.3f} / {t['p95_case_s']:.3f} | {x['peak_rss_mb']:.0f} |")
 return '\n'.join(out)

pipeline=['| Mode | TP / FP / FN / TN | Precision | Recall | F1 | Full CLI s | Peak RSS MiB |','|---|---|---:|---:|---:|---:|---:|']
for mode,x in comparison['pipeline_metrics'].items():
 rt=x['runtime'];pipeline.append(f"| {mode} | {x['tp']} / {x['fp']} / {x['fn']} / {x['tn']} | {x['precision']:.3f} | {x['recall']:.3f} | {x['f1']:.3f} | {rt['elapsed_s']:.2f} | {rt['peak_rss_mb']:.1f} |")
rows={r['id']:r for r in json.loads((I/'data/heldout_disjoint.json').read_text())}
fp=[r for r in blind['semantic_hybrid']['cases'] if r['predicted'] and not r['label']]
baseline_fp={r['id'] for r in blind['baseline']['cases'] if r['predicted'] and not r['label']}
introduced=[r for r in fp if r['id'] not in baseline_fp]
removed=baseline_fp-{r['id'] for r in fp}
examples=[]
for r in (introduced + [r for r in fp if r['id'] in baseline_fp])[:2]:
 source=rows[r['id']];p=r['passages'][0] if r['passages'] else {}
 examples.append(f"- **{r['id']}** (human label negative): “{source['left'][:180]}” ↔ “{source['right'][:180]}”. Coverage {r['score']:.3f}; passage cosine {p.get('similarity_score')}; note `{p.get('note')}`. Full text and evidence remain in the JSON.")
if not examples:examples=['No hybrid false positives occurred in this small heldout slice; this does not establish a zero population false-positive rate. STS development false positive d14 is preserved below.']
reg=comparison['pipeline_details']['semantic_hybrid']['positive_pairs']
reglines=[]
for r in reg:
 evidence=r['evidence'];p=evidence[0] if evidence else {}
 reglines.append(f"| {' ↔ '.join(r['pair'])} | {r['detected']} | {p.get('similarity_score')} | {p.get('location')} |")
freeze=json.loads((I/'final_method_freeze.json').read_text())
text=f'''# Matching improvements and reproducible comparison

The proposed **semantic_hybrid** mode fixes both confirmed misses in the actual 13-file CLI run. The original algorithm remains the default **baseline**, selectable explicitly. These changes are opt-in because broader semantic matching adds CPU/memory cost and remains fallible. No original similarity acceptance/risk thresholds, embeddings, earlier reports, or benchmark outputs were changed.

## Exact changes

1. `integrity_engine/boundaries.py` reconstructs sentence units across unfinished adjacent ASR segments. Original full_text, segment pages and floating-point timestamps stay intact; every unit is an exact original character slice. Reconstruction stops at terminal punctuation, pauses >1.5 s, 45 s, or 120 word tokens. PDF short initial unpunctuated heading lines become separate units and remain visible. This is a narrow heuristic, not layout/OCR understanding.
2. `enhanced_alignment.py` tries contiguous windows of up to three reconstructed units. This tolerates premature ASR punctuation while retaining source page and segment timestamps. Windows cannot cross PDF pages, excluded template spans or long audio pauses. Whole short text-page context is also considered, never a whole audio page across unrelated pauses. Boundary-only additions require token-LCS ≥.80 AND MiniLM cosine ≥.65. The original .65/.85 and lexical acceptance bars remain unchanged.
3. Stage one retrieves MiniLM window neighbours at cosine ≥.35, top five after adaptive self-exclusion. That **candidate threshold is not an acceptance threshold**. Within each candidate document pair, at most three neighbours per window in each direction, capped at 48 unique window pairs, go to verification. Permissive candidate retrieval alone emits no new overlap.
4. Stage two compares two pinned CPU cross-encoders. STS is `cross-encoder/stsb-distilroberta-base`; NLI is `cross-encoder/nli-deberta-v3-small`. Strict NLI requires minimum entailment across both input directions ≥.70 and maximum contradiction ≤.10. The selected hybrid accepts STS ≥.50 **or** strict bidirectional entailment ≥.70, and always requires NLI contradiction ≤.10. The .50 STS operating point was selected on development examples; it is a model score, not MiniLM cosine. No final heldout labels/results were used to select it.
5. Existing near-verbatim lexical evidence (LCS ≥.90) remains available even without semantic agreement. Reversed claims can still be textual reuse. This exception affects semantic-equivalence benchmark false positives; do not interpret retained lexical evidence as verified logical equivalence. Other baseline semantic evidence must pass verification in semantic modes.
6. Coverage merges accepted original character spans before counting tokens. Overlapping windows do not inflate it. The existing symmetric max-of-directions percentage convention remains unchanged. Models are released before the external/AI stages; the semantic score cache is bounded to 4096 pairs. Missing optional models cause an explicit error, not an undisclosed baseline fallback. Truncated cross-encoder pairs are rejected.

Default configuration and old algorithms are retained. Modes are `baseline`, `boundary`, `semantic_sts`, `semantic_nli`, `semantic_hybrid`. Existing defaults use baseline; proposed parameters are in [proposed.toml](improvements/configs/proposed.toml). Code changes are additive modules plus config/CLI wiring. No new Python dependency was installed; the existing torch/transformers stack loads both extra model weights.

## Evidence and schema

The per-document JSON and Markdown schema stays unchanged. `similarity_score` is still **MiniLM passage cosine**, including when a cross-encoder admits a low-cosine paraphrase. Verification scores/method/contradiction are recorded in the existing `note`; they do not masquerade as cosine. New verified windows keep `evidence_sentence_scores=[]`, since there is no claimed individual sentence alignment. Existing baseline sentence evidence retains its original scores. PDF page and audio timestamps are resolved from original absolute offsets, not guessed from reconstructed punctuation. Full source/query passages remain exact source slices.

| Original intended relationship | Detected in proposed CLI | Example passage cosine | Actual report location |
|---|---|---:|---|
{chr(10).join(reglines)}

Canonical output is [improvements/runs/pipeline_semantic_hybrid](improvements/runs/pipeline_semantic_hybrid/summary/latest.json); all evidence and false-positive pairs appear in [comparison.json](improvements/comparison.json). All files have JSON/Markdown reports. The baseline rerun has **{len(comparison['baseline_substantive_report_mismatches'])} substantive report mismatches** against the earlier frozen baseline (timestamps excluded).

## Development and final-test separation

- The previous 13-file corpus is now development/regression data. Its original results remain unchanged. Two known misses are regression targets, not blind successes.
- Development comparison: 19 deliberately challenging AI-authored pairs, 9 positives and 10 same-topic/contradiction negatives. These are controlled synthetic examples, not human submissions. Whole-paragraph verification was investigated first and underperformed; passage windows were chosen using development results. STS alone accepted reversed acquisition d14; strict NLI sacrificed recall. Hybrid development had one miss (d16, weight-based waste billing), preserved without removal.
- Final heldout: 100 publicly human-labelled PAWS-Wiki validation rows 0–99, used as this project's final test, with zero exact normalized sentence overlap against the old 2,000-row PAWS benchmark and current development pairs. This is not an externally blinded private evaluation; article-level/pretraining overlap cannot be ruled out. Only this slice was scored after method freeze.
- An initial proposed slice (test rows 2000–2099) shared 76 sentence strings with the old benchmark. It remains preserved **unscored** and was replaced during data audit, before model results were viewed. This was a data-independence correction, not removal of failing model cases.
- Before final inference, two non-selection corrections were recorded: verified-window metadata now leaves sentence scores empty, and the dataset provenance license wording was corrected to the upstream unrestricted-use permission (HF metadata says “other”). Input texts, labels, models, acceptance rules and selected parameters did not change. Freeze histories are preserved. Final freeze timestamp: `{freeze['utc']}`; no algorithm/config edits followed heldout results.

PAWS labels measure **semantic equivalence**, not authorised reuse or plagiarism misconduct. Thus a surface-overlap detector can correctly identify reused words while being a semantic false positive. The tables below separate the 13-file reuse task from this adversarial semantic diagnostic. No risk-label accuracy or disciplinary ground truth is invented.

## Full pipeline comparison on 13 development files

{chr(10).join(pipeline)}

This includes real PDF/DOCX extraction, three MP4 ASR runs, local KB, unchanged AI detection and reports; web is disabled. It evaluates all 78 unordered file pairs, retaining failures. Same-source exact duplicates use the existing shortcut in every mode. These results demonstrate repair of regressions, not generalization.

## Development semantic relationship benchmark

{table(dev)}

## Heldout semantic relationship benchmark

{table(blind)}

Each row is a fresh CPU process with the same supplied dependencies and cached weights; methods run sequentially to avoid mutual CPU contention. Case totals/median/p95 include chunking, embedding, candidate generation, verification, evidence and coverage on in-memory paired documents. They exclude actual file extraction, ASR, AI, reporting and initial model loading. Peak RSS includes models plus runtime, not whole-machine memory. Separate model-load and whole-worker times are in every result JSON. This protocol should not be compared directly with full-CLI wall time. Model-download time is excluded. No 10k-submission latency/cost projection is offered.

## False positives and remaining misses

{chr(10).join(examples)}

The hybrid produces **{len(fp)} semantic false positives**, including **{len(introduced)} newly introduced** relative to baseline; **{len(removed)} baseline false positives are removed**. The examples above prioritise newly introduced errors.

Every heldout error, source label, score and full evidence remains in `improvements/runs/heldout_*_final.json`. No failed case was removed, relabelled or used to adjust the method. Some semantic errors stem from retained lexical evidence, some from verifier overconfidence/partial-window overlap; their presence is a reason to keep this mode opt-in and require human review. STS development d14 specifically reverses who acquired whom while returning score .9796; the NLI contradiction signal was .9989 and the hybrid rejects it.

The selected verifier is not a logical proof system. It can infer entailment for unequal detail, veto a valid paraphrase as contradictory, or miss paraphrases outside retrieval top-k. Conservative limits avoid unbounded all-pairs cross-encoding but cap recall. The PDF first-line heuristic can mistake an unpunctuated short body line for a heading. ASR joins use segment-level intervals, not new word-accurate annotation; no confidence restoration or larger ASR model is added. Existing automatic three-document boilerplate suppression and quadratic legacy chunk aggregation/dense summary behavior remain baseline limitations outside this repair's scope.

## Tests, reproducibility and preservation

The 43-test suite passed (34 original + 9 new, one upstream torch deprecation warning). New tests cover the two confirmed mechanisms with real cached models, exact page/timestamp slices, premature ASR punctuation, long pauses/size guards, unchanged cosine floor, union coverage, contradiction/truncation rejection and invalid parameter bounds. Original sample scenarios still pass in baseline mode. Full CLI checks separately verify actual proposed-mode schema and evidence.

[improvements/README.md](improvements/README.md) contains exact commands, file layout and model revisions. Models and data have hashes and pinned revisions; benchmark workers refuse to overwrite output files and refuse heldout execution if the frozen method changed. Prior result preservation checks report **{len(comparison['changed_prior_result_files'])} changed files**, and final method checks **{len(comparison['changed_final_method_files'])} changes**. Earlier EVALUATION.md, all four validation reports, validation outputs and saved benchmarks remain byte-for-byte unchanged. A source archive retains the original implementation at commit e15af35.

Model references: [STS model card](https://huggingface.co/cross-encoder/stsb-distilroberta-base), [NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small). The former is trained for textual similarity; the latter on SNLI/MultiNLI for entailment/neutral/contradiction. Both are Apache 2.0. Dataset acknowledgement: Google LLC, Yuan Zhang, Jason Baldridge and Luheng He, [PAWS (2019)](https://github.com/google-research-datasets/paws), upstream permission preserved in `improvements/data/PAWS_LICENSE.txt`. Newly authored development fixtures are synthetic and openly reusable under the project MIT license.
'''
(R/'MATCHING_IMPROVEMENTS.md').write_text(text)
print('Wrote MATCHING_IMPROVEMENTS.md')
