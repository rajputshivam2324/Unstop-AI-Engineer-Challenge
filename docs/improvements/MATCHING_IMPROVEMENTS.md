# Matching improvements and reproducible comparison

The proposed **semantic_hybrid** mode fixes both confirmed misses in the actual 13-file CLI run. The original algorithm remains the default **baseline**, selectable explicitly. These changes are opt-in because broader semantic matching adds CPU/memory cost and remains fallible. No original similarity acceptance/risk thresholds, embeddings, earlier reports, or benchmark outputs were changed.

## Exact changes

1. `integrity_engine/boundaries.py` reconstructs sentence units across unfinished adjacent ASR segments. Original full_text, segment pages and floating-point timestamps stay intact; every unit is an exact original character slice. Reconstruction stops at terminal punctuation, pauses >1.5 s, 45 s, or 120 word tokens. PDF short initial unpunctuated heading lines become separate units and remain visible. This is a narrow heuristic, not layout/OCR understanding.
2. `enhanced_alignment.py` tries contiguous windows of up to three reconstructed units. This tolerates premature ASR punctuation while retaining source page and segment timestamps. Windows cannot cross PDF pages, excluded template spans or long audio pauses. Whole short text-page context is also considered, never a whole audio page across unrelated pauses. Boundary-only additions require token-LCS ≥.80 AND MiniLM cosine ≥.65. The original .65/.85 and lexical acceptance bars remain unchanged.
3. Stage one retrieves MiniLM window neighbours at cosine ≥.35, top five after adaptive self-exclusion. That **candidate threshold is not an acceptance threshold**. Within each candidate document pair, at most three neighbours per window in each direction, capped at 48 unique window pairs, go to verification. Permissive candidate retrieval alone emits no new overlap.
4. Stage two compares two pinned CPU cross-encoders. STS is `cross-encoder/stsb-distilroberta-base`; NLI is `cross-encoder/nli-deberta-v3-small`. Strict NLI requires minimum entailment across both input directions ≥.70 and maximum contradiction ≤.10. The selected hybrid accepts STS ≥.50 **or** strict bidirectional entailment ≥.70, and always requires NLI contradiction ≤.10. The .50 STS operating point was selected on development examples; it is a model score, not MiniLM cosine. No final heldout labels/results were used to select it.
5. Existing near-verbatim lexical evidence (LCS ≥.90) remains available even without semantic agreement. Reversed claims can still be textual reuse. This exception affects semantic-equivalence benchmark false positives; do not interpret retained lexical evidence as verified logical equivalence. Other baseline semantic evidence must pass verification in semantic modes.
6. Coverage merges accepted original character spans before counting tokens. Overlapping windows do not inflate it. The existing symmetric max-of-directions percentage convention remains unchanged. Models are released before the external/AI stages; the semantic score cache is bounded to 4096 pairs. Missing optional models cause an explicit error, not an undisclosed baseline fallback. Truncated cross-encoder pairs are rejected.

Default configuration and old algorithms are retained. Modes are `baseline`, `boundary`, `semantic_sts`, `semantic_nli`, `semantic_hybrid`. Existing defaults use baseline; proposed parameters are in [proposed.toml](../../improvements/configs/proposed.toml). Code changes are additive modules plus config/CLI wiring. No new Python dependency was installed; the existing torch/transformers stack loads both extra model weights.

## Evidence and schema

The per-document JSON and Markdown schema stays unchanged. `similarity_score` is still **MiniLM passage cosine**, including when a cross-encoder admits a low-cosine paraphrase. Verification scores/method/contradiction are recorded in the existing `note`; they do not masquerade as cosine. New verified windows keep `evidence_sentence_scores=[]`, since there is no claimed individual sentence alignment. Existing baseline sentence evidence retains its original scores. PDF page and audio timestamps are resolved from original absolute offsets, not guessed from reconstructed punctuation. Full source/query passages remain exact source slices.

| Original intended relationship | Detected in proposed CLI | Example passage cosine | Actual report location |
|---|---|---:|---|
| 01_human_speech.pdf ↔ 11_human_crossmodal.mp4 | True | 0.6897 | {'query_page': 2, 'matched_timestamp': '00:00:00 - 00:00:15'} |
| 02_alice_source.pdf ↔ 03_alice_copy.pdf | True | 1.0 | {'query_page': 1, 'matched_page': 1} |
| 06_policy_argument.docx ↔ 07_policy_rewording.docx | True | 0.6626 | {'query_page': None, 'matched_page': None} |

Canonical output is [improvements/runs/pipeline_semantic_hybrid](../../improvements/runs/pipeline_semantic_hybrid/summary/latest.json); all evidence and false-positive pairs appear in [comparison.json](../../improvements/comparison.json). All files have JSON/Markdown reports. The baseline rerun has **0 substantive report mismatches** against the earlier frozen baseline (timestamps excluded).

## Development and final-test separation

- The previous 13-file corpus is now development/regression data. Its original results remain unchanged. Two known misses are regression targets, not blind successes.
- Development comparison: 19 deliberately challenging AI-authored pairs, 9 positives and 10 same-topic/contradiction negatives. These are controlled synthetic examples, not human submissions. Whole-paragraph verification was investigated first and underperformed; passage windows were chosen using development results. STS alone accepted reversed acquisition d14; strict NLI sacrificed recall. Hybrid development had one miss (d16, weight-based waste billing), preserved without removal.
- Final heldout: 100 publicly human-labelled PAWS-Wiki validation rows 0–99, used as this project's final test, with zero exact normalized sentence overlap against the old 2,000-row PAWS benchmark and current development pairs. This is not an externally blinded private evaluation; article-level/pretraining overlap cannot be ruled out. Only this slice was scored after method freeze.
- An initial proposed slice (test rows 2000–2099) shared 76 sentence strings with the old benchmark. It remains preserved **unscored** and was replaced during data audit, before model results were viewed. This was a data-independence correction, not removal of failing model cases.
- Before final inference, two non-selection corrections were recorded: verified-window metadata now leaves sentence scores empty, and the dataset provenance license wording was corrected to the upstream unrestricted-use permission (HF metadata says “other”). Input texts, labels, models, acceptance rules and selected parameters did not change. Freeze histories are preserved. Final freeze timestamp: `2026-11-06T13:58:46.539594+00:00`; no algorithm/config edits followed heldout results.

PAWS labels measure **semantic equivalence**, not authorised reuse or plagiarism misconduct. Thus a surface-overlap detector can correctly identify reused words while being a semantic false positive. The tables below separate the 13-file reuse task from this adversarial semantic diagnostic. No risk-label accuracy or disciplinary ground truth is invented.

## Full pipeline comparison on 13 development files

| Mode | TP / FP / FN / TN | Precision | Recall | F1 | Full CLI s | Peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|
| baseline | 1 / 0 / 2 / 75 | 1.000 | 0.333 | 0.500 | 38.17 | 2026.8 |
| boundary | 2 / 0 / 1 / 75 | 1.000 | 0.667 | 0.800 | 41.58 | 2062.5 |
| semantic_nli | 3 / 0 / 0 / 75 | 1.000 | 1.000 | 1.000 | 75.71 | 2022.8 |
| semantic_hybrid | 3 / 0 / 0 / 75 | 1.000 | 1.000 | 1.000 | 96.37 | 1936.3 |

This includes real PDF/DOCX extraction, three MP4 ASR runs, local KB, unchanged AI detection and reports; web is disabled. It evaluates all 78 unordered file pairs, retaining failures. Same-source exact duplicates use the existing shortcut in every mode. These results demonstrate repair of regressions, not generalization.

## Development semantic relationship benchmark

| Mode | TP / FP / FN / TN | Precision | Recall | F1 | Case total s | Median / p95 s | Peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | 0 / 4 / 9 / 6 | 0.000 | 0.000 | 0.000 | 1.74 | 0.089 / 0.164 | 546 |
| boundary | 0 / 4 / 9 / 6 | 0.000 | 0.000 | 0.000 | 2.90 | 0.149 / 0.489 | 579 |
| semantic_sts | 8 / 1 / 1 / 9 | 0.889 | 0.889 | 0.889 | 12.79 | 0.342 / 6.836 | 844 |
| semantic_nli | 3 / 0 / 6 / 10 | 1.000 | 0.333 | 0.500 | 22.23 | 0.337 / 13.960 | 1263 |
| semantic_hybrid | 8 / 0 / 1 / 10 | 1.000 | 0.889 | 0.941 | 29.79 | 0.413 / 19.061 | 1510 |

## Heldout semantic relationship benchmark

| Mode | TP / FP / FN / TN | Precision | Recall | F1 | Case total s | Median / p95 s | Peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | 43 / 49 / 4 / 4 | 0.467 | 0.915 | 0.619 | 11.37 | 0.108 / 0.163 | 553 |
| boundary | 43 / 49 / 4 / 4 | 0.467 | 0.915 | 0.619 | 13.34 | 0.135 / 0.164 | 556 |
| semantic_sts | 41 / 51 / 6 / 2 | 0.446 | 0.872 | 0.590 | 18.64 | 0.188 / 0.239 | 806 |
| semantic_nli | 35 / 26 / 12 / 27 | 0.574 | 0.745 | 0.648 | 32.46 | 0.306 / 0.476 | 1025 |
| semantic_hybrid | 39 / 27 / 8 / 26 | 0.591 | 0.830 | 0.690 | 40.36 | 0.374 / 0.751 | 1034 |

Each row is a fresh CPU process with the same supplied dependencies and cached weights; methods run sequentially to avoid mutual CPU contention. Case totals/median/p95 include chunking, embedding, candidate generation, verification, evidence and coverage on in-memory paired documents. They exclude actual file extraction, ASR, AI, reporting and initial model loading. Peak RSS includes models plus runtime, not whole-machine memory. Separate model-load and whole-worker times are in every result JSON. This protocol should not be compared directly with full-CLI wall time. Model-download time is excluded. No 10k-submission latency/cost projection is offered.

## False positives and remaining misses

- **paws-validation:52** (human label negative): “He died of stomach cancer in Claremore , Oklahoma , New York City , where the Lynn Riggs Memorial is located on June 30 , 1954 .” ↔ “He died on June 30 , 1954 , of stomach cancer in New York City . Claremore , Oklahoma is home to the Lynn Riggs Memorial .”. Coverage 1.000; passage cosine 0.9365; note `hybrid verified candidate: score=0.9821, contradiction=0.0011; bidirectional_entailment=0.9969; reported similarity is MiniLM cosine; human review required`. Full text and evidence remain in the JSON.
- **paws-validation:78** (human label negative): “Their music is considered by many as an industrial metal with rap metal and alternative metal influences , according to previous interviews they consider themselves '' murder - roc” ↔ “Their music is considered by many as alternative metal with rap metal and industrial metal influences . According to previous interviews , they consider themselves `` murder rock '”. Coverage 1.000; passage cosine 0.9616; note `hybrid verified candidate: score=0.9911, contradiction=0.0303; bidirectional_entailment=0.9405; reported similarity is MiniLM cosine; human review required`. Full text and evidence remain in the JSON.

The hybrid produces **27 semantic false positives**, including **2 newly introduced** relative to baseline; **24 baseline false positives are removed**. The examples above prioritise newly introduced errors.

Every heldout error, source label, score and full evidence remains in `improvements/runs/heldout_*_final.json`. No failed case was removed, relabelled or used to adjust the method. Some semantic errors stem from retained lexical evidence, some from verifier overconfidence/partial-window overlap; their presence is a reason to keep this mode opt-in and require human review. STS development d14 specifically reverses who acquired whom while returning score .9796; the NLI contradiction signal was .9989 and the hybrid rejects it.

The selected verifier is not a logical proof system. It can infer entailment for unequal detail, veto a valid paraphrase as contradictory, or miss paraphrases outside retrieval top-k. Conservative limits avoid unbounded all-pairs cross-encoding but cap recall. The PDF first-line heuristic can mistake an unpunctuated short body line for a heading. ASR joins use segment-level intervals, not new word-accurate annotation; no confidence restoration or larger ASR model is added. Existing automatic three-document boilerplate suppression and quadratic legacy chunk aggregation/dense summary behavior remain baseline limitations outside this repair's scope.

## Tests, reproducibility and preservation

The 43-test suite passed (34 original + 9 new, one upstream torch deprecation warning). New tests cover the two confirmed mechanisms with real cached models, exact page/timestamp slices, premature ASR punctuation, long pauses/size guards, unchanged cosine floor, union coverage, contradiction/truncation rejection and invalid parameter bounds. Original sample scenarios still pass in baseline mode. Full CLI checks separately verify actual proposed-mode schema and evidence.

[improvements/README.md](../../improvements/README.md) contains exact commands, file layout and model revisions. Models and data have hashes and pinned revisions; benchmark workers refuse to overwrite output files and refuse heldout execution if the frozen method changed. Prior result preservation checks report **0 changed files**, and final method checks **0 changes**. Earlier EVALUATION.md, all four validation reports, validation outputs and saved benchmarks remain byte-for-byte unchanged. A source archive retains the original implementation at commit e15af35.

Model references: [STS model card](https://huggingface.co/cross-encoder/stsb-distilroberta-base), [NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small). The former is trained for textual similarity; the latter on SNLI/MultiNLI for entailment/neutral/contradiction. Both are Apache 2.0. Dataset acknowledgement: Google LLC, Yuan Zhang, Jason Baldridge and Luheng He, [PAWS (2019)](https://github.com/google-research-datasets/paws), upstream permission preserved in `improvements/data/PAWS_LICENSE.txt`. Newly authored development fixtures are synthetic and openly reusable under the project MIT license.
