# Reproduce the matching repair

Read [MATCHING_IMPROVEMENTS.md](../MATCHING_IMPROVEMENTS.md) for measurements, error cases and limitations. All previous evaluation artifacts are preserved. Baseline is still the default; the new mode is opt-in.

## Run the proposed pipeline

From repository root, use the existing environment. To cache the two pinned extra models (and the original models):

```bash
.venv/bin/python run_integrity_check.py --input_dir sample_submissions \
  --output_dir improvements/models_only --models-only --matching-mode semantic_hybrid
```

Then run with a new output directory:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python run_integrity_check.py \
  --input_dir validation/corpus --output_dir improvements/reproduction/proposed \
  --config improvements/configs/proposed.toml --skip-web
```

For the original algorithm use `--matching-mode baseline`; for just the boundary fix use `--matching-mode boundary`; the separate verifier comparators are `semantic_sts` and `semantic_nli`. No global MiniLM acceptance threshold changes. Do not use `--eval` with its default output path, which would update the historical EVALUATION.md.

## Reproduce comparison workers

Workers refuse to overwrite existing result files. Choose fresh output paths:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python improvements/scripts/benchmark_matching.py \
  --split development --mode semantic_hybrid --out improvements/reproduction/dev_hybrid.json
HF_HUB_OFFLINE=1 .venv/bin/python improvements/scripts/benchmark_matching.py \
  --split regression --mode boundary --out improvements/reproduction/regression_boundary.json
HF_HUB_OFFLINE=1 .venv/bin/python improvements/scripts/benchmark_matching.py \
  --split heldout --mode semantic_hybrid --out improvements/reproduction/heldout_hybrid.json
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests -q \
  --basetemp=improvements/reproduction/pytest_tmp
```

Repeat the worker command for each of the five modes. `run_comparison.py` and `execute_final.py` document the original sequential orchestration and deliberately refuse an overwrite of the recorded paths. `run_pipeline_comparison.py` records the four actual CLI comparisons, with exact commands, stdout/stderr, exit codes and timings. The STS comparator is measured in the relationship benchmark; full CLI runs are baseline, boundary, NLI and selected hybrid.

`benchmark_matching.py --split heldout` verifies production/config/script/data hashes against `final_method_freeze.json` before inference. If the code later changes, preserve the current outputs and treat this heldout as a regression dataset; create a new independent final set. Do not overwrite the freeze to call a newly tuned result blind. A copied fresh checkout at the frozen implementation is the intended reproduction target.

## Artifacts

- `prior_results_freeze.json`: SHA-256 of all earlier evaluation artifacts and reports.
- `baseline_source.tar`: original committed source/config/runner/requirements. Baseline mode also remains directly runnable; original algorithms were not replaced.
- `data/development.json`: 19 controlled AI-authored positive/same-topic/contradiction pairs.
- `data/heldout_disjoint.json`: final human-labelled PAWS-Wiki validation slice 0–99; no exact normalized sentence overlap with development/old benchmark.
- `data/heldout.json`: initial overlapping candidate, preserved unscored; never used in final results.
- `data/freeze.json`, `final_holdout_freeze.json`, `overlap_audit.json`: preparation and audit history. `data/ATTRIBUTION.md`/`PAWS_LICENSE.txt` record correct source permission; the initial unscored candidate's CC BY wording was mistaken.
- `models.json`: pinned new-model revisions/cache paths. NLI-small and STS-distilRoBERTa are used; an NLI-base candidate was downloaded during exploration but is not used in the selected method or timing tables.
- `final_method_freeze.json`: final code/config/data/model identities. v1/v2 preserve pre-inference metadata-correction history, not alternative tuned methods.
- `runs/development_*_final.json`, `regression_*_final.json`, `heldout_*_final.json`: all labels, scores, evidence, false positives/negatives and measured CPU time/RSS. Earlier direct paragraph-only exploratory outputs are also retained and not conflated with window retrieval results.
- `runs/pipeline_*/`: full 13-file CLI outputs with original schema and page/timestamp evidence.
- `comparison.json`: full-CLI confusion metrics, expected-positive evidence, all false positives, schema checks, baseline comparison and preservation result.

The old validation scripts intentionally freeze production code at the earlier commit; they will detect the now-authorized source changes. Their outputs were not rewritten. New comparison tools check prior *result* preservation independently of current source changes.

## Pinned model identities

| Model | Revision |
|---|---|
| cross-encoder/stsb-distilroberta-base | `6b71347df6e2b34246b53e06d6bce70ef67de368` |
| cross-encoder/nli-deberta-v3-small | `fa2804872c3b4bd748f38c0185cc85775361e735` |

Runtime dependencies are unchanged from requirements.txt and the earlier validation/dependencies.txt. New optional model weights must be downloaded before offline mode. Missing models cause explicit failure in semantic modes; they do not affect baseline. The model card scores are not calibrated misconduct probabilities. All baseline and new evidence still require human review.
