# Independent validation artifacts

Validated original commit `e15af35db37afafa9c6957c4e515c769dd08c38a`. Start with the four reports at repository root. No original code, configuration, report or saved benchmark was edited. The main 13-file test corpus and its labels were constructed before the evaluated runs; later component diagnostics are explicitly exploratory.

## Contents

- `freeze.json`, `config.frozen.toml`, `dependencies.txt`, `ffmpeg_version.txt`: environment, model revisions and original-file hashes.
- `manifest.json`: expected relationships, source offsets, authorship and fixture SHA-256, timestamped before engine inference.
- `corpus/`: 5 PDF, 5 DOCX, 3 speech MP4; exact-copy, paraphrase and spoken-copy positives, five common prompts, remaining negative pairs.
- `edge_corpus/`: valid control, empty PDF, corrupt PDF, no-audio MP4.
- `probe_corpus/`: assertion and controlled negation. Semantic negative, textual-reuse positive.
- `sources/`: full public-domain source files, fetched license notices, download manifest, preparation-only source alignment. The downloaded Time Machine text is unused. First-210-second WAV is a derived preparation intermediate.
- `runs/main`, `runs/edge`, `runs/semantic`: unchanged CLI outputs, stdout, stderr, exact commands, exit status and timings.
- `runs/pytest.log`, `runs/pytest_tmp/`: original automated tests and newly generated test reports. Original reports outside validation remain untouched.
- `runs/retrieval_reproduction/`: original 400-document benchmark rerun to a separate destination.
- `results.json`: all 78 main pair outcomes, full overlap evidence, metrics, ASR references/hypotheses, per-file statuses/risks/timings and schema checks.
- `extraction.json`, `diagnostics.json`: extracted text, pages, ASR times, sentence matrices, three-copy component probe and external controls.
- `split_audit.json`, `ai_metrics_recomputed.json`, `original_metrics_recomputed.json`: benchmark/split reconstruction and arithmetic verification. HC3 inference was not rerun; its preserved component outputs were rescored.
- `qa/`: Poppler/LibreOffice render checks; not used as new evaluated submissions.
- `preservation_check.json`: original and fixture checksum comparison.

## Reproduce pipeline measurements

Use the supplied `.venv` and already cached models. Required runtime packages are those in `dependencies.txt`; FFmpeg/ffprobe must be on PATH. The four scored model revisions are:

| Model | Frozen snapshot revision |
|---|---|
| all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| faster-whisper-tiny | `d90ca5fe260221311c53c58e660288d3deb8d356` |
| openai-community/gpt2 | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| wasitaigeneratedcom/ai-text-detector-small | `f1795c86806e6838d4afa33d0b1427f8430c9615` |

Whisper base `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66` was used only to identify source cut locations, not by the evaluated CLI. Whisper small was already cached but unused. Model-file hashes appear in freeze.json. Offline runs use the cached `main` refs at these revisions; verify them before reproduction on a different host. Original setup does not enforce these snapshot hashes itself.

From repository root, choose new output directories to preserve recorded evidence:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python run_integrity_check.py \
  --input_dir validation/corpus --output_dir validation/reproduction/main --skip-web
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python run_integrity_check.py \
  --input_dir validation/edge_corpus --output_dir validation/reproduction/edge --skip-web
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python run_integrity_check.py \
  --input_dir validation/probe_corpus --output_dir validation/reproduction/semantic --skip-web
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests -q \
  --basetemp=validation/reproduction/pytest_tmp
```

Do not add `--eval` with its default destination: that updates the original EVALUATION.md. Runtime, logger timestamps and file generation timestamps can vary; compare substantive signals and fixture hashes. The logger writes its normal new runtime log/current_run.log under logs. No 10,000-document pipeline was executed.

## Reproduce scoring and audit

```bash
PYTHONPATH=. .venv/bin/python validation/scripts/score_independent.py
.venv/bin/python validation/scripts/audit_splits.py
.venv/bin/python validation/scripts/recompute_ai_metrics.py
HF_HUB_OFFLINE=1 .venv/bin/python validation/scripts/diagnose.py
```

These score/check the preserved canonical run and rewrite only derived validation JSON/logs. `score_independent.py` fails if any frozen source/report/config or fixture hash changed. Its source contains a straightforward all-pairs scorer and AUC calculation. Review `results.json` for every expected/detected pair; absent output is retained as a false negative when its frozen label is positive.

To rerun the original retrieval benchmark safely, import `benchmarks/bench_retrieval.py` using the wrapper below, set its `RESULTS` variable to a new validation destination, and call `main()`; do not run it unredirected because that overwrites original benchmark results. Exact runnable wrapper: `scripts/reproduce_retrieval.py`.

`run_frozen.py` records the initial three CLI runs and deliberately refuses to overwrite existing run directories. `fetch_sources.py` and `build_corpus.py` document construction; they are not required to rerun the pipeline. Rebuilding fixtures may change DOCX ZIP timestamps and container hashes; do so in a fresh repository copy and retain the original manifest, rather than overwriting this frozen corpus. The plain text content and FFmpeg commands are fully specified. No random seed or model-output-dependent fixture selection is used.

## Licenses and authorship

- Recording: John F Kennedy inaugural address, US federal public domain per preserved Commons license page. MP4s use this human audio, not TTS; colour backgrounds are generated.
- Text: Kennedy speech and Lewis Carroll's Alice from Project Gutenberg, with complete source/license notices retained. No modern translation. Published human text is excerpted into newly rendered PDF containers.
- All seven library/policy documents, shared institutional prompt, cover text and semantic probes: authored by this AI assistant for this evaluation. These are openly reusable evaluation fixtures under the repository's MIT license; they are not human ground truth. Machine-generated passages must not be relabeled human in later tests.
- No private or proprietary student content was collected. Authentic in-domain human library essays and multiple independent human speakers were not obtained, limiting real-world validation.

The documents skill's bundled renderer could not import pdf2image. Installed LibreOffice converted DOCX to PDF, Poppler rendered all 11 main document pages, and each page image was inspected for clipping/missing content. Its marker helper/workspace-specific dependency loader was not available; existing environment authoring libraries were used. No dependencies were installed or production code changed to work around those host limitations.
