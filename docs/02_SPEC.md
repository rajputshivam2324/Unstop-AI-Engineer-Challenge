# 02_SPEC.md — Multimodal Submission Integrity & Plagiarism Detection Engine
**Audience:** Senior Engineer (implements exactly this, no clarifying questions).
**Inputs honored:** `/tmp/opencode/assignment.txt` (source of truth) + `docs/01_research_notes.md` (design directives D1–D12, verified facts).
**Verified environment baked in:** Linux x86_64, 12 cores, 7.1 GB RAM, CPU-only, Python 3.14.8 (`/usr/bin/python3`) + Python 3.11, pip 26.0.1, ffmpeg 8.1.3 at `/usr/bin/ffmpeg`, espeak-ng at `/usr/bin/espeak-ng`, internet (PyPI + HF reachable), ample disk.

---

## 1. Overview, Goals & Non-Goals

### 1.1 One-paragraph summary
A CLI Python pipeline that ingests a corpus of PDF/DOCX/MP4 submissions, normalizes all modalities to text (faster-whisper transcribes MP4 audio with per-segment timestamps), embeds chunks with `sentence-transformers/all-MiniLM-L6-v2` (384-d, L2-normalized), indexes them in a FAISS `IndexFlatIP` (exact cosine), retrieves suspicious peer pairs, produces sentence-level explainable alignments with page/timestamp provenance, discounts shared boilerplate via a fingerprint TemplateRegistry, computes three integrity signals (peer %, external-KB/web %, AI-generated %) with an ensemble AI detector plus an explicit "uncertain" band, and writes one JSON + one Markdown audit report per document plus an aggregate summary. A synthetic 14-file dataset (5 PDF / 6 DOCX / 3 MP4) with a ground-truth manifest and an EVALUATION.md generator (precision/recall/F1 + confusion matrix) prove correctness.

### 1.2 Goals
| ID | Goal | Evidence (assignment / research) |
|----|------|-------------------------------|
| G1 | Graceful ingestion of PDF, DOCX, MP4 with page/timestamp provenance on every chunk | Task 1; D1, D2 |
| G2 | Exact cosine peer similarity with self-match and dedup exclusion | Task 2; D4, D7 |
| G3 | Boilerplate/common-source discounting so Scenario-4 docs score LOW | Scenario 4; D6 |
| G4 | Cross-modal matching (MP4 speech ↔ PDF page) with both location types | Scenario 1; D2 |
| G5 | Paraphrase detection via semantic embeddings with fixed, model-calibrated thresholds | Scenario 2; D5 |
| G6 | 2-component ensemble AI detection (HF classifier 0.6 + GPT-2 perplexity/burstiness 0.4) with uncertainty band, length gate, and detector-source marking | Task 3; §2.1 research; D9 |
| G7 | Two-tier external engine: always-on local KB + optional free web API with mock fallback | Task 3; D8 |
| G8 | Reviewer-verifiable reports: exact passages, file, page, timestamp, chunk cosine, plain-English explanation; <30 s comprehension | Task 4; D10 |
| G9 | Self-evaluation vs ground truth manifest (P/R/F1, confusion matrix, FPR on originals, 2 failure studies) | §8; D11 |
| G10 | 10k-doc scaling brief in README (queue, index choice, cost/latency math) | Task 5; D12 |
| G11 | Zero crashes on any input file, deterministic outputs, ≤2 GB RAM on the 14-doc corpus | "Malformed gracefully"; NFRs |

### 1.3 Non-Goals (mirror assignment "Out of Scope" verbatim in spirit)
- **No web UI / frontend** — CLI + JSON/Markdown only.
- **No paid plagiarism APIs** (Turnitin, Copyleaks, etc.) — local models and free/no-key endpoints only.
- **No long-video transcription** — sample MP4s are 10–30 s; the pipeline never targets hour-scale ASR in this prototype.
- Additionally (inferred this-scope limits, documented as assumptions A-series): no OCR of scanned-image PDFs beyond reporting `empty`; no cross-language plagiarism; no agent/network installs at runtime beyond HF model downloads; no multi-node distribution.

---

## BP. Build ordering & de-scope rules

**Strict build order (build order ≠ runtime pipeline order §6; e.g. the template registry runs early at *runtime* (S3) but is *built* late, step (vii)):**
1. (i) repo skeleton, venv, `requirements.txt` install, `scripts/check_env.py` green.
2. (ii) ingestion: the 3 extractors + status contract + every E-row warning (§4.1, §7).
3. (iii) `text_processing`: normalization, sentences, chunks with provenance (§4.2).
4. (iv) embeddings + FAISS + peer pair retrieval with self-match/dedup exclusion (§4.3–4.4).
5. (v) alignment + report JSON+MD writers (§4.5, §4.9).
6. (vi) AI ensemble + external KB (§4.6–4.7).
7. (vii) template registry (§4.5).
8. (viii) dataset generator (§4.10).
9. (ix) tests + evaluation (§10, §4.11).

**Gate:** stages (i)–(v) must run end-to-end on the sample corpus (with the template registry as a no-op — Scenario 4's discount lands in step (vii)) before any work starts on (vi)–(viii).

**De-scope rules — if any stage (vi)+ is still failing after ~2 fix attempts, apply the matching rule; do not else-escalate:**
| Rule | Decision |
|---|---|
| R1 Template registry | Primary stays simhash64 (§4.5). Sanctioned fallback: exact-normalized-text **8-word shingle** clustering — hash every sliding 8-word shingle, cluster documents sharing ≥ 3 shingle hashes; same is_template semantics, no simhash code. |
| R2 Corpus-relative σ calibration | **REMOVED (unconditional).** Thresholds are fixed constants from config only. README *Future work* must include exactly one sentence noting corpus-relative σ calibration as a possible upgrade. |
| R3 AI ensemble reduction | Exactly **2** top-level components, no new deps: `p_final = 0.6·p_cls + 0.4·p_text`, where `p_text = 0.75·p_perp + 0.25·p_burst` (both sub-terms come from the already-pinned gpt2 run). |
| R4 Dependency freeze | No new models or packages beyond the §9 pins, ever, during recovery attempts. |
| R5 Test trimming | Edge-case tests reduced to: one integration test per scenario (S1–S5) + corrupted-file unit tests. Delete redundant parameterized variants; do not re-add them. |
| R6 Assumptions cap | The assumptions log stays at 12 load-bearing entries (A1–A12) + one short "further notes" bullet list; recovery work may not grow it. |
| R7 Failure studies | EVALUATION.md contains **exactly** the 2 required case studies; never more scaffolds. |

**Deadline rule (explicit):** README and EVALUATION are *deliverables of the core build*, not polish items — deadline slippage is never justified by polish, extra recovery attempts, or scope growth. A stage that exhausts its de-scope rule ships under the rule's fallback instead.

---

## 2. Repository Layout

Project root: `/home/shivam/assignment/unstopchallenge`. Package name frozen: **`integrity_engine`**.

```
/home/shivam/assignment/unstopchallenge/
├── run_integrity_check.py           # thin CLI entry: from integrity_engine.cli import main; main()
├── requirements.txt                 # exact pins (§9)
├── README.md                        # setup, methodology, LLM-provenance disclosure (§4.10), assumptions, future work (σ-calibration), 10k scaling brief
├── EVALUATION.md                    # generated by --eval, then narrative hand-added
├── LICENSE                          # MIT
├── .gitignore                       # .venv/, reports/, cache/, __pycache__, *.wav intermediates
├── config/
│   └── default.toml                 # every threshold (§4.13)
├── integrity_engine/
│   ├── __init__.py                  # exports run_pipeline
│   ├── cli.py                       # argument parsing, logging init, orchestration
│   ├── config.py                    # dataclass Config + TOML load/merge
│   ├── common.py                    # get_trimmed_logger, Timer, set_seed, list_input_files
│   ├── types.py                     # all TypedDict/dataclasses (§5)
│   ├── ingestion/
│   │   ├── __init__.py              # registry dict + extract_any()
│   │   ├── base.py                  # Extractor protocol + ProcessedDocument factory helpers
│   │   ├── pdf.py                   # pypdf extractor
│   │   ├── docx.py                  # python-docx extractor
│   │   └── mp4.py                   # ffmpeg + faster-whisper extractor
│   ├── text_processing.py           # normalize, sentences, chunks
│   ├── embeddings.py                # Embedder wrapper
│   ├── indexing.py                  # FAISS build/search, doc-pair aggregation
│   ├── alignment.py                 # TemplateRegistry + MatchedPassage builder
│   ├── ai_detection.py              # AIDetector ensemble
│   ├── external_check.py            # local KB + optional DDG tier
│   ├── scoring.py                   # risk combination, explanations
│   ├── reporting.py                 # JSON + Markdown writers
│   ├── dataset_generator.py         # synthetic corpus + manifest
│   └── evaluation.py                # metrics + EVALUATION.md writer
├── docs/
│   ├── 01_research_notes.md         # researcher findings (input to this spec); 02_SPEC.md = this file; 03_llm_generated_sample.md = LLM-provenance record for doc 06 (§4.10)
├── scripts/
│   ├── setup_env.sh                 # venv + CPU torch + requirements install (§9); check_env.py = version/import sanity printer
├── reference_kb/
│   └── kb_texts.py                  # KB source strings — 2 groups: encyclopedic + case-methodology (see §4.7)
├── sample_submissions/              # generated once by dataset_generator (via CLI flag), committed; 14 files
│   ├── 01_retail_turnaround_report.pdf
│   ├── 02_retail_turnaround_copy.docx
│   ├── 03_paraphrased_strategy.docx
│   ├── 04_original_strategy.docx
│   ├── 05_original_market_pdf.pdf
│   ├── 06_llm_generated_summary.pdf
│   ├── 07_pitch_crossmodal.pdf
│   ├── 08_pitch_crossmodal_video.mp4
│   ├── 09_third_pitch_video.mp4
│   ├── 10_corrupted_empty.pdf       # zero-byte
│   ├── 11_silent_video.mp4
│   ├── 12_case_template_A.docx
│   ├── 13_case_template_B.docx
│   ├── 14_case_template_C.docx
│   └── dataset_manifest.json
├── reports/                         # output dir (created at runtime; gitignored)
├── logs/                            # per-run DEBUG logs (created at runtime; gitignored)
├── cache/                           # HF model cache (HF_HOME override; gitignored)
└── tests/                           # pytest (§10)
```
Naming rule: dataset files carry `NN_` numeric prefixes purely for stable sort order = deterministic document_id ordering (document_id = full filename incl. extension, matching the assignment's `"submission_042.pdf"` pattern).

---

## 3. CLI Contract

`run_integrity_check.py` — argparse, `--help` text as below. **Roles:** `--eval PATH` consumes reports to produce `EVALUATION.md`; no dataset use.
```
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports --eval ./sample_submissions/dataset_manifest.json
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports --models-only
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports --generate-dataset [--seed 42]
python run_integrity_check.py --config /path/to/custom.toml --input_dir ... --output_dir ...
python run_integrity_check.py --input_dir ... --output_dir ... --skip-web
```

### Flags
| Flag | Type/default | Behavior |
|---|---|---|
| `--input_dir` | str, **required** | Non-recursive scan of this dir. Entries with extensions {`.pdf`,`.docx`,`.mp4`}, case-insensitive, sorted by filename for determinism. Subdirectories ignored. If missing → exit 1. If no supported files → exit 1 with `ERROR: no supported files (.pdf/.docx/.mp4) found in <dir>`. |
| `--output_dir` | str, **required** | Created if absent. Contains `per_doc/*.json`, `per_doc/*.md`, `summary/report_{ts}.json` where ts = UTC `YYYYMMDD_HHMMSS`. Existing files with same name are overwritten; other files untouched. |
| `--config` | str, default `config/default.toml` relative to repo root | TOML threshold overrides. Unknown TOML keys are logged as warnings and ignored (forward compat). Parse error → exit 1. |
| `--models-only` | flag | Download/precache HF models (embedding model, GPT-2, Whisper tiny, AI classifier) into `cache/hf/` (set env `HF_HOME` to that dir before imports) and print their sizes, then exit 0. No ingestion. |
| `--generate-dataset` | flag, consumes `--seed` | Regenerate the 14 canonical files + `dataset_manifest.json` inside `--input_dir` (that flag is the target dir), then exit 0. Deterministic for a given seed. Overwrites only the 14 canonical filenames + manifest; any other files in the dir are left untouched. A confirmation prompt is printed if the dir already contains a `dataset_manifest.json` with a **different** `dataset_version` (non-interactive runs then abort with exit 1). |
| `--skip-web` | flag | Disable the optional Tier-2 DDG web calls entirely (Tier-1 local KB always runs; this is also what happens automatically offline). Default: Tier-2 enabled but auto-skips on any network/parse error (never raises). |
| `--log-level` | choice{DEBUG,INFO,WARNING,ERROR}, default `INFO` | Console verbosity. File log always at DEBUG. |
| `--seed` | int, default `42` | Used by `--generate-dataset` and for sampling randomness; the engine pipeline itself is deterministic without RNG (further note FN2). |

### Exit codes
| Code | Meaning |
|---|---|
| 0 | Run completed; ≥1 report written. Individual degraded files are warnings, not errors (they are by-design entries in the sample corpus; Scenario 3 expects the run to succeed while flagging them). |
| 1 | Fatal: `--input_dir` missing/unreadable, no supported files found, unparseable `--config`, or `--output_dir` not creatable. **Per-file degraded ingestion (error/empty/silent_video) is never fatal** — Scenario 3 requires graceful continuation. |
| 2 | Argparse usage error (argparse default). |

### stdout behavior
- Human-readable log lines on **stderr** via `logging` at `--log-level`; a per-run DEBUG copy at `logs/run_{ts}.log`, plus a stable path `logs/current_run.log` (overwritten every run) for tooling.
- On **stdout**, a final machine-readable one-line JSON summary of the aggregate report: `{"summary_path": "...", "documents_evaluated": 14, "warnings": 2, "elapsed_s": 214.3}` — plus nothing else on stdout (keeps `jq` piping clean; all narrative goes to stderr/reports).
- `--models-only` prints table on stderr and writes nothing to stdout.

---

## 4. Module Architecture

### 4.0 Shared conventions
- All modules import types from `integrity_engine/types.py`; no circular imports.
- Logging: `logger = logging.getLogger("integrity_engine.<module>")`; never `print` in library code.
- No module ever raises past `ingestion` for *file-content* problems; config/IO-of-output problems may raise (CLI catches → exit 1).
- Docstring contract: every module file starts with a `<MODULE PURPOSE>`, `<PUBLIC API>`, `<DEPENDS ON>` block (rubric: Code Quality & Documentation).

### 4.1 `ingestion/` — modality → `ProcessedDocument`

**Contract:** never raise past `extract_any`; boundaries: `status ∈ {ok, empty, error, silent_video}`; every emitted page/segment keeps provenance; `warnings: list[str]` fed to report explanations.

`base.py`
```python
class Extractor(Protocol):
    def extract(self, path: Path, config: Config) -> ProcessedDocument: ...
def finish_document(path: Path, pages: list[Page], warnings: list[str],
                    modality: str, config: Config) -> ProcessedDocument
# shared tail for all extractors: computes document_id (= file name), sha256, size_bytes,
# metadata, and decides status: ok / empty (total normalized chars < 15)
# (error/silent_video are set by the module-specific caller BEFORE calling finish_document)
```
`pdf.py`
```python
def extract(path: Path, config: Config) -> ProcessedDocument
```
Implementation notes:
- `size_bytes = stat.st_size`; zero-byte → status=`error`, warnings=[f"file is empty (0 bytes); nothing to analyze"].
- `pypdf.PdfReader` (pypdf==5.x) with per-page try/except; `page.extract_text()`; if `""`, record `page` with `text=""` and append warning `page {i}: no extractable text (likely scanned image)`. Timeouts: none per-page (CPU bound is fine); per-file guard `max_pages=500`.
- Header/footer strip: compute the multiset of first- and last-nonempty-line strings across pages; any line whose normalized form occurs on ≥50% of pages (and page_count ≥ 2) is removed from every page and counted in `document_warnings` as `removed repeated header/footer line: "<trunc 40 chars>"`; after strip, if total chars < 15 → status=`empty`, warning `document contains no usable text`.

`docx.py`
```python
def extract(path: Path, config: Config) -> ProcessedDocument
```
- `docx.Document(path)`; `zipfile.BadZipFile`/`PackageNotFoundError` → status=`error`, warning `not a valid .docx container (may be legacy .doc or corrupted)`.
- Iterate `document.paragraphs` in order; a run of ≥4 consecutive `Heading`-styled short (<60 chars) paragraphs is treated as outline, not stripped (assignment strips only headers/footers); empty paragraphs collapse (no double-newlines).
- Page numbers do not exist in docx — `page=None`; `char spans` still populated. Tables: flattened row-wise, cells joined with `" | "`, appended after paragraphs (assumption A3).
- Status, empty-threshold, warnings identical to pdf.py logic (shared helper in `base.py: finish_document(path, pages, warnings)`).

`mp4.py`
```python
def extract(path: Path, config: Config) -> ProcessedDocument
```
- `ffprobe -v error -select_streams a -show_entries stream=codec_name,sample_rate,channels,duration -of json <path>`; absent audio stream → status=`silent_video`, warning `no audio stream in MP4`. `ffprobe` failure → status=`error` warning `unreadable media container`.
- Extract: `ffmpeg -y -i <mp4> -vn -ac 1 -ar 16000 -c:a pcm_s16le <tmp>.wav` (16 k mono, required by whisper). Subprocess timeout 120 s.
- Also compute loudness guard: `ffmpeg -af volumedetect` RMS mean_level dB; if < −45 dB → treat as silent *before* ASR (fast path): status=`silent_video`, warning `audio track is effectively silent (mean volume < -45 dB)`.
- ASR: `WhisperModel(model_size_or_qty="tiny", device="cpu", compute_type="int8", download_root=<hf cache>)`; `transcribe(wav, language="en", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))`.
- Segments: for each `segment` (`start`, `end`, `text` float seconds); `language` from `info.language`; if != "en" → warning `non-english audio detected ({lang}); transcription may be degraded` but continue (assumption A4).
- **Chunk provenance contract (A5):** this module emits one `Page` per ASR segment: `Page(page=None, text=segment.text, char_start=offset0, char_end=offset1)`, where `offset0/offset1` are the segment's character positions inside the concatenated transcript (computed cumulatively as segments are joined with a single space). Chunking itself happens in `text_processing.build_chunks`, which for `modality == "mp4"` groups consecutive *segments* (not sentences) and derives each chunk's `timestamp_start = segments[first].start`, `timestamp_end = segments[last].end` from the segment list held alongside. faster-whisper segment times are already in **seconds** (`segment.start` / `segment.end` floats) — store seconds, format `hh:mm:ss` only at report time.
- If VAD produces zero segments → status=`silent_video`, warning `speech-not-detected by VAD; possibly silent or music-only audio`. Status if transcript chars < 15 → `empty`, warning `transcript too short to analyze`. Cleanup tmp wav in `finally`.

`registry.py` (`ingestion/__init__.py` re-exports)
```python
EXTENSION_TO_EXTRACTOR: dict[str, Callable[[Path, Config], ProcessedDocument]] = {
    ".pdf": pdf.extract, ".docx": docx.extract, ".mp4": mp4.extract,
}
def extract_any(path: Path, config: Config) -> ProcessedDocument | None   # None if unknown extension
```
Unknown extensions (e.g. `.txt`, `.zip`, `.doc`) are *skipped* with log `skipping {name}: unsupported extension`.

### 4.2 `text_processing.py`

```python
UNICODE_MAP: dict[str, str]           # built once: NFKC + {"\ufb00"→"ff","\ufb01"→"fi","\ufb02"→"fl","\u2018"/"\u2019"→"'","\u201c"/"\u201d"→'"', "\u2013"/"\u2014"→"-", "\u00a0"→" ", "\t"→" "}
def normalize_text(raw: str) -> str
def de_hyphenate_linebreaks(text: str) -> str          # r"(\w+)-\n\s*(\w+)" → r"\1\2"
def collapse_ws(text: str) -> str                      # runs of whitespace → single " "; strip leading/trailing per paragraph
def split_sentences(text: str) -> list[Sentence]        # Sentence: (text, char_start, char_end)
def sentence_char_spans_by_pages(pages: list[Page]) -> list[Sentence]  # pages joined with "\n\n"; page attribution per sentence
def build_chunks(doc: ProcessedDocument, cfg: ChunkingConfig) -> list[Chunk]
     # text/pdf/docx path: sentence accumulate; MP4 path: segment accumulate (as decided)
def token_count(text: str) -> int        # regex \w+ tokens, digits included; documented as the ONLY token definition in the project
def doc_clean_warnings(doc: ProcessedDocument) -> list[str]   # e.g. "> 20,000 chars truncated for AI analysis"
```
`split_sentences` spec (deterministic, no model): find boundary candidates with regex `([.!?])\s+(?=[A-Z0-9"'])`; **reject** a candidate when the punctuation is part of an abbreviation — i.e., the word immediately ending at the punctuation matches the whitelist `(Mr|Mrs|Ms|Dr|Prof|vs|etc|e\.g|i\.e|Fig|No|approx|Inc|Ltd|Jr|Sr|St)\.$` — or when the sentence produced is < 20 chars and the previous one ends in a whitelisted abbreviation (merge back). Sentences > 800 chars are split at the nearest `;` / `,` midpoint (log debug).
`build_chunks` algorithm (pdf/docx): walk sentences in page order; append sentence if `tokens_so_far + len(sent) <= max_chunk_tokens(200)` else close chunk and start new one carrying the last `overlap_sentences=2` sentences, but only if those 2 sentences account for ≤ 25% of the new chunk's tokens (otherwise start clean; silent, no warning). A single sentence > 200 tokens: split at word boundaries into ≤200-token sub-chunks (rare; deterministic). Always enforce `token_count ≥ min_chunk_tokens(40)` for a chunk to exist; a leftover tail below 40 tokens is merged into the previous chunk (exceeding 200 only in that merge case). Comment style: `# CHUNK: accumulate sentences until the 200-token budget (§4.2)` — every such rule constant traces to `config`.

MP4 chunk rule: accumulate *back-to-back* ASR segments; close when token count ≥ 120 (because ASR lacks sentence reliability; target mid-band) or when `t_end - t_start ≥ 45 s` (config); overlap = replay previous 1 segment. If transcript < 15 total chars → treated by ingestion as empty earlier.

### 4.3 `embeddings.py`

```python
class Embedder:
    def __init__(self, model_name: str, cache_dir: Path, device: str = "cpu",
                 batch_size: int = 32, normalize: bool = True) -> None
    @property
    def dim(self) -> int                 # 384
    @property
    def max_seq_tokens(self) -> int      # introspect model; default 256 (attribute handle, D3)
    def encode_corpus(self, texts: list[str]) -> np.ndarray     # (N,384) float32, L2 rows (zero-norm rows left zero; numpy no NaN)
    def encode_query(self, text: str) -> np.ndarray             # (1,384) same normalization
    # introspection:
    def warmup(self) -> None             # load model at init time and time one dummy encode; log ms
```
Implementation: `SentenceTransformer(model_name, device=device, cache_folder=cache_dir)`; `encode(..., convert_to_numpy=True, normalize_embeddings=normalize, show_progress_bar=False, batch_size=batch_size)`. `max_seq_tokens` property: read `model.max_seq_length`; fall back to `256` if the attribute is missing. `encode_corpus` returns contiguous `float32`; guard: if `len(texts) == 0` return `np.zeros((0, self.dim), dtype=np.float32)`; after encoding, rows with zero norm are left as zero vectors (never renormalized — avoids divide-by-zero).
Research fact applied: cosine on unrelated MiniLM pairs typically 0.2–0.5, but absolute thresholds are model-specific (HF E5 card warns 0.7–1.0 distribution) → thresholds live in config and are *not* hardcoded in this file (D5).

### 4.4 `indexing.py`

```python
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP          # + faiss.normalize_L2 done by embedder; index.add embeds
class SelfMatchExclusion:
    def __init__(self, doc_ids: list[str]) -> None
    def is_excluded(self, i: int, j: int) -> bool   # same doc_id; ALSO same normalized file-content hash (sha256 of raw bytes)
def search_neighbors(index, query_vec: np.ndarray, top_k: int, exclude: SelfMatchExclusion) -> list[tuple[int, float]]
def pairwise_doc_similarity(doc_chunk_ids: dict[str, list[int]],
                            chunk_sims: dict[tuple[int,int], float],
                            cfg: PairwiseConfig) -> list[PairCandidate]
```
**Self-match exclusion rules (exact, D7):**
1. `(i==j)` trivially excluded.
2. `doc_id(chunk_i) == doc_id(chunk_j)` excluded (assignment's "identical self-matches").
3. **Identical-file dedup:** at ingest time compute `sha256(raw bytes)` per file via `common.sha256_file`; if two files share a hash, the second arrival is kept as a document but marked `metadata["duplicate_of"] = <first document_id>`, and its chunks are tagged `is_duplicate=True`. When iterating chunk-pairs, a pair whose **match chunk belongs to a document that is a byte-identical duplicate of the query document's canonical file** is excluded from aggregation math but still reported in `peer_matches` with `similarity_score = 1.0` and `note = "byte-identical duplicate of {canonical_id}"` (which is how Scenario *E11* stays explainable without double-counting). Assumption A2 covers the merged semantics (self-match exclusion + identical-file dedup).
4. **Small-source exclusion** (research D7 / Turnitin-style): a *matched document* whose doc-pair score falls below `pairwise.min_pair_score_floor` (0.05) is surfaced in `peer_matches` (labeled `note="below reporting floor"`), but does **not** contribute to `peer_similarity_percentage` aggregation, HIGH/MEDIUM flagging, or FPR evaluation.

`pairwise_doc_similarity` aggregation math (exact, defensible):
- Doc pair score `S(Q,M) = (Σ_{chunks c∈Q, c non-template} max_{m∈M, m non-template} cos(c,m)) / |Q|` → **coverage-weighted mean-max** (template chunks are excluded on BOTH sides — the Scenario-4 discount; a query chunk whose best match is a template chunk is dropped from |Q| as well). Rationale (goes into README method): this asks "how much of Q is covered by M"; the reverse score `S(M,Q)` is also computed and `doc_pair_score = max(S(Q,M), S(M,Q))`. Direction is preserved: each document's report sees its own side as query. Emits `PairCandidate(query=..., match=..., score=..., shared_chunk_pairs=[...], is_cross_modal=..., directionality=...)`.
- top_k per chunk query = `pairwise.chunk_query_topk` (10); only chunk pairs with `sim ≥ pairwise.chunk_min_cos` (0.55) are retained for alignment (0.55 chosen: below this, MiniLM noise dominates; if `--log-level DEBUG`, the run-wide chunk-score histogram (mean/σ/p95) is written to `reports/summary/calibration.json` for the EVALUATION.md threshold discussion).

Index choice (research D4): with 14 docs, `IndexFlatIP` is the *exact* gold standard and stays; the README scaling brief covers the IVFFlat/HNSW switch points.

### 4.5 `alignment.py` — evidence extraction + boilerplate discount

```python
class TemplateRegistry:
    """Chunks whose simhash64 fingerprint recurs in ≥ 3 distinct documents are institutional boilerplate."""
    def __init__(self, min_docs: int = 3, hamming_max: int = 3,
                 min_fingerprint_tokens: int = 40) -> None
    def register_all_chunks(self, chunks: list[Chunk]) -> None       # pass 1 of pipeline
    def is_template(self, chunk: Chunk) -> bool
    def template_fingerprints(self) -> list[int]
def simhash64(text: str) -> int                                      # 64-bit simhash over token 5-gram hashes (mmh3 optional; use builtin hash of sorted ngrams aggregated per the standard simhash weighting)
def hamming(a: int, b: int) -> int
def align_pair(query_doc, match_doc, chunk_pairs, embedder, registry) -> list[MatchedPassage]
def sentence_lcs_alignment(q_sents: list[str], m_sents: list[str]) -> list[tuple[int,int,float]]
     # inclusion rule: a sentence pair is "aligned overlap" if cosine ≥ sentence_align_cos_floor (0.65),
     # OR (0.45 ≤ cosine < 0.65 AND length-normalized token-LCS ratio ≥ sentence_lcs_token_ratio 0.60 — catches near-verbatim
     # rewordings that MiniLM under-scores). Mapping is monotonic greedy on the sorted score matrix.
```
**TemplateRegistry algorithm (concrete, D6):**
1. For every chunk with `token_count ≥ min_fingerprint_tokens(40)`, compute `simhash64(normalized tokens)`.
2. Cluster fingerprints: for each new fingerprint, if an existing fingerprint within `hamming ≤ 3`, join that cluster; else new cluster member.
3. After full pass: cluster.size ≥ `min_docs(3)` distinct document_ids → **template cluster**. Every chunk in that cluster is `is_template=True` and gets a stable `template_cluster_id = "tmpl_{cluster_index:02d}"` (if a corpus has NO clusters of size ≥3 the registry is empty and behavior reduces to standard — expected for hidden edge runs; no error).
4. Template chunks are (a) excluded from doc-coverage numerator (never counted in `peer_similarity_percentage`), (b) NOT used as query chunks for peer sieve (they can't seed a flagged pair), (c) still *reported* in `common_source_matches` if they match another template chunk or another doc (with `note="shared institutional text (discounted)"`).
   Scenario 4 prediction: 3 docs share the same ~150-word case prompt → their prompt chunks land in one ≥3-doc cluster → discounted → `peer_similarity_percentage` for 12/13/14 stays LOW → they must NOT be flagged HIGH, while prompt overlaps still appear in `peer_matches` with `match_type = "common_source"`.
   Derived requirement: `dataset_generator` must emit prompt text so that only the *prompt* block repeats; analytical bodies differ (see §4.10).
5. Fallback boilerplate catch (BT exact-string rule): if a chunk's `normalized text hash` appears in ≥3 docs verbatim → auto-template regardless of token count (use 20-token floor, since prompts could be shorter; config `verbatim_template_min_tokens=20`).

**`align_pair` output `MatchedPassage` fields (see §5 for the dataclass):** populated from the best sentence-level mapping; when both sides are template chunks `match_type="common_source"`, else `"overlap"`. Fields `query_page`/`matched_page` derive from the chunk's page attribution; `matched_timestamp` string from the covered ASR segment span formatted `"00:00:24 - 00:00:36"` (`hh:mm:ss - hh:mm:ss`, space around hyphen, exactly like the assignment sample). `similarity_score` = chunk cosine (float, from the index), NOT the mean of sentence sims (sentence cosines are kept in `evidence_sentence_scores` for transparency).
Minimum evidence bar: drop candidate matches where the combined char length of the aligned sentence sets is < `alignment.min_evidence_chars` (60; DrillBit-style noise floor); they are counted in a per-document `discounted_noise_matches: int` field on the report for honesty, never silently dropped.

### 4.6 `ai_detection.py`

```python
class AIDetector:
    """Loads the HF DeBERTa classifier and gpt2; if either is unavailable the whole AI analysis degrades to the heuristic fallback (never blended, §4.6)."""
    def __init__(self, cfg: AIDetectionConfig, cache_dir: Path) -> None
    def detect(self, text: str) -> AIDetectionResult
    def detect_doc(self, doc: ProcessedDocument) -> AIDetectionResult
def feature_heuristic_ai_score(text: str) -> tuple[float, dict]   # returns (prob 0..1, details)

MODEL_CLASSIFIER: Final[str] = "wasitaigeneratedcom/ai-text-detector-small"   # verified HF ID per research "Verified tech facts"
GPT2: Final[str] = "openai-community/gpt2"                                   # plain GPT-2 small (124M) — cheap CPU perplexity; no exotic variants
```
Ensemble math (exact; de-scope decision R3: exactly **2** top-level components, 0.6/0.4 weights, no new deps):
```
p_text     = 0.75 * p_perp + 0.25 * p_burst               # text-statistics component — both sub-terms come from the already-pinned gpt2 run
p_ai_prob  = 0.6 * p_cls + 0.4 * p_text                   # model path: classifier × 0.6 + text-statistics × 0.4
length gate: if `token_count(text) < ai.min_length_tokens` (50) → `label = "INSUFFICIENT_TEXT"`, `length_gate_triggered = True`, `ai_generated_probability = None`, explanation states the doc is too short for reliable detection (precedent: Copyleaks' 350-char minimum; GPTZero short-text warnings).
p_perp: normalized logistic of gpt2 perplexity: p_ai_perp = 1 / (1 + exp((perplexity - 22) / 9.5))   # empirical: gpt2 ppl on human text 25-60; on structured LLM text 5-20 (config constant, adjustable)
p_burst:     1 / (1 + exp((burstiness - 0.55) / 0.14))          # burstiness = stddev of per-sentence logprob / mean
```
Where `perplexity` = `exp(-1/N * Σ log p(token_i | prev))` from gpt2 loss on ≤1024-token windows (chunked doc text; if doc > 1024 tokens, mean over windows).
`burstiness` = `std(logprob_sentence)/mean(logprob_sentence) + 1e-6`; sentence-conditional continuation probs obtained by feeding sentences with a trailing newline; token log-likelihoods from `model(input_ids, labels=...)` loss per sentence.
**Bands (config):** score is `ai_generated_probability = round(100*p_ai_prob, 1)`.
- `< 35` → `"HUMAN"`; `35 ≤ score < 65` → `"UNSURE"` (explicit *uncertain* band, mirroring GPTZero 40–70 and Turnitin-withholding concept; research D9); `≥ 65` → `"AI_LIKELY"`.
- Research-facts *caveat* sentence auto-appended to every explanation when score ≥ 65 OR label UNSURE: `"Note: AI detectors are unreliable on non-native-English and polished human text (Stanford 2023: up to 61% false-positive rate on ESL essays); treat this as a lead for human review, not proof."`
**Detector-source rule (exact, reviewer-required):** `ai_detector_source = "model"` **iff both** the HF classifier *and* gpt2 load at runtime — that is the only path on which `p_ai_prob = 0.6·p_cls + 0.4·p_text` is computed. If **either** model is unavailable, the *whole* AI analysis switches to the deterministic fallback; a model component is never blended with heuristic output, and the two modes are never averaged or merged in evaluation (§4.11). 
**Fallback heuristic (`feature_heuristic_ai_score`)** — explicitly documented, deterministic, no model: `p = 0.75 if (avg_sentence_len > 11 and type_token_ratio < 0.55 and low_burstiness_indicator) else 0.25`, where `low_burstiness_indicator` = sentence-length stdev / mean < 0.35. The returned `detail` dict names exactly the three booleans and the string `"rule-based mock fallback, lower confidence, per the assignment's allowance for a documented reproducible mock wrapper"`. When the fallback is active this is encoded as `AIDetectionResult.ai_detector_source = "heuristic_fallback"` (`p_classifier`/`p_perp`/`p_burst` are then `None`), and every affected report is marked so evaluation can count it (`ai_detector_fallback_count`, §4.11): the explanation **must** include the token `heuristic (lower confidence)` and the code logs `WARNING ai: model unavailable ({missing}); report {name} marked heuristic_fallback`. When model-based, `ai_detector_source = "model"`.
Text for detect always = full normalized doc text truncated to first 20,000 chars (log note if truncated).

### 4.7 `external_check.py`

```python
class LocalReferenceKB:
    """Tier 1: always-on offline knowledge base. Tier 2: optional web search with mock fallback."""
    def __init__(self, cache_dir: Path, embedder: Embedder, cfg: ExternalCheckConfig) -> None
    def build(self) -> None                 # embed KB passages once at startup; reused by every query document
    def search_doc(self, doc: ProcessedDocument) -> ExternalEvidence
def pick_query_strings(sentences: list[Sentence], k: int, min_words: int) -> list[str]
def ddg_query(query: str, request_timeout_s: int) -> list[dict] | None
```
- `reference_kb/kb_texts.py` holds **6** plain-English public-domain-style passages in exactly **2 groups** (no LLM-flavored group — doc 06's body comes from `docs/03_llm_generated_sample.md` instead, §4.10): (a) three canonical Wikipedia-style encyclopedic blocks ("Solar power in India…", "Supply chain resilience after 2020…", "E-invoicing basics…"), (b) three case-methodology/consulting-textbook blocks ("SWOT analysis is a structured planning method…", "Porter's generic strategies…", "Break-even analysis…"). The KB never feeds dataset content **except** by declared design: doc 04 lightly paraphrases the "Supply chain resilience" passage (constant `KB_SUPPLY_CHAIN`) so the external signal has one legitimate offline hit in evaluation; doc 03 inherits that paraphrased paragraph through `PARAPHRASE_TABLE`; docs 03 and 04 are the only docs with `external_expected_min = 20.0` in the manifest. Doc 06 (`06_llm_generated_summary.pdf`) contains **no** verbatim KB passage — it stays external-clean because its text is genuinely LLM-generated.
- KB search: query sentences embedded with `Embedder.encode_query` (batched); searched against the KB `IndexFlatIP`; top-2 per query retained, filtered by cosine ≥ `external.local_kb_min_cos` (0.72). KB ids are a separate namespace (`kb::<idx>`), so no self-match case can arise.
- `pick_query_strings(sentences, k=external.query_topk, min_words=12)`: TF-IDF (pure python: term freq × inverse doc freq across all doc sentences; no sklearn) selects the top-`k` scoring sentences in the 12–35-word range; a noun-phrase light pass (regex `((?:\b[A-Z][a-z]+\b|\b[a-z]{4,}\b)(?:\s+(?:\b[A-Z][a-z]+\b|\b[a-z]{4,}\b)){1,2})`) then concatenates the top-2 noun phrases as one extra query. Hard cap: 3 queries per document.
- Tier 2 (only if not `--skip-web` and `external.allow_external_web`): `ddgs` library (see §9 pin; if the pip package ever rotates, fall back to raw `requests.get("https://api.duckduckgo.com/?q=...&format=json", timeout=external.request_timeout_s)` instant-answer endpoint as secondary; final fallback is a `MockDDG` returning an empty list and appending warning `web tier unavailable; using mock empty result`). Budget: `external.web_budget_per_doc` (2) queries per document per run, each call wrapped `try/except Exception → skip`, 0.6 s sleep between calls, only for sentences with ≥ 12 words. A web snippet is evidence if token-overlap Jaccard vs the query sentence ≥ 0.35; it carries `source_title`/`source_url`.
- Output `ExternalEvidence(score_pct, hits: list[ExternalHit], tier_used: "local_kb" | "web" | "mock" | "none")` — the tier reflects what actually produced the hits.
- `external_similarity_percentage` normalization (**A6**): coverage-style, comparable to the peer metric: `score_pct = round(100 * (Σ hit.aligned_tokens) / doc_tokens, 1)` capped to [0, 100], where `hit.aligned_tokens` = the number of query-document tokens covered by that hit's best-aligned text span (bounded by the KB passage's token count, so a single short KB hit cannot inflate coverage). Expected magnitudes: benign docs < 10; the KB-anchored docs 03 and 04 ≥ 20 (manifest `external_expected_min = 20.0`).

### 4.8 `scoring.py` — signal combination, exact arithmetic

```python
def compute_integrity_signals(doc, peer_result, external, ai_result, cfg) -> Scores
def verdict_from_signals(sig: Scores, flags: ...) -> tuple[str, str]     # (risk_label, explanation)
```
**Exact formula:**
```
final_peer_pct     = round(100 * pair_score, 1)
    where pair_score = the highest doc-pair score for this query document (coverage-weighted mean-max from §4.4), computed AFTER:
    (a) template-chunk exclusion — template chunks are removed from both numerator (matched chunk pairs) and denominator (query chunk count) per §4.5,
    (b) noise-floor drop — aligned matches below `alignment.min_evidence_chars` (60) and small-source matches below `pairwise.min_pair_score_floor` (0.05) excluded,
    then clipped to [0.0, 100.0].
final_external_pct = round(external.score_pct, 1)
final_ai_label     = ai_result.label            # AI_LIKELY requires ai_generated_probability ≥ 65
Signal precedence (stated verbatim in every explanation): peer (dominant) → external (secondary) → AI (tertiary/supporting).
risk_label:
    HIGH   if final_peer_pct ≥ 70
           or (final_peer_pct ≥ 40 and final_external_pct ≥ 30)
           or (final_peer_pct ≥ 40 and final_ai_label == "AI_LIKELY" and the pair is cross-modal)
    MEDIUM if final_peer_pct ≥ 40
           or (final_peer_pct ≥ 25 and final_external_pct ≥ 25)
           or (final_ai_label == "AI_LIKELY" and ai_generated_probability ≥ 65)   # AI alone is a review lead, not proof
    LOW    otherwise
```
Peer dominance rationale (write verbatim in report's explanation when HIGH): *"Peer overlap is the strongest evidence because it confirms reuse within the evaluated cohort; external and AI scores are supporting signals."*
`explanations` string template (all tokens mandatory; the test asserts their presence):
```
"Peer similarity: {peer}% across {n} listed source(s) ({plain_overlap_summary}).
 External: {ext}% ({tier_used}). AI: {ai_label} at {ai}%{caveat_clause}.
 Overlap types: {cross_modal_count} cross-modal, {copy_count} exact-copy, {paraphrase_count} paraphrase, {template_count} discounted common-source."
```
Semantics: `{peer}` is the discounted percentage; `{n}` = count of `peer_matches` entries at or above the small-source floor (0.05); `{plain_overlap_summary}` = `top: {file} @ {score:.2f}` when n ≥ 1, else `"no substantive peer overlap"`; a `common_source`-only report legitimately reads `"0.0% across 1 listed source(s)"` because shared boilerplate is listed but discounted — that exact phrasing is what makes the Scenario-4 behavior self-explanatory in one line.
Also emits `explanation_human_sentence = "…"` longer *plain English* paragraph (that is what the .md shows).

### 4.9 `reporting.py` — final artifacts

```python
def write_json_report(doc, signals, matches, out_dir) -> Path
def write_markdown_report(doc, signals, matches, out_dir) -> Path
def write_summary_report(all_docs, out_dir, elapsed) -> Path     # aggregate matrix + per-doc one-liner verdicts
```
**JSON for each document — field names frozen to match the assignment sample; extension fields (everything after `explanations`) are the "sensible extra fields" allowance:**
```json
{
  "document_id": "12_case_template_A.docx",
  "total_integrity_risk": "LOW",
  "scores": {"peer_similarity_percentage": 0.0, "external_similarity_percentage": 0.0, "ai_generated_probability": 13.3},
  "peer_matches": [
    {"matched_file": "13_case_template_B.docx", "similarity_score": 0.38,
     "query_passage": "…verbatim CASE_PROMPT text…", "matched_passage": "…verbatim CASE_PROMPT text…",
     "location": {"query_page": null, "matched_page": null},
     "match_type": "common_source", "evidence_sentence_scores": [0.41],
     "note": "shared institutional text (discounted)"}
  ],
  "multimodal_matches": [],
  "external_matches": [],
  "ai_analysis": {"ai_generated_probability": 13.3, "label": "HUMAN", "p_perplexity": 0.09, "p_burstiness": 0.22, "p_classifier": 0.14,
                  "length_gate_triggered": false, "ai_detector_source": "model", "caveat": ""},
  "template_registry_hits": [{"template_cluster_id": "tmpl_00", "chunk_ids": [3, 9], "note": "repeated block in ≥3 documents (discounted)"}],
  "warnings": [],
  "status": "ok",
  "discounted_noise_matches": 0,
  "explanations": "Peer similarity: 0.0% across 1 listed source(s) (top: 13_case_template_B.docx @ 0.38; common-source, discounted). External: 0.0% (none). AI: HUMAN at 13.3%. Overlap types: 0 cross-modal, 0 exact-copy, 0 paraphrase, 1 discounted common-source.",
  "engine_version": "1.0.0",
  "generated_at_utc": "2026-11-04T09:41:00Z"
}
```
Sanity check of the sample numbers: `fused = 0.6·p_cls + 0.4·(0.75·p_perp + 0.25·p_burst) = 0.6·0.14 + 0.4·(0.0675 + 0.055) = 0.133 → 13.3 → HUMAN`.
Notes: `location` is always nested inside each peer/multimodal match (as the assignment sample shows), never a top-level key; for an MP4-vs-text pair the location object carries `{"query_page": <int|null>, "matched_timestamp": "hh:mm:ss - hh:mm:ss"}` (assignment sample shape); for text-vs-text it carries `{"query_page": <int|null>, "matched_page": <int|null>}`; DOCX sides are `null` pages (A3); an `error`/`empty`/`silent_video` doc still gets a full report file with empty match arrays and its `warnings` list populated (never omitted) — `ai_generated_probability = null` there is legal JSON.
The main sample above has no cross-modal or external entries; two standalone snippets (not from that report) fix their shapes. Multimodal — from `07_pitch_crossmodal.pdf`'s report (the assignment's sample shape):
```json
{
  "matched_file": "08_pitch_crossmodal_video.mp4",
  "similarity_score": 0.88,
  "query_passage": "Operating margins contracted by 4.2% over Q3 …",
  "matched_passage": "Operating margins dropped 4.2% across the third quarter because …",
  "location": {"query_page": 2, "matched_timestamp": "00:00:04 - 00:00:21"}
}
```
External evidence — from `04_original_strategy.docx`'s report (its KB-anchored paragraph, §4.7):
```json
{
  "source_title": "kb::1", "source_url": null, "similarity": 0.79,
  "query_passage": "…04's paraphrased 'supply chain resilience' text…",
  "matched_passage": "Supply chain resilience after 2020 depends on diversified suppliers and deliberate buffers …"
}
```
**Markdown report layout (frozen; targeted at the "< 30-second verify" bar; verdict-card style inspired by HackerEarth's likelihood badge + Turnitin's per-source panel — research §3):**
```
# Submission Integrity Report: 12_case_template_A.docx

| Risk | Peer% | External% | AI% | Status |
|---|---|---|---|---|
| 🟢 LOW | 0.0 | 0.0 | 13.3 | ok |

> **Why:** Peer similarity: 0.0% across 1 listed source(s) (top: 13_case_template_B.docx @ 0.38; common-source, discounted). External: 0.0% (none). AI: HUMAN at 13.3%. Overlap types: 0 cross-modal, 0 exact-copy, 0 paraphrase, 1 discounted common-source.

## Peer Matches (1)
1. **matched → 13_case_template_B.docx** · chunk cosine **0.38** · 📄 this doc p *n/a* → source p *n/a*
   > query:   "…verbatim CASE_PROMPT text…"
   > matched: "…verbatim CASE_PROMPT text…"
   *shared institutional text (discounted)*

## Cross-Modal Matches (0) / External Reference Matches (0)
*none — both sections render exactly like the peer section (with their own match-type notes) whenever they have entries.*

## AI Analysis
- Label: **HUMAN** (13.3/100) · perplexity signal 0.09 · burstiness signal 0.22 · classifier 0.14 · detector source: `model`
- Caveat: — (auto-appended only at UNSURE / AI_LIKELY)

## Warnings
- *none*
```
Marker rules: every evidence bullet carries location markers inline — `📄 pN` for each side with a page (query side first), `⏱ hh:mm:ss - hh:mm:ss` when an MP4 side participates, and `page n/a` (italic) for DOCX sides. MP4 evidence lines must show the timestamp in the same `hh:mm:ss - hh:mm:ss` shape as the JSON. `summary/report_{ts}.json` (ts = UTC `YYYYMMDD_HHMMSS`) plus a copy at `summary/latest.json`:
```json
{"generated_at_utc": "...", "elapsed_s": 214.3,
 "documents": [{"document_id": "...", "status": "ok", "risk": "LOW",
                "peer": 0.0, "external": 12.4, "ai_label": "HUMAN", "ai": 22.0}, ...],
 "aggregate": {"n_docs": 14, "high": 6, "medium": 1, "low": 7, "degraded": 2}}
```
Expected truth balance (§4.10 manifest): HIGH 6 (01, 02, 03, 04, 07, 08) · MEDIUM 1 (06) · LOW 7 (05, 09, 10, 11, 12, 13, 14 — of which 10/11 are the degraded pair).
File layout under `--output_dir`: `per_doc/{document_id}.json`, `per_doc/{document_id}.md` (base name, no double extension), `summary/report_{ts}.json`, `summary/latest.json`, and — with `--eval` — `summary_metrics.json` at the top level of `--output_dir`.

### 4.10 `dataset_generator.py`

```python
def generate_dataset(target_dir: Path, seed: int = 42) -> list[Path]        # writes all 14 files + manifest; returns paths
def write_manifest(target_dir: Path) -> None                                # dataset_manifest.json (schema below)
def build_pdf_from_text(text: str, path: Path) -> None
def build_docx_from_text(text: str, path: Path) -> None
def build_mp4_from_text(text: str, path: Path, espeak_voice: str = "en+f2", espeak_rate: int = 155) -> None
```
`build_pdf_from_text` (further note FN1): uses `fpdf2` (pinned) — tiny, pure-python, deterministic output; multi-page split on the `\f\f` page marker embedded in the source strings; no images. `build_mp4_from_text`: `espeak-ng -v {voice} -s {rate} -w tmp.wav "{text}"` — the 2 spoken clips use different voices for plausible variety (08: `en+f2`; 09: `en+M2`), then ffmpeg composites the WAV onto a solid color background (exact command below).
**Content plan (docs 01–05 and 07–14 carry literal author-owned strings inside `dataset_generator.py`; doc 06's body is the verbatim LLM constant; each doc ≤ ~900 words so the corpus stays tiny):**

| # | File | Content plan |
|---|---|---|
| 01 | `01_retail_turnaround_report.pdf` | 4-page synthetic business case: ~900-word "Retail Turnaround" report; **page 2** contains the exact 120-word passage (code constant `CANONICAL_SHARED_PASSAGE`). |
| 02 | `02_retail_turnaround_copy.docx` | **Verbatim body copy of 01**: identical text carried in a DOCX container (intent: exact-copy pair). The two files' sha256 differ (different containers), so *text similarity* — not byte-dedup — detects this pair; the byte-dedup path (E11) is reserved for same-byte files. |
| 03 | `03_paraphrased_strategy.docx` | ~750-word "Digital Distribution Strategy" **paraphrase of 04**: same argument skeleton, sentence-level rewording via a hand-maintained literal mapping table `PARAPHRASE_TABLE` in code (deterministic synonym + clause reorder, NOT algorithmic) so chunk cosine lands ≥ 0.70 while token Jaccard of the pair stays < 0.35 (Scenario-2 test). The paraphrase table implicitly carries 04's KB-derived paragraph too (so 03's external ≥ 20 mirrors 04's). |
| 04 | `04_original_strategy.docx` | ~750-word "Digital Distribution Strategy (v1)" — the source document 03 paraphrases. **Its second paragraph is a light paraphrase of the reference-KB "Supply chain resilience" passage** (`KB_SUPPLY_CHAIN` in `reference_kb/kb_texts.py`) — the deliberate external-KB demonstration (§4.7); no other peer relations. |
| 05 | `05_original_market_pdf.pdf` | ~700-word "Indian solar market" original, 3 pages, all content authored from scratch; **no peer relations and no KB anchor → false-positive guard** (topic-adjacent to the KB "Solar power in India" block but textually unrelated — deliberate near-domain negative control; expected external < 10). |
| 06 | `06_llm_generated_summary.pdf` | ~600-word body **copied VERBATIM** from `docs/03_llm_generated_sample.md` (everything below the `---` divider) into module constant `LLM_GENERATED_TEXT`, with the provenance comment `# LLM_GENERATED_TEXT is genuine LLM output: GLM, single pass 2026-11-04; full prompt in docs/03_llm_generated_sample.md — do not hand-edit`. Rendered with fpdf2, ≤ 2 pages. No CASE_PROMPT (standalone), no KB anchor. *Observation (not a construction rule):* the generated register naturally shows uniform sentence lengths and formulaic connectors ("In summary", "Furthermore"); the detector reads that register, we did not force it. Expected: AI signal alone drives risk MEDIUM (fused ≥ 65 per manifest flag `ai_expected_min_probability: 55`). |
| 07 | `07_pitch_crossmodal.pdf` | 2-page "Q3 Margins" report; **page 2** contains `PITCH_PASSAGE` (95 words). |
| 08 | `08_pitch_crossmodal_video.mp4` | **Spoken verbatim narration** of exactly `PITCH_PASSAGE` (espeak-ng; ≤ 30 s audio at 155 wpm). Cross-modal pair with 07. |
| 09 | `09_third_pitch_video.mp4` | ~25 s pitch on carbon-tracking logistics — unique content; also proves an unrelated spoken doc stays below the peer reporting floor (replaces the dropped `09_second_pitch_video.mp4` use case; the near-peer exclusion floor stays covered by 05 in the pair-retrieval path). |
| 10 | `10_corrupted_empty.pdf` | **Zero-byte file** (`Path.touch()`). |
| 11 | `11_silent_video.mp4` | 12 s silent video (ffmpeg `anullsrc` audio), no speech → expected status `silent_video`. Deliberate: 3 MP4s is the assignment's "2–3 clips" maximum, and the silent clip is chosen as the 3rd so Scenario 3's degraded path ships inside the corpus itself (A9). |
| 12 | `12_case_template_A.docx` | Case-study template: 150-word `CASE_PROMPT` block (identical string in 12/13/14) + ~350-word original "Unit-economics analysis" body. |
| 13 | `13_case_template_B.docx` | Same `CASE_PROMPT` + ~350-word different original body ("Distribution-network SWOT"). |
| 14 | `14_case_template_C.docx` | Same `CASE_PROMPT` + ~350-word different original body ("Break-even plan"). → template trio for Scenario 4 (§4.5 rule needs ≥ 3 docs; modality is irrelevant to the rule — the trio is all-DOCX). |
Counts: PDFs = 01, 05, 06, 07, 10 (**5**); DOCX = 02, 03, 04, 12, 13, 14 (**6**); MP4 = 08, 09, 11 (**3** — the assignment's "2–3 short MP4 clips" upper bound exactly). Total **14** files (inside the assignment's 10–15 dataset range and Scenario 5's 10–20 range).
**MP4 creation command (exact):**
```bash
espeak-ng -v {voice} -s {rate} -w {tmp}.wav "{text}"
ffmpeg -y -f lavfi -i color=c=0x1F3B5C:s=640x360:r=15 -i {tmp}.wav -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac {out}.mp4
```
Silent video: `ffmpeg -y -f lavfi -i color=c=black:s=640x360:r=10 -f lavfi -i anullsrc=r=16000:cl=mono -t 12 -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac 11_silent_video.mp4`.
`dataset_manifest.json` schema (regenerated by `--generate-dataset`; shipped in repo):
```json
{
  "dataset_version": "1.1",
  "generated_with_seed": 42,
  "generated_by": "GLM (LLM), single pass 2026-11-04",
  "generation_prompt_summary": "~600-word executive case-competition submission analyzing a mid-sized organic grocery chain called GreenHarvest Foods (root causes, three-phase turnaround plan, expected margin impact, key risks), written in the register typical of LLM outputs; provenance of the document 06_llm_generated_summary.pdf only — the other 13 files are human-authored synthetic strings. Body text embedded verbatim from docs/03_llm_generated_sample.md as constant LLM_GENERATED_TEXT.",
  "ground_truth_pairs": [
    {"query": "01_retail_turnaround_report.pdf", "source": "02_retail_turnaround_copy.docx", "relation": "exact_copy", "expected_min_peer_similarity": 0.90},
    {"query": "04_original_strategy.docx", "source": "03_paraphrased_strategy.docx", "relation": "paraphrase", "expected_min_peer_similarity": 0.60, "expected_test": "matched-passage cosine > 0.80 (assignment's paraphrase band) while token Jaccard of the pair < 0.35"},
    {"query": "07_pitch_crossmodal.pdf", "source": "08_pitch_crossmodal_video.mp4", "relation": "cross_modal", "expected_min_peer_similarity": 0.70}
  ],
  "expected_integrity_risk": {
    "01_retail_turnaround_report.pdf": "HIGH",
    "02_retail_turnaround_copy.docx": "HIGH",
    "03_paraphrased_strategy.docx": "HIGH",
    "04_original_strategy.docx": "HIGH",
    "05_original_market_pdf.pdf": "LOW",
    "06_llm_generated_summary.pdf": "MEDIUM",
    "07_pitch_crossmodal.pdf": "HIGH",
    "08_pitch_crossmodal_video.mp4": "HIGH",
    "09_third_pitch_video.mp4": "LOW",
    "10_corrupted_empty.pdf": "LOW",
    "11_silent_video.mp4": "LOW",
    "12_case_template_A.docx": "LOW",
    "13_case_template_B.docx": "LOW",
    "14_case_template_C.docx": "LOW"
  },
  "ground_truth_flags": {
    "03_paraphrased_strategy.docx": {"external_expected_min": 20.0},
    "04_original_strategy.docx": {"external_expected_min": 20.0},
    "06_llm_generated_summary.pdf": {"ai_expected_min_probability": 55.0},
    "05_original_market_pdf.pdf": {"expected_peer_matches": 0}
  },
  "original_false_positive_guard": ["05_original_market_pdf.pdf", "09_third_pitch_video.mp4",
                                     "12_case_template_A.docx", "13_case_template_B.docx", "14_case_template_C.docx"],
  "degraded": [
    {"file": "10_corrupted_empty.pdf", "kind": "zero_byte_pdf", "expected_status": "error"},
    {"file": "11_silent_video.mp4", "kind": "silent_video", "expected_status": "silent_video"}
  ],
  "template_group": {"files": ["12_case_template_A.docx", "13_case_template_B.docx", "14_case_template_C.docx"], "shared_block": "CASE_PROMPT",
                     "expected": "risk LOW for all three; the trio is all-DOCX (modality irrelevant to the ≥3-shared-block rule); prompt chunks reported as match_type=common_source; no HIGH flag (Scenario 4)"},
  "content_authorship_note": "All content is synthetic and authored for this project (doc 06: generated by GLM/LLM, single pass, disclosed above); no copyrighted student or vendor data used (assignment §7 compliance)."
}
```
Truth-semantics note (encoded here, consumed by `evaluation.py`): *(i)* plagiarism-pair membership is flagged **symmetrically** — the engine reports text overlap wherever found (Turnitin-convention), so the copied source 04 is truth-HIGH too; culpability (perpetrator vs original author) is explicitly out of scope and listed as a limitation in EVALUATION.md. *(ii)* 06 is truth-MEDIUM: its AI signal alone (expected fused ≥ 65 → `AI_LIKELY`, with peer ≈ 0 and external ≈ 0) is a review lead, not disqualification-grade evidence — the fused risk sits where the §4.8 formula puts it. *(iii)* 10/11 are truth-LOW (degraded; nothing verifiable to charge). This map is the single source of truth for the confusion matrix.
**LLM-provenance disclosure requirement (reviewer-edict):** README (§Methodology) and EVALUATION.md (§Setup & reproduce) must each carry one short subsection disclosing that doc 06's body was generated in a single pass by GLM on 2026-11-04 with the prompt summarized above, pointing to `docs/03_llm_generated_sample.md` and to the manifest's `generated_by`/`generation_prompt_summary` fields — never described as human-authored "LLM-styled" text.

### 4.11 `evaluation.py`

```python
def evaluate(reports_dir: Path, manifest_path: Path, cfg) -> EvaluationSummary
def write_evaluation_md(summary, path: Path) -> None
def prf(tp, fp, fn) -> tuple[float, float, float]     # plain-python; no sklearn dependency
def confusion_matrix(y_true: list[str], y_pred: list[str], labels) -> dict[tuple[str,str], int]
```
- Pair-level P/R/F1 computed **separately per relation kind** (`exact_copy`, `paraphrase`, `cross_modal`) — a predicted peer/multimodal match counts as TP for a relation if (file pair, in either direction) appears in that relation's manifest `ground_truth_pairs` list; all *other* above-threshold reported pairs count as that relation's false positives.
- **FPR on originals:** files listed in manifest `original_false_positive_guard` (5 files: 05, 09, 12, 13, 14 — the degraded pair 10/11 is excluded because degraded docs produce no analyzable text) must show zero `peer_matches` with `match_type != "common_source"`; `FPR = (# guard files with ≥1 such match) / 5`. Common-source (template) matches are excluded by definition — that exclusion is exactly the Scenario-4 discount at work.
- **Confusion matrix** at file level: rows = truth from manifest `expected_integrity_risk`; columns = engine's `total_integrity_risk`; labels {LOW, MEDIUM, HIGH}; written to both `EVALUATION.md` and `reports/summary_metrics.json`, with the document_ids of every mislabeled cell listed beneath the matrix (rubric "honest evaluation"). Also compute accuracy and balanced accuracy.
- **AI-detection table (dedicated; reviewer-required):** `EVALUATION.md` contains an `AI-detection results` table listing, for every doc the manifest AI-flags (06) **plus** every doc the engine reports UNSURE/AI_LIKELY: `document_id`, `p_classifier`, `perplexity` (raw), `p_perp`, `burstiness` (raw), `p_burst`, fused score, `ai_detector_source`, final label. **Model-based rows and heuristic-fallback rows are never merged or averaged** — they stay separate rows, and every fallback row is annotated `heuristic_fallback`. When the HF classifier or gpt2 was unavailable at runtime, each affected report was already marked `ai_detector_source="heuristic_fallback"` and logged (§4.6); the eval generator counts those reports as `ai_detector_fallback_count` and prints the total above the table.
- Writes `EVALUATION.md` **with numbers filled in**, sections in this order: `Setup & reproduce` (exact commands from §10 + the LLM-provenance disclosure from §4.10), `Dataset overview` (counts by type + scenario table), `Runtime` (cold/warm wall-clock from the runs), `Pair retrieval metrics` (per-relation P/R/F1 table), `False-positive rate on originals`, `AI-detection results` (the table above), `Confusion matrix (risk levels)` (+ mislabeled files), `Thresholds used` (the config values + calibration histogram pointer), `Failure Case Study 1`, `Failure Case Study 2`, `Trade-off analysis` (embedding model + Flat index vs compute; mirrors README but own words). The two failure case studies are **scaffolds with instructions**: each must be completed by the engineer agent with 2 concrete paragraphs citing actual names/numbers from that run's reports (e.g. a specific false positive, a specific recall miss, an ASR artifact) — the scaffold names the two candidate framings from research D11 (accented-speech ASR errors lowering cross-modal recall; ESL/styled-text AI false positives; Symmetric flagging of the copied source 04; threshold sensitivity of the paraphrase pair) and explicitly forbids fabricating numbers. `--eval` exits 0 regardless of metric outcomes.
- Also writes `reports/summary_metrics.json` with the numeric summary machine-readable.

### 4.12 `config.py`

```python
@dataclass(frozen=True)
class ChunkingConfig:   max_chunk_tokens: int = 200; min_chunk_tokens: int = 40; overlap_sentences: int = 2; mp4_target_tokens: int = 120; mp4_max_seconds: int = 45
@dataclass(frozen=True)
class EmbeddingConfig:  model_name: str = "sentence-transformers/all-MiniLM-L6-v2"; device: str = "cpu"; batch_size: int = 32
@dataclass(frozen=True)
class PairwiseConfig:   chunk_query_topk: int = 10; chunk_min_cos: float = 0.55; doc_flag_threshold: float = 0.70; doc_report_floor: float = 0.40; min_pair_score_floor: float = 0.05
@dataclass(frozen=True)
class AlignmentConfig:  template_min_docs: int = 3; template_hamming_max: int = 3; template_min_tokens: int = 40; verbatim_template_min_tokens: int = 20; min_evidence_chars: int = 60; sentence_align_cos_floor: float = 0.65; sentence_lcs_token_ratio: float = 0.60
@dataclass(frozen=True)
class AIDetectionConfig: classifier_model: str = "wasitaigeneratedcom/ai-text-detector-small"; gpt2_model: str = "openai-community/gpt2"; perplexity_mu: float = 22.0; perplexity_k: float = 9.5; burst_mu: float = 0.55; burst_k: float = 0.14; band_human_max: float = 35.0; band_unsure_max: float = 65.0; min_length_tokens: int = 50; max_chars: int = 20000; max_tokens_per_window: int = 1024
@dataclass(frozen=True)
class ExternalCheckConfig: local_kb_min_cos: float = 0.72; web_budget_per_doc: int = 2; query_topk: int = 3; request_timeout_s: int = 6; allow_external_web: bool = True; ddg_backend: str = "duckduckgo_lite_or_fallback"
@dataclass(frozen=True)
class RuntimeConfig:    log_level: str = "INFO"; hf_home: Path | None = None; cache_dir: Path = Path("cache"); seed: int = 42; max_ingest_seconds_per_file: int = 120
@dataclass(frozen=True)
class Config:           chunking: ChunkingConfig; embedding: EmbeddingConfig; pairwise: PairwiseConfig; alignment: AlignmentConfig; ai: AIDetectionConfig; external: ExternalCheckConfig; runtime: RuntimeConfig
def load_config(path: Path | None) -> Config        # merges TOML over defaults; unknown keys warned
```
**`config/default.toml`** mirrors this dataclass with `[chunking]`, `[embedding]`, `[pairwise]`, … tables. Every "magic number" in the codebase must be referenced from `cfg`, never literal (Code Quality rubric + maintainability); `doc_flag_threshold = 0.70` is assigned per D5 (assignment's >0.80 is paraphrase-specific and MiniLM saturated scores mean ours *is* stricter at doc level: correct calibration; `pairwise.chunk_min_cos = 0.55` is the sentence-evidence floor while doc-pair aggregated score uses `doc_flag_threshold`).
Decision note: the *passage-level paraphrase band* still explicitly reports `cos > 0.80` in reports whenever that assignment-defined case arises (any matched passage with chunk cosine ≥ 0.80 is labeled `"paraphrase-band (>0.80)"`; assumption A7), keeping the assignment's language visible while doc-level flagging uses the calibrated `doc_flag_threshold`.

### 4.13 `common.py`
```python
def get_trimmed_logger(name: str, log_file: Path | None, level: str) -> logging.Logger
def list_input_files(root: Path) -> list[Path]      # sorted; unsupported ext skipped + logged
def sha256_file(p: Path) -> str
def utc_now_iso() -> str
```

---

## 5. Data Structures (single source of truth: `integrity_engine/types.py`)

```python
@dataclass
class Page:                 # a provenance unit for one page (pdf/docx) or one ASR segment (mp4)
    page: int | None             # 1-based for pdf; None for docx/mp4
    text: str
    char_start: int; char_end: int     # into joined "full document text"

@dataclass
class ProcessedDocument:
    document_id: str             # exact filename incl. extension, e.g. "06_llm_generated_summary.pdf"
    path: Path
    modality: Literal["pdf", "docx", "mp4"]
    status: Literal["ok", "empty", "error", "silent_video"]
    warnings: list[str]
    pages: list[Page]            # always ordered; page None allowed
    full_text: str               # joined via "\n\n"; char offsets in Page refer to this
    size_bytes: int
    sha256: str                  # dedup key (A2)
    metadata: dict               # mp4: {"duration_s": 13.4, "asr_language": "en", "asr_model": "tiny"}; pdf: {"page_count": 4}; docx: {"paragraph_count": 22}

@dataclass
class Chunk:
    chunk_id: str                # f"{doc_idx:02d}:{chunk_idx:03d}"
    document_id: str
    text: str
    start_char: int; end_char: int
    page_start: int | None; page_end: int | None        # inclusive; None for docx/mp4 paths
    timestamp_start: float | None; timestamp_end: float | None   # seconds, mp4 only
    sentence_ids: list[int]
    token_count: int
    is_template: bool = False
    template_cluster_id: str | None = None
    embedding: np.ndarray | None = None        # excluded from json; runtime only

@dataclass
class PairCandidate:
    query_document_id: str
    match_document_id: str
    score: float                               # doc-level coverage-weighted mean-max cosine (the peer_similarity_percentage basis)
    shared_chunk_pairs: list[tuple[str, str, float]]    # (query_chunk_id, match_chunk_id, cos)
    is_cross_modal: bool
    directionality: Literal["forward", "reverse", "both"]

@dataclass
class MatchedPassage:
    matched_file: str
    similarity_score: float
    query_passage: str
    matched_passage: str
    query_page: int | None; matched_page: int | None
    matched_timestamp: str | None           # "00:00:24 - 00:00:36"
    match_type: Literal["overlap", "common_source"]     # common_source => template-registry discount
    evidence_sentence_scores: list[float]
    note: str | None

@dataclass
class Sentence:
    text: str
    char_start: int; char_end: int     # into the joined full_text of the owning document
    page: int | None                 # owning page for pdf; None for docx/mp4

@dataclass
class ExternalHit:
    source_title: str
    source_url: str | None        # None for local KB
    similarity: float             # cosine (local KB) or Jaccard overlap (web tier)
    query_passage: str
    matched_passage: str
    source: Literal["local_kb", "web", "mock"]
    aligned_tokens: int           # query tokens covered by this hit (feeds external coverage %)

@dataclass
class ExternalEvidence:
    score_pct: float; hits: list[ExternalHit]; tier_used: Literal["local_kb", "web", "mock", "none"]

@dataclass
class AIDetectionResult:
    ai_generated_probability: float | None      # 0–100, None when length-gated or failed
    label: Literal["HUMAN", "UNSURE", "AI_LIKELY", "INSUFFICIENT_TEXT"]
    p_perplexity: float | None; p_burstiness: float | None; p_classifier: float | None
    length_gate_triggered: bool; ai_detector_source: str   # "model" | "heuristic_fallback" — never merged across sources (§4.6)
    caveat: str; detail: dict                      # e.g. perplexity raw, burstiness raw, token_count

@dataclass
class Scores:
    peer_similarity_percentage: float    # 0–100 after template/noise discount
    external_similarity_percentage: float
    ai_generated_probability: float | None
    risk: Literal["LOW", "MEDIUM", "HIGH"]

@dataclass
class EvaluationSummary:
    relation_prf: dict[str, tuple[float, float, float]]   # per relation kind
    fpr_originals: float
    ai_detector_fallback_count: int                       # reports whose AI verdict ran as heuristic_fallback (never merged with model rows, §4.11)
    confusion: dict[tuple[str, str], int]                 # (truth, predicted) -> count
    mislabeled: dict[tuple[str, str], list[str]]
    elapsed_cold_s: float | None
    elapsed_warm_s: float | None
```
`Scores` is the object serialized into the final JSON (exact schema in §4.9); types not listed here (config dataclasses, GroundTruthRecord passthroughs) are internal.

---

## 6. Pipeline Stages (exact order) & 7.1 GB budget math

```
cli.main()
 ├─ S0 parse args, load TOML config, init logging (stderr + logs/run_{ts}.log). set seed.
 ├─ S1 extract_any() every input file, sorted by name → list[ProcessedDocument]
 │     (mp4 files processed serially; every file wrapped try/except → status error on unexpected exception, log traceback. Never abort batch.)
 ├─ S2 for each doc: text_processing.build_chunks → all_chunks (skip if status != ok)
 ├─ S3 alignment.TemplateRegistry.register_all_chunks(all_chunks)   — pure text pass BEFORE embeddings (cheap; research D6)
 │     then: build Embedder, encode all chunk texts → matrix (n_chunks x 384) float32
 │     faiss.IndexFlatIP(384).add(matrix) → for each chunk (query) get top-k=10 neighbors with exclusions → chunk_sims
 ├─ S4 filter chunk pairs ≥ chunk_min_cos 0.55 → pairwise_doc_similarity → list[PairCandidate]; sort doc-pairs by score desc
 ├─ S5 for each doc (every one, not only flagged ones):
 │      • peer_matches = align_pair(...) for its pair candidates with score ≥ pairwise.doc_report_floor (0.40);
 │        only score ≥ pairwise.doc_flag_threshold (0.70) counts toward HIGH flagging
 │      • multimodal_matches = the subset of matches where either side is an .mp4 (assignment shows this as its own array)
 │      • external_check.search_doc (local KB always; web ≤2 calls unless skipped/offline)
 │      • ai_detection.detect_doc
 ├─ S6 scoring.compute_integrity_signals → risk + explanation
 ├─ S7 reporting.write_json_report + write_markdown_report per doc; write_summary_report
 └─ S8 (if --eval) evaluation.evaluate against dataset_manifest.json → EVALUATION.md + summary_metrics.json
```

**RAM math (documented in README scaling brief §8):**
- Corpus: 14 files ≈ 5,900 tokens total ≈ 40 chunks × 384 dims × 4 bytes ≈ **60 KB** of vectors; peak process RSS target < **2 GB** incl. torch runtime (~600 MB), MiniLM (~90 MB), gpt2 fp32 (~550 MB), the DeBERTa-small AI classifier (~140 MB), faster-whisper tiny int8 (~80 MB) — worst theoretical sum ≈ 1.6 GB, comfortably inside 7.1 GB and the NFR ≤2 GB. Discipline: models are instantiated in sequence (ASR → free → AI models) and Python references dropped (`del` + `gc.collect()` after stage S1) so ASR memory is released before the AI models load; the engineer must verify with `scripts/check_env.py --rss` which prints `resource.getrusage(...).ru_maxrss` on exit.
- The 10k-doc brief speaks to production; this prototype code path does not attempt 10k. Time budget: ingestion ≈ 18 s (2 spoken clips × tiny/int8 ≈ 2–8× real-time on 12 cores → a 25-s clip ≈ 4–12 s; the silent clip short-circuits before ASR), embeddings ≈ 4 s, retrieval ≈ 1 s, per-doc external checks ≈ 10 s total (KB embeddings reused; the web tier issues ≤ 2 calls per doc — ≤ 28 across 14 docs — only if a network is present), AI ≈ 11 s (gpt2 perplexity ≈ 0.8 s × 14 docs), writing ≈ 2 s → total ~60–120 s **after model preload** (NFR ≤5 min warm); first run adds ~1.1 GB of HF downloads (ST + gpt2 + classifier + whisper tiny) — inside the ≤20-minute cold target (A11; downloads use HF `resume_download`).

---

## 7. Edge-Case Behavior Table

| # | Input case | Ingestion status | Report shows | Log line (exact wording) | Pipeline effect |
|---|---|---|---|---|---|
| E1 | Zero-byte file (any ext) | `error` | JSON+MD report with empty matches, `warnings:["file is empty (0 bytes); nothing to analyze"]` | `WARNING  ingestion: file is empty (0 bytes); nothing to analyze: {name}` | No chunk; excluded from template pass; scan continues |
| E2 | Silent MP4 (no audio stream) | `silent_video` | `multimodal_matches: []`, status silent_video | `WARNING  mp4: no audio stream in MP4: {name}` | No chunks; no crash |
| E3 | MP4 with audio but VAD finds no speech (music-only) | `silent_video` | status silent_video; `warnings:["speech-not-detected by VAD; possibly silent or music-only audio"]` | `WARNING  mp4: speech-not-detected by VAD: {name}` | No chunks |
| E4 | MP4 with very quiet audio (RMS < −45 dB) but non-zero stream | `silent_video` | as E2 | `WARNING  mp4: audio track is effectively silent (mean volume < -45 dB): {name}` | ASR fast-path skipped (saves wall time) |
| E5 | Non-English audio | `ok` | chunks built; `warnings:["non-english audio detected ({lang}); transcription may be degraded"]` | `WARNING  mp4: non-english audio detected ({lang}): {name}` | Proceeds; explanation caveat auto-appended "ASR quality degraded for non-English audio" |
| E6 | Corrupted PDF container | `error` | JSON+MD with `warnings:["unreadable pdf container ({exc_summary})"]` | `WARNING  pdf: unreadable pdf container: {name} ({cls}: {msg})` | No chunks |
| E7 | PDF that is a scanned image (no text layer) | `empty` | status empty; every page warning individually | `WARNING  pdf: page {i}: no extractable text (likely scanned image) {name}` | No chunks; no OCR (non-goal) |
| E8 | Corrupt / legacy `.doc` renamed `.docx` | `error` | warnings `["not a valid .docx container (may be legacy .doc or corrupted)"]` | `WARNING  docx: not a valid .docx container: {name}` | No chunks |
| E9 | Text-only DOCX/PDF with < 15 chars after normalization | `empty` | status empty; warnings `["document contains no usable text"]` | `WARNING  {ext}: document contains no usable text: {name}` | No chunks |
| E10 | Unknown extension (`.txt`, `.zip`, `.exe`) | — | no report file (it is not a supported submission) | `INFO  skipping {name}: unsupported extension` | Excluded from `documents` list |
| E11 | Two byte-identical files (different names) | both `ok` | first is canonical; second's peer_matches against first show `similarity_score` 1.0 with `match_type:"overlap"` and note `"byte-identical duplicate"`; risk HIGH for both (exact-copy pair) | `INFO  dedup: {b} is byte-identical to {a}` | Both remain in summary; neither is "excluded" from reporting—only from double-counting (A2) |
| E12 | Very long MP4 (> 60 s; violates sample-data norm but hidden test may include it) | `ok` | chunks generated normally | `INFO  mp4: long audio ({dur}s); processing within prototype limits` | Hard timeout 120 s per file at ingestion; with tiny/int8, 5-min audio ≈ 40 s CPU — acceptable; > 120 s wall clock → status error warning `asr exceeded per-file time budget` (never crashes) |
| E13 | Single-file input dir (only one doc) | `ok` | peer_matches empty | `INFO  only one document in corpus; peer comparisons are vacuous` | External + AI still run (they are doc-local) |
| E14 | Input dir missing / unreadable | — | no reports | `ERROR  input_dir does not exist or is not readable: {path}`; exit 1 | — |
| E15 | `--eval` with mismatched manifest entries (a manifest file absent from reports dir) | — | EVALUATION.md notes it | `WARNING  eval: manifest entry {doc} missing from reports; counted as FAIL for recall` | Not a crash |
| E16 | Every model download unreachable (offline first run) | — | If the **embedding model** is unavailable: no reports are written, clear error + README `Troubleshooting: offline first run` guidance, exit 1 (it is the one hard requirement, being the peer engine's core). If only *whisper/gpt2/classifier/DDG* are unavailable: reports still written with degradations — `ai_analysis.ai_detector_source="heuristic_fallback"` + logged warning per affected report, mp4 files get `error` status, web tier silently mocked | `ERROR  core model unavailable (offline and not cached): {model}`; `WARNING  ai: model unavailable ({missing}); report {name} marked heuristic_fallback` | A8 degradation hierarchy: embedding model = **fatal**; whisper tiny = per-mp4 `error`; gpt2/classifier = heuristic-fallback AI mode (never merged with model results, §4.11); DDG web = silent mock. Never a bare Python traceback to the user |

Assignment-Scenario mapping: **S1** ↔ files 07+08 (§4.10 plan + `multimodal_matches` with `matched_timestamp`), **S2** ↔ files 03+04 (`paraphrase` pair; any chunk evidence in the >0.80 band is labeled `paraphrase-band (>0.80)` in reports; both docs also anchor the external-KB demo), **S3** ↔ files 10+11 (rows E1/E2/E4 — the zero-byte PDF and the silent MP4 ship inside the corpus), **S4** ↔ files 12/13/14 (§4.5 template discount → LOW risk, not a HIGH false alarm), **S5** ↔ the whole 14-file corpus run + `summary/report_{ts}.json`.

---

## 8. Non-Functional Requirements

| NFR | Requirement | Verification |
|---|---|---|
| N1 | CPU-only execution; no CUDA calls anywhere in code; `device="cpu"` hard default | `grep -rn "cuda" integrity_engine/ run_integrity_check.py` → only in an allowlist comment; run completes with `CUDA_VISIBLE_DEVICES=""` |
| N2 | ≤ 2 GB peak RSS for the 14-doc run | `scripts/check_env.py` prints `resource.getrusage(RUSAGE_SELF).ru_maxrss` after pipeline; record in README |
| N3 | Full-run wall time ≤ 20 min cold (incl. model downloads ≈1.1 GB), ≤ 5 min warm on the 12-core box | Measured twice and recorded in EVALUATION.md "Runtime" table |
| N4 | Zero unhandled exceptions from file content — any surprise becomes `status="error"` + warnings + exit 0 | pytest `test_cli_never_crashes_on_garbage.py` feeds random bytes files of every extension |
| N5 | Determinism: same input dir + same seed → byte-identical JSON reports (except `generated_at_utc`, `elapsed_s`) | `test_determinism.py` runs pipeline twice on sample corpus, diffs normalized JSON |
| N6 | PEP 8 + typing (mypy-clean intended but not gating), all public functions typed | `ruff check .` (line-length 100) passes; type hints on every public signature |
| N7 | Logs: console at chosen level (stderr) + full DEBUG per-run file under `logs/` | `logs/run_*.log` exists after run; contains model version lines and per-file outcomes |
| N8 | Modern layout hygiene: no `src`-less namespace collisions; `integrity_engine` importable from repo root without install (`rootdir` on sys.path) | `pytest` green from repo root with no `pip install -e .` (documented; `run_*.py` adds no path hack because root already is sys.path) |
| N9 | Reproducible env: `scripts/setup_env.sh` + `requirements.txt` + `scripts/check_env.py` prints versions matrix | Fresh-clone smoke test instructions in README §Quickstart produce a green run |

---

## 9. Dependencies (exact `requirements.txt`)

```text
# Core inference stack (CPU). Install torch FIRST from the CPU wheel index (see scripts/setup_env.sh)
# so no CUDA runtime is pulled; then this file installs the rest (pin equality enforced).
numpy==2.2.6
faiss-cpu==1.15.1
sentence-transformers==6.2.0
torch==2.15.0                  # already satisfied by setup_env.sh (CPU wheel from download.pytorch.org/whl/cpu)
transformers==4.49.0           # gpt2 perplexity + wasitaigeneratedcom/ai-text-detector-small classifier
faster-whisper==1.1.0          # ctranslate2-backed ASR; voice activity filter; segment timestamps
pypdf==5.4.0
python-docx==1.1.2
fpdf2==2.8.2                   # only for dataset_generator (synthetic PDF authoring); ends up unused at runtime
ddgs==9.6.1                    # optional Tier-2 web search; every call is try/except, never required
```
Rationale table (README mirrors this):
| dep | why | alternative rejected |
|---|---|---|
| faiss-cpu | assignment explicitly suggests FAISS; exact Flat IP = gold standard at this scale (research D4) | NumPy cosine matrix (fine but rubric names FAISS; also scales to the 10k brief with same interface) |
| sentence-transformers | assignment names all-MiniLM-L6-v2; simplest API, normalization built-in | raw transformers mean-pooling (more code, no gain) |
| faster-whisper | 4× faster than openai-whisper, int8 CPU-friendly, segment timestamps (research §2.2/D2) | openai-whisper (slower), vosk (no punctuation → worse chunking) |
| transformers (gpt2 + classifier) | research-verified ensemble pieces (§2.1, D9) | zero-shot Fast-DetectGPT (needs bigger LM; over budget) |
| ddgs | only *no-key* free search option with reusable library; optional (mock fallback) | Bing/Google APIs (keys/paid), DDG instant-answer-only API (too thin alone) |
| fpdf2 | deterministic tiny PDF writing for the synthetic dataset | reportlab (heavier); hand-rolled pdf bytes (fragile) |
| pypdf / python-docx | assignment-listed extraction libs | pdfminer.six (slower, no real gain here) |
**Python interpreter decision (A10):** primary target is a dedicated **Python 3.11 venv** (`python3.11 -m venv .venv`) because the ctranslate2/torch binary wheel ecosystem is maximally mature there; `scripts/check_env.py` prints all versions. If wheel resolution of any pin fails on 3.11, the documented fallback order is: (1) bump that single pin to the newest compatible release and record it in README §Assumptions, (2) retry with Python 3.14 venv (already verified: torch 2.15.0, faiss-cpu 1.15.1, sentence-transformers 6.2.0 import fine there per environment probe). `scripts/setup_env.sh` implements both targets behind `PYTHON_BIN` env var (default `python3.11`).
Not used (explicit): gTTS (espeak-ng subprocess covers TTS offline, zero deps), pandas, rich/typer (argparse+plain logging suffice; NFR N6), langchain/llamaindex (FAQ says raw ST/FAISS is preferred), sklearn (metrics are 20 lines of plain python).

---

## 10. Test Plan (pytest)

Layout under `tests/`; every test ≤ ~30 s; no network in unit tests (Tier-2 web tests use the mock; model-dependent tests marked `@pytest.mark.slow_model` are acceptable to run in the final full pass because models are already cached).

| Test file | Cases |
|---|---|
| `test_text_processing.py` | normalize: ligatures/hyphen/whitespace/unicode; sentence split guards (abbreviations, mid-sentence 800 char); chunk token bounds 200 & overlap; docx path `<15 tokens empty`; char-offset/Page provenance exactness |
| `test_ingestion_pdf.py` | generated PDF → pages + text match source string; zero-byte → error; scanned-image simulated (rasterized-only pdf via fpdf2 image-only page) → empty; repeated header line stripped |
| `test_ingestion_docx.py` | paragraphs → text; corrupt `.docx` → error; heading noise tolerated; tables concatenation |
| `test_ingestion_mp4.py` | silent mp4 → silent_video; generated spoken mp4 → transcript contains ≥1 expected keyword; `mp4` char spans align `timestamp start ≤ first transit end ≤ last`; unknown-lang token warning present |
| `test_embeddings.py` | L2 norm rows (max abs‖v‖−1 ≤1e-5); `dim==384`; identical texts → cos 1.0; unrelated pair cos < 0.7 (log actual); zero-length input no crash |
| `test_indexing.py` | self-match exclusion (same doc); byte-duplicate exclusion logic (A2); top-k correctness with k=10; doc pair aggregation math golden value on fixture vectors; small-source floor |
| `test_template_registry.py` | three synthetic docs sharing 120-token block → cluster size 3 → `is_template`; two docs sharing → not template; hamming cluster limit; verbatim short block rule; template coverage excluded from doc score |
| `test_alignment.py` | exact-copy pair alignment: sentence identity mapping; paraphrase pair: aligned sentences under the 0.65 / token-LCS rule on ≥ 2 sentence pairs; `min_evidence_chars` drop; timestamp/page assignment; `MatchedPassage` field types |
| `test_ai_detection.py` | length gate (10-word text → INSUFFICIENT_TEXT); the genuine LLM constant `LLM_GENERATED_TEXT` → label AI_LIKELY on the model path (`@pytest.mark.slow_model`); golden ensemble math `0.6·p_cls + 0.4·(0.75·p_perp + 0.25·p_burst)` on fixture probabilities; detector source is `"model"` when both models load and `"heuristic_fallback"` (with the exact `WARNING ai: model unavailable…` log line) when either is monkeypatched off; caveat string presence |
| `test_external_check.py` | KB hit for 04's KB-derived paragraph (and its inherited copy in 03) ≥ config threshold; benign doc 05 → score < 10; TF-IDF sentence picker; DDG mock returns [] gracefully; web budget cap respected (query counter) |
| `test_scoring.py` | risk table: HIGH/ MEDIUM/ LOW boundaries; template discount keeps Scenario-4 docs LOW; explanations template string contains all mandatory tokens |
| `test_report_schema.py` | produced JSON keys exactly == assignment sample's top-level keys + allowed extras; boolean/number/null types; md contains verdict card + evidence bullets with ⏱ markers; mp4 evidence line contains `matched_timestamp` format `\d{2}:\d{2}:\d{2} - \d{2}:\d{2}:\d{2}` |
| `test_cli.py` | exit codes: 0 healthy, 1 missing dir, 2 argparse misuse; stdout is exactly one JSON line; `--models-only` produces no reports |
| `test_cli_never_crashes_on_garbage.py` | random-byte files of each ext in a tmp dir → exit 0, reports written, statuses all error-ish, no traceback in stderr |
| `test_determinism.py` | two consecutive runs (same seed, cached models) → normalized JSON equal |
| `test_e2e_scenarios.py` **(integration, the money test)** | runs the real pipeline on the real `sample_submissions/`; asserts: (S1) 07+08 pair appears in `multimodal_matches` with `matched_timestamp` and `query_page`; (S2) 03+04 detected as a pair with doc score ≥ 0.60 (report floor) and any ≥ 0.80 chunk evidence labeled `paraphrase-band (>0.80)`; (S3) statuses contain `error` (10) and `silent_video` (11), exit 0, no crash; (S4) 12/13/14 all risk ≠ HIGH; their template chunks appear as `common_source`; (S5) 14 JSON + 14 MD reports + summary written; the 5 files in `original_false_positive_guard` → 0 non-common-source peer matches; LLM doc 06 → `ai_generated_probability ≥ 55` and `ai_detector_source == "model"` (steady state) with external ≈ 0; docs 03+04 → external ≥ 20 each |
| `test_dataset_generator.py` | regenerate with seed 42 in tmp dir → file count 14, manifest valid JSON and contains `expected_integrity_risk` for all 14 files plus the `generated_by`/`generation_prompt_summary` provenance fields; doc 06's embedded text equals the body after `---` in `docs/03_llm_generated_sample.md` (provenance assertion); zero-byte file is zero bytes; the 2 spoken MP4s' audio durations within 8–35 s; PDF/DOCX/MP4 counts = 5/6/3, byte-level determinism (rerun produces identical files) |

**Commands the engineer must run and have pass before finishing (recorded in README §Verification):**
```bash
python3.11 -m venv .venv && source .venv/bin/activate
bash scripts/setup_env.sh                     # installs torch CPU wheel + rest
python scripts/check_env.py                   # prints version matrix incl. faiss, torch, st, fw
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports --generate-dataset --seed 42   # (first-time data authoring)
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports --models-only
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports --eval ./sample_submissions/dataset_manifest.json
python -m pytest tests/ -q && ruff check . && ruff format --check .     # full suite green; lint clean
```

---

## 11. Acceptance Criteria Checklist (assignment "Submission Checklist" + rubric extras)

| # | Item (assignment) | Satisfying artifact |
|---|---|---|
| 1 | Working ingestion pipeline PDF/DOCX/MP4 | `integrity_engine/ingestion/*` + tests `test_ingestion_*` + E2E S1 |
| 2 | Vector similarity search with self-match exclusion | `indexing.py` SelfMatchExclusion + `test_indexing.py` |
| 3 | AI-generation and external similarity modules | `ai_detection.py` (2-component ensemble + bands + `ai_detector_source` model/heuristic_fallback marking), `external_check.py` (KB + optional web, mock fallback) |
| 4 | Human-readable explainability report: exact chunk matches + timestamps/pages | `reporting.py` MD + JSON; `test_report_schema.py`; MD verdict card < 30 s rule |
| 5 | Synthetic dataset 10–15 files + ground-truth manifest in repo | §4.10 table (14 files: 5 PDF / 6 DOCX / 3 MP4) + `sample_submissions/dataset_manifest.json` incl. the `generated_by`/`generation_prompt_summary` provenance fields; README §Methodology + EVALUATION.md disclose the LLM-generation methodology (§4.10) |
| 6 | EVALUATION.md with metrics + ≥2 failure analyses | `evaluation.py` writer + scaffold; engineer fills narratives from real run (no fabrication) |
| 7 | 10k scaling brief | README §"Scaling to 10,000 submissions" (structure: S0 preprocessing queue → extraction → chunk+embed → upsert; queue tech: SQS/Celery; index: IVFFlat/HNSW thresholds with formulas (`nlist ∈ [4√N,16√N]`) or hosted Qdrant/pgvector metadata split like research D12; cost/latency bottlenecks: ASR largest, embedding parallelizable, report I/O; GPU-vs-CPU numbers quoted from research §2.6) |
| 8 | Single root command on sample data | `run_integrity_check.py` (README Quickstart one-liner) |
| 9 | Assumptions heading in README | README §Assumptions == spec §12 list, slightly prose-ified |
| 10 | Long-video scope respected (10–60 s clips) | Dataset MP4s ≤ 30 s; NFR test |
| 11 | No paid APIs | Pinned open-source only; `--skip-web` default-safe behavior |
| 12 | Graceful malformed handling everywhere | E1–E16 + `test_cli_never_crashes_on_garbage.py` |
| 13 | Clean modular code + docs + error handling | §4 docstring contracts, config centralization, logs, ruff clean |
| 14 | Multi-file aggregate matrix + per-file audit reports | `summary/report_*.json` + per-doc files (Scenario 5) |
| 15 | Explainable cross-modal evidence w/ page AND timestamp | §4.9 md bullet marker `⏱` + `📄` |
| 16 | AI-detection uncertainty honesty (Stanford ESL caveat) | mandatory caveat token in ai explanation + md; per-report `ai_detector_source` field; EVALUATION.md AI-detection table with never-merged fallback rows + `ai_detector_fallback_count` (§4.11) |
| 17 | Rubric "Quality & Rigor of Own Evaluation": confusion matrix + FPR on originals + honest ambiguity | evaluation.py outputs (confusion matrix w/ per-cell filenames, originals FPR, 2 failure studies) |
| 18 | Rubric "Architecture & Scaling Reasoning" (10%) | README scaling brief includes the index-switch math (Flat → IVFFlat/HNSW with `nlist ∈ [4√N, 16√N]`), CPU/GPU throughput anchors, and the async-queue design from research D12 + §2.6 |

---

## 12. Assumptions Log (each of the 12 also listed in README §Assumptions verbatim; capped per de-scope rule R6)

| ID | Ambiguity in assignment | Decision |
|---|---|---|
| A1 | "peer similarity percentage" is not defined mathematically | Coverage-weighted mean-max cosine over non-template chunks (§4.4); rationale documented in README + report explanation. |
| A2 | "identical self-matches excluded" could mean files or chunks | Both semantics merged: intra-doc chunks excluded; byte-identical files (by sha256) reported as exact-copy pair (emitted at 1.0, not double-counted in aggregation). |
| A3 | DOCX pages are undefined | `page=None`; provenance is paragraph-level char spans; PDFs carry pages; MP4s carry timestamps. |
| A4 | Language of video/audio | English-only assumption; non-English gets a warning + degraded-ASR caveat, still processed. |
| A5 | Chunking granularity for transcripts | Segment-aligned chunking ≤ 200 tokens with 1-segment overlap (better provenance than page-style chunking). |
| A6 | External % normalization | Coverage-style (Σ aligned tokens / doc tokens), comparable to the peer's philosophy; KB-anchored docs 03/04 expected ≥ 20, benign docs < 10. |
| A7 | Paraphrase result banding | The assignment's "cosine > 0.80" is a **chunk/passage-level** band: matched passages with chunk cosine ≥ 0.80 are labeled `paraphrase-band (>0.80)`; doc-level flagging uses `doc_flag_threshold` 0.70 (HIGH) with full evidence reporting from 0.40 up. |
| A8 | Offline behavior | Embedding model missing → exit 1 fatal (core requirement); whisper missing → mp4 files degrade to `error`; gpt2/classifier missing → the whole AI analysis switches to `heuristic_fallback` (marked per report, logged, never blended with model results, §4.6/§4.11); DDG unreachable → silent mock. |
| A9 | MP4 count "2–3 short clips" | **Exactly 3** = the assignment's upper bound; one of the 3 is deliberately silent so Scenario 3's degraded path ships inside the 14-file corpus (which stays inside both the 10–15 dataset range and Scenario 5's 10–20 range). |
| A10 | Python version on this box | 3.11 venv primary for wheel maturity; 3.14 documented fallback (env probe showed torch 2.15.0/faiss 1.15.1/ST 6.2.0 import there). |
| A11 | Model sizes, download budget | all-MiniLM-L6-v2 (~90 MB) + gpt2 (~550 MB) + AI classifier (~150 MB) + whisper-tiny (~75 MB) ≈ 0.9 GB disk, ≤ 20-min cold run target on this 12-core box. |
| A12 | "AI-generated doc" ground truth | Doc 06's body is **genuinely LLM-generated** (GLM, single pass 2026-11-04; prompt + provenance recorded in `docs/03_llm_generated_sample.md` and the manifest's `generated_by`/`generation_prompt_summary` fields); it is embedded verbatim (`LLM_GENERATED_TEXT`), disclosed in README/EVALUATION, and never hand-edited or re-authored. |

**Further notes** (single details, intentionally *not* load-bearing — list stays at 12 entries; new material lands here, never as a new A-row):
- **FN1** — Synthetic PDFs are authored with fpdf2 (tiny, pure-python, deterministic output).
- **FN2** — `--seed` affects only dataset generation; the engine pipeline itself is deterministic without RNG.
- **FN3** — `multimodal_matches` are populated mutually (the assignment example shows only a text doc querying a video; both files' reports reference the pair).
- **FN4** — Header/footer stripping applies to PDFs only (≥50% page-repeat heuristic); DOCX headers/footers live in separate XML parts and are ignored.
- **FN5** — Aggregate summaries are written as `summary/report_{UTC ts}.json`, with `summary/latest.json` overwritten each run.
- **FN6** — Text overlap is flagged **symmetrically** (Turnitin convention): the copied source 04 is truth-HIGH too; culpability inference is out of scope (stated limitation in EVALUATION.md).
- **FN7** — `--eval` compares engine output vs manifest ground truth as defined; no external scorer.
- **FN8** — The 10k brief is a written deliverable only; the prototype does not implement 10k execution.
- **FN9** — Repo hygiene: `reports/`, `logs/`, `cache/`, `.venv/` gitignored; `sample_submissions/` committed (≤ 5 MB).

---

## 13. Module dependency graph (for import order sanity)
```
cli ─► config ─► types
cli ─► common ─► config
cli ─► ingestion/* ─► types, common, config
cli ─► text_processing ─► types, config
cli ─► embeddings ─► config, types
cli ─► indexing ─► types, config
cli ─► alignment ─► types, config, embeddings (sentence scoring cache)
cli ─► ai_detection ─► config, common
cli ─► external_check ─► config, types, embeddings
cli ─► scoring ─► types, config
cli ─► reporting ─► types, config
cli ─► evaluation ─► types, config            (only when --eval)
cli ─► dataset_generator ─► config            (only when --generate-dataset)
# module-specific imports NEVER import each other except through types/config/embeddings;
# keeps the graph acyclic and test-mockable.
```

---

*End of spec. Everything above is binding; if a detail conflicts with the assignment text, the assignment wins and the engineer must note the deviation in README §Assumptions.*
