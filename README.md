# Multimodal Submission Integrity & Plagiarism Detection Engine

A CPU-only Python pipeline that reads a folder of PDF, DOCX and MP4 submissions and
produces one JSON report and one Markdown report per file, plus a batch summary.
Each report contains three signals, **peer similarity %**, **external similarity %**
and **AI-generated likelihood %**, and a LOW / MEDIUM / HIGH risk. Every overlap is
backed by the exact passages, the PDF page or video timestamp on each side, and the
similarity score.

```bash
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
```

**Results at a glance** (details and every table: [EVALUATION.md](EVALUATION.md))

| check | result |
|---|---|
| Assignment scenarios S1–S5 on the 14-file sample corpus | 5 / 5 pass |
| Sample corpus: ground-truth pairs (exact copy, paraphrase, cross-modal MP4↔PDF) | 3 / 3 found, 0 false-positive pairs, 0 / 5 originals falsely matched |
| Sample corpus: risk level vs manifest | 12 / 14 correct (both misses explained in EVALUATION.md) |
| 400-doc benchmark (HC3 human text + injected copies / MRPC paraphrases), at the shipped threshold | precision 0.96, recall 0.94, pair AUC 0.99998; 12 template docs → 0 pair score |
| AI detector, HC3 held-out (105 human + 105 ChatGPT) | precision 0.99, recall 0.99, human FPR 1 % |
| AI detector, pre-2019 Wikipedia (human) vs machine-written docs | 0 / 30 human texts AI_LIKELY, but 11 / 30 UNSURE; the classifier alone puts 10 / 30 above 0.5 |
| Runtime, sample corpus | 74–87 s wall-clock across runs, 2.05–2.09 GB peak RSS, 6-core laptop CPU, no GPU |
| Tests | 34 pytest tests (scenarios, corrupted inputs, core math, determinism) |

## Quickstart

Requirements: Linux or macOS, Python 3.11, `ffmpeg` on PATH. `flite` and
`espeak-ng` are needed only to regenerate the sample MP4s.

```bash
bash scripts/setup_env.sh          # venv + CPU torch + requirements + model download (~2.5 GB)
.venv/bin/python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
```

Useful flags:

| flag | effect |
|---|---|
| `--eval ./sample_submissions/dataset_manifest.json` | score the run against ground truth and refresh `EVALUATION.md` |
| `--skip-web` | disable the optional DuckDuckGo tier (the local reference KB always runs) |
| `--models-only` | download / cache all models and exit |
| `--generate-dataset [--seed 42]` | rebuild the 14 sample files + manifest (byte-deterministic) |
| `--config my.toml` | override any threshold in `config/default.toml` |

Exit codes: `0` run completed, including runs with corrupted inputs; `1` fatal
(missing input dir, no supported files, bad config, embedding model unavailable);
`2` usage error. Stdout carries one JSON line (`summary_path`,
`documents_evaluated`, `warnings`, `elapsed_s`, `peak_rss_mb`). Logs go to stderr
and to `logs/run_<UTC>.log`.

## Output

```
reports/
├── per_doc/<file>.json     # assignment schema + extension fields
├── per_doc/<file>.md       # reviewer view: verdict card, side-by-side passages
├── summary/latest.json     # all verdicts + document×document peer-score matrix + stage timings
├── summary/SUMMARY.md      # the same as a readable table
└── summary_metrics.json    # with --eval
```

Excerpt from `reports/per_doc/07_pitch_crossmodal.pdf.json` (Scenario 1):

```json
{
  "document_id": "07_pitch_crossmodal.pdf",
  "total_integrity_risk": "HIGH",
  "scores": {"peer_similarity_percentage": 100.0, "external_similarity_percentage": 0.0,
             "ai_generated_probability": 65.1},
  "peer_matches": [],
  "multimodal_matches": [{
    "matched_file": "08_pitch_crossmodal_video.mp4",
    "similarity_score": 0.90,
    "query_passage": "Margin bridge and countermeasures\nOperating margins contracted by 4.2% over Q3 because inbound freight per carton rose nineteen percent ...",
    "matched_passage": "Operating margins contracted by 4.2% over Q3 because inbound freight per carton rose 19% ...",
    "location": {"query_page": 2, "matched_timestamp": "00:00:00 - 00:00:25"},
    "overlap_kind": "cross_modal",
    "evidence_sentence_scores": [0.91, 1.0, 0.96],
    "lexical_jaccard": 0.89,
    "note": "spoken/written cross-modal match (lexical Jaccard 0.89, token-LCS 0.87; ASR noise expected)"
  }],
  "explanations": "Peer similarity: 100.0% across 1 listed source(s) (top: 08_pitch_crossmodal_video.mp4 @ 0.90; cross-modal). ..."
}
```

The first six keys match the assignment schema in order. `location` uses
`query_page` / `matched_page` for document sides and `query_timestamp` /
`matched_timestamp` for video sides. Extension fields follow `explanations`:
`status`, `warnings`, `external_matches`, `ai_analysis` (every component and
`ai_detector_source`), `template_registry_hits`, `doc_pair_scores` and `provenance`.
Corrupted and silent files still get a full report, with `status` and
`warnings` filled in.

## Methodology

```
files ─► extract (pdf/docx/mp4) ─► normalize + sentences + chunks (page / timestamp provenance)
      ─► template registry (shared boilerplate) ─► MiniLM embeddings ─► FAISS IndexFlatIP
      ─► candidate pairs (chunk kNN ∪ sentence kNN, self-matches excluded)
      ─► sentence alignment ─► evidence runs ─► peer % ─┐
      ─► local reference KB (+ optional web) ─► external % ─┼─► risk rule ─► JSON + MD
      ─► DeBERTa classifier + GPT-2 statistics ─► AI % ─┘
```

**1. Ingestion** (`integrity_engine/ingestion/`). PDFs are read with `pypdf` page by
page. A line is treated as a repeated header or footer and removed only if it is
the first or last line on at least 2 pages and on at least 50 % of pages. A
page with no text layer gets the warning "no extractable text (likely scanned
image)". DOCX files are read with `python-docx` in true document order, with tables
flattened to `cell | cell` rows. MP4s go through `ffprobe` (no audio stream →
`silent_video`), `ffmpeg volumedetect` (mean volume below −45 dB → `silent_video`
without running ASR), 16 kHz mono WAV extraction, then faster-whisper `tiny` int8 with
VAD. Each ASR segment keeps its start and end seconds. No file problem raises past
the extractor: zero-byte, corrupt, legacy `.doc` renamed to `.docx`, random bytes,
no-audio and music-only files all become `error`, `empty` or `silent_video` with a
specific warning, and the batch continues.

**2. Normalization and chunking** (`text_processing.py`). NFKC normalization,
ligature and quote mapping, de-hyphenation across line breaks, and whitespace
collapse. Sentences are split with a deterministic regex that knows common
abbreviations. Text chunks are built from whole sentences up to 200 tokens with a
2-sentence overlap. 200 is used rather than the assignment's 200–400 because
MiniLM truncates input at 256 word-pieces. MP4 chunks group ASR segments up to 120
tokens or 45 s, with a 1-segment overlap. Every chunk and sentence carries absolute
character offsets, so any passage maps back to its PDF page or its ASR time span.

**3. Embeddings and index** (`embeddings.py`, `indexing.py`).
`all-MiniLM-L6-v2` (384-d) vectors are L2-normalized. FAISS `IndexFlatIP` on
normalized vectors gives exact cosine similarity. **Self-match exclusion:** a
neighbour is dropped if it is the same chunk or comes from the same document.
Byte-identical files (same sha256) are linked as duplicates and reported at
similarity 1.0 without being counted twice.

**4. Peer similarity** (`alignment.py`). Candidate document pairs come from two
FAISS searches: chunk top-10 neighbours, and sentence top-10 neighbours at cosine
≥ 0.65. For each candidate pair, every sentence of one document is aligned against
every sentence of the other. A sentence pair is kept at cosine ≥ 0.65, or at
cosine ≥ 0.45 if the token-LCS ratio is ≥ 0.60. The mapping is monotonic (no
crossing). Consecutive aligned sentences form **evidence runs**. A run counts as
evidence if it has ≥ 2 sentences, or is a single sentence of ≥ 12 tokens at cosine
≥ 0.85 or token-LCS ≥ 0.80. The single-sentence bar is high because single-sentence
cosine is unreliable: on MRPC, cosine ≥ 0.65 has a false-positive rate of 0.67.

> **peer similarity %** = 100 × (tokens of the document inside evidence runs against
> source *S*) / (all non-template tokens of the document), maximised over sources and
> over both directions of the pair.

Every run becomes one reported passage. Each passage records the passage cosine,
the per-sentence cosines, lexical Jaccard and token-LCS, and a label: `verbatim`
(token-LCS ≥ 0.90), `paraphrase`, or `cross_modal` (one side is an MP4). The two
lexical numbers are shown so a reviewer can see when a match is semantic (high
cosine, low Jaccard).

**5. Boilerplate discount (Scenario 4).** The assignment's case-prompt problem is the
"sanctioned copying" case from the winnowing paper (Schleimer et al., SIGMOD 2003):
register boilerplate as a special document and ignore matches against it. Our
`TemplateRegistry` does this without being given the prompt. A chunk whose
simhash64 fingerprint (or exact normalized text) occurs in ≥ 3 different
submissions is marked template, and so is any ≥ 6-token sentence found verbatim in
≥ 3 submissions. Template text is removed from both the numerator and the
denominator of peer %. It is still listed in `peer_matches` with `match_type:
"common_source"` and the note "shared institutional text (discounted)", so a
reviewer can see why it did not count. Two documents sharing a block are **not**
discounted: two students sharing text is the thing we are looking for.

**6. External similarity** (`external_check.py`). Tier 1 is always on: a local
reference knowledge base of 6 passages (encyclopedic topics plus case-method
staples such as SWOT, Porter and break-even), split into 34 sentences. Each document
sentence is compared to every KB sentence, and a hit needs cosine ≥ 0.70. Tier 2 is
optional: at most 2 DuckDuckGo queries per document, built from the highest TF-IDF
sentences. A web snippet counts if its token Jaccard with the query is ≥ 0.35.
Every network error falls back to a documented mock that returns no results.
External % = share of the document's tokens covered by hits, with overlapping spans
merged.

**7. AI-generated likelihood** (`ai_detection.py`). The score combines two
components:
`p = 0.6·p_classifier + 0.4·(0.75·p_perplexity + 0.25·p_burstiness)`.
The classifier is `wasitaigeneratedcom/ai-text-detector-small` (DeBERTa-v3-large;
ai_score = 1 − P(human), length-weighted over 512-token windows). The two
statistical components are logistic curves over GPT-2 perplexity and the
coefficient of variation of per-sentence log-loss. Their constants were fitted on a
held-out calibration split of HC3. Labels are HUMAN (< 35), UNSURE (35–65) or
AI_LIKELY (≥ 65). Texts under 50 tokens get `INSUFFICIENT_TEXT`. If either model
cannot load, the whole AI analysis switches to a documented rule-based heuristic,
and every affected report is marked `ai_detector_source: "heuristic_fallback"`.
The two modes are never blended. Every UNSURE or AI_LIKELY report carries the ESL
false-positive caveat (Liang et al. 2023).

**8. Risk rule** (`scoring.py`, thresholds in `[scoring]`). Peer evidence dominates,
external evidence is secondary, and AI is supporting:

* **HIGH** if peer ≥ 70 %, or peer ≥ 40 % with external ≥ 30 %, or peer ≥ 40 % with
  AI_LIKELY on a cross-modal pair.
* **MEDIUM** if peer ≥ 20 %, or peer ≥ 10 % with external ≥ 25 %, or AI_LIKELY alone.
* **LOW** otherwise.

AI alone can never produce HIGH.

**9. Explainability.** The Markdown report is designed for a check in under 30
seconds. It starts with a one-line verdict. Each match then gets a
"this doc 📄 p2 → source ⏱ 00:00:00 - 00:00:25" header, the two passages one above
the other, and a note that says *what kind* of overlap it is. Discounted template
blocks, external hits, AI components and warnings follow in separate sections.

## Scaling to 10,000 submissions

All per-unit costs below were **measured on this prototype's hardware** (6-core
Ryzen 5 4600H, CPU only; see EVALUATION.md). The 10k figures are projections from
those rates, and cloud prices are approximate on-demand list prices. Assumed
workload: 10,000 submissions, each a ~1,500-word report (≈ 75 sentences, 8 chunks,
4 classifier windows), and half of them with a 3-minute video.

**Architecture**

```
upload ─► object store (S3/GCS), sha256 = job id (idempotent; exact duplicates found for free)
       ─► queue (SQS or Celery+Redis), one message per file
       ─► ingest workers   CPU: pdf/docx   |  ASR pool: mp4 (CPU int8, or GPU at volume)
       ─► embed workers    batch sentences + chunks; write vectors + provenance rows
       ─► per-challenge stores: Postgres (docs, sentences, page/timestamp, scores)
                                FAISS chunk index (Flat) + sentence index (HNSW) on disk
       ─► template pass    official prompt registered up front as a "special document",
                           plus ≥3-document recurrence detection over the whole challenge
       ─► candidate stage  kNN per sentence/chunk (k=10) → candidate doc pairs (no N² loop)
       ─► evidence workers sentence alignment per candidate pair → peer %, passages
       ─► AI workers       GPU pool (dominant cost) │ external workers (KB + rate-limited web)
       ─► report writer    JSON/MD to object store; reviewer queue sorted by risk
```

Each stage only consumes the previous stage's output, so stages scale
independently and can be retried by job id. Late submissions query the existing
indexes and are then appended, so the batch never needs to be recomputed from
scratch.

**Compute: where the time goes** (single 6-core CPU box, measured rate → 10k projection)

| stage | measured rate | 10k projection | GPU? |
|---|---|---|---|
| ASR, whisper-tiny int8 | real-time factor 0.07 | 5,000 × 180 s × 0.07 ≈ **17.5 box-hours** | optional; CPU int8 is already 14× faster than real time |
| AI classifier, DeBERTa-v3-large | ~2 s per 512-token window | 40,000 windows ≈ **22 box-hours** | **yes**: the dominant cost, and it batches well on a GPU |
| GPT-2 statistics | 0.4 s per document | ≈ 1.1 box-hours | no |
| MiniLM embeddings | 160 sentences/s | 830k texts ≈ 1.4 box-hours | no |
| sentence kNN, 750k vectors | Flat: 4,774 q/s at 60k, linear in N → ~380 q/s at 750k ≈ 33 min. HNSW (M=32, ef=128): 6,663 q/s at 60k, recall 0.993 → a few minutes | see left | no |
| alignment + evidence | 0.27 s per document (400-doc benchmark) | ≈ 45 min | no |

Total: about 43 box-hours on CPU. With 32 workers (4 × 8-vCPU instances) the batch
takes about 1.5–2 hours. Moving the classifier to one GPU instance removes about
half of that.

**Vector store and index choice.** At 10k submissions the indexes hold about 80k
chunk vectors (123 MB) and about 750k sentence vectors (1.15 GB float32). The chunk
index stays **Flat**: it is exact, it builds in under 0.1 s, and it is cheap at this
size. The sentence index is the one that hits the scaling wall. Flat search is
O(N) per query, and all-vs-all search over 750k vectors is O(N²). HNSW
(M=32, efSearch=128) gave recall@10 = 0.993 at 0.41 ms per query in our 60k test,
against 7.7 ms for Flat. IVF-Flat with nlist = 4√N needed nprobe = 32 to reach
0.94 recall, so HNSW is the better trade-off here. Two cautions. First, every
recall point lost is evidence that never gets aligned, so keep efSearch high and
spot-check recall against Flat on each challenge. Second, sentences that occur in
many documents (the case prompt) produce giant neighbour lists and dominate kNN
time. Remove template sentences *before* the kNN stage, not after. For multi-node
serving or SQL-joinable provenance (pages and timestamps next to vectors), Qdrant or
pgvector with HNSW is the managed equivalent. FAISS on local disk is enough for a
batch job.

**Cost.** CPU path: 43 box-hours ≈ 43 × ~$0.34/h (8-vCPU c6i.2xlarge class) ≈
**$15 per 10k-submission challenge**. With a T4-class GPU (~$0.53/h) for the
classifier and ASR, the GPU stages shrink by an estimated order of magnitude (not
measured here) and the total stays around $10–15, but finishes in well under an
hour. Storage is negligible: about 1.3 GB of vectors and a few GB of media per
challenge. Web search is the cost to watch. At 2 queries per document, 10k
documents make 20k queries, which free endpoints will rate-limit. In production
this tier needs either a paid search API or an offline crawl of likely sources
(Wikipedia dumps, the challenge's own reading list) added to the KB tier.

**Latency and throughput bottlenecks.** (1) The AI classifier: 1.7 GB per process,
so worker count is limited by RAM before CPU. Give it a GPU, or replace it with a
smaller detector tuned on in-domain data. (2) ASR for long videos: cap clip length,
use VAD to skip silence (already done), and only escalate low-confidence clips to
larger Whisper models. (3) The candidate stage: template sentences must be filtered
before kNN, as above. (4) Re-indexing: changing the embedding model invalidates
every vector, so store `model_name` and version next to each vector and rebuild
per challenge.

**Pitfalls.** Treat reports as evidence for a human reviewer, never as automatic
penalties, especially the AI signal (EVALUATION.md failure study 2). Thresholds
depend on the embedding model; re-check them with `benchmarks/bench_retrieval.py`
after any model change. Keep each challenge isolated: index per challenge, with
cross-challenge search only as an explicit opt-in.

## Assumptions

1. **A1 Peer similarity %** is not defined by the assignment. Here it is the share
   of a document's non-template tokens covered by accepted sentence-evidence runs
   against one source (max over sources and both directions). *Change from the
   spec*: the spec used a chunk-level coverage-weighted mean-max cosine. The
   400-document benchmark measured pair AUC 0.637 for that approach and 0.99998 for
   sentence evidence, so the score was moved.
2. **A2 Self-match exclusion** means both "same chunk / same document" and
   "byte-identical file under another name". The latter is reported at 1.0 but
   counted once.
3. **A3 DOCX has no pages**, so DOCX locations are `null`. PDFs carry 1-based pages
   and MP4s carry `hh:mm:ss - hh:mm:ss` from ASR segments.
4. **A4 English only.** Non-English audio gets a warning and is still processed.
5. **A5 Transcript chunking** groups ASR segments (not sentences) so every chunk has
   exact start and end times.
6. **A6 External %** uses the same coverage idea as peer % (tokens covered by
   KB/web hits ÷ document tokens).
7. **A7 The assignment's "cosine > 0.80"** is treated as a passage-level band.
   Passages ≥ 0.80 are labelled `paraphrase-band (>0.80)`, but detection does not
   require it: MRPC shows 0.80 catches only 30 % of low-overlap paraphrases.
8. **A8 Degradation order when offline:** missing embedding model → exit 1 (it is
   the core of the engine); missing Whisper → MP4 files get `error`; missing
   classifier or GPT-2 → heuristic fallback, marked in each report; no network →
   web tier mocked.
9. **A9 Dataset:** 14 files = 5 PDF / 6 DOCX / 3 MP4. One MP4 is silent, so
   Scenario 3 ships inside the corpus. Spoken clips are 25–26 s.
10. **A10 Python 3.11** venv for wheel maturity. Two pins changed from the spec:
    `transformers` 4.49 → 5.22.1 (required by torch 2.15 / sentence-transformers
    6.2), and `primp` pinned to 0.15.0 because primp 2.x breaks ddgs 9.6.1
    ("Invalid impersonate").
11. **A11 Model budget:** MiniLM 90 MB, GPT-2 0.55 GB, the AI classifier 1.7 GB
    (the spec assumed 0.14 GB; the model named "small" is DeBERTa-v3-large), and
    whisper-tiny 75 MB. Peak RSS is 2.05–2.09 GB vs the spec's 2 GB target.
12. **A12 "LLM-generated document"** = doc 06, generated by GLM in one pass
    (prompt in `docs/03_llm_generated_sample.md`). The other synthetic documents
    were drafted with an AI coding assistant, so they are machine-written too. They
    are valid *plagiarism* ground truth (who copied whom is controlled), but not
    human ground truth for the AI detector. HC3 and pre-2019 Wikipedia are used for
    that instead.

Further notes:

* The MEDIUM peer threshold is 20 % instead of the spec's 40 %. Measured: at 40 %,
  a student who copied a third of a peer's text scored LOW (recall 0.31). At 20 %,
  precision is 0.96 and recall 0.94.
* The AI perplexity and burstiness constants are fitted on HC3 rather than set by
  hand. Burstiness carries no useful signal (AUC 0.29) and is kept only because the
  spec fixes the two-component formula.
* Overlap is flagged on both sides of a pair (Turnitin convention). Deciding which
  student copied is out of scope.
* `--seed` only affects dataset generation. The engine has no randomness, and two
  runs give identical reports (`tests/test_determinism.py`).
* *Future work*: corpus-relative σ calibration of thresholds (per-challenge
  baselines) could replace the fixed constants.

## Known limitations

* **Garbled transcripts fail silently.** At WER 0.66 the cross-modal copy was
  invisible and the reports came out clean (failure study 1). There is no ASR
  confidence warning yet.
* **The AI signal is unreliable on polished business prose.** Labels on the corpus
  sit within ±1 point of the band edge, and the classifier scores ≥ 0.5 on a third
  of real human Wikipedia intros (failure study 2).
* **Heavy paraphrase is under-scored.** 03 is detected at 62 % coverage but misses
  the HIGH threshold. Sentences rewritten below cosine 0.65 are not aligned.
* **No OCR**: scanned PDFs are reported as `empty`. No cross-language matching.
* **Word order:** MiniLM cannot distinguish "A acquired B" from "B acquired A"
  (PAWS). The engine relies on multi-sentence runs and human review.
* **Web tier:** free DuckDuckGo results are noisy and rate-limited. It is a
  demonstration, not a production source index.

## Repository layout

```
run_integrity_check.py      CLI entry point
integrity_engine/           cli · config · types · ingestion/{pdf,docx,mp4} · text_processing
                            embeddings · indexing · alignment · external_check · ai_detection
                            scoring · reporting · evaluation · metrics · dataset_generator · corpus_content
reference_kb/kb_texts.py    local reference knowledge base (6 passages)
config/default.toml         every threshold, with the reason for its value
sample_submissions/         14 files + dataset_manifest.json (ground truth)
benchmarks/                 public-dataset benchmarks (see benchmarks/README.md); results/*.json committed
tests/                      pytest: scenarios S1–S5, corrupted inputs, core math, determinism
docs/                       research notes, design spec, LLM provenance record for doc 06
scripts/                    setup_env.sh, check_env.py
```

`python -m pytest tests -q` runs all 34 tests in about 100 s; models must be cached.
`ruff check .` passes with the config in `ruff.toml`.

## AI-assistant disclosure

This project was built with heavy use of AI coding assistants, as the assignment
allows if disclosed:

* **Research, spec and code.** An AI agent workflow (research agent → spec writer →
  engineering agent, run inside Kiro / OpenCode with Claude and GLM models) produced
  `docs/01_research_notes.md` and `docs/02_SPEC.md`, and wrote most of the code,
  tests, benchmarks and documentation. A human reviewed the spec and requested the
  five changes listed in the spec history (MP4 count, a genuine LLM document, PDF
  count, AI-detector honesty, de-scoping rules).
* **Sample data.** Doc 06 was generated by GLM on purpose, as the corpus' declared
  LLM document. The other synthetic documents were also drafted by the assistant,
  which is why they are not used as human ground truth (Assumption A12).
* **Numbers.** Every figure in EVALUATION.md and this README comes from a script in
  this repository (`--eval` or `benchmarks/`). None was written by hand or by a
  model without being measured.

I am prepared to explain every design decision and line of code in a follow-up
interview.
