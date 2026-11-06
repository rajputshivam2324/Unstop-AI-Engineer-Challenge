# Assignment validation

Validated commit `e15af35db37afafa9c6957c4e515c769dd08c38a` on 2026-11-06. **Estimated score: 74/100; closest tier: Meets Expectations.** The prototype is modular, handles malformed inputs and reports real evidence, but two mandatory matching scenarios fail on the new corpus. The existing sample suite passes. This is an independent engineering estimate, not an official hiring decision.

The authority is [AI Engineer Assignment.pdf](../../AI%20Engineer%20Assignment.pdf), particularly sections 4–9. Its task weights (20/25/20/20/15) differ from its final scoring rubric; the estimate below uses the requested final rubric (25/20/20/15/10/10). Internal `docs/02_SPEC.md` choices such as a 2 GB memory target are not additional assignment requirements. No interview, portal submission, private repository access, 72-hour deadline, or actual reviewer completion time can be verified locally.

## Scope and preservation

Production code, configuration, README, EVALUATION, samples, original reports and saved benchmarks were frozen before new fixtures were made. [validation/freeze.json](../../validation/freeze.json) records the commit, initial clean Git state, hashes and model snapshots; [preservation_check.json](../../validation/preservation_check.json) checks 92 original files unchanged. New work is confined to `validation/` and these four new Markdown reports. The engine ran with default config, cached original models and `--skip-web`; local KB remained active. `--eval` was deliberately avoided because its default overwrites EVALUATION.md. Existing tests write their new reports under `validation/runs/pytest_tmp/`.

PASS means verified for the stated scope; PARTIAL means a deliverable exists with a verified gap or a material untested limitation; FAIL means an explicit test expectation was missed. A pass is not a population-level accuracy claim.

## Five mandatory tasks

| Task / requirement | Status | Evidence and qualification |
|---|---|---|
| T1: PDF extraction, pages | PASS | All five new text PDFs status `ok`; source page 2 preserved in `validation/extraction.json`; pypdf page walk in `ingestion/pdf.py`. Image-only handling is warning/empty, not OCR, as allowed by graceful handling. |
| T1: DOCX extraction and order | PASS | Five new DOCX status `ok`; `ingestion/docx.py` walks paragraphs and tables in XML order. All paragraph fixture text was retained. Nested tables, floating text boxes and DOCX header/footer extraction were not independently exercised. |
| T1: MP4 audio and ASR metadata | PASS | Genuine speech in all three 10–15 s clips extracted; tiny/int8; segment times stored. WER 7.1%, 5.6%, 33.3% against archival reference text. These are three clips of one speaker, not an accent benchmark. |
| T1: normalization and overlapping chunks | PARTIAL | NFKC, ligatures, whitespace, dehyphenation and deterministic sentence splitting exist; 18 main chunks; PDF/DOCX cap nominally 200 word tokens, MP4 120 tokens/45 s with overlap. Regex words are not model wordpieces; tail merge can exceed 200. Heading and ASR segment boundaries break S1 matching. |
| T1: malformed/empty inputs | PASS | Separate edge run exits 0: empty PDF/error, corrupt PDF/error, no-audio MP4/silent_video, valid control/ok; all four JSON and MD reports. Existing corruption tests also pass. |
| T2: open dense embeddings and vector index | PASS | Original MiniLM-L6-v2, 384 dimensions, L2 normalization, FAISS IndexFlatIP; no replacement embeddings. Source and tests verify dot product/cosine math. |
| T2: self-match exclusion | PASS | No self evidence in independent reports; self matrix entries omitted; same-document chunk and sentence neighbours excluded. Tests exercise exclusions and duplicate identity. Fixed neighbour overfetch can reduce recall, not admit self evidence. |
| T2: internal peer percentage and matching | PARTIAL | Exact copy found, pair score 1.0. S1 and S2 both score zero. Final percentage is max of directional evidence coverage, not cosine and not necessarily share of the query file. Candidate restriction does not guarantee every document pair is exhaustively aligned. |
| T3: local external reference mechanism | PASS | Six local passages/34 sentence units; direct positive control returns 100% and eight exact source hits. New library/literature material returns 0%, consistent with this small KB's scope. This is a local reference demonstration, not broad internet coverage. |
| T3: optional web mechanism | PARTIAL | Live original `ddg_query` returned search results with titles, URLs and snippets in 1.96 s. Failure transport control disclosed mock with zero hits. Complete live document-to-web scoring/recall was not benchmarked. Assignment permits a local KB or documented fallback. |
| T3: AI estimate and uncertainty | PARTIAL | Classifier+GPT-2 models loaded, no fallback; all seven evaluator-authored AI files AI_LIKELY, one unique pure-human literary excerpt HUMAN. Mixed speech PDF UNSURE; all short clips INSUFFICIENT_TEXT. Too few independent human works to validate calibration. Percentages are uncalibrated composite scores. |
| T3: three distinct scores and risk | PASS | Required score keys present for every main and edge file; risk combines peer/external/AI without making AI alone HIGH. Unknown/unreadable files retain LOW plus explicit status/warnings; LOW is not clearance. |
| T4: exact passages and references | PARTIAL | Detected exact copy has exact extracted text and page 1 on both sides. Original sample cross-modal test verifies page 2 and timestamp fields. New S1 has no evidence or location output, because detection fails. Contradiction probe falsely says “meaning matches”. |
| T4: reviewer context | PARTIAL | Per-file verdict, source text, cosine and locations are present. JSON keeps full text; Markdown previews clip at 400 chars, and labels passage cosine “chunk cosine”. No timed human reviewer study establishes the 30-second target. External hits lack query page/timestamp fields. |
| T5: 10,000-document brief | PARTIAL | README covers CPU/GPU, asynchronous queues, storage, Flat/HNSW, latency, cost and bottlenecks; measured 60k-vector benchmark supports index trade-offs. Full 10k ingestion was not run. Current chunk aggregation has a quadratic document loop and summary has quadratic output size; production topology is a proposal. |

## Five mandatory scenarios

| Scenario | Status | Independent observation |
|---|---|---|
| S1 cross-modal duplication | FAIL | `01_human_speech.pdf` p2 ↔ `11_human_crossmodal.mp4` 0–15 s is a deliberate spoken copy. Both peer 0, LOW, no multimodal evidence, no warning. ASR recognizes nearly all words. Original synthetic S1 passes existing tests. |
| S2 reworded DOCX argument | FAIL | `06_policy_argument.docx` ↔ `07_policy_rewording.docx`: same distinctive causal-evaluation argument, lexical Jaccard 0.0593, document cosine 0.6352, maximum sentence cosine 0.5628; no peer match. Both MEDIUM solely from AI. A MEDIUM risk does not count as successful retrieval. |
| S3 corrupted / empty | PASS | Edge run processes 4/4, including all three degraded cases; specific warnings and reports, exit 0. |
| S4 five shared prompts and independent answers | PASS | Exactly 150 whitespace words shared across PDF04/PDF05/DOCX08/09/10; answers separately composed by evaluator, synthetic rather than human. All five peer 0, each has one template registry hit and four discounted common-source partners; MEDIUM is from known AI authorship, none HIGH. |
| S5 mixed batch | PASS | 13 submissions = 5 PDF + 5 DOCX + 3 MP4; 13 JSON, 13 MD, JSON and Markdown summaries. 13 matrix rows/156 off-diagonal entries, diagonal deliberately absent. |

## Deliverables and reproducibility

| Deliverable / check | Status | Evidence |
|---|---|---|
| Modular Python source and root CLI | PASS | `integrity_engine/`, `run_integrity_check.py`; three new CLI runs exit 0; all original modules unchanged. |
| Original sample files and ground truth | PARTIAL | Original 14-file sample and manifest present. 5 PDF/6 DOCX/3 MP4 satisfy format minima; 2 speech clips about 25–26 s and silent clip 12 s meet PDF's sample duration allowance. Original S4 has three documents, not five. Some original manifest AI/risk labels are inconsistent with disclosed machine authorship. |
| README: setup, methods, assumptions, evaluation | PASS | Substantial documented methodology, assumptions, explicit AI-assistant disclosure and failure analysis. Several precision/scale descriptions need correction, recorded separately. |
| Requirements, configuration, downloads | PARTIAL | Exact top-level package pins and setup/check scripts exist. `pip check` passes in supplied environment; installed dependency list recorded. Transitive dependencies and remote model revisions are not pinned by original setup; fresh empty-cache setup was not executed. Python 3.11 requirement is documented, within assignment's Python 3.9+ allowance. |
| EVALUATION: metrics and ≥2 failures, trade-offs | PASS | Original metrics recomputed from preserved reports; two substantial documented failures, threshold history and model/index trade-offs. Selection leakage limits generalization; this is not a missing evaluation. |
| Automated tests | PASS | **34 passed, 1 dependency deprecation warning, 107.61 s**; `validation/runs/pytest.log`. Tests demonstrate regression on original fixtures; independent S1/S2 misses show coverage limitations. |
| Style / environment sanity | PASS | Ruff checks original engine, tests, benchmarks, scripts and runner: all checks passed. `pip check`: no broken requirements. Neither substitutes for clean installation. |
| Source provenance | PARTIAL | Original corpus openly discloses synthetic authorship; KB passages are “public-domain-style”, not traceable original publications. Independent source URLs/licenses/hashes are recorded. Actual human competition submissions were unavailable and not used. |
| Submission portal / reviewer access / interview | PARTIAL | Outside local verification; not claimed complete. |

## Rubric estimate

| Actual criterion | Award / available | Evidence for deduction |
|---|---:|---|
| Correctness and core functionality | 21 / 25 | Good extraction, indexing math and exclusions; severe new semantic and boundary-related misses, ambiguous coverage interpretation. |
| Multimodal scenarios and edge cases | 11 / 20 | S3–S5 pass; independent S1/S2 fail, no match-quality warning; three-copy suppression in component probe. |
| Evaluation rigor | 14 / 20 | Systematic public benchmarks and honest failures; retrieval development/test reuse, artificial mixed-topic documents, only three sample positives. This new independent audit does not retroactively make the original benchmark held-out. |
| Explainability | 12 / 15 | Full evidence and provenance where detected; absent evidence for misses, wrong semantic assertion for negation, incomplete external locations. |
| Architecture and scaling reasoning | 7 / 10 | Useful concrete design and measured ANN comparisons; quadratic current paths and unmeasured end-to-end cost assumptions. |
| Code quality and documentation | 9 / 10 | Modular code, successful tests/lint, detailed docs; incomplete clean-install/model version reproducibility and stale descriptions. |
| **Total** | **74 / 100** | Judgment uncertainty approximately ±5 points; no mechanical tier cutoffs are supplied in the PDF. |

**Closest tier: Meets Expectations.** Extraction, reports, tests and multiple scenarios work, but the independent matching failures are more substantial than “minor edge-case misses”. This build does not demonstrate the flawless cross-modal behavior or production-ready scaling required for Exceptional. See [independent results](INDEPENDENT_EVALUATION.md), [benchmark audit](OVERFITTING_ANALYSIS.md), and [issues](VALIDATION_ISSUES.md).
