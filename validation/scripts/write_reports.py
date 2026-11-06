"""Render the validation narrative from preserved measurements; no production changes."""
from pathlib import Path
import json
V=Path(__file__).resolve().parents[1];R=V.parent
x=json.loads((V/'results.json').read_text());f=json.loads((V/'freeze.json').read_text());d=json.loads((V/'diagnostics.json').read_text());m=json.loads((V/'manifest.json').read_text());split=json.loads((V/'split_audit.json').read_text())
commit=f['commit'];met=x['metrics_any_reported_evidence'];summaries={k:json.loads((V/f'runs/{k}/summary/latest.json').read_text()) for k in ('main','edge','semantic')}
import re
# Reports live in docs/validation/ since the docs restructure; repo-root links get ../../
def write(name,s): (R/'docs/validation'/name).write_text(re.sub(r'\]\((validation/|improvements/|AI%20Engineer%20Assignment\.pdf|EVALUATION\.md|README\.md)',r'](../../\1',s.strip())+'\n',encoding='utf-8')
write('ASSIGNMENT_VALIDATION.md',f'''# Assignment validation

Validated commit `{commit}` on 2026-11-06. **Estimated score: 74/100; closest tier: Meets Expectations.** The prototype is modular, handles malformed inputs and reports real evidence, but two mandatory matching scenarios fail on the new corpus. The existing sample suite passes. This is an independent engineering estimate, not an official hiring decision.

The authority is [AI Engineer Assignment.pdf](AI%20Engineer%20Assignment.pdf), particularly sections 4–9. Its task weights (20/25/20/20/15) differ from its final scoring rubric; the estimate below uses the requested final rubric (25/20/20/15/10/10). Internal `docs/02_SPEC.md` choices such as a 2 GB memory target are not additional assignment requirements. No interview, portal submission, private repository access, 72-hour deadline, or actual reviewer completion time can be verified locally.

## Scope and preservation

Production code, configuration, README, EVALUATION, samples, original reports and saved benchmarks were frozen before new fixtures were made. [validation/freeze.json](validation/freeze.json) records the commit, initial clean Git state, hashes and model snapshots; [preservation_check.json](validation/preservation_check.json) checks 92 original files unchanged. New work is confined to `validation/` and these four new Markdown reports. The engine ran with default config, cached original models and `--skip-web`; local KB remained active. `--eval` was deliberately avoided because its default overwrites EVALUATION.md. Existing tests write their new reports under `validation/runs/pytest_tmp/`.

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
''')
# Data-driven main results and per-file observations.
perfile='\n'.join(f"| `{r['file']}` | {r['status']} | {r['peer_pct']:.1f} | {r['external_pct']:.1f} | {r['ai_pct'] if r['ai_pct'] is not None else 'n/a'} / {r['ai_label']} | {r['risk']} | {r['ingestion_s_from_pipeline_log']} | {len(r['warnings'])} |" for r in x['per_file'])
positives='\n'.join(f"| {' ↔ '.join(r['files'])} | {r['expected_relationship']} | {r['doc_pair_score']:.4f} | {', '.join(r['detected_relationships']) or 'none'} | {' / '.join(r['risks'])} | {r['status']} |" for r in x['pairs'] if r['expected_relationship']!='negative')
runtimes='\n'.join(f"| {k} | {s['aggregate']['n_docs']} | {s['runtime']['elapsed_s']} | {s['runtime']['peak_rss_mb']} | 0 |" for k,s in summaries.items())
wer='\n'.join(f"| {r['file']} | {r['duration_s']} | {100*r['wer']:.1f}% | {r['segments']} |" for r in x['asr'])
write('INDEPENDENT_EVALUATION.md',f'''# Independent evaluation

**The frozen engine detected 1 of 3 intended positive pairs: TP=1, FP=0, FN=2, TN=75. Precision 100%, recall 33.3%, F1 50.0%.** The single success is a byte-identical exact-copy pair. S1 and S2 fail; S3, S4 and S5 pass. These 13 purpose-selected submissions are an assignment coverage test, not a representative estimate of plagiarism prevalence or production accuracy.

Commit `{commit}`; models and dependencies in [freeze.json](validation/freeze.json) and [dependencies.txt](validation/dependencies.txt). All thresholds/scoring/embeddings unchanged. [manifest.json](validation/manifest.json) records labels, fixture hashes, authentic versus synthetic authorship and expected source locations before engine execution. No failing case was removed or rewritten after results. The separate two-file negation corpus was also built before execution. Three-copy and external transport controls are later **exploratory component tests**, excluded from primary metrics.

## Corpus and provenance

5 PDF, 5 DOCX and 3 MP4. Video durations are 15.00, 12.00 and 10.52 s (the 10.5 s requested cut occupies whole video frames). All have genuine English speech. `01` uses John F Kennedy's 1961 address on page 2; `02/03` use Lewis Carroll's *Alice's Adventures in Wonderland*. These human works differ from HC3/MRPC and the existing business fixtures. The three audio cuts are non-overlapping sections of **one archival recording and one speaker**, packaged over static generated video; they are not three independent speakers. Seven library/policy documents and the speech PDF cover were written by this AI evaluation assistant. They are **synthetic**, including independently composed library answers and intentional paraphrasing; they are not authentic human student work.

Downloads succeeded using curl after requests received HTTP 403 for the audio. The human recording is marked US federal public domain by its [Commons source/license page](https://commons.wikimedia.org/wiki/File:JFK_inaugural_address.ogg). Human text sources: [Kennedy address](https://www.gutenberg.org/ebooks/3) and [Carroll text](https://www.gutenberg.org/ebooks/11). Full fetched files and Gutenberg license notices are preserved in `validation/sources/`; [downloads.json](validation/sources/downloads.json) records URLs, byte counts and SHA-256. No modern translation or private student material is used. Original author deaths/publication ages and the US-government recording support this use; this is not a legal opinion on every jurisdiction. A third text, *The Time Machine*, was downloaded during preparation but not used in any scored submission.

Speech cut locations were prepared from the archival transcript plus a separate cached Whisper **base** alignment of the source audio. The evaluated engine remained on Whisper **tiny**. This preparation alignment is not independent human timestamp annotation; it locates broad clip boundaries only. Reference wording comes from the published human speech. No listening study or word-by-word human timestamp labeling was performed.

## Fixed measurement protocol

The unit is an **unordered file pair**, all C(13,2)=78 retained. Three positives are exact copy, intentional paraphrase and cross-modal copy; all other 75 pairs are negative for substantive peer reuse. Shared authorised prompt evidence is excluded. Primary detection means at least one `match_type=overlap` passage in either file's JSON; this follows the original corpus evaluator and shipped 0.05 evidence reporting floor. Secondary metrics use document-pair coverage ≥0.20, the pre-existing benchmark/risk threshold; they are identical here. No threshold sweep or tuning was done on these fixtures. Passage cosine and document coverage are different quantities.

| Expected / detected | Positive | Negative |
|---|---:|---:|
| Positive | 1 TP | 2 FN |
| Negative | 0 FP | 75 TN |

Precision = 1/(1+0) = **100%**; recall = 1/(1+2) = **33.3%**; F1 = 2/(2+0+2) = **50.0%**. Pair AUC = **0.6667**, including absent evidence as score zero. Approximate Wilson 95% intervals: precision **20.7–100%**, recall **6.1–79.2%**. These binomial intervals are illustrative: constructed pairs share files and are not independent random draws. They do not establish statistical significance versus the original benchmark. Accuracy 76/78 would obscure the two important misses, so it is not the headline.

Within the library-topic group (04–10), all 20 negative pairs are correctly rejected, including all 10 pairs sharing the institutional prompt. This is useful same-topic coverage, but remains evaluator-authored and curated. No score is a finding of plagiarism misconduct; the authentic exact copy and speech reuse are deliberately authorised tests.

## Positive-case results and exact evidence

| Pair | Expected | Coverage score | Detected label | Risks | Outcome |
|---|---|---:|---|---|---|
{positives}

**Exact copy (PASS):** both JSON reports quote the exact extracted Alice passage, cosine 1.0000, query_page=1 and matched_page=1. The beginning is “Alice was beginning to get very tired of sitting by her sister on the bank”; it ends with the White Rabbit running close by her. Both files have identical bytes, so this success also exercises the engine's duplicate shortcut. It should not be presented as evidence of difficult semantic retrieval. Full exact text and both directions are preserved in [results.json](validation/results.json), not truncated to this excerpt.

**S1 (FAIL):** expected source is PDF page **2**, spoken clip local interval **0–15 s**, archival audio offset **51–66 s**. Published source sentence:

> The world is very different now, for man holds in his mortal hands the power to abolish all forms of human poverty and all forms of human life.

Tiny transcript:

> The world is very different now for man holds it is mortal hands the power to abolish all forms of human poverty and all forms of human life

No detected passage, PDF page reference or video match timestamp was emitted. Both reports say peer 0%, LOW and no warnings. Thus **matching page/timestamp accuracy cannot pass**; there is no reported match to localize. Extraction itself correctly retains the source on p2 and segment times [0.18,6.02], [6.78,14.86], within 0.18/0.14 s of the broad clip endpoints. These extraction boundaries are not measured word-accurate citation localization.

Diagnosis: the punctuation-free PDF heading joins the source sentence; the transcript is split at ASR segment boundaries. That PDF unit has cosine 0.4942 and 0.6004 against the two transcript units; neither survives the shipped cosine/LCS alignment rule. The transcript WER is only 7.1%, so this failure cannot be dismissed as wholly unintelligible speech. Evidence in [diagnostics.json](validation/diagnostics.json) records the full sentence matrix; no remedy was applied.

**S2 (FAIL):** both DOCX texts communicate the same unusual argument: wiping fines clears accounting debt by definition and is not proof of punctual returns; measure new-loan lateness, use a neighbouring unchanged council in the same weeks to address seasonality, hold lost-item charges constant, and protect access for former debtors. Lexical Jaccard **0.0593**. Whole-document cosine **0.6352**, maximum sentence cosine **0.5628**; no accepted aligned sentences and coverage zero. The assignment's illustrative >0.80 semantic band is not reached. Both MEDIUM risks come from the AI signal and do not identify the counterpart or prove successful paraphrase retrieval. Full input text, sentence matrix and evidence absence are preserved.

## Every main submission

| File | Status | Peer % | External % | AI score / band | Risk | Ingest seconds | Warnings |
|---|---|---:|---:|---|---|---:|---:|
{perfile}

All main statuses are ok and warnings empty; offline cached-model notices and one torch deprecation warning appear in stderr, outside per-document warnings. Ingestion timings above are rounded original pipeline log values: 0.0 does not mean zero elapsed work. The engine does not instrument total per-file embedding/AI/report runtime; shared-stage and batch timings are recorded instead. A per-file end-to-end duration is therefore **unavailable**, not estimated or fabricated.

## S3, S4, S5 and runtime

S3 separate run: empty `20_empty.pdf` and corrupt `21_corrupt.pdf` → error with one warning each; `22_silent.mp4` → silent_video with “no audio stream in MP4”; `05_valid_control.pdf` → ok. Four reports of each format and full summary; exit 0. Degraded files receive LOW plus explicit status/warnings/INSUFFICIENT_TEXT; that is absence of analysis, not evidence of originality.

S4: exactly 150 shared prompt words across five files (04,05,08,09,10) with separately composed answers. Every peer score is zero; each lists four common-source partners and a template registry entry. All are MEDIUM only because the text really is AI-authored. None is heavily flagged by the prompt.

S5: main run emits 13 JSON + 13 Markdown reports, aggregate JSON/Markdown and a 13×13 conceptual matrix with 156 off-diagonal values; self diagonal is omitted. Evidence strings for every reported overlap are exact substrings of extracted query/source text, pages correct in the detected Alice pair; no self matches. All 78 pair outcomes are machine-readable in `validation/results.json`, including 75 TN and both FN.

| Run | Files | Engine wall seconds | Peak RSS MiB | Exit |
|---|---:|---:|---:|---:|
{runtimes}

Runtime is measured on the supplied 12-logical-CPU machine with warm model caches. It is not directly comparable to the longer original corpus or a cold install. Full subprocess wall times, exact CLI commands, stage timings and stdout/stderr are in `validation/runs/*/`. Peak RSS is process-reported, not whole-machine memory.

| Human clip | MP4 duration s | WER | ASR segment seconds |
|---|---:|---:|---|
{wer}

WER uses the existing benchmark's lowercased alphanumeric tokenization against published reference text; substitutions/insertions/deletions all count. These snippets are short and one speaker; do not extrapolate to all real speech. All three clips are too short for the shipped 50-word AI gate, which correctly reports insufficient text.

## Supplemental checks and limits

- **Negation probe:** deliberate controlled reversal of three factual claims; coverage 1.0, passage cosine 0.9624, sentence cosines 0.8909/0.8464/0.7218, HIGH on both. This is a true overlap positive because the text was intentionally transformed, but a **semantic false positive**: the note says “meaning matches while wording differs” although each proposition is reversed. Not included as a plagiarism FP in primary metrics. See `validation/runs/semantic/`.
- **Three-copy component probe:** distinctive flood-basin analysis copied into three different nonidentical text documents becomes three template chunks; all three pairs have zero directional coverage. This tests untrusted recurrence being mistaken for authorised boilerplate. Component-level evidence is exploratory; no full CLI or AI risk measurement is claimed for it.
- **Local KB:** exact synthetic KB positive control produces 100%, eight source hits; mechanism pass, not a web-scale test. Original live web transport returned real search results; deterministic failure control returns no hits and mock=true. Main runs stayed offline from web scoring.
- **AI:** 7/7 known AI documents AI_LIKELY; 1/1 unique pure-human literary work HUMAN. Duplicate Alice excluded as an independent human; speech PDF excluded for mixed authorship. Bands/uncertainty/fallback provenance work, but one human negative cannot validate false-positive rate or calibrated probabilities. No fresh human library essays were available.

No independent LOW/MEDIUM/HIGH “ground truth” was frozen, because similarity evidence and academic sanctions are different questions. Accordingly, this audit does **not** invent a risk accuracy comparable to the original 85.7%; it reports every actual risk and retrieval confusion separately. All statements about source copying refer to intentional fixture construction, not accusations.

Reproduction commands and source details: [validation/README.md](validation/README.md). Production and original result preservation: [preservation_check.json](validation/preservation_check.json).
''')
write('OVERFITTING_ANALYSIS.md',f'''# Overfitting and benchmark selection analysis

**Benchmark-dependent algorithm/threshold selection is proven; intentional memorization and pretrained model train/test contamination are not proven.** The frozen engine fails two deliberately different independent positive cases. That establishes limits on these examples, not a measured population-wide generalization gap. The original public benchmarks are useful development evidence and should not be described as untouched held-out plagiarism performance.

## What can be established

| Audit question | Finding | Strength / evidence |
|---|---|---|
| Did evaluation data influence algorithms or thresholds? | Yes. Sentence-evidence coverage replaced chunk aggregation after AUC comparisons; MEDIUM peer threshold moved 40→20%; evidence-run rules were selected using retrieval/MRPC behavior. | **Confirmed** in EVALUATION “Thresholds used”, config comments, scoring docstring and Git history. EVALUATION already admits no separate held-out plagiarism corpus. |
| Do calibration and AI test splits overlap? | Reconstructed 45 calibration and 105 test question IDs have zero overlap; normalized complete human/AI texts also have zero overlap. | **Confirmed for these splits**, [split_audit.json](validation/split_audit.json). No assertion that all near-duplicate passages/topics are disjoint. |
| Does benchmark source material recur across evaluations? | Retrieval shares 28 normalized answer segments with AI calibration texts and 62 with AI test texts. Its paraphrases come from MRPC test, which also supplies the sentence benchmark. | **Confirmed source reuse**, not automatically training leakage between different tasks. No independent retrieval holdout exists. |
| Did pretrained models train on the tests? | No local model training is present. Full pretrained training corpora cannot be reconstructed here. | **Unknown**, not evidence of cleanliness or of contamination. The classifier's public model card does not give a complete training manifest. |
| Are synthetic positive samples easy? | Forty exact-copy pairs inject an entire answer; forty paraphrase pairs concatenate four positive MRPC pairs, inserted in similarly constructed three-answer documents. Sample MP4s are synthetic TTS and were revised after a failed speech test. | **Confirmed construction**; artificial structure and repeated multiple-sentence evidence favor the selected rule. “Too easy for all systems” is not formally established. |
| Are negatives mostly unrelated? | 80 labelled positives among 79,800 pairs; 79,720 negatives from documents assembled out of shuffled human Q&A answers across finance/open_qa/wiki_csai. There is no curated same-prompt/contradiction negative stratum in the retrieval benchmark. | **Construction bias confirmed**; the exact fraction of semantically unrelated pairs was not human-labelled. It is reasonable to infer many are easy negatives, not to assign an invented percentage. |
| Are thresholds optimized for a particular dataset? | Retrieval score sweep includes .05/.1/.2/.3/.4/.7; .2 and run rules selected with the same corpus in view. External .70 config comment explicitly compares doc04 to next-best sample cosine .674. HC3 logistic constants fitted only on calibration. | **Confirmed threshold provenance**; legitimate calibration for HC3 is separate from same-benchmark method selection. No quantitative “slightly optimistic” correction can be estimated. |
| Does reported performance generalize? | New exact copy found; new speech/PDF and DOCX paraphrase missed; boilerplate/degraded handling works. | **Observed counterexamples**, insufficient for a broad real-world precision/recall claim. |

No “training set” should be invented for retrieval: it is a fixed pretrained embedding system plus hand-selected rules. Its relevant contamination is **evaluation feedback used for method selection**, rather than a fitted local model memorizing labels.

## Original numbers versus independent measurements

| Quantity | Original reported | Independent audit |
|---|---:|---|
| Retrieval precision at coverage ≥.20 | 96.2% | Main corpus 100%, based on only one TP and zero FP; not evidence of better precision. |
| Retrieval recall | 93.8% | 33.3% (1/3); one exact duplicate succeeds, different speech/paraphrase fail. |
| Retrieval F1 | 94.9% | 50.0%; different purpose-selected data and small denominator. |
| Document-pair AUC | ≈1.0 (0.99998) | 0.6667 on all 78 new pairs, ties at zero receive half credit. |
| Original sample risk accuracy | 85.7% (12/14) | Recomputed from original preserved reports. No new risk accuracy fabricated; independent expected sanctions were not defined. |
| AI held-out HC3 | P/R ≈99.0% | Recomputed from cached components: TP104/FP1/FN1/TN104. Not fresh model inference. Independent AI sample has only one unique human work. |

The 400-document run is rerun into `validation/runs/retrieval_reproduction/`, preserving original benchmark files. The rerun exactly reproduces the saved threshold sweep and AUC: at .20, precision 0.961538, recall 0.937500, F1 0.949367, 3 FP (75 TP, 5 FN); AUC 0.9999854177. Recorded stages total 146.34 s; runtime is not expected to match across runs. These figures validate reproduction, not independence. Unlike the main corpus, that script operates on in-memory `ProcessedDocument`s: it does not evaluate PDF/DOCX container extraction, MP4 ASR, external search, fused risk or report localization. Its document AUC cannot substantiate those capabilities.

AUC is insensitive to class prevalence alone. The concern is **negative difficulty/composition**, not simply the presence of 79,720 negatives. At .20, the original saved figures correspond to 75 TP, 3 FP and 5 FN; almost all zero-score negatives make ranking easy. Those are real results on that benchmark and not proof of fraudulent arithmetic. The source deduplication also drops answers sharing a six-word sentence, removing real repeated-content complications; this improves label validity but narrows the test distribution.

## Calibration and model exposure

The HC3 code splits by question before fitting perplexity/burstiness curves; both answers stay together. This is the right local separation, and the zero-overlap reconstruction supports it. It does not prove the pretrained detector never saw HC3, nor that repeated observations of test results did not influence higher-level design. Cache entries use question/answer keys without text, revision or config fingerprints, making future changed datasets/seed workflows vulnerable to stale cache reuse. No stale-output mismatch was established in this session.

The detector's [model card](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small) confirms a DeBERTa-v3-large architecture, four classes and 1−P(human) score; it does not establish a full training-data separation from HC3. MiniLM's [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) describes broad pretraining/fine-tuning sources and a 256-wordpiece limit. Familiar public-domain text can also have appeared in pretrained language-model corpora; “independent from this repository's development corpus” does not mean “guaranteed unseen in pretraining”.

Original Wikipedia OOD evidence is informative but not a random held-out sample of student writing: 30 old business intros versus 10 machine-written sample files. It already reveals 11/30 human UNSURE and 10/30 classifier scores above .5. AI template answers cluster near the 65 band edge. That demonstrates domain-sensitive fusion; it does not validate the reported percentages as calibrated probabilities.

## Confirmed robustness gaps versus possible causes

**Confirmed behavior:** independent low-lexical paraphrase retrieval fails; human speech fragmented into ASR units misses despite a substantially correct transcript; untrusted three-document repetition suppresses distinctive copied analysis; negation is called same-meaning paraphrase; 10k implementation claims omit a quadratic document loop and output matrix.

**Plausible causes/bias:** evidence rules selected for concatenated MRPC/HC3 documents favor that construction; synthetic natural-pace TTS understates real ASR boundary/noise variation; broad unrelated-topic negatives make near-perfect all-pair AUC less diagnostic; external floor has sample-aware selection. These are supported explanations, not experimentally isolated causal attributions.

**Not proven:** deliberate cheating, exact training-set memorization, model fine-tuning on independent files, hidden source-specific lookup logic, or a population-sized numerical overfitting penalty. There is no evidence to allege those. The authors explicitly disclose benchmark-driven selection and synthetic data, which is good evaluation practice; the remaining error is treating development results as stronger generalization evidence than they warrant.

## Recommendations for future evaluation only

Freeze revisions, rules and evaluation labels before a larger external test. Group splits by underlying source/prompt/author, not merely by rendered file. Include legitimate same-topic independent answers, meaning reversals, short and patchwork borrowing, multiple real speakers, headings/tables/ASR fragmentation and ≥3-way copied analysis. Report per-relation recall, hard-negative strata, passage localization and extraction failures separately from AI/risk. Keep any future calibration split outside the final held-out set, and avoid choosing another threshold using the files in this audit. This corpus is now a regression/development corpus and cannot become a fresh test again after fixes.
''')
write('VALIDATION_ISSUES.md',f'''# Validation issues

Scope: commit `{commit}`; production changes are **not made**. Recommendations below are proposed future work and require a new independent test set after implementation. Evidence resides under `validation/`; original reports remain unchanged.

## Confirmed defects and behavioral limitations

| ID | Severity | Finding | Reproduction / evidence | Recommended fix |
|---|---|---|---|---|
| V01 | High | Genuine speech copy produces peer 0, LOW, no evidence/warning. | Main `01_human_speech.pdf` p2 ↔ `11_human_crossmodal.mp4`; published sentence almost intact in transcript, WER .0714. `validation/diagnostics.json`: heading merged with PDF sentence; two ASR units, cos .4942/.6004. `alignment.py:537`, text segmentation/ingestion metadata. | Preserve heading boundaries and align sentences across ASR segments, including one-to-many spans; expose transcript confidence and distinguish no evidence from a reliable negative. Verify against new speakers/files; do not tune to this sole clip. |
| V02 | High | Distinctive DOCX argument reworded with low lexical overlap is missed entirely. | Main 06↔07, Jaccard .0593, max sentence cos .5628, doc cosine .6352; no partner reported, risk MEDIUM solely from AI. | Evaluate alternative candidate/evidence methods on a calibration corpus; use stronger semantic verification where justified. Simply lowering the cosine floor has known precision costs. |
| V03 | High | Copying shared by three nonidentical submissions can be automatically hidden as boilerplate. | Exploratory in-memory trio in `diagnose.py`/`diagnostics.json`: three copied flood-basin sentences + different tails; 3 template chunks, all three pair coverages 0. `alignment.py:108/172/537`. | Trust official prompt/source registration separately from untrusted submission recurrence. Keep recurring unknown analysis visible for review instead of treating recurrence as authorization. |
| V04 | Medium | Reversed claims are falsely described as same-meaning paraphrase. | `validation/runs/semantic`: arsenic removal, below-limit result and recommendation all negated; coverage1, cosine .9624, both HIGH; note says “meaning matches”. | Use neutral overlap language unless entailment/contradiction has been checked; preserve factual differences. Current retrieval is legitimately detecting textual reuse, so this is not counted as a plagiarism FP. |
| V05 | Medium | A symmetric pair score is described as if it measures the queried document's copied share. | `alignment.py:741` uses max(forward,reverse); `_peer_evidence` uses best pair. A small fully shared source can yield 100% on a much longer file, even when its own covered fraction is small. | Report directional coverage separately and label the symmetric score as pair similarity. This is a documented design convention with misleading report semantics, not a broken cosine calculation. |
| V06 | Medium | Current implementation retains a quadratic document loop despite README's “no N² loop” candidate-stage claim. | `indexing.py:135–136` visits every directed file pair before sentence candidate filtering; `cli.py` also allocates dense off-diagonal zeros; summary writer emits the dense Markdown matrix. | Iterate only retrieved candidate doc pairs; use sparse matrix output/overview for large batches. Benchmark end-to-end, including memory and serialization, before costing 10k jobs. |
| V07 | Medium | Query location missing from external evidence. | `ExternalHit` and report serialization retain text/title/URL/cosine but no query page or timestamp (`types.py`, `reporting.py:100`). | Retain query spans and PDF/ASR provenance on external hits, and source metadata where available. Local KB IDs are cross-checkable in source but not a citation to an authentic publication. |
| V08 | Low | Some report wording reflects obsolete algorithms. | `reporting.py:169` calls re-embedded passage cosine “chunk cosine”; summary blank legend says chunk floor despite sentence-evidence scoring; several types/index docstrings describe earlier chunk scores. | Update labels/documentation to the actual coverage and passage-cosine definitions. Keep missing, discounted and rejected evidence distinct. |
| V09 | Medium | Poor or incomplete transcript can yield a clean-looking negative without quality warning. | New S1 has no warnings despite failed expected detection; clip13 WER .3333, status ok, no quality caveat. Only language/VAD/budget failures warn. | Include ASR confidence/coverage and explicit review state; avoid presenting LOW as a reliable all-clear for unvalidated ASR. |

## Evaluation and reproducibility limitations

| ID | Severity | Limitation | Evidence and recommendation |
|---|---|---|---|
| E01 | High | Reported retrieval metrics are development-set performance. | Method and .20 threshold chosen using same 400 docs; no separate plagiarism holdout. Keep these figures, label selection honestly, reserve a new source-group holdout. Independent P100/R33.3/F1 50 is also tiny, not a substitute production estimate. |
| E02 | Medium | AI generalization and calibration are not established. | HC3 locally separated; pretrained exposure unknown. Original OOD prose shows uncertain human texts and score compression near 65. New sample has one unique pure-human literary work, seven evaluator-generated positives. Add independently sourced in-domain human texts, report abstention, validate operating point on separate calibration. |
| E03 | Medium | Original risk accuracy has inconsistent truth semantics. | Original manifest treats some known AI-authored “originals” as LOW, while shipped AI alone gives MEDIUM; 05 is a reported mismatch but not evidence of a human AI false positive. Separate reuse, AI provenance and review policy labels. Do not define misconduct from text alone. |
| E04 | Medium | Full setup from an empty environment is unverified. | Supplied venv passes `pip check`, Ruff and all 34 tests. Original top-level pins exist, but transitive resolution and model `main` are not locked. Record model snapshot revisions and hashes, transitive lock and verify a clean install in a separate environment. No install failure is alleged here. |
| E05 | Medium | Scaling figures are projections, with inconsistent workload assumptions. | README uses 8 chunks/75 sentences per 1,500-word doc; older EVALUATION trade-off prose uses ~3 chunks/30 sentences. GPU speed-up, worker scaling and cloud prices are not newly measured. Task count/cost reasoning must include quadratic current stages, IO and shared overheads. |
| E06 | Low | Original S4 demonstrates three shared-prompt files, not specified five. | Original tests explicitly expect two common partners. New five-document test passes, supporting implementation capability while exposing original coverage shortfall. |
| E07 | Low | Benchmark component cache lacks data/model fingerprinting. | `bench_ai_detection.score_texts` uses qid/answer keys only. Changing source files or model revisions may silently reuse stale scores. Hash dataset text, preprocessing, model and relevant settings in cache metadata. Current reconstructed metrics agree; no present stale-cache error demonstrated. |
| E08 | Low | No full per-file duration or validated reviewer-time measurement. | CLI times shared stages and file ingestion; not all per-file inference. Report known stage/batch timing, instrument future performance evaluations, and time actual human citation checks. |

## Checks that passed and boundaries

34 original pytest tests passed (one torch deprecation warning), original-code Ruff passed and installed requirements are consistent. Main CLI processed 13/13; edge CLI processed all four inputs; exact-copy pages and text are correct; all five prompt-sharing answers have peer0 and no HIGH; local KB positive control gives100%; live web transport returns real results; unavailable web transport control discloses an empty mock. These successes should be retained alongside failures.

LOW on corrupted/silent files is paired with explicit failure status and warning. It is a report-consumer risk if a caller filters only the risk field, not a claim that the extractor silently succeeded. OCR, translated plagiarism, hours of audio and a production frontend are not required by this assignment; their absence is not scored as a mandatory defect.

The new corpus is mostly synthetic document writing plus two authentic human works and one archival speaker. It cannot prove real-world performance, diagnose every causal contribution or establish classifier training contamination. The canonical document renderer failed because the supplied runtime lacks `pdf2image`; fixtures were instead converted by installed LibreOffice, rendered with Poppler and all 11 main document pages visually inspected. Artifact marker helper was absent. These are evaluation-host tooling limitations, not project runtime defects. Existing venv authoring packages were used without installing or changing production dependencies.

All 92 frozen original files and every frozen main/edge/semantic fixture hash are checked in `validation/preservation_check.json`. No thresholds, production code, original reports or benchmark results were modified.
''')
print('Wrote four reports')
