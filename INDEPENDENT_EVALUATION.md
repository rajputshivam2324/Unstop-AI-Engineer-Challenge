# Independent evaluation

**The frozen engine detected 1 of 3 intended positive pairs: TP=1, FP=0, FN=2, TN=75. Precision 100%, recall 33.3%, F1 50.0%.** The single success is a byte-identical exact-copy pair. S1 and S2 fail; S3, S4 and S5 pass. These 13 purpose-selected submissions are an assignment coverage test, not a representative estimate of plagiarism prevalence or production accuracy.

Commit `e15af35db37afafa9c6957c4e515c769dd08c38a`; models and dependencies in [freeze.json](validation/freeze.json) and [dependencies.txt](validation/dependencies.txt). All thresholds/scoring/embeddings unchanged. [manifest.json](validation/manifest.json) records labels, fixture hashes, authentic versus synthetic authorship and expected source locations before engine execution. No failing case was removed or rewritten after results. The separate two-file negation corpus was also built before execution. Three-copy and external transport controls are later **exploratory component tests**, excluded from primary metrics.

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
| 01_human_speech.pdf ↔ 11_human_crossmodal.mp4 | cross_modal | 0.0000 | none | LOW / LOW | FN |
| 02_alice_source.pdf ↔ 03_alice_copy.pdf | exact_copy | 1.0000 | verbatim | HIGH / HIGH | TP |
| 06_policy_argument.docx ↔ 07_policy_rewording.docx | paraphrase | 0.0000 | none | MEDIUM / MEDIUM | FN |

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
| `01_human_speech.pdf` | ok | 0.0 | 0.0 | 64.6 / UNSURE | LOW | 0.0 | 0 |
| `02_alice_source.pdf` | ok | 100.0 | 0.0 | 5.3 / HUMAN | HIGH | 0.0 | 0 |
| `03_alice_copy.pdf` | ok | 100.0 | 0.0 | 5.3 / HUMAN | HIGH | 0.0 | 0 |
| `04_library_hours.pdf` | ok | 0.0 | 0.0 | 65.7 / AI_LIKELY | MEDIUM | 0.0 | 0 |
| `05_library_storage.pdf` | ok | 0.0 | 0.0 | 65.6 / AI_LIKELY | MEDIUM | 0.0 | 0 |
| `06_policy_argument.docx` | ok | 0.0 | 0.0 | 65.9 / AI_LIKELY | MEDIUM | 0.0 | 0 |
| `07_policy_rewording.docx` | ok | 0.0 | 0.0 | 65.8 / AI_LIKELY | MEDIUM | 0.0 | 0 |
| `08_library_transport.docx` | ok | 0.0 | 0.0 | 65.7 / AI_LIKELY | MEDIUM | 0.0 | 0 |
| `09_library_privacy.docx` | ok | 0.0 | 0.0 | 66.0 / AI_LIKELY | MEDIUM | 0.0 | 0 |
| `10_library_access.docx` | ok | 0.0 | 0.0 | 66.0 / AI_LIKELY | MEDIUM | 0.0 | 0 |
| `11_human_crossmodal.mp4` | ok | 0.0 | 0.0 | n/a / INSUFFICIENT_TEXT | LOW | 2.5 | 0 |
| `12_human_allies.mp4` | ok | 0.0 | 0.0 | n/a / INSUFFICIENT_TEXT | LOW | 1.4 | 0 |
| `13_human_revolution.mp4` | ok | 0.0 | 0.0 | n/a / INSUFFICIENT_TEXT | LOW | 1.4 | 0 |

All main statuses are ok and warnings empty; offline cached-model notices and one torch deprecation warning appear in stderr, outside per-document warnings. Ingestion timings above are rounded original pipeline log values: 0.0 does not mean zero elapsed work. The engine does not instrument total per-file embedding/AI/report runtime; shared-stage and batch timings are recorded instead. A per-file end-to-end duration is therefore **unavailable**, not estimated or fabricated.

## S3, S4, S5 and runtime

S3 separate run: empty `20_empty.pdf` and corrupt `21_corrupt.pdf` → error with one warning each; `22_silent.mp4` → silent_video with “no audio stream in MP4”; `05_valid_control.pdf` → ok. Four reports of each format and full summary; exit 0. Degraded files receive LOW plus explicit status/warnings/INSUFFICIENT_TEXT; that is absence of analysis, not evidence of originality.

S4: exactly 150 shared prompt words across five files (04,05,08,09,10) with separately composed answers. Every peer score is zero; each lists four common-source partners and a template registry entry. All are MEDIUM only because the text really is AI-authored. None is heavily flagged by the prompt.

S5: main run emits 13 JSON + 13 Markdown reports, aggregate JSON/Markdown and a 13×13 conceptual matrix with 156 off-diagonal values; self diagonal is omitted. Evidence strings for every reported overlap are exact substrings of extracted query/source text, pages correct in the detected Alice pair; no self matches. All 78 pair outcomes are machine-readable in `validation/results.json`, including 75 TN and both FN.

| Run | Files | Engine wall seconds | Peak RSS MiB | Exit |
|---|---:|---:|---:|---:|
| main | 13 | 35.45 | 2042.1 | 0 |
| edge | 4 | 13.49 | 1909.5 | 0 |
| semantic | 2 | 6.31 | 605.7 | 0 |

Runtime is measured on the supplied 12-logical-CPU machine with warm model caches. It is not directly comparable to the longer original corpus or a cold install. Full subprocess wall times, exact CLI commands, stage timings and stdout/stderr are in `validation/runs/*/`. Peak RSS is process-reported, not whole-machine memory.

| Human clip | MP4 duration s | WER | ASR segment seconds |
|---|---:|---:|---|
| 11_human_crossmodal.mp4 | 15.0 | 7.1% | [[0.18, 6.02], [6.78, 14.86]] |
| 12_human_allies.mp4 | 12.0 | 5.6% | [[0.0, 11.38]] |
| 13_human_revolution.mp4 | 10.52 | 33.3% | [[0.46, 10.39]] |

WER uses the existing benchmark's lowercased alphanumeric tokenization against published reference text; substitutions/insertions/deletions all count. These snippets are short and one speaker; do not extrapolate to all real speech. All three clips are too short for the shipped 50-word AI gate, which correctly reports insufficient text.

## Supplemental checks and limits

- **Negation probe:** deliberate controlled reversal of three factual claims; coverage 1.0, passage cosine 0.9624, sentence cosines 0.8909/0.8464/0.7218, HIGH on both. This is a true overlap positive because the text was intentionally transformed, but a **semantic false positive**: the note says “meaning matches while wording differs” although each proposition is reversed. Not included as a plagiarism FP in primary metrics. See `validation/runs/semantic/`.
- **Three-copy component probe:** distinctive flood-basin analysis copied into three different nonidentical text documents becomes three template chunks; all three pairs have zero directional coverage. This tests untrusted recurrence being mistaken for authorised boilerplate. Component-level evidence is exploratory; no full CLI or AI risk measurement is claimed for it.
- **Local KB:** exact synthetic KB positive control produces 100%, eight source hits; mechanism pass, not a web-scale test. Original live web transport returned real search results; deterministic failure control returns no hits and mock=true. Main runs stayed offline from web scoring.
- **AI:** 7/7 known AI documents AI_LIKELY; 1/1 unique pure-human literary work HUMAN. Duplicate Alice excluded as an independent human; speech PDF excluded for mixed authorship. Bands/uncertainty/fallback provenance work, but one human negative cannot validate false-positive rate or calibrated probabilities. No fresh human library essays were available.

No independent LOW/MEDIUM/HIGH “ground truth” was frozen, because similarity evidence and academic sanctions are different questions. Accordingly, this audit does **not** invent a risk accuracy comparable to the original 85.7%; it reports every actual risk and retrieval confusion separately. All statements about source copying refer to intentional fixture construction, not accusations.

Reproduction commands and source details: [validation/README.md](validation/README.md). Production and original result preservation: [preservation_check.json](validation/preservation_check.json).
