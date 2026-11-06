# Overfitting and benchmark selection analysis

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
