# Benchmarks

The 14-file sample corpus has only 3 positive pairs and 1 declared LLM document, so
its metrics have very wide error bars. These scripts measure each part of the
engine on public, labelled data with the engine's own code paths. Results land
in `benchmarks/results/*.json`; `run_integrity_check.py --eval` copies the summary
tables into `EVALUATION.md`.

| script | question | data | runtime (this box, CPU) |
|---|---|---|---|
| `fetch_data.py` | download everything below into `benchmarks/data/` (gitignored) | HF Hub | ~1 min |
| `bench_ai_detection.py` | how well does the AI-detection ensemble separate human from ChatGPT text? Also calibrates the perplexity/burstiness curves on a held-out split. | HC3: 150 questions × (human, ChatGPT), 30 % calibration / 70 % test | ~8 min |
| `bench_ai_ood.py` | same, on texts the detector was unlikely to be tuned on | 30 Wikipedia business intros (revisions before 2019) vs the corpus' machine-written docs | ~3 min |
| `bench_paraphrase.py` | is MiniLM cosine a better paraphrase signal than Jaccard / token-LCS, and what do the 0.65 / 0.80 thresholds buy? | MRPC test (1,725 pairs), PAWS test (first 2,000 pairs) | ~1 min |
| `bench_retrieval.py` | does the full peer pipeline find copied / paraphrased documents in a 400-doc corpus without flagging shared templates? | 400 docs of HC3 human answers with 40 injected verbatim copies, 40 MRPC-paraphrase passages, 12 template docs | ~3 min |
| `bench_index_scaling.py` | Flat vs IVF vs HNSW latency / recall / memory on real vectors | ~60k MiniLM sentence vectors from HC3 | ~4 min |
| `bench_asr.py` | WER of whisper tiny/base on the spoken clips, and the effect of TTS speed | corpus scripts re-synthesized | ~1 min |

Run everything:

```bash
.venv/bin/python benchmarks/fetch_data.py
for b in ai_detection ai_ood paraphrase retrieval index_scaling asr; do
  HF_HUB_OFFLINE=1 .venv/bin/python benchmarks/bench_$b.py
done
.venv/bin/python run_integrity_check.py --input_dir ./sample_submissions \
  --output_dir ./reports --eval ./sample_submissions/dataset_manifest.json
```

All sampling uses fixed seeds (42), so reruns give the same numbers up to
floating-point noise in model inference.

Licences: HC3 is CC BY-SA 4.0; MRPC and PAWS are distributed through the GLUE /
Google Research Hugging Face datasets; Wikipedia text is CC BY-SA. None of this
data is committed to the repository.
