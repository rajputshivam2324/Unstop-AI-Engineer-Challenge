# Batch integrity summary

14 documents · elapsed 79.37 s · peak RSS 2107.4 MiB

| # | Document | Status | Risk | Peer % | External % | AI % (label) | Top match |
|---|---|---|---|---|---|---|---|
| 1 | `01_retail_turnaround_report.pdf` | ok | 🔴 HIGH | 100.0 | 0.0 | 65.7 (AI_LIKELY) | 02_retail_turnaround_copy.docx (1.00) |
| 2 | `02_retail_turnaround_copy.docx` | ok | 🔴 HIGH | 100.0 | 0.0 | 65.7 (AI_LIKELY) | 01_retail_turnaround_report.pdf (1.00) |
| 3 | `03_paraphrased_strategy.docx` | ok | 🟠 MEDIUM | 62.3 | 0.0 | 65.3 (AI_LIKELY) | 04_original_strategy.docx (0.62) |
| 4 | `04_original_strategy.docx` | ok | 🔴 HIGH | 62.3 | 32.4 | 65.8 (AI_LIKELY) | 03_paraphrased_strategy.docx (0.62) |
| 5 | `05_original_market_pdf.pdf` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 66.2 (AI_LIKELY) | — |
| 6 | `06_llm_generated_summary.pdf` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 65.3 (AI_LIKELY) | — |
| 7 | `07_pitch_crossmodal.pdf` | ok | 🔴 HIGH | 100.0 | 0.0 | 65.1 (AI_LIKELY) | 08_pitch_crossmodal_video.mp4 (1.00) |
| 8 | `08_pitch_crossmodal_video.mp4` | ok | 🔴 HIGH | 100.0 | 0.0 | 65.3 (AI_LIKELY) | 07_pitch_crossmodal.pdf (1.00) |
| 9 | `09_third_pitch_video.mp4` | ok | 🟢 LOW | 0.0 | 0.0 | 17.9 (HUMAN) | — |
| 10 | `10_corrupted_empty.pdf` | error | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |
| 11 | `11_silent_video.mp4` | silent_video | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |
| 12 | `12_case_template_A.docx` | ok | 🟢 LOW | 0.0 | 0.0 | 64.8 (UNSURE) | — |
| 13 | `13_case_template_B.docx` | ok | 🟢 LOW | 0.0 | 0.0 | 64.8 (UNSURE) | — |
| 14 | `14_case_template_C.docx` | ok | 🟢 LOW | 0.0 | 0.0 | 64.7 (UNSURE) | — |

## Peer similarity matrix (doc-pair score, row = query document)

Columns use the two-digit file prefix. Template chunks are excluded; blank = no chunk pair above the cosine floor.

| query \ source | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `01` retail_turnaround_rep | — | **1.00** |  |  |  |  |  |  |  |  |  |  |  |  |
| `02` retail_turnaround_cop | **1.00** | — |  |  |  |  |  |  |  |  |  |  |  |  |
| `03` paraphrased_strategy. |  |  | — | 0.62 |  |  |  |  |  |  |  |  |  |  |
| `04` original_strategy.doc |  |  | 0.62 | — |  |  |  |  |  |  |  |  |  |  |
| `05` original_market_pdf.p |  |  |  |  | — |  |  |  |  |  |  |  |  |  |
| `06` llm_generated_summary |  |  |  |  |  | — |  |  |  |  |  |  |  |  |
| `07` pitch_crossmodal.pdf |  |  |  |  |  |  | — | **1.00** |  |  |  |  |  |  |
| `08` pitch_crossmodal_vide |  |  |  |  |  |  | **1.00** | — |  |  |  |  |  |  |
| `09` third_pitch_video.mp4 |  |  |  |  |  |  |  |  | — |  |  |  |  |  |
| `10` corrupted_empty.pdf |  |  |  |  |  |  |  |  |  | — |  |  |  |  |
| `11` silent_video.mp4 |  |  |  |  |  |  |  |  |  |  | — |  |  |  |
| `12` case_template_A.docx |  |  |  |  |  |  |  |  |  |  |  | — |  |  |
| `13` case_template_B.docx |  |  |  |  |  |  |  |  |  |  |  |  | — |  |
| `14` case_template_C.docx |  |  |  |  |  |  |  |  |  |  |  |  |  | — |

## Stage timings (s)

| ingestion | chunking_and_templates | embedding_and_search | pair_scoring | alignment | external_check | ai_detection | scoring_and_reports |
|---|---|---|---|---|---|---|---|
| 5.75 | 0.06 | 1.53 | 4.43 | 2.29 | 2.3 | 62.99 | 0.01 |
