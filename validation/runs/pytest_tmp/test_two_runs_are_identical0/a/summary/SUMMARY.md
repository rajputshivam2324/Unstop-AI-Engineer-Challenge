# Batch integrity summary

5 documents · elapsed 12.99 s · peak RSS 1422.2 MiB

| # | Document | Status | Risk | Peer % | External % | AI % (label) | Top match |
|---|---|---|---|---|---|---|---|
| 1 | `07_pitch_crossmodal.pdf` | ok | 🔴 HIGH | 100.0 | 0.0 | 25.0 (HUMAN) | 08_pitch_crossmodal_video.mp4 (1.00) |
| 2 | `08_pitch_crossmodal_video.mp4` | ok | 🔴 HIGH | 100.0 | 0.0 | 25.0 (HUMAN) | 07_pitch_crossmodal.pdf (1.00) |
| 3 | `12_case_template_A.docx` | ok | 🟢 LOW | 0.0 | 0.0 | 25.0 (HUMAN) | — |
| 4 | `13_case_template_B.docx` | ok | 🟢 LOW | 0.0 | 0.0 | 25.0 (HUMAN) | — |
| 5 | `14_case_template_C.docx` | ok | 🟢 LOW | 0.0 | 0.0 | 25.0 (HUMAN) | — |

## Peer similarity matrix (doc-pair score, row = query document)

Columns use the two-digit file prefix. Template chunks are excluded; blank = no chunk pair above the cosine floor.

| query \ source | 07 | 08 | 12 | 13 | 14 |
|---|---|---|---|---|---|
| `07` pitch_crossmodal.pdf | — | **1.00** |  |  |  |
| `08` pitch_crossmodal_vide | **1.00** | — |  |  |  |
| `12` case_template_A.docx |  |  | — |  |  |
| `13` case_template_B.docx |  |  |  | — |  |
| `14` case_template_C.docx |  |  |  |  | — |

## Stage timings (s)

| ingestion | chunking_and_templates | embedding_and_search | pair_scoring | alignment | external_check | ai_detection | scoring_and_reports |
|---|---|---|---|---|---|---|---|
| 4.85 | 0.02 | 0.43 | 0.59 | 1.13 | 0.8 | 5.17 | 0.0 |
