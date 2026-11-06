# Batch integrity summary

13 documents · elapsed 38.17 s · peak RSS 2026.8 MiB

| # | Document | Status | Risk | Peer % | External % | AI % (label) | Top match |
|---|---|---|---|---|---|---|---|
| 1 | `01_human_speech.pdf` | ok | 🟢 LOW | 0.0 | 0.0 | 64.6 (UNSURE) | — |
| 2 | `02_alice_source.pdf` | ok | 🔴 HIGH | 100.0 | 0.0 | 5.3 (HUMAN) | 03_alice_copy.pdf (1.00) |
| 3 | `03_alice_copy.pdf` | ok | 🔴 HIGH | 100.0 | 0.0 | 5.3 (HUMAN) | 02_alice_source.pdf (1.00) |
| 4 | `04_library_hours.pdf` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 65.7 (AI_LIKELY) | — |
| 5 | `05_library_storage.pdf` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 65.6 (AI_LIKELY) | — |
| 6 | `06_policy_argument.docx` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 65.9 (AI_LIKELY) | — |
| 7 | `07_policy_rewording.docx` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 65.8 (AI_LIKELY) | — |
| 8 | `08_library_transport.docx` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 65.7 (AI_LIKELY) | — |
| 9 | `09_library_privacy.docx` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 66.0 (AI_LIKELY) | — |
| 10 | `10_library_access.docx` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 66.0 (AI_LIKELY) | — |
| 11 | `11_human_crossmodal.mp4` | ok | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |
| 12 | `12_human_allies.mp4` | ok | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |
| 13 | `13_human_revolution.mp4` | ok | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |

## Peer similarity matrix (doc-pair score, row = query document)

Columns use the two-digit file prefix. Template chunks are excluded; blank = no chunk pair above the cosine floor.

| query \ source | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `01` human_speech.pdf | — |  |  |  |  |  |  |  |  |  |  |  |  |
| `02` alice_source.pdf |  | — | **1.00** |  |  |  |  |  |  |  |  |  |  |
| `03` alice_copy.pdf |  | **1.00** | — |  |  |  |  |  |  |  |  |  |  |
| `04` library_hours.pdf |  |  |  | — |  |  |  |  |  |  |  |  |  |
| `05` library_storage.pdf |  |  |  |  | — |  |  |  |  |  |  |  |  |
| `06` policy_argument.docx |  |  |  |  |  | — |  |  |  |  |  |  |  |
| `07` policy_rewording.docx |  |  |  |  |  |  | — |  |  |  |  |  |  |
| `08` library_transport.doc |  |  |  |  |  |  |  | — |  |  |  |  |  |
| `09` library_privacy.docx |  |  |  |  |  |  |  |  | — |  |  |  |  |
| `10` library_access.docx |  |  |  |  |  |  |  |  |  | — |  |  |  |
| `11` human_crossmodal.mp4 |  |  |  |  |  |  |  |  |  |  | — |  |  |
| `12` human_allies.mp4 |  |  |  |  |  |  |  |  |  |  |  | — |  |
| `13` human_revolution.mp4 |  |  |  |  |  |  |  |  |  |  |  |  | — |

## Stage timings (s)

| ingestion | chunking_and_templates | embedding_and_search | pair_scoring | alignment | external_check | ai_detection | scoring_and_reports |
|---|---|---|---|---|---|---|---|
| 6.66 | 0.03 | 5.55 | 0.33 | 2.16 | 0.73 | 22.71 | 0.01 |
