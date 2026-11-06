# Batch integrity summary

13 documents · elapsed 96.37 s · peak RSS 1936.3 MiB

| # | Document | Status | Risk | Peer % | External % | AI % (label) | Top match |
|---|---|---|---|---|---|---|---|
| 1 | `01_human_speech.pdf` | ok | 🔴 HIGH | 100.0 | 0.0 | 64.6 (UNSURE) | 11_human_crossmodal.mp4 (1.00) |
| 2 | `02_alice_source.pdf` | ok | 🔴 HIGH | 100.0 | 0.0 | 5.3 (HUMAN) | 03_alice_copy.pdf (1.00) |
| 3 | `03_alice_copy.pdf` | ok | 🔴 HIGH | 100.0 | 0.0 | 5.3 (HUMAN) | 02_alice_source.pdf (1.00) |
| 4 | `04_library_hours.pdf` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 65.7 (AI_LIKELY) | — |
| 5 | `05_library_storage.pdf` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 65.6 (AI_LIKELY) | — |
| 6 | `06_policy_argument.docx` | ok | 🔴 HIGH | 100.0 | 0.0 | 65.9 (AI_LIKELY) | 07_policy_rewording.docx (1.00) |
| 7 | `07_policy_rewording.docx` | ok | 🔴 HIGH | 100.0 | 0.0 | 65.8 (AI_LIKELY) | 06_policy_argument.docx (1.00) |
| 8 | `08_library_transport.docx` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 65.7 (AI_LIKELY) | — |
| 9 | `09_library_privacy.docx` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 66.0 (AI_LIKELY) | — |
| 10 | `10_library_access.docx` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 66.0 (AI_LIKELY) | — |
| 11 | `11_human_crossmodal.mp4` | ok | 🔴 HIGH | 100.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | 01_human_speech.pdf (1.00) |
| 12 | `12_human_allies.mp4` | ok | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |
| 13 | `13_human_revolution.mp4` | ok | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |

## Peer similarity matrix (doc-pair score, row = query document)

Columns use the two-digit file prefix. Template chunks are excluded; blank = no chunk pair above the cosine floor.

| query \ source | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `01` human_speech.pdf | — |  |  |  |  |  |  |  |  |  | **1.00** |  |  |
| `02` alice_source.pdf |  | — | **1.00** |  |  |  |  |  |  |  |  |  |  |
| `03` alice_copy.pdf |  | **1.00** | — |  |  |  |  |  |  |  |  |  |  |
| `04` library_hours.pdf |  |  |  | — |  |  |  |  |  |  |  |  |  |
| `05` library_storage.pdf |  |  |  |  | — |  |  |  |  |  |  |  |  |
| `06` policy_argument.docx |  |  |  |  |  | — | **1.00** |  |  |  |  |  |  |
| `07` policy_rewording.docx |  |  |  |  |  | **1.00** | — |  |  |  |  |  |  |
| `08` library_transport.doc |  |  |  |  |  |  |  | — |  |  |  |  |  |
| `09` library_privacy.docx |  |  |  |  |  |  |  |  | — |  |  |  |  |
| `10` library_access.docx |  |  |  |  |  |  |  |  |  | — |  |  |  |
| `11` human_crossmodal.mp4 | **1.00** |  |  |  |  |  |  |  |  |  | — |  |  |
| `12` human_allies.mp4 |  |  |  |  |  |  |  |  |  |  |  | — |  |
| `13` human_revolution.mp4 |  |  |  |  |  |  |  |  |  |  |  |  | — |

## Stage timings (s)

| ingestion | chunking_and_templates | embedding_and_search | pair_scoring | alignment | external_check | ai_detection | scoring_and_reports |
|---|---|---|---|---|---|---|---|
| 7.05 | 0.03 | 10.1 | 50.33 | 4.24 | 1.34 | 23.27 | 0.01 |
