# Batch integrity summary

4 documents · elapsed 13.49 s · peak RSS 1909.5 MiB

| # | Document | Status | Risk | Peer % | External % | AI % (label) | Top match |
|---|---|---|---|---|---|---|---|
| 1 | `05_valid_control.pdf` | ok | 🟠 MEDIUM | 0.0 | 0.0 | 65.6 (AI_LIKELY) | — |
| 2 | `20_empty.pdf` | error | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |
| 3 | `21_corrupt.pdf` | error | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |
| 4 | `22_silent.mp4` | silent_video | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |

## Peer similarity matrix (doc-pair score, row = query document)

Columns use the two-digit file prefix. Template chunks are excluded; blank = no chunk pair above the cosine floor.

| query \ source | 05 | 20 | 21 | 22 |
|---|---|---|---|---|
| `05` valid_control.pdf | — |  |  |  |
| `20` empty.pdf |  | — |  |  |
| `21` corrupt.pdf |  |  | — |  |
| `22` silent.mp4 |  |  |  | — |

## Stage timings (s)

| ingestion | chunking_and_templates | embedding_and_search | pair_scoring | alignment | external_check | ai_detection | scoring_and_reports |
|---|---|---|---|---|---|---|---|
| 0.32 | 0.0 | 6.35 | 0.08 | 0.0 | 0.4 | 6.34 | 0.0 |
