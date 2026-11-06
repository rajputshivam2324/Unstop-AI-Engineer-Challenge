# Batch integrity summary

4 documents · elapsed 12.51 s · peak RSS 637.0 MiB

| # | Document | Status | Risk | Peer % | External % | AI % (label) | Top match |
|---|---|---|---|---|---|---|---|
| 1 | `bad_0.pdf` | error | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |
| 2 | `bad_1.docx` | error | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |
| 3 | `bad_2.mp4` | error | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |
| 4 | `bad_3.pdf` | error | 🟢 LOW | 0.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | — |

## Peer similarity matrix (doc-pair score, row = query document)

Columns use the two-digit file prefix. Template chunks are excluded; blank = no chunk pair above the cosine floor.

| query \ source | ba | ba | ba | ba |
|---|---|---|---|---|
| `ba` _0.pdf | — |  |  |  |
| `ba` _1.docx |  | — |  |  |
| `ba` _2.mp4 |  |  | — |  |
| `ba` _3.pdf |  |  |  | — |

## Stage timings (s)

| ingestion | chunking_and_templates | embedding_and_search | pair_scoring | alignment | external_check | ai_detection | scoring_and_reports |
|---|---|---|---|---|---|---|---|
| 0.13 | 0.0 | 11.38 | 0.0 | 0.0 | 0.64 | 0.36 | 0.0 |
