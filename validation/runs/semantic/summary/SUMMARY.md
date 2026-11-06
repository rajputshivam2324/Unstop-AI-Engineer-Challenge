# Batch integrity summary

2 documents · elapsed 6.31 s · peak RSS 605.7 MiB

| # | Document | Status | Risk | Peer % | External % | AI % (label) | Top match |
|---|---|---|---|---|---|---|---|
| 1 | `30_assertion.pdf` | ok | 🔴 HIGH | 100.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | 31_negation.pdf (1.00) |
| 2 | `31_negation.pdf` | ok | 🔴 HIGH | 100.0 | 0.0 | n/a (INSUFFICIENT_TEXT) | 30_assertion.pdf (1.00) |

## Peer similarity matrix (doc-pair score, row = query document)

Columns use the two-digit file prefix. Template chunks are excluded; blank = no chunk pair above the cosine floor.

| query \ source | 30 | 31 |
|---|---|---|
| `30` assertion.pdf | — | **1.00** |
| `31` negation.pdf | **1.00** | — |

## Stage timings (s)

| ingestion | chunking_and_templates | embedding_and_search | pair_scoring | alignment | external_check | ai_detection | scoring_and_reports |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 5.68 | 0.07 | 0.04 | 0.38 | 0.13 | 0.0 |
