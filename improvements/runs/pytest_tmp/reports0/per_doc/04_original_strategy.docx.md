# Submission Integrity Report: `04_original_strategy.docx`

| Risk | Peer % | External % | AI % (label) | Status |
|---|---|---|---|---|
| 🔴 **HIGH** | 62.3 | 32.4 | 65.8 (AI_LIKELY) | ok |

> **Verdict in one line:** 62.3% of peer similarity comes from overlap with 03_paraphrased_strategy.docx (chunk cosine 0.85). External/online similarity 32.4% via the local reference knowledge base (best match: kb::1). AI-generated estimate 65.8% (flagged as likely machine-generated). Note: AI detectors are unreliable on non-native-English and polished human text (Stanford 2023: up to 61% false-positive rate on ESL essays); treat this as a lead for human review, not proof. Peer overlap is the strongest evidence because it confirms reuse within the evaluated cohort; external and AI scores are supporting signals.

<sub>Machine explanation: Peer similarity: 62.3% across 1 listed source(s) (top: 03_paraphrased_strategy.docx @ 0.85; overlap). External: 32.4% (local_kb). AI: AI_LIKELY at 65.8%. Overlap types: 0 cross-modal, 0 exact-copy, 2 paraphrase, 0 discounted common-source.</sub>

## Peer matches (2)
1. **→ 03_paraphrased_strategy.docx** · chunk cosine **0.85** · this doc *page n/a (docx)* → source *page n/a (docx)* · aligned sentences: 5 (min cos 0.69)
   - **this doc:** "Applying those findings to appliances, we recommend a two-supplier cushion on compressors and control boards, plus a 21-day reserve on the 12 fastest-moving SKUs. Route economics favor a hybrid model. The D2C storefront carries only 60 SKUs, picked for installation simplicity and delivery density, and omits dealer-exclusive floor-stock lines. Order routing assigns orders to stores when delivery is…"
   - **source:** "Three safeguards follow for appliances: split compressor and control-board supply across two vendors, and hold a 21-day reserve on the 12 quickest-turning SKUs so short shocks do not force panic purchases. Math on routes says the hybrid channel wins. Only 60 products ride the D2C storefront, picked for straightforward installation and dense delivery pockets, while dealer-owned floor-stock lines re…"
   - *semantic paraphrase paraphrase-band (>0.80): meaning matches while wording differs (lexical Jaccard 0.33, token-LCS 0.41)*
2. **→ 03_paraphrased_strategy.docx** · chunk cosine **0.84** · this doc *page n/a (docx)* → source *page n/a (docx)* · aligned sentences: 5 (min cos 0.62)
   - **this doc:** "Forecasts put first-year D2C volume at 34 thousand units, or 7 percent of total, incremental to dealer demand because the catalog skews to spare-parts-adjacent segments. Cash conversion improves: direct orders settle in 2 days against 29 days for dealer credit terms, funding the reserve inventory without new credit lines. Three risks need owners: dealer defection in three metro clusters, service c…"
   - **source:** "Volume modeling expects 34 thousand incremental units in year one, roughly 7 percent of the total, and additive to dealer business because the catalog leans toward spares-adjacent buyer needs. Working-capital timing helps as well: direct orders clear in 2 days instead of the 29-day dealer credit cycle, so the reserve inventory funds itself without new credit lines. Ownership goes to three known da…"
   - *semantic paraphrase paraphrase-band (>0.80): meaning matches while wording differs (lexical Jaccard 0.43, token-LCS 0.52)*

## Cross-modal matches (video ↔ document) (0)
*none*

## External reference matches (7) — tier: `local_kb`
1. **kb::1** (local reference KB) · cosine **0.86**
   - **this doc:** "Our supply review found what the post-2020 literature already knew: The pandemic years exposed how fragile long, single-source supply chains can be."
   - **reference:** "Supply chain resilience after 2020: The pandemic years exposed how fragile long, single-source supply chains can be."
2. **kb::1** (local reference KB) · cosine **1.00**
   - **this doc:** "Firms that depended on one supplier or one shipping route faced complete stoppages when borders closed, while companies with diversified supplier networks continued to operate."
   - **reference:** "Firms that depended on one supplier or one shipping route faced complete stoppages when borders closed, while companies with diversified supplier networks continued to operate."
3. **kb::1** (local reference KB) · cosine **1.00**
   - **this doc:** "The first lesson is supplier diversification: spreading orders across several vendors in different regions reduces the risk that a single disruption halts production."
   - **reference:** "The first lesson is supplier diversification: spreading orders across several vendors in different regions reduces the risk that a single disruption halts production."
4. **kb::1** (local reference KB) · cosine **1.00**
   - **this doc:** "The second lesson is deliberate buffer inventory: holding modest reserves of critical parts absorbs short shocks without forcing panic purchases at peak prices."
   - **reference:** "The second lesson is deliberate buffer inventory: holding modest reserves of critical parts absorbs short shocks without forcing panic purchases at peak prices."
5. **kb::1** (local reference KB) · cosine **1.00**
   - **this doc:** "The third lesson is visibility: monitoring supplier health, shipment delays, and tier-two dependencies lets managers act before a shortage spreads."
   - **reference:** "The third lesson is visibility: monitoring supplier health, shipment delays, and tier-two dependencies lets managers act before a shortage spreads."
6. **kb::1** (local reference KB) · cosine **1.00**
   - **this doc:** "Resilience has a cost, and firms balance buffer spending against expected savings from lean operations."
   - **reference:** "Resilience has a cost, and firms balance buffer spending against expected savings from lean operations."
7. **kb::1** (local reference KB) · cosine **1.00**
   - **this doc:** "Post-2020 studies show that firms which treated resilience as a strategic investment, rather than a compliance expense, recovered faster than competitors and retained more customers during the disruption period."
   - **reference:** "Post-2020 studies show that firms which treated resilience as a strategic investment, rather than a compliance expense, recovered faster than competitors and retained more customers during the disruption period."

## AI-generated text analysis
- Label **AI_LIKELY** · fused score 65.8/100 · detector source `model`
- Components: classifier 0.9995 · perplexity signal 0.0 (raw ppl 99.878) · burstiness signal 0.5861 (raw 0.131)
- ⚠️ Note: AI detectors are unreliable on non-native-English and polished human text (Stanford 2023: up to 61% false-positive rate on ESL essays); treat this as a lead for human review, not proof.

## Warnings
- *none*
- 1 tiny match(es) below the evidence-length noise floor were counted but not listed.

## File provenance
- size 37963 bytes · sha256 `8c910040149e56c0…` · chunks 4
