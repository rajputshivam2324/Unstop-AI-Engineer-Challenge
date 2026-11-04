"""<MODULE PURPOSE>
All author-owned corpus content for the 14 sample submissions (§4.10 content
plan). Doc 06's body is the verbatim LLM constant below (the deliberate
"LLM-generated submission"). Every other document is synthetic text stored as
literals here; HONESTY NOTE: those literals were drafted inside an AI coding
assistant session (not typed by a human), so they are "synthetic originals", not
verified human writing. EVALUATION.md therefore does not use them as human
ground truth for the AI detector; see benchmarks/ for the human-text control set.

<PUBLIC API> DOC01_PAGES, DOC03_SENTENCES, DOC04_SENTENCES, DOC05_PAGES,
DOC07_PAGES, PITCH_PASSAGE, PITCH_SPOKEN_PASSAGE, DOC09_TEXT, DOC12_BODY..., CASE_PROMPT,
PARAPHRASE_TABLE, LLM_GENERATED_TEXT, build_doc04_text, build_doc03_text

<DEPENDS ON> reference_kb.kb_texts (the declared KB-derived paragraph only).
"""

from __future__ import annotations

import re

from reference_kb.kb_texts import KB_SUPPLY_CHAIN

# --- doc 06 provenance (reviewer-edict, hard fact) ----------------------------
# LLM_GENERATED_TEXT is genuine LLM output: GLM, single pass 2026-11-04; full
# prompt in docs/03_llm_generated_sample.md — do not hand-edit. Embedded 1:1
# (everything below the `---` divider of that file).
LLM_GENERATED_TEXT = """## Executive Analysis: GreenHarvest Foods — A Structured Turnaround Framework

GreenHarvest Foods, a mid-sized organic grocery chain operating 42 stores across three regions,
is currently experiencing a sustained decline in profitability. This analysis identifies the
principal root causes of underperformance and proposes a structured, three-phase turnaround
plan designed to restore operating margins within a twenty-four-month horizon. In addition, the
analysis outlines the expected margin impact of each phase and highlights the key risks that
require active mitigation.

### Root Causes of Underperformance

First, the profitability decline can be attributed to a combination of structural and
operational factors. Structurally, GreenHarvest Foods is exposed to significant cost pressure in
its supply chain, driven by rising produce prices and a fragmented network of regional
distributors. Operationally, the chain suffers from consistently high shrinkage rates, which are
distinguished by excessive spoilage of perishable inventory and weak demand forecasting.
Furthermore, the in-store customer experience has deteriorated relative to competitors, as
evidenced by declining foot traffic and lower basket sizes over the past eight quarters.

Second, the company's operating expense base remains largely fixed. Because rent, labor, and
energy costs are insensitive to sales volume, the decline in revenue translates disproportionately
into margin erosion. In summary, the combination of gross-margin compression and an inflexible
cost structure has amplified the profitability decline.

### Phase One: Stabilize (Months 1–8)

The initial phase concentrates on immediate margin protection. Specifically, the plan recommends
consolidating regional distributors into two primary partners, which is expected to unlock
volume-based pricing and reduce procurement costs by an estimated 3.2 percentage points of
revenue. In parallel, the company should implement improved inventory rotation and tighten
perishables ordering policies, targeting a reduction in shrinkage of 1.5 percentage points.
Finally, renegotiation of the five highest-cost store leases provides an additional, measurable
margin lever without requiring capital expenditure.

### Phase Two: Optimize (Months 9–16)

The second phase addresses revenue quality. Notably, the plan introduces a loyalty-driven
personalization engine that uses purchase-history data to promote higher-margin categories.
Moreover, store formats will be reorganized to allocate more floor space to prepared foods and
private-label products, both of which carry structurally higher margins. It is also recommended
that the chain adopt dynamic markdown optimization to recover value from near-expiry inventory
while protecting brand perception. Collectively, these initiatives are expected to contribute
an additional 2.4 percentage points of operating margin.

### Phase Three: Differentiate (Months 17–24)

The final phase establishes durable competitive positioning. Concretely, GreenHarvest Foods
should expand its private-label assortment to approximately 22 percent of revenue, intensify its
local-farm sourcing program as a brand differentiator, and pilot a subscription-based delivery
model in its two highest-density markets. Because these initiatives build on data assets created
in Phase Two, their execution risk is meaningfully reduced.

### Financial Impact and Key Risks

Overall, the phased plan targets a cumulative operating-margin improvement of approximately
5.6 percentage points by the end of Month 24, restoring the chain to competitive parity. The key
risks are supplier concentration during the distributor transition, execution capacity within
store teams, and potential demand sensitivity to assortment changes. In conclusion, disciplined
sequencing, transparent weekly performance tracking, and a single integrated reporting dashboard
are essential to ensuring that the turnaround delivers its intended financial outcome."""

# --- doc 01 / doc 02 (exact-copy pair) ---------------------------------------
_CANONICAL_SHARED_PASSAGE = (
    "The turnaround hinges on restoring shrink discipline in fresh categories. Store clusters "
    "with dedicated replenishment owners cut waste by 38 percent within eleven weeks, while "
    "pharmacy-led traffic promotions lifted weekday footfall by 6.4 percent after launch. "
    "Weekly cycle counts replaced quarterly full counts, letting buyers act before stock "
    "spoiled and freeing 2.1 million of working capital in the first quarter. farmers' "
    "inspection came down to two failure modes: late deliveries and missing cold-chain "
    "telemetry, and both closed with dual-sourcing plus sensor retrofits."
)

DOC01_PAGE1 = (
    "Retail Turnaround: Monsoon Lane Grocers\n\n"
    "Prepared for the steering committee. This four-page report reviews the 14-month "
    "turnaround of an 18-store regional grocery chain losing money on perishables.\n\n"
    "Background. Monsoon Lane Grocers entered the year with flat same-store sales, a "
    "shrink rate of 4.8 percent in fresh categories, and ceiling-high trade payables. "
    "The board approved a turnaround budget of 6.2 million spread across logistics, "
    "store operations, and pricing. This report documents what was tried, what worked, "
    "and what remains on the critical path.\n\n"
    "Method. The team instrumented 42 processes across five clusters, ran weekly "
    "leadership reviews, and modeled margins at store level using point-of-sale and "
    "cradle-to-grave waste scans."
)
# page 2 holds the canonical shared passage verbatim (embedded inside surrounding text)
DOC01_PAGE2 = (
    "Findings and evidence\n\n" + _CANONICAL_SHARED_PASSAGE + "\n\n"
    "The shrink result spread unevenly: cluster A and cluster B captured the full decline, "
    "cluster D lagged because its prime suppliers were already dual-sourced. The committee "
    "asked for a repeatable playbook, so the fare-freeze policy and the waste-scan cadence "
    "were standardized across all 18 stores."
)
DOC01_PAGE3 = (
    "Operating detail\n\n"
    "Pricing. Price gaps against the nearest rival were closed on 210 head items, funded "
    "by a supplier rebate pool and the freed working capital. Gross margin fell 0.3 points "
    "in category funds but rose 0.9 points on waste reduction, netting a positive contribution "
    "by month five.\n\n"
    "Logistics. Consolidating two depots into one cross-dock removed 220 route miles per "
    "day. On-time delivery to stores improved from 87 percent to 96 percent, with the "
    "residue concentrated in the night-barking produce window at the two most remote "
    "clusters.\n\n"
    "People. Store cluster leads got a single page scorecard: shrink, on-shelf availability, "
    "safety incidents, and mystery-shop fronts. Bonus pools moved to cluster level, which "
    "quietly defused the corridor-versus-courtyard rivalry between store teams."
)
DOC01_PAGE4 = (
    "Risks and next quarter\n\n"
    "Three risks dominate: rebate dependence on a single anchor supplier, cold-chain "
    "sensor coverage still at 70 percent of stores, and supervisor attrition at 19 percent "
    "in cluster D. The committee's ask for next quarter is digitized supplier scorecards, "
    "a pay-banding review for floor leads, and a tightened night-slapping produce handoff "
    "at cluster D.\n\n"
    "With shrink held below 2.6 percent and the new logistics stack at steady state, the "
    "plan projects an operating margin of 3.1 percent for the full year, with downside "
    "sensitivity dominated by bread-and-butter category inflation."
)

DOC01_PAGES = [DOC01_PAGE1, DOC01_PAGE2, DOC01_PAGE3, DOC01_PAGE4]
DOC02_TEXT = (
    "Retail Turnaround: Monsoon Lane Grocers\n\n"
    "Prepared for the steering committee. This four-page report reviews the 14-month "
    "turnaround of an 18-store regional grocery chain losing money on perishables.\n\n"
    "Background. Monsoon Lane Grocers entered the year with flat same-store sales, a "
    "shrink rate of 4.8 percent in fresh categories, and ceiling-high trade payables. "
    "The board approved a turnaround budget of 6.2 million spread across logistics, "
    "store operations, and pricing. This report documents what was tried, what worked, "
    "and what remains on the critical path.\n\n"
    "Method. The team instrumented 42 processes across five clusters, ran weekly "
    "leadership reviews, and modeled margins at store level using point-of-sale and "
    "cradle-to-grave waste scans.\n\n"
    "Findings and evidence\n\n" + _CANONICAL_SHARED_PASSAGE + "\n\n"
    "The shrink result spread unevenly: cluster A and cluster B captured the full decline, "
    "cluster D lagged because its prime suppliers were already dual-sourced. The committee "
    "asked for a repeatable playbook, so the fare-freeze policy and the waste-scan cadence "
    "were standardized across all 18 stores.\n\n"
    "Operating detail\n\n"
    "Pricing. Price gaps against the nearest rival were closed on 210 head items, funded "
    "by a supplier rebate pool and the freed working capital. Gross margin fell 0.3 points "
    "in category funds but rose 0.9 points on waste reduction, netting a positive contribution "
    "by month five.\n\n"
    "Logistics. Consolidating two depots into one cross-dock removed 220 route miles per "
    "day. On-time delivery to stores improved from 87 percent to 96 percent, with the "
    "residue concentrated in the night-barking produce window at the two most remote "
    "clusters.\n\n"
    "People. Store cluster leads got a single page scorecard: shrink, on-shelf availability, "
    "safety incidents, and mystery-shop fronts. Bonus pools moved to cluster level, which "
    "quietly defused the corridor-versus-courtyard rivalry between store teams.\n\n"
    "Risks and next quarter\n\n"
    "Three risks dominate: rebate dependence on a single anchor supplier, cold-chain "
    "sensor coverage still at 70 percent of stores, and supervisor attrition at 19 percent "
    "in cluster D. The committee's ask for next quarter is digitized supplier scorecards, "
    "a pay-banding review for floor leads, and a tightened night-slapping produce handoff "
    "at cluster D.\n\n"
    "With shrink held below 2.6 percent and the new logistics stack at steady state, the "
    "plan projects an operating margin of 3.1 percent for the full year, with downside "
    "sensitivity dominated by bread-and-butter category inflation."
)

# --- doc 04 (the ORIGINAL that doc 03 paraphrases) ----------------------------
DOC04_SENTENCES: list[str] = [
    "Digital Distribution Strategy (v1): taking a mid-market home-appliance brand direct "
    "to customer doorsteps without cannibalizing its certified-dealer channel.",
    "The brand sells 480 thousand units a year through 1,900 dealers, and any new route "
    "must respect that base.",
    # KB-derived section (declared by design): light paraphrase of KB_SUPPLY_CHAIN
    KB_SUPPLY_CHAIN.replace(
        "Supply chain resilience after 2020: ",
        "Our supply review found what the post-2020 literature already knew: "),
    "Applying those findings to appliances, we recommend a two-supplier cushion on "
    "compressors and control boards, plus a 21-day reserve on the 12 fastest-moving SKUs.",
    "Route economics favor a hybrid model. The D2C storefront carries only 60 SKUs, "
    "picked for installation simplicity and delivery density, and omits dealer-exclusive "
    "floor-stock lines.",
    "Order routing assigns orders to stores when delivery is within 40 kilometers of a "
    "dealer, cutting the last-mile cost to 7.2 percent of order value versus 11.8 percent "
    "for the courier-only path.",
    "Dealer participation is the political keystone. Dealers capturing a share of "
    "routed orders earn a fulfillment fee of 4.5 percent and keep the installation "
    "revenue, which our pilot showed lifts dealer approval to 74 percent.",
    "We price the D2C line at manufacturer-retail parity with dealer shelf prices, and "
    "we honor service contracts through the same certified network.",
    "Advertising spend shifts 15 percent from broad regional slots to search-link and "
    "influencer content for the 25 modeled SKUs.",
    "Forecasts put first-year D2C volume at 34 thousand units, or 7 percent of total, "
    "incremental to dealer demand because the catalog skews to spare-parts-adjacent "
    "segments.",
    "Cash conversion improves: direct orders settle in 2 days against 29 days for "
    "dealer credit terms, funding the reserve inventory without new credit lines.",
    "Three risks need owners: dealer defection in three metro clusters, service "
    "capacity in tier-2 cities, and warehouse slots for bulky goods in three states.",
    "We steer launch readiness on four metrics: dealer approval, last-mile cost share, "
    "install-attempt success rate, and weekly reserve coverage of the core SKUs.",
    "The committee decision requested is a 9-month pilot across two states with a "
    "budget of 3.4 million and a quarterly kill review.",
]

# --- doc 03: hand-maintained sentence-level paraphrase table -----------------
# One entry per sentence of DOC04 so argument skeleton matches while surface
# wording is reworded (partial synonym + clause reorder; NOT algorithmic).
PARAPHRASE_TABLE: list[str] = [
    "A distribution overhaul plan for moving a mid-market home-appliance brand into "
    "direct-to-consumer delivery while leaving the certified-dealer footprint "
    "intact, aimed at growth without channel conflict.",
    "Roughly 480 thousand units move through 1,900 dealer counters each year, and "
    "the new path has to be built around that reality rather than around it.",
    "Our logistics checkpoint confirmed the well-documented post-2020 lesson that "
    "depending on one supplier or one shipping lane can freeze production "
    "altogether, while outfits with supplier networks spread across several "
    "regions kept shipping through the crunch. Spare-part buffers and early "
    "warnings of supplier trouble let managers step in before a shortage spreads "
    "and multiplies.",
    "Three safeguards follow for appliances: split compressor and control-board "
    "supply across two vendors, and hold a 21-day reserve on the 12 "
    "quickest-turning SKUs so short shocks do not force panic purchases.",
    "Math on routes says the hybrid channel wins. Only 60 products ride the D2C "
    "storefront, picked for straightforward installation and dense delivery "
    "pockets, while dealer-owned floor-stock lines remain off the site.",
    "Where a buyer sits within 40 kilometers of a store, the order hands over to "
    "that store, dragging the last-mile cost down to 7.2 percent of order value "
    "rather than the 11.8 percent that a courier-only alternative pays.",
    "Keeping dealers inside the plan matters most politically. Stores that fulfill "
    "routed orders collect a 4.5 percent fulfillment fee and also retain the "
    "installation income, and the trial run lifted their approval to 74 percent.",
    "Shelf pricing stays identical on both routes, and servicing still flows "
    "through the same certified technician network for warranty consistency.",
    "Media money re-balances: 15 percent of the regional broadcast budget moves "
    "toward search phrases and creator content for the 25 modeled products.",
    "Volume modeling expects 34 thousand incremental units in year one, roughly 7 "
    "percent of the total, and additive to dealer business because the catalog "
    "leans toward spares-adjacent buyer needs.",
    "Working-capital timing helps as well: direct orders clear in 2 days instead "
    "of the 29-day dealer credit cycle, so the reserve inventory funds itself "
    "without new credit lines.",
    "Ownership goes to three known dangers: dealers defecting in three metro "
    "areas, technician capacity outside big cities, and bulky-goods warehouse "
    "space in three states.",
    "Launch tracking watches four gauges week by week: dealer approval, last-mile "
    "cost share, install-attempt success, and reserve coverage on core SKUs.",
    "What we ask of the committee is a two-state, 9-month pilot costing 3.4 "
    "million, with a quarterly checkpoint empowered to kill it early.",
]


def build_doc04_text() -> str:
    return "\n".join(DOC04_SENTENCES)


def build_doc03_text() -> str:
    assert len(PARAPHRASE_TABLE) == len(DOC04_SENTENCES)
    return "\n".join(PARAPHRASE_TABLE)


# --- doc 05 (original false-positive guard) -----------------------------------
DOC05_PAGES = [
    "Indian Solar Market Watch, Q2\n\n"
    "An independent quarterly note on auction clearing prices, rooftop economics, and "
    "state-level tendering. Written for internal portfolio review.\n\n"
    "Clearing prices. Utility-scale tenders this quarter cleared between 2.41 and 2.62 "
    "per unit, a band we last saw before module logistics kiranas two years ago. The "
    "tightening traces to ocean freight on panels and a duty adjustment on imported "
    "cells. Project bids from four developers were disqualified for bank-guarantee "
    "formatting, which thinning the deep-pocket pool and nudged prices upward.",
    "Rooftop systems. Payback math for commercial rooftops improved nine days this "
    "quarter, mostly on faster net-metering approvals in two states. Residential "
    "systems still lean on subsidies, and installer chatter suggests paperwork, not "
    "hardware, is the binding constraint. Our field survey of 60 installers found the "
    "average quoting turnaround at 11 days, an improvement from 16.\n\n"
    "Storage. Battery tenders remain exploratory. Two pumped-water proposals advance, "
    "and one firm is piloting vanadium flow batteries next to a hospital cluster where "
    "diesel back-up currently runs 80 hours a month.",
    "Portfolio view. We hold three developers with 1.1 gigawatts of pipeline and are "
    "watching a fourth whose land bank lacks road access. The transmission queue is the "
    "system bottleneck: interconnection delays now average five months beyond "
    "commissioning dates. Next review will quantify how module price bands shift our "
    "yield-on-cost estimates for both existing and pending assets.",
]

# --- docs 07 / 08 (cross-modal pair) ------------------------------------------
PITCH_PASSAGE = (
    "Operating margins contracted by 4.2% over Q3 because inbound freight per carton "
    "rose nineteen percent and the prepaid fuel surcharge stuck at 9 dollars. We held "
    "list prices flat through the quarter, so the squeeze showed up almost entirely in "
    "gross margin rather than in demand. Two countermeasures shipped in October: a "
    "regional carrier swap on the western lanes cuts per-carton freight by a projected "
    "11 percent, and volume-tier surcharge caps lock our effective rate through peak "
    "season. We expect Q4 operating margin to recover to 12.8 percent if carton volume "
    "holds above 340 thousand and the western swap beds in by mid-November."
)

# The MP4 narrator reads the FIRST THREE sentences of PITCH_PASSAGE aloud (78 words,
# ~25 s at natural pace). Reading a whole 103-word paragraph would need either >30 s
# of audio (assignment bound) or sped-up TTS, which measurably destroyed ASR quality
# (see EVALUATION.md failure study 1).
PITCH_SPOKEN_PASSAGE = " ".join(
    re.split(r"(?<=[.!?])\s+", PITCH_PASSAGE)[:3])

DOC07_PAGES = [
    "Q3 Margins Review (Operating Deep Dive)\n\n"
    "Snapshot. Group operating margin printed 11.1 percent for the quarter, down from "
    "15.3 percent a year ago. Revenue grew 2 percent while operating margin fell, "
    "which puts the story firmly on the cost side. Page 2 quantifies the margin bridge "
    "and the countermeasures already in flight.\n\n"
    "Revenue mix did its job: subscription renewals came in at 94 percent and bulk "
    "orders added one-off volume in August. The drag came from logistics inflation and "
    "a fuel surcharge scheduled to reset in December. Nothing in the quarter signals a "
    "demand problem.",
    "Margin bridge and countermeasures\n\n" + PITCH_PASSAGE + "\n\n"
    "Sensitivity. If the western carrier swap slips past mid-December, Q4 margin "
    "recovers only to 12.1 percent absent further pricing moves. The treasury view "
    "holds us harmless on FX through natural hedges, and the audit committee asked for "
    "a freight-contract review memo by the second week of December.",
]

DOC09_TEXT = (
    "Carbon tracking for city logistics. Our pilot puts a small sensor node on every "
    "delivery cart and logs kilograms of CO2 per route, per stop. Dispatch software "
    "reorders drops to cut idling, and the dashboard gives planners a daily carbon "
    "budget that resets at 6 a.m. In the six-week trial across two depots, rerouted "
    "mornings trimmed idle time 22 percent and diesel use 13 percent. Next step is a "
    "three-city rollout funded by the airline offset pool."
)

# --- docs 12 / 13 / 14 (template trio, Scenario 4) ----------------------------
# CASE_PROMPT: problem statement + question instructions + citations (the shared
# institutional block the assignment describes). Kept >= 200 tokens so the whole
# block forms chunk 00 verbatim-identical across the trio -> template cluster.
CASE_PROMPT = (
    "Unit Economics Case Prompt (shared template block).\n\n"
    "You are advising the operations committee of a mid-sized logistics firm. The "
    "case concerns a delivery network serving 12 cities from 3 depots. Depots differ "
    "in size and automation. Orders arrive in two waves, morning and evening, and the "
    "fleet is a mix of owned vans and contracted riders. Your brief contains monthly "
    "cost and volume data for the last 9 months.\n\n"
    "Question instructions. Answer both questions. Question 1: estimate the "
    "contribution margin per delivered order for each depot and identify which cost "
    "line explains the difference between the strongest and weakest depot. Question 2: "
    "propose two operational interventions with quantified estimates of their effect "
    "on the break-even volume of the weakest depot. Support every estimate with "
    "arithmetic shown line by line, and state the assumption behind each number you "
    "cite.\n\n"
    "Citations. Reference the operations handbook sections 4 and 7 for cost-line "
    "definitions, and the depot p&l template for the monthly data. Do not consult "
    "external market reports; the case is self-contained. Submit the analysis as a "
    "two-page memo addressed to the committee, and attach the arithmetic appendix "
    "as a separate exhibit labeled with your candidate code."
)

DOC12_BODY = (
    "Unit-economics analysis (memo A).\n\n"
    "Depot B wins on contribution because its rider acquisition cost is spread over "
    "4,100 evening orders against 2,300 at depot C. Morning waves break even at 60 "
    "percent utilization, but evening waves at depot C stall near 40 percent, so the "
    "second wave is where the loss executes.\n\n"
    "Fix one: move the depot C evening wave to smaller contracted riders with "
    "pickup-point handoffs, cutting delivery fee per order from 31 to 24 on 1,800 "
    "monthly orders.\n\n"
    "Fix two: pool depot B's idle vans with depot C's overflowing evening batch on "
    "the two densest pincode clusters; the share of van orders rises to 55 percent "
    "and drops fuel cost per stop.\n\n"
    "Arithmetic summary attached in the appendix table, per the handbook. The "
    "committee is asked to approve both interventions and re-check break-even at the "
    "next monthly close."
)

DOC13_BODY = (
    "Distribution-network SWOT (memo B).\n\n"
    "Strength: depot A's 2 automated sorters cover 61 percent of volume with a defect "
    "rate of 0.4 percent, the network's best. Weakness: depot C pays 18 percent above "
    "market on contracted rider fees because its contract rolled over during the "
    "monsoon crunch.\n\n"
    "Opportunity: the two densest pincode clusters can absorb an evening micro-hub, "
    "which cuts 9 kilometers per van route and about 7 percent of fuel spend.\n\n"
    "Threat: the rider contract auto-renews in 11 weeks; without renegotiation, the "
    "fee gap funds no interventions at all.\n\n"
    "Recommended sequence: renegotiate first, micro-hub second. The committee gets a "
    "one-page decision sheet with both verdicts priced."
)

DOC14_BODY = (
    "Break-even plan for depot C (memo C).\n\n"
    "Fixed costs at depot C run 2.9 million monthly: shed rent, three supervisors, "
    "and the navigation-system lease. Contribution per evening order is 46 before the "
    "rider-fee fix and 53 after, holding volume at 2,300.\n\n"
    "Break-even volume moves from 63,000 evening orders a month (~ inbox math logged "
    "in appendix) down to 54,700 with both fixes in place, a 13 percent reduction.\n\n"
    "Sensitivity: if the rider contract renews 6 percent higher, break-even slides "
    "back by 4,200 orders, so renegotiation timing gates everything else.\n\n"
    "The plan fits the handbook's cost-line definitions, and the committee decision "
    "requested is approval of the renegotiation mandate this week."
)
