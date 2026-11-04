"""<MODULE PURPOSE>
Local reference knowledge-base passages (Tier 1 external source, spec §4.7):
6 public-domain-style passages in exactly 2 groups — (a) encyclopedic
Wikipedia-register blocks, (b) case-methodology/consulting-textbook blocks.
No LLM-flavored group: doc 06's body comes solely from
docs/03_llm_generated_sample.md (verbatim, §4.10).

The KB never feeds dataset content EXCEPT by declared design: doc 04 lightly
paraphrases KB_SUPPLY_CHAIN (constant below) so the external signal has one
legitimate offline hit; doc 03 inherits that paraphrase through
dataset_generator.PARAPHRASE_TABLE. Only docs 03 and 04 carry
external_expected_min = 20.0 in the manifest.

<PUBLIC API> KB_GROUPS, KB_SUPPLY_CHAIN (plus all 6 named constants)
<DEPENDS ON> none (plain strings only).
"""

from __future__ import annotations

KB_SOLAR = (
    "Solar power in India: India is one of the largest producers of solar energy in "
    "the world. The country's installed solar capacity has expanded rapidly since "
    "2015, supported by large-scale auctions, falling photovoltaic module prices, and "
    "government targets for renewable capacity. Utility-scale parks dominate the "
    "market, although rooftop installations are growing faster in percentage terms. "
    "Solar generation follows strong seasonal patterns, peaking during the summer "
    "months when daylight hours are longest. Grid integration remains a challenge, "
    "because solar output varies with weather and time of day and requires storage "
    "or flexible backup. Analysts note that domestic manufacturing of solar modules "
    "has increased, reducing dependence on imported panels. Land acquisition and "
    "transmission bottlenecks continue to slow some projects, particularly in "
    "densely populated states. India's solar program also supports rural "
    "electrification through off-grid installations and solar-powered irrigation "
    "pumps, which reduce diesel consumption for farmers."
)

KB_SUPPLY_CHAIN = (
    "Supply chain resilience after 2020: The pandemic years exposed how fragile "
    "long, single-source supply chains can be. Firms that depended on one supplier "
    "or one shipping route faced complete stoppages when borders closed, while "
    "companies with diversified supplier networks continued to operate. The first "
    "lesson is supplier diversification: spreading orders across several vendors in "
    "different regions reduces the risk that a single disruption halts production. "
    "The second lesson is deliberate buffer inventory: holding modest reserves of "
    "critical parts absorbs short shocks without forcing panic purchases at peak "
    "prices. The third lesson is visibility: monitoring supplier health, shipment "
    "delays, and tier-two dependencies lets managers act before a shortage spreads. "
    "Resilience has a cost, and firms balance buffer spending against expected "
    "savings from lean operations. Post-2020 studies show that firms which treated "
    "resilience as a strategic investment, rather than a compliance expense, "
    "recovered faster than competitors and retained more customers during the "
    "disruption period."
)

KB_EINVOICING = (
    "E-invoicing basics: Electronic invoicing replaces paper or PDF invoices with "
    "structured digital documents that flow directly between accounting systems. "
    "A compliant electronic invoice contains unique identifiers, buyer and seller "
    "tax registration numbers, line-item details, and tax breakdowns in a machine "
    "readable format. Because the document arrives as data instead of an image, "
    "matching against purchase orders and goods receipt notes is automatic, which "
    "cuts processing time and manual entry errors. Governments in many countries "
    "mandate electronic invoicing for business-to-business transactions to narrow "
    "the tax gap and speed audits. Businesses typically gain faster payments, "
    "better cash-flow visibility, and lower storage costs, while finance teams "
    "spend less effort chasing missing invoice details."
)

KB_SWOT = (
    "SWOT analysis is a structured planning method for evaluating the strategic "
    "position of a business or project. It examines four dimensions: strengths, "
    "which are internal capabilities that give an advantage; weaknesses, internal "
    "factors that hold performance back; opportunities, external trends the "
    "organization could exploit; and threats, outside risks that could damage "
    "results. The method is popular because it is simple and encourages a balanced "
    "view rather than a single lens. In practice, teams list factors in a "
    "two-by-two grid, then prioritize items with the largest potential impact. "
    "Common pitfalls include listing vague items, mixing internal and external "
    "factors, and treating the grid as an end in itself. Practitioners recommend "
    "pairing SWOT with quantified evidence and following it with concrete action "
    "plans rather than leaving it as a brainstorm."
)

KB_PORTER = (
    "Porter's generic strategies describe three broad ways a firm can outcompete "
    "rivals: cost leadership, which means delivering acceptable quality at the "
    "lowest price in an industry; differentiation, which means offering something "
    "perceived as unique and charging a premium for it; and focus, which serves one "
    "narrow segment better than broad-market competitors do. Porter argued that "
    "being stuck in the middle, neither low cost nor clearly differentiated, "
    "typically brings below-average profitability. Each strategy demands different "
    "resources and organization: cost leadership rewards scale, efficient "
    "operations, and tight purchasing; differentiation rewards design, brand, and "
    "service innovation; focus rewards deep knowledge of the chosen niche."
)

KB_BREAK_EVEN = (
    "Break-even analysis calculates the point at which a product or business stops "
    "losing money and starts contributing profit. The break-even quantity equals "
    "fixed costs divided by the contribution margin per unit, where the "
    "contribution margin is the selling price minus the variable cost of one unit. "
    "Below the break-even volume, every additional unit still fails to cover "
    "shared fixed costs such as rent, salaries, and equipment; above it, each unit "
    "adds to profit. Managers use break-even analysis to test whether a price cut "
    "can be recovered by volume, to size a new investment, and to check how much "
    "sales can fall before the business slips into loss. Sensitivity analysis on "
    "price and variable cost shows which levers move the break-even point most."
)

KB_GROUPS: dict[str, list[tuple[str, str]]] = {
    "encyclopedic": [
        ("kb::0", KB_SOLAR),
        ("kb::1", KB_SUPPLY_CHAIN),
        ("kb::2", KB_EINVOICING),
    ],
    "case_methodology": [
        ("kb::3", KB_SWOT),
        ("kb::4", KB_PORTER),
        ("kb::5", KB_BREAK_EVEN),
    ],
}


def all_passages() -> list[tuple[str, str]]:
    """Flat ordered list of (kb_id, text) pairs for embedding."""
    flat: list[tuple[str, str]] = []
    for group in KB_GROUPS.values():
        flat.extend(group)
    return flat
