"""
fp_scenarios.py — Scenario engine for financial planning projections.

Generates Conservative / Balanced / Aggressive projections using different
return and saving-rate assumptions. All math is deterministic; LLM adds narrative.
"""
from __future__ import annotations
import json, os
from typing import List, Dict, Any

from fp_schemas import ClientProfile, ScenarioProjection
import fp_calculators as calc


_RULES_PATH = os.path.join(os.path.dirname(__file__), "data", "rule_configs", "planning_rules.json")


def _load_rules() -> Dict[str, Any]:
    with open(_RULES_PATH) as f:
        return json.load(f)


SCENARIO_CONFIGS = {
    "Conservative": {
        "return":          0.05,   # nominal annualized (low-risk portfolio)
        "inflation":       0.035,
        "savings_boost":   0.0,    # no change to current rate
        "replacement":     0.85,   # needs more in retirement (risk-averse)
        "swr":             0.035,  # more conservative SWR
        "description":     "Lower-return portfolio (bonds-heavy), no increase in savings rate, conservative SWR of 3.5%.",
    },
    "Balanced": {
        "return":          0.07,
        "inflation":       0.030,
        "savings_boost":   0.02,   # client slightly increases savings rate
        "replacement":     0.80,
        "swr":             0.040,
        "description":     "Diversified 60/40 portfolio. Modest 2% savings rate increase. 4% SWR (Bengen rule).",
    },
    "Aggressive": {
        "return":          0.09,
        "inflation":       0.025,
        "savings_boost":   0.05,   # meaningful savings increase
        "replacement":     0.75,
        "swr":             0.045,
        "description":     "Growth-oriented portfolio (equities-heavy). Significant savings increase. Higher SWR tolerated.",
    },
}


def generate_scenarios(profile: ClientProfile) -> List[ScenarioProjection]:
    """
    For each scenario, project retirement corpus and compare to corpus needed.
    Returns a list of ScenarioProjection sorted Conservative → Balanced → Aggressive.
    """
    years_to_ret = max(1, profile.retirement.target_retirement_age - profile.age)
    base_annual_contrib = calc.calc_annual_contribution(profile)
    total_income = profile.total_annual_income()

    projections = []
    for name, cfg in SCENARIO_CONFIGS.items():
        boosted_rate = min(
            profile.retirement.contribution_rate_pct / 100 + cfg["savings_boost"],
            0.30   # cap at 30% contribution rate
        )
        employer_match = profile.gross_annual_income * (profile.retirement.employer_match_pct / 100)
        annual_contrib = profile.gross_annual_income * boosted_rate + employer_match

        proj = calc.calc_retirement_projection(
            current_balance    = profile.assets.retirement_total(),
            annual_contribution= annual_contrib,
            years_to_retirement= years_to_ret,
            annual_return      = cfg["return"],
            inflation          = cfg["inflation"],
        )

        target_income  = total_income * cfg["replacement"]
        corpus_needed  = calc.calc_retirement_corpus_needed(target_income, cfg["swr"])
        gap            = corpus_needed - proj["corpus_nominal"]

        # Monthly savings needed to close gap (additional above current)
        extra_needed = max(0.0, gap)
        monthly_extra = calc.calc_goal_monthly_savings_needed(
            extra_needed, 0.0, years_to_ret * 12, cfg["return"] / 12
        ) if extra_needed > 0 else 0.0

        # Build human-readable summary
        if gap <= 0:
            status_txt = f"On track — projected surplus of ${abs(gap):,.0f}."
        else:
            status_txt = f"Projected shortfall of ${gap:,.0f}. Would need ~${monthly_extra:,.0f}/month extra savings."

        projections.append(ScenarioProjection(
            name               = name,
            assumptions        = {
                "Annual return":          f"{cfg['return']*100:.0f}%",
                "Inflation":              f"{cfg['inflation']*100:.1f}%",
                "Savings rate":           f"{boosted_rate*100:.0f}%",
                "Income replacement":     f"{cfg['replacement']*100:.0f}%",
                "Safe withdrawal rate":   f"{cfg['swr']*100:.1f}%",
            },
            retirement_corpus  = proj["corpus_nominal"],
            corpus_needed      = corpus_needed,
            gap                = gap,
            monthly_savings_needed = monthly_extra,
            summary            = (
                f"{cfg['description']} "
                f"Projected corpus at age {profile.retirement.target_retirement_age}: "
                f"${proj['corpus_nominal']:,.0f}. "
                f"Target corpus ({cfg['replacement']*100:.0f}% replacement, "
                f"{cfg['swr']*100:.1f}% SWR): ${corpus_needed:,.0f}. "
                + status_txt
            ),
        ))

    return projections
