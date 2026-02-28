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


def _build_growth_series(
    current_balance: float,
    annual_contrib: float,
    years: int,
    annual_return: float,
) -> List[float]:
    """Year-by-year projected retirement balance (index 0 = today)."""
    series = [current_balance]
    bal = current_balance
    for _ in range(years):
        bal = bal * (1 + annual_return) + annual_contrib
        series.append(round(bal, 0))
    return series


def generate_scenarios(profile: ClientProfile) -> List[ScenarioProjection]:
    """
    For each scenario, project retirement corpus and compare to corpus needed
    (net of estimated Social Security benefit).
    Returns a list of ScenarioProjection sorted Conservative → Balanced → Aggressive.
    """
    years_to_ret  = max(1, profile.retirement.target_retirement_age - profile.age)
    total_income  = profile.total_annual_income()
    employer_match_base = profile.gross_annual_income * (profile.retirement.employer_match_pct / 100)

    # Base contribution (no scenario boost) for display purposes
    base_employee_contrib = profile.gross_annual_income * (profile.retirement.contribution_rate_pct / 100)
    base_total_annual     = base_employee_contrib + employer_match_base
    base_monthly          = base_total_annual / 12

    # Social Security estimate (today's dollars, FRA=67)
    ss_annual = calc.estimate_social_security_benefit(profile.age, total_income, claiming_age=67)

    projections = []
    for name, cfg in SCENARIO_CONFIGS.items():
        boosted_rate = min(
            profile.retirement.contribution_rate_pct / 100 + cfg["savings_boost"],
            0.30   # cap at 30%
        )
        annual_contrib   = profile.gross_annual_income * boosted_rate + employer_match_base
        boosted_monthly  = annual_contrib / 12

        proj = calc.calc_retirement_projection(
            current_balance     = profile.assets.retirement_total(),
            annual_contribution = annual_contrib,
            years_to_retirement = years_to_ret,
            annual_return       = cfg["return"],
            inflation           = cfg["inflation"],
        )

        # Target income in future nominal dollars; subtract inflation-adjusted SS benefit
        target_income_today = total_income * cfg["replacement"]
        target_income_fut   = target_income_today * ((1 + cfg["inflation"]) ** years_to_ret)
        ss_future           = ss_annual * ((1 + cfg["inflation"]) ** years_to_ret)
        net_target_income   = max(0.0, target_income_fut - ss_future)
        corpus_needed       = calc.calc_retirement_corpus_needed(net_target_income, cfg["swr"])
        gap                 = corpus_needed - proj["corpus_nominal"]

        # Additional monthly savings needed to close any remaining gap
        extra_needed  = max(0.0, gap)
        monthly_extra = calc.calc_goal_monthly_savings_needed(
            extra_needed, 0.0, years_to_ret * 12, cfg["return"] / 12
        ) if extra_needed > 0 else 0.0

        # Year-by-year growth series for charting
        growth_series = _build_growth_series(
            profile.assets.retirement_total(), annual_contrib, years_to_ret, cfg["return"]
        )

        if gap <= 0:
            status_txt = f"On track — projected surplus of ${abs(gap):,.0f}."
        else:
            status_txt = (f"Projected shortfall of ${gap:,.0f} "
                          f"(after SS ~${ss_annual:,.0f}/yr credit). "
                          f"Would need ~${monthly_extra:,.0f}/month additional savings.")

        projections.append(ScenarioProjection(
            name               = name,
            assumptions        = {
                "Annual return":          f"{cfg['return']*100:.0f}%",
                "Inflation":              f"{cfg['inflation']*100:.1f}%",
                "Savings rate":           f"{boosted_rate*100:.0f}%",
                "Income replacement":     f"{cfg['replacement']*100:.0f}%",
                "Safe withdrawal rate":   f"{cfg['swr']*100:.1f}%",
            },
            retirement_corpus       = proj["corpus_nominal"],
            corpus_needed           = corpus_needed,
            gap                     = gap,
            monthly_savings_needed  = monthly_extra,
            base_monthly_contrib    = round(base_monthly, 2),
            boosted_monthly_contrib = round(boosted_monthly, 2),
            ss_annual_benefit       = ss_annual,
            corpus_growth           = growth_series,
            summary                 = (
                f"{cfg['description']} "
                f"Est. SS benefit: ${ss_annual:,.0f}/yr. "
                f"Projected corpus at age {profile.retirement.target_retirement_age}: "
                f"${proj['corpus_nominal']:,.0f}. "
                f"Target corpus ({cfg['replacement']*100:.0f}% income, net of SS, "
                f"{years_to_ret}yr inflation adj., {cfg['swr']*100:.1f}% SWR): "
                f"${corpus_needed:,.0f}. " + status_txt
            ),
        ))

    return projections
