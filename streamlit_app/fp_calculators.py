"""
fp_calculators.py — Deterministic financial planning calculations.

All functions are pure (no side effects, no LLM calls).
Inputs come from ClientProfile; outputs are plain numbers or dicts.
Formulas reference standard planning assumptions; sources noted inline.
"""
from __future__ import annotations
import math
from typing import Dict, Any, Optional


# ── Emergency fund ────────────────────────────────────────────────────────────

def calc_emergency_fund_months(liquid_assets: float, monthly_expenses: float) -> float:
    """How many months of expenses are covered by liquid assets."""
    if monthly_expenses <= 0:
        return 0.0
    return round(liquid_assets / monthly_expenses, 2)


# ── Debt ratios ───────────────────────────────────────────────────────────────

def calc_dti(total_monthly_debt_payments: float, gross_monthly_income: float) -> float:
    """Total Debt-to-Income ratio (CFPB / mortgage underwriting standard)."""
    if gross_monthly_income <= 0:
        return 0.0
    return round(total_monthly_debt_payments / gross_monthly_income, 4)


def calc_housing_ratio(monthly_housing_payment: float, gross_monthly_income: float) -> float:
    """Front-end (housing) ratio = housing payment / gross monthly income."""
    if gross_monthly_income <= 0:
        return 0.0
    return round(monthly_housing_payment / gross_monthly_income, 4)


def calc_consumer_debt_burden(consumer_debt_total: float, gross_annual_income: float) -> float:
    """Non-mortgage debt as % of gross annual income."""
    if gross_annual_income <= 0:
        return 0.0
    return round(consumer_debt_total / gross_annual_income, 4)


# ── Cash flow ─────────────────────────────────────────────────────────────────

def calc_monthly_cash_flow(
    gross_monthly_income: float,
    monthly_expenses_total: float,
    monthly_debt_payments_total: float,
    effective_tax_rate: float = 0.25,
) -> float:
    """
    Approximate monthly surplus / deficit after taxes, expenses, and debt payments.
    Uses a flat effective tax rate as an estimate — not a precise tax calculation.
    """
    net_income = gross_monthly_income * (1 - effective_tax_rate)
    return round(net_income - monthly_expenses_total - monthly_debt_payments_total, 2)


# ── Savings rate ──────────────────────────────────────────────────────────────

def calc_savings_rate(
    annual_retirement_contributions: float,
    annual_other_savings: float,
    gross_annual_income: float,
) -> float:
    """Total savings (retirement + other) as % of gross income."""
    if gross_annual_income <= 0:
        return 0.0
    return round((annual_retirement_contributions + annual_other_savings) / gross_annual_income, 4)


# ── Insurance ────────────────────────────────────────────────────────────────

def calc_life_insurance_coverage_ratio(coverage: float, gross_annual_income: float) -> float:
    """Life insurance face amount as multiple of gross annual income."""
    if gross_annual_income <= 0:
        return 0.0
    return round(coverage / gross_annual_income, 2)


# ── Net worth ────────────────────────────────────────────────────────────────

def calc_net_worth_target(age: int, gross_annual_income: float) -> float:
    """
    Target net worth using Stanley-Danko benchmark, interpolated linearly.
    Source: "The Millionaire Next Door" income-proportional benchmarks.
    """
    _benchmarks = {30: 0.5, 35: 1.0, 40: 2.0, 45: 3.0,
                   50: 5.0, 55: 8.0, 60: 12.0, 65: 15.0}
    if age <= 30:
        return gross_annual_income * 0.5
    if age >= 65:
        return gross_annual_income * 15.0
    # Linear interpolation between bracket ages
    ages = sorted(_benchmarks.keys())
    for i in range(len(ages) - 1):
        a0, a1 = ages[i], ages[i + 1]
        if a0 <= age <= a1:
            m0, m1 = _benchmarks[a0], _benchmarks[a1]
            frac = (age - a0) / (a1 - a0)
            mult = m0 + frac * (m1 - m0)
            return round(gross_annual_income * mult, 2)
    return gross_annual_income * _benchmarks[ages[-1]]


# ── Retirement projection ─────────────────────────────────────────────────────

def calc_retirement_projection(
    current_balance:      float,
    annual_contribution:  float,
    years_to_retirement:  int,
    annual_return:        float,   # nominal, e.g. 0.07
    inflation:            float = 0.03,
) -> Dict[str, float]:
    """
    Project retirement corpus using FV formula.

    FV_balance = current_balance * (1+r)^n
    FV_contributions = annual_contribution * [(1+r)^n - 1] / r  (end-of-year payments)

    Returns corpus (nominal), corpus_real (inflation-adjusted), and annual_income (4% SWR).
    """
    if years_to_retirement <= 0:
        return {"corpus_nominal": current_balance,
                "corpus_real": current_balance,
                "annual_income_4pct": current_balance * 0.04}

    r = annual_return
    n = years_to_retirement

    fv_balance = current_balance * ((1 + r) ** n)
    if r > 0:
        fv_contributions = annual_contribution * (((1 + r) ** n - 1) / r)
    else:
        fv_contributions = annual_contribution * n

    corpus_nominal = round(fv_balance + fv_contributions, 2)
    # Deflate by inflation to get today's purchasing power
    corpus_real    = round(corpus_nominal / ((1 + inflation) ** n), 2)
    annual_income  = round(corpus_nominal * 0.04, 2)   # 4% safe withdrawal rate

    return {
        "corpus_nominal":    corpus_nominal,
        "corpus_real":       corpus_real,
        "annual_income_4pct": annual_income,
        "fv_balance":        round(fv_balance, 2),
        "fv_contributions":  round(fv_contributions, 2),
    }


def calc_retirement_corpus_needed(
    target_annual_income: float,   # amount needed per year in retirement
    safe_withdrawal_rate: float = 0.04,
) -> float:
    """
    Required corpus at retirement using SWR.
    corpus = target_income / SWR   (Bengen 4% rule)
    """
    if safe_withdrawal_rate <= 0:
        return 0.0
    return round(target_annual_income / safe_withdrawal_rate, 2)


def calc_annual_contribution(profile_obj) -> float:  # type: ignore[annotation-unchecked]
    """
    Total annual retirement contributions from profile:
    (401k contribution rate + employer match) * gross income + other_income contributions.
    Assumes contributions come from primary income only.
    """
    from fp_schemas import ClientProfile
    p: ClientProfile = profile_obj
    employee_contrib = p.gross_annual_income * (p.retirement.contribution_rate_pct / 100)
    employer_match   = p.gross_annual_income * (p.retirement.employer_match_pct / 100)
    return round(employee_contrib + employer_match, 2)


# ── Goal funding gap ──────────────────────────────────────────────────────────

def calc_goal_monthly_savings_needed(
    target_amount:   float,
    current_savings: float,
    months:          int,
    annual_return:   float = 0.05,
) -> float:
    """
    Monthly savings needed to reach a goal, given current savings and investment return.
    Uses future-value of an annuity-due approach.
    Returns 0 if already funded.
    """
    if months <= 0:
        return max(0.0, target_amount - current_savings)

    shortfall = target_amount - current_savings * ((1 + annual_return / 12) ** months)
    if shortfall <= 0:
        return 0.0

    r = annual_return / 12
    if r > 0:
        monthly = shortfall * r / (((1 + r) ** months) - 1)
    else:
        monthly = shortfall / months

    return round(max(0.0, monthly), 2)


# ── Social Security estimate ──────────────────────────────────────────────────

def estimate_social_security_benefit(
    age: int,
    gross_annual_income: float,
    claiming_age: int = 67,
) -> float:
    """
    Rough annual Social Security benefit estimate in today's dollars.
    Uses SSA average replacement rates (NOT the actual AIME/PIA formula).

    Replacement rates by income tier (FRA = 67):
      ≤ $30k → ~54%  |  $30–60k → ~40%  |  $60–120k → ~30%  |  > $120k → ~27%
    Capped at 2024 maximum of ~$52,000/yr at FRA.

    Claiming age adjustments (vs FRA=67):
      Age 62: −30% | 65: −13.3% | 67: 0% | 70: +24%

    NOTE: Educational approximation only. Use SSA.gov for actual estimates.
    """
    income = gross_annual_income
    if income <= 0:
        return 0.0

    # Replacement rate by income tier
    if income <= 30_000:
        rate = 0.54
    elif income <= 60_000:
        rate = 0.40
    elif income <= 120_000:
        rate = 0.30
    else:
        rate = 0.27

    annual_benefit = min(income * rate, 52_000.0)

    # Claiming age adjustment relative to FRA (67)
    _age_adj = {62: -0.300, 63: -0.250, 64: -0.167, 65: -0.133, 66: -0.067,
                67:  0.000, 68:  0.080, 69:  0.160, 70:  0.240}
    adj = _age_adj.get(max(62, min(70, claiming_age)), 0.0)
    return round(annual_benefit * (1 + adj), 2)


def check_401k_limit(annual_employee_contrib: float, age: int) -> Dict[str, Any]:
    """
    Check whether employee 401k contributions exceed IRS 2024 limits.
    Catch-up contributions allowed for age >= 50.

    Returns:
        over_limit (bool), limit (float), excess (float), catch_up_eligible (bool)
    """
    catch_up = age >= 50
    limit = 30_500.0 if catch_up else 23_000.0
    excess = max(0.0, annual_employee_contrib - limit)
    return {
        "over_limit":        excess > 0,
        "limit":             limit,
        "excess":            round(excess, 2),
        "catch_up_eligible": catch_up,
    }


def calc_avalanche_payoff(
    debts: list,          # list of {"name": str, "balance": float, "rate": float, "min_payment": float}
    extra_monthly: float = 0.0,
) -> Dict[str, Any]:
    """
    Debt avalanche: extra payments applied to highest-interest debt first.
    Returns months to full payoff and total interest paid.
    Capped at 360 months (30 years).
    """
    import copy
    d = [x for x in copy.deepcopy(debts) if x.get("balance", 0) > 0]
    if not d:
        return {"months": 0, "total_interest": 0.0}

    months = 0
    total_interest = 0.0

    while any(x["balance"] > 0 for x in d) and months < 360:
        months += 1
        # Sort highest rate first for extra payment targeting
        active_sorted = sorted([x for x in d if x["balance"] > 0], key=lambda x: -x["rate"])
        remaining_extra = extra_monthly

        for debt in d:
            if debt["balance"] <= 0:
                continue
            interest = debt["balance"] * (debt["rate"] / 12)
            total_interest += interest
            debt["balance"] += interest
            payment = min(debt["min_payment"], debt["balance"])
            debt["balance"] = round(max(0.0, debt["balance"] - payment), 2)

        for debt in active_sorted:
            if debt["balance"] <= 0 or remaining_extra <= 0:
                break
            paid = min(remaining_extra, debt["balance"])
            debt["balance"] = round(max(0.0, debt["balance"] - paid), 2)
            remaining_extra -= paid

    return {"months": months, "total_interest": round(total_interest, 2)}


def calc_snowball_payoff(
    debts: list,
    extra_monthly: float = 0.0,
) -> Dict[str, Any]:
    """
    Debt snowball: extra payments applied to smallest-balance debt first.
    Returns months to full payoff and total interest paid.
    Capped at 360 months.
    """
    import copy
    d = [x for x in copy.deepcopy(debts) if x.get("balance", 0) > 0]
    if not d:
        return {"months": 0, "total_interest": 0.0}

    months = 0
    total_interest = 0.0

    while any(x["balance"] > 0 for x in d) and months < 360:
        months += 1
        active_sorted = sorted([x for x in d if x["balance"] > 0], key=lambda x: x["balance"])
        remaining_extra = extra_monthly

        for debt in d:
            if debt["balance"] <= 0:
                continue
            interest = debt["balance"] * (debt["rate"] / 12)
            total_interest += interest
            debt["balance"] += interest
            payment = min(debt["min_payment"], debt["balance"])
            debt["balance"] = round(max(0.0, debt["balance"] - payment), 2)

        for debt in active_sorted:
            if debt["balance"] <= 0 or remaining_extra <= 0:
                break
            paid = min(remaining_extra, debt["balance"])
            debt["balance"] = round(max(0.0, debt["balance"] - paid), 2)
            remaining_extra -= paid

    return {"months": months, "total_interest": round(total_interest, 2)}


# ── Tax estimates (rough) ─────────────────────────────────────────────────────

def estimate_effective_tax_rate(gross_annual_income: float, filing_status: str = "single") -> float:
    """
    Very rough effective federal income tax rate estimate for a given income level.
    NOT a substitute for actual tax calculation. Used only for cash-flow approximation.
    Rates are approximate 2024 effective rates (federal only).
    """
    income = gross_annual_income
    if filing_status in ("married", "widowed"):
        # Rough brackets for MFJ
        if income <= 30000:    return 0.08
        if income <= 80000:    return 0.12
        if income <= 160000:   return 0.17
        if income <= 250000:   return 0.21
        if income <= 400000:   return 0.24
        return 0.29
    else:
        # Single / divorced
        if income <= 15000:    return 0.08
        if income <= 45000:    return 0.12
        if income <= 90000:    return 0.18
        if income <= 150000:   return 0.22
        if income <= 250000:   return 0.26
        return 0.32


def calc_total_tax_rate(
    gross_annual_income: float,
    filing_status: str = "single",
    state_income_tax_rate: float = 0.0,
) -> float:
    """
    Combined effective tax rate = federal effective rate + state flat rate.
    Capped at 0.60 to prevent unrealistic results.
    """
    federal = estimate_effective_tax_rate(gross_annual_income, filing_status)
    return min(federal + state_income_tax_rate, 0.60)
