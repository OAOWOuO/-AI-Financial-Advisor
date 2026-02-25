"""
fp_rules.py — Transparent rules engine for financial planning gap analysis.

Rules are loaded from planning_rules.json so thresholds are visible and editable.
Each check returns a list of PlanningIssue objects.

Design principle: deterministic rules, not LLM decisions.
The LLM is used only for narrative explanation, never for threshold judgments.
"""
from __future__ import annotations
import json
import os
from typing import List, Dict, Any

from fp_schemas import ClientProfile, PlanningIssue, IssueSeverity, IssueCategory
import fp_calculators as calc


_RULES_PATH = os.path.join(os.path.dirname(__file__), "data", "rule_configs", "planning_rules.json")

def _load_rules() -> Dict[str, Any]:
    with open(_RULES_PATH, "r") as f:
        return json.load(f)


class RulesEngine:
    """
    Evaluates a ClientProfile against configurable planning thresholds.

    Usage:
        engine = RulesEngine()
        issues = engine.run_all_checks(profile)
    """

    def __init__(self, rules_path: str = _RULES_PATH):
        with open(rules_path) as f:
            self.rules = json.load(f)

    # ─── Public entry point ───────────────────────────────────────────────────

    def run_all_checks(self, profile: ClientProfile) -> List[PlanningIssue]:
        issues: List[PlanningIssue] = []
        issues.extend(self._check_emergency_fund(profile))
        issues.extend(self._check_debt(profile))
        issues.extend(self._check_cash_flow(profile))
        issues.extend(self._check_insurance(profile))
        issues.extend(self._check_retirement(profile))
        issues.extend(self._check_net_worth(profile))
        issues.extend(self._check_goals(profile))
        # Sort: CRITICAL first, then HIGH, MEDIUM, LOW, INFO
        _order = {IssueSeverity.CRITICAL: 0, IssueSeverity.HIGH: 1,
                  IssueSeverity.MEDIUM: 2,  IssueSeverity.LOW: 3, IssueSeverity.INFO: 4}
        issues.sort(key=lambda i: _order.get(i.severity, 9))
        return issues

    # ─── Individual checks ────────────────────────────────────────────────────

    def _check_emergency_fund(self, p: ClientProfile) -> List[PlanningIssue]:
        issues = []
        r = self.rules["emergency_fund"]
        monthly_exp = p.monthly_expenses.total()
        liquid      = p.assets.liquid()
        months      = calc.calc_emergency_fund_months(liquid, monthly_exp)

        min_m = r["minimum_months"]
        rec_m = r["recommended_months"]
        hi_m  = r["high_risk_months"]

        # Single-income or self-employed households need more buffer
        needs_higher = (p.marital_status == "single" or p.dependents > 0)
        target_m = hi_m if needs_higher else rec_m

        if months < min_m:
            issues.append(PlanningIssue(
                category    = IssueCategory.EMERGENCY_FUND,
                severity    = IssueSeverity.CRITICAL,
                title       = "Emergency fund critically low",
                detail      = (f"Only {months:.1f} months of expenses are covered by liquid assets. "
                               f"Minimum recommended is {min_m} months."),
                metric_value= f"{months:.1f} months",
                benchmark   = f"≥ {min_m} months (minimum)",
                action_hint = f"Build liquid savings to at least ${monthly_exp * min_m:,.0f}.",
            ))
        elif months < target_m:
            issues.append(PlanningIssue(
                category    = IssueCategory.EMERGENCY_FUND,
                severity    = IssueSeverity.HIGH,
                title       = f"Emergency fund below {target_m}-month target",
                detail      = (f"Currently {months:.1f} months covered. "
                               f"For {'single-income households or families with dependents' if needs_higher else 'this household profile'}, "
                               f"{target_m} months is recommended."),
                metric_value= f"{months:.1f} months",
                benchmark   = f"≥ {target_m} months (recommended)",
                action_hint = f"Target savings of ${monthly_exp * target_m:,.0f}.",
            ))
        else:
            issues.append(PlanningIssue(
                category    = IssueCategory.EMERGENCY_FUND,
                severity    = IssueSeverity.INFO,
                title       = "Emergency fund adequate",
                detail      = f"{months:.1f} months of expenses covered — meets the {rec_m}-month benchmark.",
                metric_value= f"{months:.1f} months",
                benchmark   = f"≥ {rec_m} months",
            ))
        return issues

    def _check_debt(self, p: ClientProfile) -> List[PlanningIssue]:
        issues = []
        r  = self.rules["debt"]
        gm = p.gross_monthly_income()
        dti = calc.calc_dti(p.monthly_debt_payments.total(), gm)
        hr  = calc.calc_housing_ratio(p.monthly_debt_payments.mortgage, gm)
        consumer_burden = calc.calc_consumer_debt_burden(
            p.liabilities.consumer_total(), p.total_annual_income()
        )

        # DTI check
        if dti > r["max_total_dti"]:
            issues.append(PlanningIssue(
                category    = IssueCategory.DEBT,
                severity    = IssueSeverity.CRITICAL,
                title       = "Debt-to-income ratio dangerously high",
                detail      = (f"Total DTI of {dti*100:.1f}% exceeds the maximum threshold of "
                               f"{r['max_total_dti']*100:.0f}%. This level of debt burden severely "
                               f"limits financial flexibility and may jeopardize loan qualification."),
                metric_value= f"DTI = {dti*100:.1f}%",
                benchmark   = f"≤ {r['max_total_dti']*100:.0f}%",
                action_hint = "Prioritize aggressive debt paydown or income growth.",
            ))
        elif dti > r["recommended_dti"]:
            issues.append(PlanningIssue(
                category    = IssueCategory.DEBT,
                severity    = IssueSeverity.MEDIUM,
                title       = "Debt-to-income ratio above recommended level",
                detail      = (f"DTI is {dti*100:.1f}%, above the recommended {r['recommended_dti']*100:.0f}%. "
                               "While manageable, it limits savings capacity and emergency resilience."),
                metric_value= f"DTI = {dti*100:.1f}%",
                benchmark   = f"≤ {r['recommended_dti']*100:.0f}% recommended",
                action_hint = "Target debt reduction to improve cash flow.",
            ))
        else:
            issues.append(PlanningIssue(
                category    = IssueCategory.DEBT,
                severity    = IssueSeverity.INFO,
                title       = "Debt-to-income ratio within range",
                detail      = f"DTI of {dti*100:.1f}% is within the recommended threshold of {r['recommended_dti']*100:.0f}%.",
                metric_value= f"DTI = {dti*100:.1f}%",
                benchmark   = f"≤ {r['recommended_dti']*100:.0f}%",
            ))

        # Credit card / high-interest consumer debt flag
        if p.liabilities.credit_cards > 0:
            issues.append(PlanningIssue(
                category    = IssueCategory.DEBT,
                severity    = IssueSeverity.HIGH,
                title       = "High-interest credit card debt present",
                detail      = (f"Carrying ${p.liabilities.credit_cards:,.0f} in credit card debt. "
                               "Credit card interest rates typically exceed 20% APR — "
                               "this is the highest-priority debt to eliminate."),
                metric_value= f"${p.liabilities.credit_cards:,.0f}",
                benchmark   = "$0 (target)",
                action_hint = "Pay off credit cards before investing beyond 401k match.",
            ))

        return issues

    def _check_cash_flow(self, p: ClientProfile) -> List[PlanningIssue]:
        issues = []
        gm   = p.gross_monthly_income()
        tax_rate = 0.22 if p.marital_status == "married" else 0.25
        cf   = calc.calc_monthly_cash_flow(
            gm, p.monthly_expenses.total(),
            p.monthly_debt_payments.total(), tax_rate
        )
        if cf < 0:
            issues.append(PlanningIssue(
                category    = IssueCategory.CASH_FLOW,
                severity    = IssueSeverity.CRITICAL,
                title       = "Monthly cash flow is negative",
                detail      = (f"Estimated monthly deficit of ${abs(cf):,.0f}. "
                               "Spending (including debt payments) exceeds after-tax income. "
                               "This situation is unsustainable and will erode savings over time."),
                metric_value= f"-${abs(cf):,.0f}/month",
                benchmark   = "Positive surplus",
                action_hint = "Identify discretionary cuts or income growth opportunities immediately.",
            ))
        elif cf < gm * 0.10:
            issues.append(PlanningIssue(
                category    = IssueCategory.CASH_FLOW,
                severity    = IssueSeverity.MEDIUM,
                title       = "Monthly cash flow surplus is thin",
                detail      = (f"Estimated monthly surplus is ${cf:,.0f} "
                               f"({cf/gm*100:.1f}% of gross income). "
                               "Little room for unexpected expenses or accelerated savings."),
                metric_value= f"${cf:,.0f}/month",
                benchmark   = "≥ 10% of gross income",
                action_hint = "Review discretionary spending to widen the buffer.",
            ))
        else:
            issues.append(PlanningIssue(
                category    = IssueCategory.CASH_FLOW,
                severity    = IssueSeverity.INFO,
                title       = "Cash flow surplus is adequate",
                detail      = f"Estimated monthly surplus of ${cf:,.0f} provides reasonable flexibility.",
                metric_value= f"${cf:,.0f}/month",
                benchmark   = "≥ 10% of gross income",
            ))
        return issues

    def _check_insurance(self, p: ClientProfile) -> List[PlanningIssue]:
        issues = []
        r = self.rules["insurance"]

        # Health insurance
        if not p.insurance.has_health:
            issues.append(PlanningIssue(
                category    = IssueCategory.INSURANCE,
                severity    = IssueSeverity.CRITICAL,
                title       = "No health insurance coverage",
                detail      = "A single medical event without insurance can cause financial catastrophe. Obtain coverage immediately.",
                action_hint = "Explore employer plan, ACA marketplace, or Medicaid.",
            ))

        # Life insurance
        if p.gross_annual_income > 0:
            coverage_ratio = calc.calc_life_insurance_coverage_ratio(
                p.insurance.life_coverage_amount, p.total_annual_income()
            )
            has_dependents = p.dependents > 0 or p.marital_status in ("married",)
            min_mult = r["life_insurance_dependent_multiple"] if has_dependents else r["life_insurance_recommended_multiple"]

            if not p.insurance.has_life and has_dependents:
                issues.append(PlanningIssue(
                    category    = IssueCategory.INSURANCE,
                    severity    = IssueSeverity.CRITICAL,
                    title       = "No life insurance — dependents at risk",
                    detail      = ("No life insurance coverage detected while the household has dependents. "
                                   f"Recommended coverage: {min_mult}× annual income = "
                                   f"${p.total_annual_income() * min_mult:,.0f}."),
                    metric_value= "$0 coverage",
                    benchmark   = f"{min_mult}× income = ${p.total_annual_income()*min_mult:,.0f}",
                    action_hint = "Obtain term life insurance immediately.",
                ))
            elif p.insurance.has_life and coverage_ratio < min_mult:
                issues.append(PlanningIssue(
                    category    = IssueCategory.INSURANCE,
                    severity    = IssueSeverity.HIGH,
                    title       = "Life insurance coverage may be insufficient",
                    detail      = (f"Current coverage of ${p.insurance.life_coverage_amount:,.0f} "
                                   f"is {coverage_ratio:.1f}× income. "
                                   f"Recommended: {min_mult}× for {'a household with dependents' if has_dependents else 'this profile'}."),
                    metric_value= f"{coverage_ratio:.1f}× income",
                    benchmark   = f"{min_mult}× income",
                    action_hint = "Review and increase term life coverage.",
                ))

        # Disability insurance
        if not p.insurance.has_disability and p.gross_annual_income > 30000:
            issues.append(PlanningIssue(
                category    = IssueCategory.INSURANCE,
                severity    = IssueSeverity.HIGH,
                title       = "No disability income insurance",
                detail      = ("A long-term disability is statistically more likely than early death "
                               "for working-age adults. Loss of income without disability coverage "
                               "can deplete savings rapidly."),
                action_hint = "Explore employer group disability or individual DI policy (60% income replacement).",
            ))

        # LTC for older clients
        if p.age >= 50 and not p.insurance.has_ltc:
            issues.append(PlanningIssue(
                category    = IssueCategory.INSURANCE,
                severity    = IssueSeverity.MEDIUM,
                title       = "Long-term care insurance not in place",
                detail      = (f"At age {p.age}, it is advisable to evaluate long-term care insurance. "
                               "Average nursing home costs exceed $100K/year and are not covered by Medicare for extended stays."),
                action_hint = "Obtain LTC insurance quotes before age 60 when premiums are lower.",
            ))

        return issues

    def _check_retirement(self, p: ClientProfile) -> List[PlanningIssue]:
        issues = []
        r = self.rules["retirement"]

        years_to_ret = max(0, p.retirement.target_retirement_age - p.age)
        annual_contrib = p.gross_annual_income * (p.retirement.contribution_rate_pct / 100)
        employer_match = p.gross_annual_income * (p.retirement.employer_match_pct / 100)

        # Check if capturing full employer match
        if (p.retirement.employer_match_pct > 0
                and p.retirement.contribution_rate_pct < p.retirement.employer_match_pct):
            issues.append(PlanningIssue(
                category    = IssueCategory.RETIREMENT,
                severity    = IssueSeverity.HIGH,
                title       = "Not capturing full employer 401k match",
                detail      = (f"Employer matches up to {p.retirement.employer_match_pct:.0f}% of income, "
                               f"but current contribution is only {p.retirement.contribution_rate_pct:.0f}%. "
                               "Uncaptured match = "
                               f"${(p.retirement.employer_match_pct - p.retirement.contribution_rate_pct)/100 * p.gross_annual_income:,.0f}/year of free money left uncollected."),
                metric_value= f"{p.retirement.contribution_rate_pct:.0f}% contribution",
                benchmark   = f"≥ {p.retirement.employer_match_pct:.0f}% to capture full match",
                action_hint = f"Increase 401k contribution to at least {p.retirement.employer_match_pct:.0f}% immediately.",
            ))

        # Retirement savings trajectory
        current_balance = p.assets.retirement_total()
        balanced_projection = calc.calc_retirement_projection(
            current_balance, annual_contrib + employer_match,
            years_to_ret, r["balanced_real_return"], r["inflation_assumption"]
        )
        target_income  = p.total_annual_income() * r["replacement_ratio"]
        corpus_needed  = calc.calc_retirement_corpus_needed(target_income, r["safe_withdrawal_rate"])
        gap = corpus_needed - balanced_projection["corpus_nominal"]

        if gap > 0 and years_to_ret > 0:
            severity = IssueSeverity.HIGH if gap > corpus_needed * 0.3 else IssueSeverity.MEDIUM
            issues.append(PlanningIssue(
                category    = IssueCategory.RETIREMENT,
                severity    = severity,
                title       = "Projected retirement corpus below target",
                detail      = (f"At current savings rate, the balanced-scenario projection is "
                               f"${balanced_projection['corpus_nominal']:,.0f} at age {p.retirement.target_retirement_age}. "
                               f"Target corpus (80% income, 4% SWR): ${corpus_needed:,.0f}. "
                               f"Projected shortfall: ${gap:,.0f}."),
                metric_value= f"Projected: ${balanced_projection['corpus_nominal']:,.0f}",
                benchmark   = f"Target: ${corpus_needed:,.0f}",
                action_hint = "Consider increasing savings rate, delaying retirement, or adjusting income targets.",
            ))
        elif years_to_ret > 0:
            issues.append(PlanningIssue(
                category    = IssueCategory.RETIREMENT,
                severity    = IssueSeverity.INFO,
                title       = "Retirement savings trajectory looks adequate",
                detail      = (f"Balanced-scenario projection: ${balanced_projection['corpus_nominal']:,.0f}. "
                               f"Target: ${corpus_needed:,.0f}. On track."),
                metric_value= f"Projected: ${balanced_projection['corpus_nominal']:,.0f}",
                benchmark   = f"Target: ${corpus_needed:,.0f}",
            ))

        return issues

    def _check_net_worth(self, p: ClientProfile) -> List[PlanningIssue]:
        issues = []
        nw        = p.net_worth()
        nw_target = calc.calc_net_worth_target(p.age, p.total_annual_income())
        ratio     = nw / nw_target if nw_target > 0 else 0

        if ratio < 0.5:
            severity = IssueSeverity.HIGH if ratio < 0.25 else IssueSeverity.MEDIUM
            issues.append(PlanningIssue(
                category    = IssueCategory.INVESTMENT,
                severity    = severity,
                title       = "Net worth significantly below age benchmark",
                detail      = (f"Current net worth: ${nw:,.0f}. "
                               f"Benchmark for age {p.age}: ${nw_target:,.0f} "
                               f"({ratio*100:.0f}% of target)."),
                metric_value= f"${nw:,.0f}",
                benchmark   = f"${nw_target:,.0f} (Stanley-Danko benchmark)",
                action_hint = "Accelerate savings rate and/or reduce liabilities.",
            ))
        elif ratio >= 1.5:
            issues.append(PlanningIssue(
                category    = IssueCategory.INVESTMENT,
                severity    = IssueSeverity.INFO,
                title       = "Net worth exceeds age benchmark",
                detail      = f"Net worth of ${nw:,.0f} is {ratio:.1f}× the age-{p.age} benchmark.",
                metric_value= f"${nw:,.0f}",
                benchmark   = f"${nw_target:,.0f}",
            ))

        return issues

    def _check_goals(self, p: ClientProfile) -> List[PlanningIssue]:
        issues = []
        gm = p.gross_monthly_income()
        monthly_surplus = calc.calc_monthly_cash_flow(
            gm, p.monthly_expenses.total(), p.monthly_debt_payments.total()
        )

        for goal in p.goals:
            if goal.target_amount <= 0 or goal.timeline_months <= 0:
                continue
            needed = calc.calc_goal_monthly_savings_needed(
                goal.target_amount, 0.0, goal.timeline_months
            )
            if needed > monthly_surplus * 0.50:
                issues.append(PlanningIssue(
                    category    = IssueCategory.GOALS,
                    severity    = IssueSeverity.MEDIUM,
                    title       = f"Goal may be challenging: {goal.description}",
                    detail      = (f"Reaching ${goal.target_amount:,.0f} in {goal.timeline_months} months "
                                   f"requires saving ~${needed:,.0f}/month, "
                                   f"which represents {needed/gm*100:.0f}% of gross monthly income."),
                    metric_value= f"Needs ${needed:,.0f}/month",
                    benchmark   = f"≤ {monthly_surplus*0.50:,.0f}/month (50% of surplus)",
                    action_hint = "Consider extending the timeline or reducing the target amount.",
                ))
        return issues
