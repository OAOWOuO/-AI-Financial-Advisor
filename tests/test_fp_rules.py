"""
Unit tests for streamlit_app/fp_rules.py (RulesEngine).

No LLM, no API keys required.
RulesEngine reads planning_rules.json from data/rule_configs/ — committed to repo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))

from fp_schemas import (  # noqa: E402
    ClientProfile, MonthlyExpenses, Assets, Liabilities,
    MonthlyDebtPayments, Insurance, RetirementInfo, IssueSeverity,
)
from fp_rules import RulesEngine  # noqa: E402


# Shared engine instance (reads JSON once)
ENGINE = RulesEngine()


def _base_profile() -> ClientProfile:
    """Return a ClientProfile that passes most checks — tweaked per test."""
    p = ClientProfile()
    p.age = 40
    p.marital_status = "married"
    p.dependents = 0
    p.gross_annual_income = 80000
    p.monthly_expenses = MonthlyExpenses(housing=2000, food=600, utilities=200)
    p.assets = Assets(checking_savings=30000, retirement_401k=100000)
    p.liabilities = Liabilities()
    p.monthly_debt_payments = MonthlyDebtPayments()
    p.insurance = Insurance(
        has_health=True, has_life=True,
        life_coverage_amount=800000, has_disability=True,
    )
    p.retirement = RetirementInfo(
        contribution_rate_pct=10, employer_match_pct=5,
        target_retirement_age=65,
    )
    return p


# ── Emergency fund ────────────────────────────────────────────────────────────

class TestEmergencyFundCheck:
    def test_critical_below_minimum(self):
        # 1500 liquid / 1000 monthly = 1.5 months < 3 minimum
        p = _base_profile()
        p.monthly_expenses = MonthlyExpenses(housing=1000)
        p.assets = Assets(checking_savings=1500)
        issues = ENGINE._check_emergency_fund(p)
        assert issues[0].severity == IssueSeverity.CRITICAL

    def test_high_below_recommended_for_married(self):
        # 4.0 months — meets 3-month minimum but below 6-month married target
        p = _base_profile()
        p.monthly_expenses = MonthlyExpenses(housing=1000)
        p.assets = Assets(checking_savings=4000)
        issues = ENGINE._check_emergency_fund(p)
        assert issues[0].severity == IssueSeverity.HIGH

    def test_info_when_adequate_for_married(self):
        # 7.0 months >= 6-month target for married household
        p = _base_profile()
        p.monthly_expenses = MonthlyExpenses(housing=1000)
        p.assets = Assets(checking_savings=7000)
        issues = ENGINE._check_emergency_fund(p)
        assert issues[0].severity == IssueSeverity.INFO

    def test_single_requires_9_months(self):
        # Single household needs 9 months (high-risk threshold)
        # 7 months satisfies married target but not single target
        p = _base_profile()
        p.marital_status = "single"
        p.monthly_expenses = MonthlyExpenses(housing=1000)
        p.assets = Assets(checking_savings=7000)
        issues = ENGINE._check_emergency_fund(p)
        assert issues[0].severity == IssueSeverity.HIGH

    def test_returns_exactly_one_issue(self):
        p = _base_profile()
        issues = ENGINE._check_emergency_fund(p)
        assert len(issues) == 1


# ── Debt ─────────────────────────────────────────────────────────────────────

class TestDebtCheck:
    # gm = 80000/12 ≈ 6667; thresholds: >43%=CRITICAL, >36%=MEDIUM, ≤36%=INFO

    def test_critical_when_dti_over_43pct(self):
        p = _base_profile()
        p.monthly_debt_payments = MonthlyDebtPayments(mortgage=3000)  # ~45%
        issues = ENGINE._check_debt(p)
        dti = next(i for i in issues if "Debt-to-income" in i.title)
        assert dti.severity == IssueSeverity.CRITICAL

    def test_medium_when_dti_between_36_and_43pct(self):
        p = _base_profile()
        p.monthly_debt_payments = MonthlyDebtPayments(mortgage=2500)  # ~37.5%
        issues = ENGINE._check_debt(p)
        dti = next(i for i in issues if "Debt-to-income" in i.title)
        assert dti.severity == IssueSeverity.MEDIUM

    def test_info_when_dti_within_range(self):
        p = _base_profile()
        p.monthly_debt_payments = MonthlyDebtPayments(mortgage=1500)  # ~22.5%
        issues = ENGINE._check_debt(p)
        dti = next(i for i in issues if "Debt-to-income" in i.title)
        assert dti.severity == IssueSeverity.INFO

    def test_high_credit_card_flag_when_balance_present(self):
        p = _base_profile()
        p.liabilities = Liabilities(credit_cards=8000)
        p.monthly_debt_payments = MonthlyDebtPayments(credit_cards=200)
        issues = ENGINE._check_debt(p)
        cc = next((i for i in issues if "credit card" in i.title.lower()), None)
        assert cc is not None
        assert cc.severity == IssueSeverity.HIGH

    def test_no_credit_card_flag_when_zero_balance(self):
        p = _base_profile()  # all liabilities are 0
        issues = ENGINE._check_debt(p)
        cc = next((i for i in issues if "credit card" in i.title.lower()), None)
        assert cc is None


# ── Cash flow ─────────────────────────────────────────────────────────────────

class TestCashFlowCheck:
    def test_critical_when_negative_cash_flow(self):
        # Income $1000/mo, expenses $800, debt $500 → clearly negative
        p = _base_profile()
        p.gross_annual_income = 12000
        p.monthly_expenses = MonthlyExpenses(housing=800)
        p.monthly_debt_payments = MonthlyDebtPayments(credit_cards=500)
        issues = ENGINE._check_cash_flow(p)
        assert issues[0].severity == IssueSeverity.CRITICAL

    def test_info_when_healthy_surplus(self):
        # Income $10000/mo, expenses $2000, no debt → large surplus
        p = _base_profile()
        p.gross_annual_income = 120000
        p.monthly_expenses = MonthlyExpenses(housing=2000)
        p.monthly_debt_payments = MonthlyDebtPayments()
        issues = ENGINE._check_cash_flow(p)
        assert issues[0].severity == IssueSeverity.INFO

    def test_returns_exactly_one_issue(self):
        p = _base_profile()
        issues = ENGINE._check_cash_flow(p)
        assert len(issues) == 1


# ── Insurance ─────────────────────────────────────────────────────────────────

class TestInsuranceCheck:
    def test_critical_no_health_insurance(self):
        p = _base_profile()
        p.insurance = Insurance(has_health=False)
        issues = ENGINE._check_insurance(p)
        h = next(i for i in issues if "health" in i.title.lower())
        assert h.severity == IssueSeverity.CRITICAL

    def test_critical_no_life_with_dependents(self):
        p = _base_profile()
        p.dependents = 2
        p.insurance = Insurance(has_health=True, has_life=False)
        issues = ENGINE._check_insurance(p)
        li = next(
            i for i in issues
            if "life insurance" in i.title.lower() and "dependents" in i.title.lower()
        )
        assert li.severity == IssueSeverity.CRITICAL

    def test_high_insufficient_life_coverage(self):
        # 15× required for married/dependents; 300k/80k = 3.75× → too low
        p = _base_profile()
        p.dependents = 1
        p.insurance = Insurance(
            has_health=True, has_life=True,
            life_coverage_amount=300000, has_disability=True,
        )
        issues = ENGINE._check_insurance(p)
        li = next(
            (i for i in issues if "insufficient" in i.title.lower()), None
        )
        assert li is not None
        assert li.severity == IssueSeverity.HIGH

    def test_high_no_disability_insurance(self):
        p = _base_profile()
        p.insurance = Insurance(
            has_health=True, has_life=True,
            life_coverage_amount=800000, has_disability=False,
        )
        issues = ENGINE._check_insurance(p)
        di = next((i for i in issues if "disability" in i.title.lower()), None)
        assert di is not None
        assert di.severity == IssueSeverity.HIGH

    def test_medium_ltc_for_age_50_plus(self):
        p = _base_profile()
        p.age = 55
        p.insurance = Insurance(
            has_health=True, has_life=True,
            life_coverage_amount=800000, has_disability=True, has_ltc=False,
        )
        issues = ENGINE._check_insurance(p)
        ltc = next((i for i in issues if "long-term care" in i.title.lower()), None)
        assert ltc is not None
        assert ltc.severity == IssueSeverity.MEDIUM

    def test_no_ltc_flag_under_50(self):
        p = _base_profile()
        p.age = 45
        p.insurance = Insurance(
            has_health=True, has_life=True,
            life_coverage_amount=800000, has_disability=True, has_ltc=False,
        )
        issues = ENGINE._check_insurance(p)
        ltc = next((i for i in issues if "long-term care" in i.title.lower()), None)
        assert ltc is None


# ── Retirement ────────────────────────────────────────────────────────────────

class TestRetirementCheck:
    def test_high_when_not_capturing_full_match(self):
        p = _base_profile()
        p.retirement = RetirementInfo(
            contribution_rate_pct=2, employer_match_pct=5,
            target_retirement_age=65,
        )
        issues = ENGINE._check_retirement(p)
        match_issue = next((i for i in issues if "match" in i.title.lower()), None)
        assert match_issue is not None
        assert match_issue.severity == IssueSeverity.HIGH

    def test_no_match_issue_when_capturing_full_match(self):
        p = _base_profile()
        p.retirement = RetirementInfo(
            contribution_rate_pct=6, employer_match_pct=5,
            target_retirement_age=65,
        )
        issues = ENGINE._check_retirement(p)
        match_issue = next((i for i in issues if "match" in i.title.lower()), None)
        assert match_issue is None

    def test_always_returns_at_least_one_issue(self):
        p = _base_profile()
        issues = ENGINE._check_retirement(p)
        assert len(issues) >= 1


# ── run_all_checks ────────────────────────────────────────────────────────────

class TestRunAllChecks:
    def test_returns_non_empty_list(self):
        p = _base_profile()
        issues = ENGINE.run_all_checks(p)
        assert isinstance(issues, list)
        assert len(issues) > 0

    def test_critical_issues_sorted_first(self):
        p = _base_profile()
        p.insurance = Insurance(has_health=False)   # CRITICAL
        p.monthly_expenses = MonthlyExpenses(housing=1000)
        p.assets = Assets(checking_savings=500)     # 0.5 months < 3 → CRITICAL
        issues = ENGINE.run_all_checks(p)
        assert issues[0].severity == IssueSeverity.CRITICAL

    def test_severity_order_is_non_decreasing(self):
        """No HIGH/MEDIUM/LOW/INFO issue should appear before a CRITICAL one."""
        _rank = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.HIGH: 1,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 3,
            IssueSeverity.INFO: 4,
        }
        p = _base_profile()
        p.insurance = Insurance(has_health=False)
        issues = ENGINE.run_all_checks(p)
        ranks = [_rank.get(i.severity, 9) for i in issues]
        assert ranks == sorted(ranks)
