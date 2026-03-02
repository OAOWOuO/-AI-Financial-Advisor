"""
Unit tests for pure functions in streamlit_app/fp_report.py.

No LLM, no OpenAI API required.
Tested: build_quant_checks, build_recommendations, _parse_llm_sections, _build_snapshot
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))

from fp_schemas import (  # noqa: E402
    ClientProfile, MonthlyExpenses, Assets, Liabilities,
    MonthlyDebtPayments, Insurance, RetirementInfo,
    PlanningIssue, IssueSeverity, IssueCategory,
)
from fp_report import (  # noqa: E402
    build_quant_checks, build_recommendations,
    _parse_llm_sections, _build_snapshot,
)


def _profile() -> ClientProfile:
    p = ClientProfile()
    p.name = "Test Client"
    p.age = 40
    p.marital_status = "married"
    p.gross_annual_income = 80000
    p.monthly_expenses = MonthlyExpenses(housing=2000, food=600)
    p.assets = Assets(checking_savings=30000, retirement_401k=100000)
    p.liabilities = Liabilities()
    p.monthly_debt_payments = MonthlyDebtPayments()
    p.insurance = Insurance(has_health=True, has_life=True, life_coverage_amount=800000)
    p.retirement = RetirementInfo(contribution_rate_pct=10, employer_match_pct=5)
    return p


# ── build_quant_checks ────────────────────────────────────────────────────────

class TestBuildQuantChecks:
    def test_returns_non_empty_list(self):
        checks = build_quant_checks(_profile())
        assert isinstance(checks, list)
        assert len(checks) > 0

    def test_all_checks_have_required_fields(self):
        for c in build_quant_checks(_profile()):
            assert c.label
            assert c.value
            assert c.benchmark
            assert c.status in ("OK", "WARNING", "CRITICAL")

    def test_emergency_fund_critical_below_3_months(self):
        p = _profile()
        p.monthly_expenses = MonthlyExpenses(housing=1000)
        p.assets = Assets(checking_savings=1500)   # 1.5 months
        checks = build_quant_checks(p)
        ef = next(c for c in checks if "Emergency" in c.label)
        assert ef.status == "CRITICAL"

    def test_emergency_fund_ok_above_6_months(self):
        p = _profile()
        p.monthly_expenses = MonthlyExpenses(housing=1000)
        p.assets = Assets(checking_savings=8000)   # 8 months
        checks = build_quant_checks(p)
        ef = next(c for c in checks if "Emergency" in c.label)
        assert ef.status == "OK"

    def test_dti_ok_within_36pct(self):
        p = _profile()
        p.monthly_debt_payments = MonthlyDebtPayments(mortgage=1500)  # 22.5%
        checks = build_quant_checks(p)
        dti = next(c for c in checks if "DTI" in c.label or "Debt" in c.label)
        assert dti.status == "OK"

    def test_dti_critical_over_43pct(self):
        p = _profile()
        p.monthly_debt_payments = MonthlyDebtPayments(mortgage=3000)  # ~45%
        checks = build_quant_checks(p)
        dti = next(c for c in checks if "DTI" in c.label or "Debt" in c.label)
        assert dti.status == "CRITICAL"

    def test_cash_flow_critical_when_negative(self):
        p = _profile()
        p.gross_annual_income = 12000  # gm = $1000/mo
        p.monthly_expenses = MonthlyExpenses(housing=800)
        p.monthly_debt_payments = MonthlyDebtPayments(credit_cards=500)
        checks = build_quant_checks(p)
        cf = next(c for c in checks if "Cash Flow" in c.label)
        assert cf.status == "CRITICAL"


# ── build_recommendations ─────────────────────────────────────────────────────

class TestBuildRecommendations:
    def _issues(self):
        return [
            PlanningIssue(
                category=IssueCategory.EMERGENCY_FUND,
                severity=IssueSeverity.CRITICAL,
                title="Emergency fund critically low",
                detail="Only 1.5 months covered.",
                action_hint="Build to $3,000.",
            ),
            PlanningIssue(
                category=IssueCategory.DEBT,
                severity=IssueSeverity.INFO,
                title="DTI within range",
                detail="DTI is 22%.",
            ),
            PlanningIssue(
                category=IssueCategory.INSURANCE,
                severity=IssueSeverity.HIGH,
                title="No disability insurance",
                detail="Consider DI policy.",
                action_hint="Get disability insurance.",
            ),
        ]

    def test_info_issues_excluded(self):
        recs = build_recommendations(_profile(), self._issues(), [])
        assert not any("DTI" in r.action for r in recs)

    def test_critical_gets_immediate_timeline(self):
        recs = build_recommendations(_profile(), self._issues(), [])
        critical_rec = next(r for r in recs if "3,000" in r.action)
        assert critical_rec.timeline == "0–30 days"

    def test_high_gets_1_3_months_timeline(self):
        recs = build_recommendations(_profile(), self._issues(), [])
        high_rec = next(r for r in recs if "disability" in r.action.lower())
        assert high_rec.timeline == "1–3 months"

    def test_capped_at_10(self):
        many = [
            PlanningIssue(
                category=IssueCategory.CASH_FLOW, severity=IssueSeverity.HIGH,
                title=f"Issue {i}", detail="Detail.",
            )
            for i in range(15)
        ]
        recs = build_recommendations(_profile(), many, [])
        assert len(recs) <= 10

    def test_priority_starts_at_one(self):
        recs = build_recommendations(_profile(), self._issues(), [])
        assert recs[0].priority == 1

    def test_empty_issues_returns_empty_list(self):
        assert build_recommendations(_profile(), [], []) == []


# ── _parse_llm_sections ───────────────────────────────────────────────────────

class TestParseLlmSections:
    _SAMPLE = (
        "## Executive Summary\n"
        "The client has strong cash flow but insufficient emergency fund.\n\n"
        "## Case Reasoning\n"
        "Similar to the Rogers family case.\n\n"
        "## Follow-up Questions\n"
        "- How much do you spend monthly?\n"
        "- Do you have dependents?\n\n"
        "## Missing Information\n"
        "- Life insurance policy details\n"
    )

    def test_extracts_executive_summary(self):
        result = _parse_llm_sections(self._SAMPLE)
        assert "strong cash flow" in result["executive_summary"]

    def test_extracts_case_reasoning(self):
        result = _parse_llm_sections(self._SAMPLE)
        assert "Rogers" in result["case_reasoning"]

    def test_follow_up_as_list(self):
        result = _parse_llm_sections(self._SAMPLE)
        assert isinstance(result["follow_up"], list)
        assert any("spend monthly" in q for q in result["follow_up"])

    def test_missing_info_as_list(self):
        result = _parse_llm_sections(self._SAMPLE)
        assert isinstance(result["missing_info"], list)
        assert any("Life insurance" in m for m in result["missing_info"])

    def test_absent_section_returns_empty(self):
        result = _parse_llm_sections("## Executive Summary\nHello.")
        assert result["case_reasoning"] == ""

    def test_empty_string_returns_empty_sections(self):
        result = _parse_llm_sections("")
        assert result["executive_summary"] == ""
        assert result["follow_up"] == []


# ── _build_snapshot ───────────────────────────────────────────────────────────

class TestBuildSnapshot:
    def test_contains_name(self):
        assert "Test Client" in _build_snapshot(_profile())

    def test_contains_age(self):
        assert "40" in _build_snapshot(_profile())

    def test_contains_income(self):
        assert "80,000" in _build_snapshot(_profile())

    def test_contains_net_worth(self):
        p = _profile()
        snapshot = _build_snapshot(p)
        nw = p.net_worth()
        assert f"{nw:,.0f}" in snapshot
