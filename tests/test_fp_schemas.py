"""
Unit tests for streamlit_app/fp_schemas.py.

All tests are pure (no I/O, no side effects, no LLM).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))

from fp_schemas import (  # noqa: E402
    ClientProfile, MonthlyExpenses, Assets, Liabilities,
    MonthlyDebtPayments, Insurance, RetirementInfo, Goal,
)


# ── MonthlyExpenses ───────────────────────────────────────────────────────────

class TestMonthlyExpenses:
    def test_total_sums_all_fields(self):
        e = MonthlyExpenses(
            housing=1000, food=500, transportation=200,
            utilities=100, healthcare=50, childcare=0,
            entertainment=100, personal=50, subscriptions=30, other=70,
        )
        assert e.total() == 2100.0

    def test_zero_defaults(self):
        assert MonthlyExpenses().total() == 0.0

    def test_partial_fields(self):
        e = MonthlyExpenses(housing=2000, food=600)
        assert e.total() == 2600.0


# ── Assets ────────────────────────────────────────────────────────────────────

class TestAssets:
    def test_liquid_includes_checking_and_brokerage_only(self):
        a = Assets(
            checking_savings=5000, investments_brokerage=10000,
            retirement_401k=50000, retirement_ira=20000,
        )
        assert a.liquid() == 15000.0

    def test_liquid_excludes_retirement(self):
        a = Assets(retirement_401k=100000, retirement_ira=50000)
        assert a.liquid() == 0.0

    def test_retirement_total(self):
        a = Assets(retirement_401k=100000, retirement_ira=25000)
        assert a.retirement_total() == 125000.0

    def test_total_includes_all_fields(self):
        a = Assets(
            checking_savings=1000, investments_brokerage=2000,
            retirement_401k=3000, retirement_ira=4000,
            real_estate_equity=5000, college_529=6000, other=7000,
        )
        assert a.total() == 28000.0

    def test_zero_defaults(self):
        a = Assets()
        assert a.liquid() == 0.0
        assert a.total() == 0.0


# ── Liabilities ───────────────────────────────────────────────────────────────

class TestLiabilities:
    def test_total(self):
        li = Liabilities(
            mortgage=200000, car_loans=15000, student_loans=10000,
            credit_cards=5000, other=1000,
        )
        assert li.total() == 231000.0

    def test_consumer_total_excludes_mortgage(self):
        li = Liabilities(
            mortgage=200000, car_loans=15000, student_loans=10000,
            credit_cards=5000, other=1000,
        )
        assert li.consumer_total() == 31000.0

    def test_zero_defaults(self):
        assert Liabilities().total() == 0.0
        assert Liabilities().consumer_total() == 0.0


# ── MonthlyDebtPayments ───────────────────────────────────────────────────────

class TestMonthlyDebtPayments:
    def test_total(self):
        mdp = MonthlyDebtPayments(
            mortgage=1500, car=300, student_loans=200,
            credit_cards=100, other=50,
        )
        assert mdp.total() == 2150.0

    def test_zero_defaults(self):
        assert MonthlyDebtPayments().total() == 0.0


# ── ClientProfile defaults ────────────────────────────────────────────────────

class TestClientProfileDefaults:
    def test_default_name(self):
        assert ClientProfile().name == "Client"

    def test_default_age(self):
        assert ClientProfile().age == 35

    def test_default_marital_status(self):
        assert ClientProfile().marital_status == "single"

    def test_default_income_zero(self):
        p = ClientProfile()
        assert p.gross_annual_income == 0.0
        assert p.spouse_annual_income == 0.0

    def test_default_goals_empty(self):
        assert ClientProfile().goals == []


# ── Derived helper methods ────────────────────────────────────────────────────

class TestClientProfileDerivedMethods:
    def _make(self):
        p = ClientProfile()
        p.gross_annual_income = 60000
        p.spouse_annual_income = 40000
        p.other_annual_income = 2000
        return p

    def test_gross_monthly_income(self):
        p = self._make()
        assert p.gross_monthly_income() == (60000 + 40000 + 2000) / 12

    def test_total_annual_income(self):
        assert self._make().total_annual_income() == 102000.0

    def test_net_worth_negative(self):
        p = ClientProfile()
        p.assets = Assets(checking_savings=50000, retirement_401k=100000)
        p.liabilities = Liabilities(mortgage=200000, credit_cards=5000)
        assert p.net_worth() == -55000.0

    def test_net_worth_positive(self):
        p = ClientProfile()
        p.assets = Assets(checking_savings=300000)
        p.liabilities = Liabilities(mortgage=100000)
        assert p.net_worth() == 200000.0

    def test_zero_income_gross_monthly(self):
        assert ClientProfile().gross_monthly_income() == 0.0


# ── from_dict / to_dict roundtrip ─────────────────────────────────────────────

class TestClientProfileRoundtrip:
    def test_full_roundtrip(self):
        p = ClientProfile()
        p.name = "Alice"
        p.age = 42
        p.gross_annual_income = 90000.0
        p.monthly_expenses = MonthlyExpenses(housing=2000, food=800)
        p.assets = Assets(checking_savings=20000, retirement_401k=150000)
        p.liabilities = Liabilities(mortgage=250000)
        p.insurance = Insurance(
            has_health=True, has_life=True,
            life_coverage_amount=500000, has_disability=True,
        )
        p.retirement = RetirementInfo(
            contribution_rate_pct=10, employer_match_pct=5,
            target_retirement_age=65,
        )
        p.goals = [Goal(
            type="home_purchase", description="Buy a house",
            target_amount=100000, timeline_months=24,
        )]

        p2 = ClientProfile.from_dict(p.to_dict())

        assert p2.name == "Alice"
        assert p2.age == 42
        assert p2.gross_annual_income == 90000.0
        assert p2.monthly_expenses.housing == 2000.0
        assert p2.assets.checking_savings == 20000.0
        assert p2.liabilities.mortgage == 250000.0
        assert p2.insurance.has_life is True
        assert p2.insurance.life_coverage_amount == 500000.0
        assert p2.retirement.contribution_rate_pct == 10.0
        assert len(p2.goals) == 1
        assert p2.goals[0].type == "home_purchase"

    def test_from_dict_handles_empty_dict(self):
        p = ClientProfile.from_dict({})
        assert p.name == "Client"
        assert p.age == 35
        assert p.goals == []

    def test_from_dict_nested_assets(self):
        d = {"assets": {"checking_savings": 12000, "retirement_401k": 50000}}
        p = ClientProfile.from_dict(d)
        assert p.assets.checking_savings == 12000.0
        assert p.assets.retirement_401k == 50000.0
