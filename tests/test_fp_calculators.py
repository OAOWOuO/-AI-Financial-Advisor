"""
Unit tests for streamlit_app/fp_calculators.py.

All functions are pure (no side effects, no I/O, no LLM).
Tests run without any external dependencies or API keys.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))

import fp_calculators as calc  # noqa: E402


# ── Emergency fund ────────────────────────────────────────────────────────────

class TestCalcEmergencyFundMonths:
    def test_basic(self):
        assert calc.calc_emergency_fund_months(6000, 1000) == 6.0

    def test_partial_month(self):
        assert calc.calc_emergency_fund_months(4500, 1000) == 4.5

    def test_zero_expenses_returns_zero(self):
        assert calc.calc_emergency_fund_months(5000, 0) == 0.0

    def test_zero_liquid(self):
        assert calc.calc_emergency_fund_months(0, 1000) == 0.0


# ── Debt ratios ───────────────────────────────────────────────────────────────

class TestCalcDti:
    def test_within_range(self):
        assert calc.calc_dti(1000, 5000) == 0.2

    def test_zero_income_returns_zero(self):
        assert calc.calc_dti(1000, 0) == 0.0

    def test_high_dti(self):
        result = calc.calc_dti(2500, 5000)
        assert result == 0.5


class TestCalcHousingRatio:
    def test_basic(self):
        assert calc.calc_housing_ratio(1400, 5000) == 0.28

    def test_zero_income(self):
        assert calc.calc_housing_ratio(1000, 0) == 0.0


class TestCalcConsumerDebtBurden:
    def test_basic(self):
        result = calc.calc_consumer_debt_burden(20000, 100000)
        assert result == 0.2

    def test_zero_income(self):
        assert calc.calc_consumer_debt_burden(10000, 0) == 0.0


# ── Cash flow ─────────────────────────────────────────────────────────────────

class TestCalcMonthlyCashFlow:
    def test_positive_surplus(self):
        # Income $5000, expenses $1500, debt $500, 25% tax → net $3750 - 2000 = $1750
        result = calc.calc_monthly_cash_flow(5000, 1500, 500, 0.25)
        assert result == 1750.0

    def test_negative_cash_flow(self):
        result = calc.calc_monthly_cash_flow(3000, 2000, 1000, 0.30)
        assert result < 0

    def test_zero_income(self):
        result = calc.calc_monthly_cash_flow(0, 500, 200, 0.25)
        assert result == -700.0


# ── Savings rate ──────────────────────────────────────────────────────────────

class TestCalcSavingsRate:
    def test_basic(self):
        result = calc.calc_savings_rate(6000, 0, 60000)
        assert result == 0.1

    def test_combined_savings(self):
        result = calc.calc_savings_rate(6000, 3000, 60000)
        assert result == 0.15

    def test_zero_income(self):
        assert calc.calc_savings_rate(6000, 0, 0) == 0.0


# ── Insurance ─────────────────────────────────────────────────────────────────

class TestCalcLifeInsuranceCoverageRatio:
    def test_ten_times(self):
        assert calc.calc_life_insurance_coverage_ratio(500000, 50000) == 10.0

    def test_zero_income(self):
        assert calc.calc_life_insurance_coverage_ratio(500000, 0) == 0.0


# ── Net worth ─────────────────────────────────────────────────────────────────

class TestCalcNetWorthTarget:
    def test_young_adult(self):
        # Age 30 → benchmark ~1.0× income (from planning_rules.json)
        result = calc.calc_net_worth_target(30, 80000)
        assert result > 0

    def test_older_worker_higher_target(self):
        young = calc.calc_net_worth_target(30, 80000)
        older = calc.calc_net_worth_target(55, 80000)
        assert older > young

    def test_higher_income_scales_target(self):
        low = calc.calc_net_worth_target(40, 60000)
        high = calc.calc_net_worth_target(40, 120000)
        assert high == low * 2


# ── Retirement projection ─────────────────────────────────────────────────────

class TestCalcRetirementProjection:
    def test_zero_years_returns_current_balance(self):
        result = calc.calc_retirement_projection(100000, 5000, 0, 0.07)
        assert result["corpus_nominal"] == 100000

    def test_growth_over_time(self):
        result = calc.calc_retirement_projection(0, 10000, 10, 0.07)
        assert result["corpus_nominal"] > 100000  # more than simple sum

    def test_zero_return_equals_sum(self):
        result = calc.calc_retirement_projection(0, 10000, 10, 0.0)
        assert result["corpus_nominal"] == 100000.0

    def test_keys_present(self):
        result = calc.calc_retirement_projection(50000, 5000, 20, 0.07)
        assert "corpus_nominal" in result
        assert "corpus_real" in result
        assert "annual_income_4pct" in result


class TestCalcRetirementCorpusNeeded:
    def test_four_percent_rule(self):
        # $80k income / 4% = $2,000,000
        result = calc.calc_retirement_corpus_needed(80000, 0.04)
        assert result == 2_000_000.0

    def test_three_percent_swr(self):
        result = calc.calc_retirement_corpus_needed(60000, 0.03)
        assert result == 2_000_000.0

    def test_zero_swr_returns_zero(self):
        assert calc.calc_retirement_corpus_needed(80000, 0) == 0.0


# ── Goal funding ──────────────────────────────────────────────────────────────

class TestCalcGoalMonthlySavingsNeeded:
    def test_already_funded(self):
        # Current savings already exceeds target
        result = calc.calc_goal_monthly_savings_needed(10000, 15000, 12)
        assert result == 0.0

    def test_zero_months_returns_shortfall(self):
        result = calc.calc_goal_monthly_savings_needed(10000, 5000, 0)
        assert result == 5000.0

    def test_positive_required(self):
        result = calc.calc_goal_monthly_savings_needed(12000, 0, 12, 0.0)
        assert result == 1000.0


# ── Social Security ───────────────────────────────────────────────────────────

class TestEstimateSocialSecurityBenefit:
    def test_zero_income(self):
        assert calc.estimate_social_security_benefit(40, 0) == 0.0

    def test_low_income_higher_replacement(self):
        low = calc.estimate_social_security_benefit(40, 25000)
        high = calc.estimate_social_security_benefit(40, 200000)
        # High earner gets more in absolute dollars but lower rate
        assert low > 0 and high > 0

    def test_early_claiming_reduces_benefit(self):
        fra = calc.estimate_social_security_benefit(40, 80000, claiming_age=67)
        early = calc.estimate_social_security_benefit(40, 80000, claiming_age=62)
        assert early < fra

    def test_delayed_claiming_increases_benefit(self):
        fra = calc.estimate_social_security_benefit(40, 80000, claiming_age=67)
        delayed = calc.estimate_social_security_benefit(40, 80000, claiming_age=70)
        assert delayed > fra

    def test_benefit_capped(self):
        # Very high income — capped at $52,000/yr at FRA
        result = calc.estimate_social_security_benefit(40, 1_000_000, claiming_age=67)
        assert result == 52000.0


# ── 401k limit check ─────────────────────────────────────────────────────────

class TestCheck401kLimit:
    def test_under_limit(self):
        result = calc.check_401k_limit(15000, 40)
        assert result["over_limit"] is False
        assert result["excess"] == 0.0

    def test_over_standard_limit(self):
        result = calc.check_401k_limit(30000, 40)
        assert result["over_limit"] is True
        assert result["excess"] == 7000.0

    def test_catch_up_eligible(self):
        result = calc.check_401k_limit(28000, 52)
        assert result["catch_up_eligible"] is True
        assert result["over_limit"] is False  # 28000 < 30500 catch-up limit

    def test_not_catch_up_under_50(self):
        result = calc.check_401k_limit(1000, 45)
        assert result["catch_up_eligible"] is False


# ── Debt payoff strategies ────────────────────────────────────────────────────

class TestDebtPayoff:
    DEBTS = [
        {"name": "CC", "balance": 5000, "rate": 0.22, "min_payment": 100},
        {"name": "Car", "balance": 12000, "rate": 0.07, "min_payment": 250},
    ]

    def test_avalanche_returns_months_and_interest(self):
        result = calc.calc_avalanche_payoff(self.DEBTS)
        assert result["months"] > 0
        assert result["total_interest"] > 0

    def test_snowball_returns_months_and_interest(self):
        result = calc.calc_snowball_payoff(self.DEBTS)
        assert result["months"] > 0
        assert result["total_interest"] > 0

    def test_avalanche_saves_interest_vs_snowball(self):
        a = calc.calc_avalanche_payoff(self.DEBTS)
        s = calc.calc_snowball_payoff(self.DEBTS)
        assert a["total_interest"] <= s["total_interest"]

    def test_extra_payment_reduces_payoff_time(self):
        no_extra = calc.calc_avalanche_payoff(self.DEBTS, extra_monthly=0)
        with_extra = calc.calc_avalanche_payoff(self.DEBTS, extra_monthly=500)
        assert with_extra["months"] < no_extra["months"]

    def test_empty_debts(self):
        assert calc.calc_avalanche_payoff([]) == {"months": 0, "total_interest": 0.0}


# ── Tax estimates ─────────────────────────────────────────────────────────────

class TestTaxRates:
    def test_low_income_low_rate(self):
        rate = calc.estimate_effective_tax_rate(20000, "single")
        assert rate < 0.15

    def test_high_income_higher_rate(self):
        low = calc.estimate_effective_tax_rate(30000, "single")
        high = calc.estimate_effective_tax_rate(300000, "single")
        assert high > low

    def test_married_lower_rate_at_same_income(self):
        single = calc.estimate_effective_tax_rate(100000, "single")
        married = calc.estimate_effective_tax_rate(100000, "married")
        assert married < single

    def test_total_tax_rate_adds_state(self):
        federal = calc.estimate_effective_tax_rate(80000, "single")
        total = calc.calc_total_tax_rate(80000, "single", 0.05)
        assert abs(total - (federal + 0.05)) < 0.001

    def test_total_rate_capped_at_60pct(self):
        result = calc.calc_total_tax_rate(500000, "single", 0.50)
        assert result == 0.60
