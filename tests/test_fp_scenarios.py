"""
Unit tests for streamlit_app/fp_scenarios.py.

No I/O, no LLM, no Streamlit — pure projection math only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))

from fp_schemas import ClientProfile, Assets, RetirementInfo  # noqa: E402
from fp_scenarios import (  # noqa: E402
    _build_growth_series, generate_scenarios, SCENARIO_CONFIGS,
)


# ── _build_growth_series ──────────────────────────────────────────────────────

class TestBuildGrowthSeries:
    def test_zero_years_returns_starting_balance_only(self):
        result = _build_growth_series(100000, 5000, 0, 0.07)
        assert result == [100000]

    def test_length_is_years_plus_one(self):
        result = _build_growth_series(0, 10000, 10, 0.07)
        assert len(result) == 11  # index 0 = today, then 10 future years

    def test_positive_return_grows_over_time(self):
        result = _build_growth_series(100000, 0, 10, 0.10)
        assert result[-1] > result[0]

    def test_zero_return_equals_contributions_only(self):
        # 0% return, $10k/yr for 5 years → $50k
        result = _build_growth_series(0, 10000, 5, 0.0)
        assert result[-1] == 50000.0

    def test_series_is_monotonically_increasing_with_positive_return(self):
        result = _build_growth_series(50000, 5000, 20, 0.07)
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1]


# ── generate_scenarios ────────────────────────────────────────────────────────

class TestGenerateScenarios:
    def _profile(self) -> ClientProfile:
        p = ClientProfile()
        p.age = 40
        p.gross_annual_income = 80000
        p.assets = Assets(retirement_401k=100000)
        p.retirement = RetirementInfo(
            contribution_rate_pct=10,
            employer_match_pct=5,
            target_retirement_age=65,
        )
        return p

    def test_returns_three_scenarios(self):
        assert len(generate_scenarios(self._profile())) == 3

    def test_scenario_names_match_config(self):
        names = {s.name for s in generate_scenarios(self._profile())}
        assert names == set(SCENARIO_CONFIGS.keys())

    def test_aggressive_corpus_exceeds_conservative(self):
        scenarios = {s.name: s for s in generate_scenarios(self._profile())}
        assert (scenarios["Aggressive"].retirement_corpus
                > scenarios["Conservative"].retirement_corpus)

    def test_gap_equals_needed_minus_corpus(self):
        for s in generate_scenarios(self._profile()):
            assert abs(s.gap - (s.corpus_needed - s.retirement_corpus)) < 1.0

    def test_corpus_growth_series_present(self):
        for s in generate_scenarios(self._profile()):
            assert s.corpus_growth is not None
            assert len(s.corpus_growth) > 1

    def test_monthly_savings_needed_non_negative(self):
        for s in generate_scenarios(self._profile()):
            assert s.monthly_savings_needed >= 0.0

    def test_ss_benefit_positive_for_income_earner(self):
        for s in generate_scenarios(self._profile()):
            assert s.ss_annual_benefit >= 0.0

    def test_zero_years_to_retirement_does_not_crash(self):
        p = self._profile()
        p.retirement.target_retirement_age = p.age  # already at retirement age
        scenarios = generate_scenarios(p)
        assert len(scenarios) == 3


# ── SCENARIO_CONFIGS sanity check ─────────────────────────────────────────────

class TestScenarioConfigs:
    def test_all_required_keys_present(self):
        required = {"return", "inflation", "savings_boost", "replacement", "swr"}
        for name, cfg in SCENARIO_CONFIGS.items():
            missing = required - cfg.keys()
            assert not missing, f"{name} missing keys: {missing}"

    def test_return_rates_are_ordered(self):
        # Conservative < Balanced < Aggressive
        rates = [SCENARIO_CONFIGS[n]["return"]
                 for n in ("Conservative", "Balanced", "Aggressive")]
        assert rates == sorted(rates)

    def test_swr_within_plausible_range(self):
        for name, cfg in SCENARIO_CONFIGS.items():
            assert 0.02 <= cfg["swr"] <= 0.06, f"{name} SWR out of range"
