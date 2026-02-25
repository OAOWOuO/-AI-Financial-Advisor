"""
fp_schemas.py — Data models for the Financial Planning Assistant.

All schemas are plain Python dataclasses so there are no extra dependencies.
Each field includes a comment explaining its semantics.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum


# ─────────────────────── Enumerations ───────────────────────

class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE     = "moderate"
    AGGRESSIVE   = "aggressive"


class MaritalStatus(str, Enum):
    SINGLE   = "single"
    MARRIED  = "married"
    DIVORCED = "divorced"
    WIDOWED  = "widowed"


class GoalType(str, Enum):
    EMERGENCY_FUND  = "emergency_fund"
    DEBT_PAYOFF     = "debt_payoff"
    HOME_PURCHASE   = "home_purchase"
    COLLEGE_SAVINGS = "college_savings"
    RETIREMENT      = "retirement"
    ESTATE_PLANNING = "estate_planning"
    LTC_PLANNING    = "ltc_planning"
    OTHER           = "other"


class IssueSeverity(str, Enum):
    CRITICAL = "CRITICAL"   # act immediately
    HIGH     = "HIGH"       # act within 30 days
    MEDIUM   = "MEDIUM"     # act within 1–6 months
    LOW      = "LOW"        # optional / nice-to-have
    INFO     = "INFO"       # informational only


class IssueCategory(str, Enum):
    CASH_FLOW       = "Cash Flow"
    EMERGENCY_FUND  = "Emergency Fund"
    DEBT            = "Debt Management"
    INSURANCE       = "Insurance"
    RETIREMENT      = "Retirement"
    TAX             = "Tax"
    INVESTMENT      = "Investing"
    ESTATE          = "Estate / Legal"
    GOALS           = "Goal Planning"


# ─────────────────────── Sub-objects ───────────────────────

@dataclass
class MonthlyExpenses:
    housing:        float = 0.0   # rent or P&I payment
    food:           float = 0.0
    transportation: float = 0.0
    utilities:      float = 0.0
    healthcare:     float = 0.0
    childcare:      float = 0.0
    entertainment:  float = 0.0
    personal:       float = 0.0
    subscriptions:  float = 0.0
    other:          float = 0.0

    def total(self) -> float:
        return (self.housing + self.food + self.transportation + self.utilities
                + self.healthcare + self.childcare + self.entertainment
                + self.personal + self.subscriptions + self.other)


@dataclass
class Assets:
    checking_savings:    float = 0.0
    investments_brokerage: float = 0.0
    retirement_401k:     float = 0.0
    retirement_ira:      float = 0.0
    real_estate_equity:  float = 0.0
    college_529:         float = 0.0
    other:               float = 0.0

    def liquid(self) -> float:
        """Easily accessible funds (checking + savings + brokerage)."""
        return self.checking_savings + self.investments_brokerage

    def retirement_total(self) -> float:
        return self.retirement_401k + self.retirement_ira

    def total(self) -> float:
        return (self.checking_savings + self.investments_brokerage
                + self.retirement_401k + self.retirement_ira
                + self.real_estate_equity + self.college_529 + self.other)


@dataclass
class Liabilities:
    mortgage:     float = 0.0
    car_loans:    float = 0.0
    student_loans: float = 0.0
    credit_cards: float = 0.0
    other:        float = 0.0

    def total(self) -> float:
        return (self.mortgage + self.car_loans + self.student_loans
                + self.credit_cards + self.other)

    def consumer_total(self) -> float:
        """Non-mortgage debt."""
        return self.car_loans + self.student_loans + self.credit_cards + self.other


@dataclass
class MonthlyDebtPayments:
    mortgage:     float = 0.0
    car:          float = 0.0
    student_loans: float = 0.0
    credit_cards: float = 0.0
    other:        float = 0.0

    def total(self) -> float:
        return self.mortgage + self.car + self.student_loans + self.credit_cards + self.other


@dataclass
class Insurance:
    has_health:             bool  = False
    has_life:               bool  = False
    life_coverage_amount:   float = 0.0   # face value
    has_disability:         bool  = False
    has_renters_homeowners: bool  = False
    has_ltc:                bool  = False  # long-term care


@dataclass
class RetirementInfo:
    contribution_rate_pct:  float = 0.0   # % of gross income going to 401k/403b
    employer_match_pct:     float = 0.0   # employer match (% of gross)
    target_retirement_age:  int   = 65


@dataclass
class Goal:
    type:            str   = "other"
    description:     str   = ""
    target_amount:   float = 0.0
    timeline_months: int   = 0
    priority:        int   = 1


# ─────────────────────── Main client profile ───────────────────────

@dataclass
class ClientProfile:
    """Normalized client profile — single source of truth for all modules."""

    # Demographics
    name:           str          = "Client"
    age:            int          = 35
    marital_status: str          = "single"
    dependents:     int          = 0

    # Income (annual)
    gross_annual_income:  float  = 0.0
    spouse_annual_income: float  = 0.0
    other_annual_income:  float  = 0.0

    # Expenses / assets / liabilities
    monthly_expenses:       MonthlyExpenses     = field(default_factory=MonthlyExpenses)
    assets:                 Assets              = field(default_factory=Assets)
    liabilities:            Liabilities         = field(default_factory=Liabilities)
    monthly_debt_payments:  MonthlyDebtPayments = field(default_factory=MonthlyDebtPayments)

    # Insurance, retirement, goals
    insurance:  Insurance      = field(default_factory=Insurance)
    retirement: RetirementInfo = field(default_factory=RetirementInfo)
    goals:      List[Goal]     = field(default_factory=list)

    # Preferences
    risk_tolerance:   str = "moderate"
    situation_summary: str = ""

    # ── Derived helpers ──────────────────────────────────────
    def gross_monthly_income(self) -> float:
        return (self.gross_annual_income + self.spouse_annual_income
                + self.other_annual_income) / 12

    def total_annual_income(self) -> float:
        return self.gross_annual_income + self.spouse_annual_income + self.other_annual_income

    def net_worth(self) -> float:
        return self.assets.total() - self.liabilities.total()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClientProfile":
        """Reconstruct from a flat dict (e.g. loaded from JSON)."""
        profile = cls()
        profile.name           = d.get("name", "Client")
        profile.age            = int(d.get("age", 35))
        profile.marital_status = d.get("marital_status", "single")
        profile.dependents     = int(d.get("dependents", 0))

        profile.gross_annual_income  = float(d.get("gross_annual_income", 0))
        profile.spouse_annual_income = float(d.get("spouse_annual_income", 0))
        profile.other_annual_income  = float(d.get("other_annual_income", 0))

        me = d.get("monthly_expenses", {})
        profile.monthly_expenses = MonthlyExpenses(
            housing        = float(me.get("housing", 0)),
            food           = float(me.get("food", 0)),
            transportation = float(me.get("transportation", 0)),
            utilities      = float(me.get("utilities", 0)),
            healthcare     = float(me.get("healthcare", 0)),
            childcare      = float(me.get("childcare", 0)),
            entertainment  = float(me.get("entertainment", 0)),
            personal       = float(me.get("personal", 0)),
            subscriptions  = float(me.get("subscriptions", 0)),
            other          = float(me.get("other", 0)),
        )

        a = d.get("assets", {})
        profile.assets = Assets(
            checking_savings      = float(a.get("checking_savings", 0)),
            investments_brokerage = float(a.get("investments_brokerage", 0)),
            retirement_401k       = float(a.get("retirement_401k", 0)),
            retirement_ira        = float(a.get("retirement_ira", 0)),
            real_estate_equity    = float(a.get("real_estate_equity", 0)),
            college_529           = float(a.get("college_529", 0)),
            other                 = float(a.get("other", 0)),
        )

        li = d.get("liabilities", {})
        profile.liabilities = Liabilities(
            mortgage      = float(li.get("mortgage", 0)),
            car_loans     = float(li.get("car_loans", 0)),
            student_loans = float(li.get("student_loans", 0)),
            credit_cards  = float(li.get("credit_cards", 0)),
            other         = float(li.get("other", 0)),
        )

        mdp = d.get("monthly_debt_payments", {})
        profile.monthly_debt_payments = MonthlyDebtPayments(
            mortgage      = float(mdp.get("mortgage", 0)),
            car           = float(mdp.get("car", 0)),
            student_loans = float(mdp.get("student_loans", 0)),
            credit_cards  = float(mdp.get("credit_cards", 0)),
            other         = float(mdp.get("other", 0)),
        )

        ins = d.get("insurance", {})
        profile.insurance = Insurance(
            has_health             = bool(ins.get("has_health", False)),
            has_life               = bool(ins.get("has_life", False)),
            life_coverage_amount   = float(ins.get("life_coverage_amount", 0)),
            has_disability         = bool(ins.get("has_disability", False)),
            has_renters_homeowners = bool(ins.get("has_renters_homeowners", False)),
            has_ltc                = bool(ins.get("has_ltc", False)),
        )

        ret = d.get("retirement", {})
        profile.retirement = RetirementInfo(
            contribution_rate_pct = float(ret.get("contribution_rate_pct", 0)),
            employer_match_pct    = float(ret.get("employer_match_pct", 0)),
            target_retirement_age = int(ret.get("target_retirement_age", 65)),
        )

        profile.goals = [
            Goal(
                type            = g.get("type", "other"),
                description     = g.get("description", ""),
                target_amount   = float(g.get("target_amount", 0)),
                timeline_months = int(g.get("timeline_months", 0)),
                priority        = int(g.get("priority", 1)),
            )
            for g in d.get("goals", [])
        ]

        profile.risk_tolerance    = d.get("risk_tolerance", "moderate")
        profile.situation_summary = d.get("situation_summary", "")
        return profile


# ─────────────────────── Output types ───────────────────────

@dataclass
class PlanningIssue:
    category:    str    # IssueCategory value
    severity:    str    # IssueSeverity value
    title:       str
    detail:      str
    metric_value: Optional[str] = None   # e.g. "DTI = 41%"
    benchmark:   Optional[str] = None   # e.g. "recommended ≤ 36%"
    action_hint: Optional[str] = None   # short suggested action


@dataclass
class Recommendation:
    priority:        int    # 1 = highest
    action:          str
    reason:          str
    timeline:        str    # "0–30 days" / "1–6 months" / "6–24 months"
    expected_benefit: str
    tradeoff:        str
    source_refs:     List[str] = field(default_factory=list)


@dataclass
class QuantCheck:
    label:       str
    value:       str
    benchmark:   str
    status:      str   # "OK" / "WARNING" / "CRITICAL"
    detail:      str


@dataclass
class ScenarioProjection:
    name:              str    # Conservative / Balanced / Aggressive
    assumptions:       Dict[str, Any]
    retirement_corpus: float  # projected value at retirement
    corpus_needed:     float  # needed for target income
    gap:               float  # corpus_needed - retirement_corpus (positive = shortfall)
    monthly_savings_needed: float
    summary:           str


@dataclass
class PlanningReport:
    client_name:     str
    generated_at:    str
    client_snapshot: str

    issues:          List[PlanningIssue]
    recommendations: List[Recommendation]
    quant_checks:    List[QuantCheck]
    scenarios:       List[ScenarioProjection]

    # Source-grounded reasoning
    retrieved_sources: List[Dict[str, Any]]
    case_reasoning:    str   # LLM narrative connecting sources to recommendations

    # LLM-generated narrative sections
    executive_summary: str
    missing_info:      List[str]
    follow_up_questions: List[str]

    DISCLAIMER = (
        "⚠️ DISCLAIMER: This report is for educational purposes only. "
        "It does not constitute legal, tax, or investment advice. "
        "Consult a licensed CFP, CPA, or attorney before making financial decisions."
    )
