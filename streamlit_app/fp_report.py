"""
fp_report.py — LLM-powered report generator for the Financial Planning Assistant.

Design: deterministic calculations + rules engine produce structured data first.
The LLM is invoked only to write the narrative explanation — never to make threshold
decisions or produce numbers. This keeps the reasoning transparent and auditable.
"""
from __future__ import annotations
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from fp_schemas import (
    ClientProfile, PlanningIssue, Recommendation, QuantCheck,
    ScenarioProjection, PlanningReport, IssueSeverity
)
import fp_calculators as calc


CHAT_MODEL = "gpt-4o-mini"


# ── Quantitative checks builder ────────────────────────────────────────────────

def build_quant_checks(profile: ClientProfile) -> List[QuantCheck]:
    """Build a table of key numeric ratios with pass/warn/fail status."""
    checks = []
    gm     = profile.gross_monthly_income()

    # Emergency fund
    liquid  = profile.assets.liquid()
    monthly_exp = profile.monthly_expenses.total()
    ef_months   = calc.calc_emergency_fund_months(liquid, monthly_exp)
    checks.append(QuantCheck(
        label     = "Emergency Fund Coverage",
        value     = f"{ef_months:.1f} months",
        benchmark = "≥ 6 months",
        status    = "OK" if ef_months >= 6 else ("WARNING" if ef_months >= 3 else "CRITICAL"),
        detail    = f"Liquid assets ${liquid:,.0f} ÷ monthly expenses ${monthly_exp:,.0f}",
    ))

    # DTI
    dti = calc.calc_dti(profile.monthly_debt_payments.total(), gm)
    checks.append(QuantCheck(
        label     = "Debt-to-Income (DTI)",
        value     = f"{dti*100:.1f}%",
        benchmark = "≤ 36% recommended",
        status    = "OK" if dti <= 0.36 else ("WARNING" if dti <= 0.43 else "CRITICAL"),
        detail    = f"Total monthly debt ${profile.monthly_debt_payments.total():,.0f} ÷ gross monthly income ${gm:,.0f}",
    ))

    # Housing ratio
    hr = calc.calc_housing_ratio(profile.monthly_debt_payments.mortgage, gm)
    if profile.monthly_debt_payments.mortgage > 0:
        checks.append(QuantCheck(
            label     = "Housing Cost Ratio",
            value     = f"{hr*100:.1f}%",
            benchmark = "≤ 28% recommended",
            status    = "OK" if hr <= 0.28 else "WARNING",
            detail    = f"Mortgage payment ${profile.monthly_debt_payments.mortgage:,.0f} ÷ gross monthly income ${gm:,.0f}",
        ))

    # Net worth vs benchmark
    nw        = profile.net_worth()
    nw_target = calc.calc_net_worth_target(profile.age, profile.total_annual_income())
    checks.append(QuantCheck(
        label     = "Net Worth vs Benchmark",
        value     = f"${nw:,.0f}",
        benchmark = f"${nw_target:,.0f} (age-{profile.age} benchmark)",
        status    = "OK" if nw >= nw_target else ("WARNING" if nw >= nw_target * 0.5 else "CRITICAL"),
        detail    = f"Net worth = ${profile.assets.total():,.0f} assets − ${profile.liabilities.total():,.0f} liabilities",
    ))

    # Retirement savings rate
    annual_contrib = calc.calc_annual_contribution(profile)
    savings_rate   = annual_contrib / profile.total_annual_income() if profile.total_annual_income() else 0
    checks.append(QuantCheck(
        label     = "Retirement Savings Rate",
        value     = f"{savings_rate*100:.1f}%",
        benchmark = "≥ 15% recommended",
        status    = "OK" if savings_rate >= 0.15 else ("WARNING" if savings_rate >= 0.10 else "CRITICAL"),
        detail    = f"Annual contributions ${annual_contrib:,.0f} ÷ total income ${profile.total_annual_income():,.0f}",
    ))

    # Cash flow
    tax_rate = calc.calc_total_tax_rate(
        profile.total_annual_income(), profile.marital_status, profile.state_income_tax_rate
    )
    cf = calc.calc_monthly_cash_flow(gm, monthly_exp, profile.monthly_debt_payments.total(), tax_rate)
    checks.append(QuantCheck(
        label     = "Estimated Monthly Cash Flow",
        value     = f"${cf:,.0f}",
        benchmark = "> $0 (surplus)",
        status    = "OK" if cf > gm * 0.10 else ("WARNING" if cf > 0 else "CRITICAL"),
        detail    = "After-tax income minus living expenses and debt payments (approximate)",
    ))

    # Life insurance coverage ratio
    if profile.gross_annual_income > 0:
        li_ratio = calc.calc_life_insurance_coverage_ratio(
            profile.insurance.life_coverage_amount, profile.total_annual_income()
        )
        min_mult = 15 if profile.dependents > 0 else 10
        checks.append(QuantCheck(
            label     = "Life Insurance Coverage Ratio",
            value     = f"{li_ratio:.1f}× income" if li_ratio > 0 else "None",
            benchmark = f"≥ {min_mult}× income",
            status    = "OK" if li_ratio >= min_mult else ("WARNING" if li_ratio >= 5 else "CRITICAL"),
            detail    = f"Coverage ${profile.insurance.life_coverage_amount:,.0f} ÷ annual income ${profile.total_annual_income():,.0f}",
        ))

    return checks


# ── Prioritized recommendations ────────────────────────────────────────────────

def build_recommendations(
    profile: ClientProfile,
    issues: List[PlanningIssue],
    scenarios: List[ScenarioProjection],
) -> List[Recommendation]:
    """
    Convert issues into actionable, prioritized recommendations.
    Priority 1 = most urgent (CRITICAL issues).
    """
    recs: List[Recommendation] = []
    priority = 1

    for issue in issues:
        if issue.severity == IssueSeverity.INFO:
            continue   # informational — no action required

        timeline_map = {
            IssueSeverity.CRITICAL: "0–30 days",
            IssueSeverity.HIGH:     "1–3 months",
            IssueSeverity.MEDIUM:   "3–6 months",
            IssueSeverity.LOW:      "6–24 months",
        }

        recs.append(Recommendation(
            priority         = priority,
            action           = issue.action_hint or issue.title,
            reason           = issue.detail,
            timeline         = timeline_map.get(issue.severity, "1–6 months"),
            expected_benefit = f"Address {issue.category} gap.",
            tradeoff         = "May require reallocation of discretionary spending.",
            source_refs      = [],
        ))
        priority += 1

    return recs[:10]   # cap at 10 recommendations


# ── LLM narrative generator ────────────────────────────────────────────────────

def _build_prompt(
    profile: ClientProfile,
    issues: List[PlanningIssue],
    quant_checks: List[QuantCheck],
    scenarios: List[ScenarioProjection],
    retrieved_docs: List[Dict[str, Any]],
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> str:
    issue_lines = "\n".join(
        f"  [{iss.severity}] {iss.category} — {iss.title}: {iss.detail}"
        for iss in issues[:8]
    )
    quant_lines = "\n".join(
        f"  {q.label}: {q.value} (benchmark: {q.benchmark}) [{q.status}]"
        for q in quant_checks
    )
    scenario_lines = "\n".join(
        f"  {s.name}: corpus ${s.retirement_corpus:,.0f}, needed ${s.corpus_needed:,.0f}, gap ${s.gap:,.0f}"
        for s in scenarios
    )
    source_lines = "\n".join(
        f"  [{i+1}] {d['metadata'].get('source','?')} (topic: {d['metadata'].get('topic','?')}, "
        f"score: {d['score']}) — \"{d['text'][:200]}...\""
        for i, d in enumerate(retrieved_docs[:5])
    )

    # Build case-based reasoning context
    case_lines = ""
    if similar_cases:
        case_parts = []
        for i, c in enumerate(similar_cases[:3]):
            meta = c.get("metadata", {})
            struct = c.get("structured", {})
            reasons = "; ".join(c.get("reasons", []))
            recs = "; ".join(struct.get("candidate_recommendations", [])[:3])
            case_parts.append(
                f"  [{i+1}] {meta.get('title', c['case_id'])} "
                f"(similarity: {c.get('score_pct', 0):.0f}%) — Match: {reasons}. "
                f"Key lessons: {recs}"
            )
        case_lines = "\n".join(case_parts)

    return f"""You are a senior financial planning analyst. Your task is to write an educational planning report.

CLIENT PROFILE:
Name: {profile.name}, Age: {profile.age}, Status: {profile.marital_status}, Dependents: {profile.dependents}
Income: ${profile.total_annual_income():,.0f}/year
Net Worth: ${profile.net_worth():,.0f}
Situation: {profile.situation_summary}

KEY PLANNING ISSUES IDENTIFIED (by rules engine):
{issue_lines}

QUANTITATIVE CHECKS:
{quant_lines}

RETIREMENT SCENARIOS:
{scenario_lines}

SIMILAR CASES FROM BUILT-IN CASE LIBRARY (use for analogy, not prescription):
{case_lines if case_lines else "  No case library matches available."}

RETRIEVED REFERENCE MATERIALS (uploaded documents):
{source_lines if source_lines else "  No documents uploaded yet. Base reasoning on general planning principles."}

INSTRUCTIONS:
1. Write a concise EXECUTIVE SUMMARY (3-5 sentences) covering the client's overall financial health.
2. Write a CASE REASONING section explaining: (a) which similar cases are most analogous and why, \
(b) which uploaded reference materials are most relevant. Be explicit about the analogical reasoning.
3. List 4-6 FOLLOW-UP QUESTIONS the planner should ask the client to complete the analysis.
4. List 2-3 MISSING INFORMATION items required before finalizing the plan.

FORMAT RULES:
- Use plain text with clear section headers: ## Executive Summary, ## Case Reasoning, ## Follow-up Questions, ## Missing Information
- Do NOT invent specific numbers not in the data above.
- Mark uncertain assumptions explicitly with [ASSUMPTION].
- When referencing similar cases, say "In a similar case (Case Library)..." to make analogical reasoning transparent.
- Be concrete and actionable, but avoid overconfident language.
- Keep total response under 700 words.

IMPORTANT: End every response with this exact disclaimer:
⚠️ This analysis is for educational purposes only. It does not constitute legal, tax, or investment advice. Consult a licensed CFP, CPA, or attorney before making any financial decisions.
"""


def generate_report(
    profile: ClientProfile,
    issues: List[PlanningIssue],
    quant_checks: List[QuantCheck],
    scenarios: List[ScenarioProjection],
    retrieved_docs: List[Dict[str, Any]],
    openai_client,
    model: str = CHAT_MODEL,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> PlanningReport:
    """
    Assemble a complete PlanningReport.
    Deterministic calculations happen before this function.
    Only narrative sections come from LLM.
    similar_cases: output of fp_case_retriever.retrieve_similar_cases()
    """
    recs = build_recommendations(profile, issues, scenarios)

    prompt = _build_prompt(profile, issues, quant_checks, scenarios, retrieved_docs,
                           similar_cases=similar_cases)

    try:
        completion = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are a professional financial planning analyst writing educational reports."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        llm_text = completion.choices[0].message.content or ""
    except Exception as e:
        llm_text = f"[LLM unavailable: {e}]\n\nPlease review the quantitative checks and issues above."

    # Parse LLM output into sections
    sections = _parse_llm_sections(llm_text)

    # Build case reasoning connecting sources
    case_reasoning = sections.get("case_reasoning",
        "Reasoning is grounded in the rule-based checks above. Upload planning reference documents to enable source-based reasoning.")

    return PlanningReport(
        client_name      = profile.name,
        generated_at     = datetime.now().strftime("%Y-%m-%d %H:%M"),
        client_snapshot  = _build_snapshot(profile),
        issues           = issues,
        recommendations  = recs,
        quant_checks     = quant_checks,
        scenarios        = scenarios,
        retrieved_sources= retrieved_docs,
        case_reasoning   = case_reasoning,
        executive_summary= sections.get("executive_summary", ""),
        missing_info     = sections.get("missing_info", []),
        follow_up_questions = sections.get("follow_up", []),
    )


def _parse_llm_sections(text: str) -> Dict[str, Any]:
    """
    Extract structured sections from LLM markdown output.
    Handles variations in header formatting.
    """
    sections: Dict[str, Any] = {}

    def _between(heading_pattern: str) -> str:
        m = re.search(heading_pattern, text, re.IGNORECASE)
        if not m:
            return ""
        start = m.end()
        # Find next ## heading
        next_m = re.search(r"\n##\s", text[start:])
        end = start + next_m.start() if next_m else len(text)
        return text[start:end].strip()

    sections["executive_summary"] = _between(r"##\s*Executive Summary")
    sections["case_reasoning"]    = _between(r"##\s*Case Reasoning")

    # Follow-up questions — parse as bullet list
    fup_raw = _between(r"##\s*Follow.up Questions?")
    sections["follow_up"] = [
        re.sub(r"^[\-\*\d\.\s]+", "", line).strip()
        for line in fup_raw.splitlines()
        if line.strip() and not line.startswith("##")
    ]

    # Missing information
    mi_raw = _between(r"##\s*Missing Information")
    sections["missing_info"] = [
        re.sub(r"^[\-\*\d\.\s]+", "", line).strip()
        for line in mi_raw.splitlines()
        if line.strip() and not line.startswith("##")
    ]

    return sections


def _build_snapshot(profile: ClientProfile) -> str:
    return (
        f"{profile.name} | Age {profile.age} | {profile.marital_status.capitalize()} | "
        f"{profile.dependents} dependent(s) | "
        f"Income: ${profile.total_annual_income():,.0f}/yr | "
        f"Net Worth: ${profile.net_worth():,.0f} | "
        f"Risk: {profile.risk_tolerance}"
    )
