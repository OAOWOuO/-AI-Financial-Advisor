"""
financial_planner.py — Financial Planning Case Assistant

Replaces the original Case Q&A tab.
Architecture: RAG retrieval + rules engine + scenario engine + LLM narrative.

Tabs:
  1. Client Input       — structured form + JSON load/save
  2. Case Library       — uploaded documents, search, metadata
  3. Planning Analysis  — gap analysis, rule checks, quantitative ratios
  4. Recommendation Report — full prioritized planning report
  5. Explainability     — retrieved sources + reasoning chain
  6. Settings           — model, retrieval params, rule preset
"""
from __future__ import annotations
import json, os
import streamlit as st


# ── Helper: get OpenAI client ─────────────────────────────────────────────────

def _get_openai_client():
    api_key = ""
    # 1. Streamlit Cloud secrets
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass
    # 2. Environment variable (already set in shell or loaded by dotenv elsewhere)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    # 3. Load from .env file in parent directory (local dev)
    if not api_key:
        try:
            from dotenv import load_dotenv as _load_dotenv
            _load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
            api_key = os.getenv("OPENAI_API_KEY", "")
        except ImportError:
            pass
    if not api_key:
        return None
    import openai
    return openai.OpenAI(api_key=api_key)


# ── Session state defaults ────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "fp_profile_dict":   None,
        "fp_issues":         None,
        "fp_quant_checks":   None,
        "fp_scenarios":      None,
        "fp_report":         None,
        "fp_retrieved_docs": [],
        "fp_similar_cases":  None,
        "fp_model":          "gpt-4o-mini",
        "fp_top_k":          8,
        "fp_topic_filter":   "all",
        "fp_case_top_k":     3,
        "fp_case_mode":      "hybrid",
        "fp_use_cases":      True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── CSS injection ─────────────────────────────────────────────────────────────

def _fp_inject_css() -> None:
    st.markdown("""<style>
/* ════ FP CTA buttons — prominent gradient ════ */
.fp-cta-btn > div > button, .fp-cta-btn > button {
    background: linear-gradient(135deg, #1a3a5c 0%, #1f6feb 100%) !important;
    border: 1px solid #388bfd !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 8px !important;
    box-shadow: 0 0 16px rgba(31,111,235,0.35) !important;
    transition: all 0.22s ease !important;
}
.fp-cta-btn > div > button:hover, .fp-cta-btn > button:hover {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    box-shadow: 0 0 28px rgba(56,139,253,0.55) !important;
    transform: translateY(-2px) !important;
}
.fp-cta-btn > div > button:active, .fp-cta-btn > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 0 10px rgba(31,111,235,0.25) !important;
}
/* ════ Section headers with left accent ════ */
.fp-section-header {
    border-left: 3px solid #388bfd;
    padding-left: 10px;
    margin: 18px 0 10px 0;
    font-size: 15px;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: 0.2px;
}
/* ════ Status / severity badge ════ */
.fp-badge {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 9px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
/* ════ Issue row card ════ */
.fp-issue-row {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
/* ════ Case insight card ════ */
.fp-case-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
    transition: border-color 0.2s ease;
}
.fp-case-card:hover { border-color: #58a6ff; }
/* ════ Metric card ════ */
.fp-metric {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 14px;
    text-align: center;
}
/* ════ Tab container spacing ════ */
[data-testid="stTabs"] [data-testid="stTab"] {
    font-size: 13px !important;
}
</style>""", unsafe_allow_html=True)


# ── Colour palette ────────────────────────────────────────────────────────────

_COLORS = {
    "CRITICAL": "#ff6b6b",
    "HIGH":     "#f0883e",
    "MEDIUM":   "#d29922",
    "LOW":      "#58a6ff",
    "INFO":     "#3fb950",
    "OK":       "#3fb950",
    "WARNING":  "#d29922",
}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLIENT INPUT
# ══════════════════════════════════════════════════════════════════════════════

def _tab_client_input() -> None:
    st.markdown("#### 📋 Client Profile Input")
    st.caption("Fill in the client's financial situation, or load a sample profile.")

    # Load sample / upload JSON
    col_load, col_up, col_clear = st.columns([2, 2, 1])
    with col_load:
        samples_dir = os.path.join(os.path.dirname(__file__), "data", "sample_clients")
        samples = sorted([f for f in os.listdir(samples_dir) if f.endswith(".json")]) if os.path.isdir(samples_dir) else []
        sel = st.selectbox("Load sample profile", ["— select —"] + samples, key="fp_sample_sel")
        if sel != "— select —":
            with open(os.path.join(samples_dir, sel)) as f:
                st.session_state.fp_profile_dict = json.load(f)
            st.success(f"Loaded: {sel}")

    with col_up:
        uploaded = st.file_uploader("Or upload a client JSON", type=["json"], key="fp_profile_upload")
        if uploaded:
            st.session_state.fp_profile_dict = json.load(uploaded)
            st.success("Profile loaded from file.")

    with col_clear:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Clear", key="fp_clear_profile"):
            st.session_state.fp_profile_dict = None
            for k in ["fp_issues","fp_quant_checks","fp_scenarios","fp_report","fp_retrieved_docs"]:
                st.session_state[k] = None if k != "fp_retrieved_docs" else []
            st.rerun()

    st.divider()

    # ── Build form ──────────────────────────────────────────────────────────
    d = st.session_state.fp_profile_dict or {}

    with st.form("fp_profile_form"):
        st.markdown("##### Demographics & Income")
        c1, c2, c3, c4 = st.columns(4)
        name     = c1.text_input("Full Name / Label",   value=d.get("name","Client"))
        age      = c2.number_input("Age",               value=int(d.get("age",35)),   min_value=18, max_value=90)
        status   = c3.selectbox("Marital Status",       ["single","married","divorced","widowed"],
                                 index=["single","married","divorced","widowed"].index(d.get("marital_status","single")))
        deps     = c4.number_input("Dependents",        value=int(d.get("dependents",0)), min_value=0, max_value=20)

        c5, c6, c7 = st.columns(3)
        income_p = c5.number_input("Gross Annual Income ($)",     value=float(d.get("gross_annual_income",0)),    min_value=0.0, step=1000.0)
        income_s = c6.number_input("Spouse Annual Income ($)",    value=float(d.get("spouse_annual_income",0)),   min_value=0.0, step=1000.0)
        income_o = c7.number_input("Other Annual Income ($)",     value=float(d.get("other_annual_income",0)),    min_value=0.0, step=500.0)

        st.markdown("##### Monthly Expenses")
        me = d.get("monthly_expenses", {})
        ex_cols = st.columns(5)
        housing   = ex_cols[0].number_input("Housing",       value=float(me.get("housing",0)),       min_value=0.0, step=50.0)
        food      = ex_cols[1].number_input("Food",          value=float(me.get("food",0)),           min_value=0.0, step=50.0)
        transport = ex_cols[2].number_input("Transport",     value=float(me.get("transportation",0)), min_value=0.0, step=50.0)
        utilities = ex_cols[3].number_input("Utilities",     value=float(me.get("utilities",0)),      min_value=0.0, step=25.0)
        healthcare= ex_cols[4].number_input("Healthcare",    value=float(me.get("healthcare",0)),     min_value=0.0, step=25.0)
        ex_cols2  = st.columns(5)
        childcare = ex_cols2[0].number_input("Childcare",    value=float(me.get("childcare",0)),      min_value=0.0, step=50.0)
        entertain = ex_cols2[1].number_input("Entertainment",value=float(me.get("entertainment",0)),  min_value=0.0, step=25.0)
        personal  = ex_cols2[2].number_input("Personal",     value=float(me.get("personal",0)),       min_value=0.0, step=25.0)
        subs      = ex_cols2[3].number_input("Subscriptions",value=float(me.get("subscriptions",0)),  min_value=0.0, step=10.0)
        other_exp = ex_cols2[4].number_input("Other",        value=float(me.get("other",0)),          min_value=0.0, step=25.0)

        st.markdown("##### Assets")
        a = d.get("assets", {})
        as_cols = st.columns(4)
        chk  = as_cols[0].number_input("Checking / Savings ($)", value=float(a.get("checking_savings",0)),       min_value=0.0, step=500.0)
        brok = as_cols[1].number_input("Brokerage / Invest ($)",  value=float(a.get("investments_brokerage",0)), min_value=0.0, step=500.0)
        k401 = as_cols[2].number_input("401k / 403b ($)",         value=float(a.get("retirement_401k",0)),       min_value=0.0, step=500.0)
        ira  = as_cols[3].number_input("IRA ($)",                  value=float(a.get("retirement_ira",0)),        min_value=0.0, step=500.0)
        as_cols2 = st.columns(3)
        re_eq= as_cols2[0].number_input("Real Estate Equity ($)",  value=float(a.get("real_estate_equity",0)),   min_value=0.0, step=1000.0)
        c529 = as_cols2[1].number_input("529 / College Savings ($)",value=float(a.get("college_529",0)),         min_value=0.0, step=500.0)
        oth_a= as_cols2[2].number_input("Other Assets ($)",         value=float(a.get("other",0)),               min_value=0.0, step=500.0)

        st.markdown("##### Liabilities & Monthly Debt Payments")
        li = d.get("liabilities", {}); mdp = d.get("monthly_debt_payments", {})
        li_cols = st.columns(5)
        mort_b = li_cols[0].number_input("Mortgage Balance ($)",    value=float(li.get("mortgage",0)),      min_value=0.0, step=1000.0)
        car_b  = li_cols[1].number_input("Car Loan Balance ($)",    value=float(li.get("car_loans",0)),     min_value=0.0, step=500.0)
        sl_b   = li_cols[2].number_input("Student Loans ($)",       value=float(li.get("student_loans",0)),min_value=0.0, step=500.0)
        cc_b   = li_cols[3].number_input("Credit Cards ($)",        value=float(li.get("credit_cards",0)), min_value=0.0, step=100.0)
        oth_l  = li_cols[4].number_input("Other Debt ($)",          value=float(li.get("other",0)),        min_value=0.0, step=100.0)

        dp_cols = st.columns(5)
        mort_p = dp_cols[0].number_input("Mortgage Pmt/mo ($)",     value=float(mdp.get("mortgage",0)),      min_value=0.0, step=50.0)
        car_p  = dp_cols[1].number_input("Car Pmt/mo ($)",          value=float(mdp.get("car",0)),           min_value=0.0, step=25.0)
        sl_p   = dp_cols[2].number_input("Student Loan Pmt/mo ($)", value=float(mdp.get("student_loans",0)),min_value=0.0, step=25.0)
        cc_p   = dp_cols[3].number_input("CC Min Pmt/mo ($)",       value=float(mdp.get("credit_cards",0)), min_value=0.0, step=25.0)
        oth_p  = dp_cols[4].number_input("Other Pmt/mo ($)",        value=float(mdp.get("other",0)),        min_value=0.0, step=25.0)

        st.markdown("##### Insurance")
        ins = d.get("insurance", {})
        in_cols = st.columns(3)
        has_health = in_cols[0].checkbox("Health Insurance",         value=bool(ins.get("has_health",False)))
        has_life   = in_cols[1].checkbox("Life Insurance",           value=bool(ins.get("has_life",False)))
        has_dis    = in_cols[2].checkbox("Disability Insurance",     value=bool(ins.get("has_disability",False)))
        in_cols2   = st.columns(3)
        has_prop   = in_cols2[0].checkbox("Renters / Homeowners",    value=bool(ins.get("has_renters_homeowners",False)))
        has_ltc    = in_cols2[1].checkbox("Long-Term Care",          value=bool(ins.get("has_ltc",False)))
        life_amt   = in_cols2[2].number_input("Life Coverage Amount ($)", value=float(ins.get("life_coverage_amount",0)), min_value=0.0, step=10000.0)

        st.markdown("##### Retirement")
        ret = d.get("retirement", {})
        r_cols = st.columns(3)
        contrib_rate = r_cols[0].number_input("401k Contribution Rate (%)", value=float(ret.get("contribution_rate_pct",0)),   min_value=0.0, max_value=30.0, step=0.5)
        match_rate   = r_cols[1].number_input("Employer Match (%)",         value=float(ret.get("employer_match_pct",0)),      min_value=0.0, max_value=20.0, step=0.5)
        ret_age      = r_cols[2].number_input("Target Retirement Age",      value=int(ret.get("target_retirement_age",65)),    min_value=50,  max_value=80)

        st.markdown("##### Risk Tolerance & Situation")
        risk  = st.selectbox("Risk Tolerance", ["conservative","moderate","aggressive"],
                              index=["conservative","moderate","aggressive"].index(d.get("risk_tolerance","moderate")))
        situation = st.text_area("Situation Summary (free text)", value=d.get("situation_summary",""), height=90)

        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True, type="primary")

    if submitted:
        st.session_state.fp_profile_dict = {
            "name": name, "age": age, "marital_status": status, "dependents": deps,
            "gross_annual_income": income_p, "spouse_annual_income": income_s, "other_annual_income": income_o,
            "monthly_expenses": {"housing": housing, "food": food, "transportation": transport,
                                  "utilities": utilities, "healthcare": healthcare, "childcare": childcare,
                                  "entertainment": entertain, "personal": personal,
                                  "subscriptions": subs, "other": other_exp},
            "assets": {"checking_savings": chk, "investments_brokerage": brok, "retirement_401k": k401,
                       "retirement_ira": ira, "real_estate_equity": re_eq, "college_529": c529, "other": oth_a},
            "liabilities": {"mortgage": mort_b, "car_loans": car_b, "student_loans": sl_b,
                            "credit_cards": cc_b, "other": oth_l},
            "monthly_debt_payments": {"mortgage": mort_p, "car": car_p, "student_loans": sl_p,
                                       "credit_cards": cc_p, "other": oth_p},
            "insurance": {"has_health": has_health, "has_life": has_life, "life_coverage_amount": life_amt,
                          "has_disability": has_dis, "has_renters_homeowners": has_prop, "has_ltc": has_ltc},
            "retirement": {"contribution_rate_pct": contrib_rate, "employer_match_pct": match_rate,
                           "target_retirement_age": ret_age},
            "goals": d.get("goals", []),
            "risk_tolerance": risk, "situation_summary": situation,
        }
        # Reset downstream results when profile changes
        for k in ["fp_issues","fp_quant_checks","fp_scenarios","fp_report"]:
            st.session_state[k] = None
        st.success("Profile saved. Go to Planning Analysis to run checks.")

    # Download current profile
    if st.session_state.fp_profile_dict:
        st.download_button(
            "⬇️ Download Profile JSON",
            data=json.dumps(st.session_state.fp_profile_dict, indent=2),
            file_name=f"fp_profile_{st.session_state.fp_profile_dict.get('name','client').replace(' ','_')}.json",
            mime="application/json",
            key="fp_dl_profile",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CASE LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

def _tab_case_library() -> None:
    import fp_retriever as retriever

    st.markdown("#### 📚 Case Library & Knowledge Base")
    st.caption("Upload planning reference documents, case studies, or worksheets. "
               "Indexed chunks are used to ground recommendations.")

    client = _get_openai_client()
    if not client:
        st.error("**OPENAI_API_KEY** not set. Add it to your `.env` or Streamlit secrets.")
        return

    # ── Upload + metadata ──────────────────────────────────────────────────
    with st.expander("➕ Add Documents", expanded=retriever.store_count() == 0):
        up_cols = st.columns([3,1,1,1,1])
        files = up_cols[0].file_uploader(
            "Upload documents (PDF, MD, TXT, HTML)",
            type=["pdf","md","txt","html"], accept_multiple_files=True, key="fp_doc_upload"
        )
        topic      = up_cols[1].selectbox("Topic",        retriever.TOPIC_TAGS,     key="fp_up_topic")
        src_type   = up_cols[2].selectbox("Source Type",  retriever.SOURCE_TYPES,   key="fp_up_srctype")
        reliability= up_cols[3].selectbox("Reliability",  retriever.RELIABILITY,    key="fp_up_rel")
        doc_date   = up_cols[4].text_input("Date (YYYY)", value="",                  key="fp_up_date")

        if files:
            to_index = [f for f in files if not st.session_state.get(f"fp_indexed_{f.name}_{f.size}")]
            if to_index:
                if st.button(f"Index {len(to_index)} document(s)", key="fp_index_btn", type="primary"):
                    with st.spinner("Embedding and indexing…"):
                        total = 0
                        for f in to_index:
                            n, err = retriever.ingest_bytes(
                                f.read(), f.name, client,
                                topic=topic, source_type=src_type,
                                reliability=reliability, date=doc_date,
                            )
                            if err:
                                st.warning(f"{f.name}: {err}")
                            total += n
                            st.session_state[f"fp_indexed_{f.name}_{f.size}"] = True
                    st.success(f"Indexed {total} chunks from {len(to_index)} document(s).")
                    st.rerun()
            else:
                st.info("All uploaded files are already indexed this session.")

    # ── Document table ──────────────────────────────────────────────────────
    sources = retriever.list_sources()
    if sources:
        st.markdown(f"**{retriever.store_count()} chunks** from **{len(sources)} document(s)** in memory")

        import pandas as pd
        df = pd.DataFrame(sources)[["source","topic","source_type","reliability","date","chunks"]]
        df.columns = ["Document","Topic","Type","Reliability","Date","Chunks"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Search
        st.divider()
        st.markdown("##### Search Library")
        q_cols = st.columns([4,1,1])
        query      = q_cols[0].text_input("Search query", placeholder="e.g. emergency fund rules", key="fp_lib_query")
        top_k      = q_cols[1].number_input("Top-k", value=5, min_value=1, max_value=20, key="fp_lib_topk")
        topic_f    = q_cols[2].selectbox("Filter topic", ["all"] + retriever.TOPIC_TAGS, key="fp_lib_topic")

        if query:
            with st.spinner("Searching…"):
                results = retriever.retrieve(query, client, top_k=top_k, topic_filter=topic_f if topic_f != "all" else None)
            if results:
                for i, r in enumerate(results):
                    m = r["metadata"]
                    with st.expander(f"[{i+1}] {m.get('source','?')} · page {m.get('page','?')} · score {r['score']:.3f}"):
                        st.markdown(f"""
<div style="font-size:12px;color:#8b949e;margin-bottom:6px;">
Topic: <b>{m.get('topic','?')}</b> &nbsp;|&nbsp; Type: <b>{m.get('source_type','?')}</b> &nbsp;|&nbsp; Reliability: <b>{m.get('reliability_level','?')}</b>
</div>
<div style="font-size:13px;color:#c9d1d9;background:#161b22;padding:10px;border-radius:6px;">{r['text']}</div>
""", unsafe_allow_html=True)
            else:
                st.info("No relevant chunks found for this query.")

        if st.button("🗑️ Clear entire document store", key="fp_clear_docs"):
            retriever.clear_store()
            st.rerun()
    else:
        st.info("No documents indexed yet. Upload planning references above to enable source-grounded recommendations.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PLANNING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _tab_planning_analysis() -> None:
    from fp_schemas import ClientProfile
    from fp_rules import RulesEngine
    from fp_scenarios import generate_scenarios
    import fp_report as rpt

    st.markdown("#### 📊 Planning Analysis — Gap Analysis & Scenario Projections")

    d = st.session_state.get("fp_profile_dict")
    if not d:
        st.warning("No client profile loaded. Go to **Client Input** first.")
        return

    profile = ClientProfile.from_dict(d)

    st.markdown("""
<div style="background:#0d1117;border:1px solid #1f6feb;border-radius:10px;padding:16px 20px;margin-bottom:16px;">
  <div style="font-size:13px;color:#8b949e;margin-bottom:8px;">
  Ready to analyse <strong style="color:#e6edf3;">{}</strong> · Age {} · {} · Income ${:,.0f}/yr
  </div>
  <div style="font-size:11px;color:#6e7681;">Runs: emergency fund check · DTI · housing ratio · net worth benchmark ·
  retirement savings rate · cash flow · insurance · goals · scenario projections</div>
</div>""".format(profile.name, profile.age, profile.marital_status.capitalize(),
                 profile.total_annual_income()), unsafe_allow_html=True)

    st.markdown('<div class="fp-cta-btn">', unsafe_allow_html=True)
    run_clicked = st.button(
        "⚡ Run Full Planning Analysis", key="fp_run_analysis",
        type="primary", use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if run_clicked:
        with st.spinner("Running rules engine and scenario projections…"):
            engine   = RulesEngine()
            issues   = engine.run_all_checks(profile)
            quant    = rpt.build_quant_checks(profile)
            scenarios= generate_scenarios(profile)
            st.session_state.fp_issues       = issues
            st.session_state.fp_quant_checks = quant
            st.session_state.fp_scenarios    = scenarios
            st.session_state.fp_similar_cases = None   # reset case cache
        st.success(f"Analysis complete — {len(issues)} planning issues identified.")

    issues    = st.session_state.get("fp_issues")
    quant     = st.session_state.get("fp_quant_checks")
    scenarios = st.session_state.get("fp_scenarios")

    if not issues:
        st.info("Click **Run Planning Analysis** to generate results.")
        return

    # ── Quantitative checks ─────────────────────────────────────────────────
    st.markdown("##### Quantitative Checks")
    for q in quant:
        sc = _COLORS.get(q.status, "#8b949e")
        st.markdown(f"""
<div style="background:#0d1117;border:1px solid {sc};border-radius:8px;
  padding:10px 14px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;">
  <div>
    <span style="font-size:13px;font-weight:700;color:#e6edf3;">{q.label}</span>
    <span style="font-size:11px;color:#6e7681;margin-left:10px;">{q.detail}</span>
  </div>
  <div style="text-align:right;">
    <span style="font-size:16px;font-weight:700;color:{sc};">{q.value}</span>
    <span style="font-size:11px;color:#8b949e;margin-left:8px;">vs {q.benchmark}</span>
    &nbsp;<span style="background:{sc}22;border:1px solid {sc};color:{sc};
      border-radius:4px;padding:2px 8px;font-size:10px;font-weight:700;">{q.status}</span>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Issues ──────────────────────────────────────────────────────────────
    st.markdown('<div class="fp-section-header">Planning Issues Detected</div>', unsafe_allow_html=True)
    critical = [i for i in issues if i.severity != "INFO"]
    info_only = [i for i in issues if i.severity == "INFO"]
    if not critical:
        st.success("✅ No significant planning gaps detected.")
    _sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "⚪"}
    for iss in critical:
        sc = _COLORS.get(iss.severity, "#8b949e")
        icon = _sev_icon.get(iss.severity, "⚪")
        with st.expander(
            f"{icon} **{iss.severity}** · {iss.category} · {iss.title}",
            expanded=(iss.severity in ("CRITICAL", "HIGH"))
        ):
            cols = st.columns([3, 1])
            cols[0].markdown(iss.detail)
            if iss.metric_value or iss.benchmark:
                cols[1].markdown(f"""
<div class="fp-metric" style="border-color:{sc};">
  <div style="font-size:20px;font-weight:700;color:{sc};">{iss.metric_value or '—'}</div>
  <div style="font-size:10px;color:#8b949e;margin-top:2px;">{iss.benchmark or ''}</div>
</div>""", unsafe_allow_html=True)
            if iss.action_hint:
                st.markdown(f"""
<div style="background:#161b22;border-left:3px solid #388bfd;padding:8px 12px;border-radius:0 6px 6px 0;font-size:13px;color:#c9d1d9;margin-top:6px;">
  💡 <strong>Suggested action:</strong> {iss.action_hint}
</div>""", unsafe_allow_html=True)
    if info_only:
        with st.expander(f"ℹ️ {len(info_only)} informational note(s)", expanded=False):
            for iss in info_only:
                st.caption(f"**{iss.title}:** {iss.detail}")

    # ── Scenarios ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="fp-section-header">Retirement Scenario Projections</div>', unsafe_allow_html=True)
    sc_cols = st.columns(3)
    sc_colors = {"Conservative": "#58a6ff", "Balanced": "#d29922", "Aggressive": "#3fb950"}
    for col, scenario in zip(sc_cols, scenarios):
        c = sc_colors.get(scenario.name, "#8b949e")
        gap_color = "#3fb950" if scenario.gap <= 0 else "#f85149"
        gap_icon  = "✅" if scenario.gap <= 0 else "⚠️"
        gap_label = f"Surplus ${abs(scenario.gap):,.0f}" if scenario.gap <= 0 else f"Shortfall ${scenario.gap:,.0f}"
        assump_str = " · ".join(f"{k}: {v}" for k, v in list(scenario.assumptions.items())[:3])
        with col:
            st.markdown(f"""
<div style="background:#0d1117;border:1px solid {c};border-radius:10px;padding:18px;text-align:center;">
  <div style="font-size:11px;color:{c};text-transform:uppercase;font-weight:700;letter-spacing:1px;margin-bottom:4px;">{scenario.name}</div>
  <div style="font-size:10px;color:#6e7681;margin-bottom:14px;">{assump_str}</div>
  <div style="font-size:11px;color:#6e7681;text-transform:uppercase;letter-spacing:.5px;">Projected Corpus</div>
  <div style="font-size:26px;font-weight:700;color:#e6edf3;margin:6px 0;">${scenario.retirement_corpus:,.0f}</div>
  <div style="background:#21262d;height:1px;margin:10px 0;"></div>
  <div style="font-size:11px;color:#8b949e;">Target: <span style="color:#c9d1d9;">${scenario.corpus_needed:,.0f}</span></div>
  <div style="font-size:15px;font-weight:700;color:{gap_color};margin-top:10px;">{gap_icon} {gap_label}</div>
  {f'<div style="font-size:11px;color:#8b949e;margin-top:6px;background:#161b22;padding:6px 8px;border-radius:4px;">+${scenario.monthly_savings_needed:,.0f}/mo needed</div>' if scenario.monthly_savings_needed > 0 else '<div style="font-size:11px;color:#3fb950;margin-top:6px;">On track</div>'}
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CASE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def _tab_case_insights() -> None:
    from fp_schemas import ClientProfile
    import fp_case_retriever as case_ret

    st.markdown("#### 🏛️ Case Insights — Similar Cases from Built-in Library")
    st.caption(
        "The system retrieves the most analogous financial planning cases from the built-in library "
        "and explains why they match. These cases inform — but do not replace — the deterministic rules engine."
    )

    client = _get_openai_client()
    d      = st.session_state.get("fp_profile_dict")
    issues = st.session_state.get("fp_issues")

    if not d:
        st.warning("Load a client profile in **Client Input** first.")
        return

    profile = ClientProfile.from_dict(d)

    # ── Library status badge ─────────────────────────────────────────────
    index = case_ret.load_case_index()
    n_cases = len(index)
    is_emb  = case_ret.is_indexed()
    st.markdown(f"""
<div style="display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;">
  <span style="background:#1f6feb22;border:1px solid #1f6feb;color:#58a6ff;border-radius:5px;padding:4px 12px;font-size:12px;font-weight:700;">
    📚 {n_cases} built-in cases
  </span>
  <span style="background:{'#3fb95022' if is_emb else '#30363d'};border:1px solid {'#3fb950' if is_emb else '#30363d'};
    color:{'#3fb950' if is_emb else '#6e7681'};border-radius:5px;padding:4px 12px;font-size:12px;font-weight:700;">
    {'✅ Embeddings ready' if is_emb else '⏳ Not yet indexed'}
  </span>
  {'<span style="background:#3fb95022;border:1px solid #3fb950;color:#3fb950;border-radius:5px;padding:4px 12px;font-size:12px;font-weight:700;">✅ Analysis done</span>' if issues else '<span style="background:#d2992222;border:1px solid #d29922;color:#d29922;border-radius:5px;padding:4px 12px;font-size:12px;font-weight:700;">⚠ Run Analysis first for best matches</span>'}
</div>""", unsafe_allow_html=True)

    if not issues:
        st.info("For best case matching, run **Planning Analysis** first so the system can match your client's specific planning issues.")

    # ── Controls ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([3, 1, 1])
    top_k = c2.number_input("Cases to show", value=st.session_state.fp_case_top_k,
                             min_value=1, max_value=6, key="fp_ci_topk")
    mode  = c3.selectbox("Mode", ["hybrid", "structured"], key="fp_ci_mode",
                          help="hybrid = embedding + rules; structured = rules only (no API needed)")

    if not client and mode == "hybrid":
        st.warning("OpenAI API key required for hybrid mode. Switching to structured matching.")
        mode = "structured"

    with c1:
        st.markdown('<div class="fp-cta-btn">', unsafe_allow_html=True)
        find_clicked = st.button("🔍 Find Similar Cases", key="fp_find_cases",
                                  type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if find_clicked:
        with st.spinner(f"Indexing case library and retrieving top {top_k} matches…"):
            if client and not is_emb and mode == "hybrid":
                store = case_ret.build_case_embeddings(client)
                if store.get("error"):
                    st.warning(f"Embedding note: {store['error']}. Using structured matching.")
                    mode = "structured"
            similar = case_ret.retrieve_similar_cases(
                profile, issues or [], client or type("_", (), {"embeddings": None})(),
                top_k=top_k, mode=mode
            )
            st.session_state.fp_similar_cases = similar
            st.session_state.fp_case_top_k    = top_k
            st.session_state.fp_case_mode     = mode

    similar = st.session_state.get("fp_similar_cases")

    # ── Library overview (no results yet) ────────────────────────────────
    if not similar:
        if index:
            st.markdown('<div class="fp-section-header">Case Library Overview</div>', unsafe_allow_html=True)
            import pandas as pd
            df = pd.DataFrame([{
                "Title": m["title"],
                "Household": m["household_type"].replace("_", " ").title(),
                "Life Stage": m["life_stage"].replace("_", " ").title(),
                "Topics": ", ".join(m["major_topics"][:3]),
                "Income Range": f"${m['income_range'][0]/1e3:.0f}K–${m['income_range'][1]/1e3:.0f}K",
            } for m in index])
            st.dataframe(df, use_container_width=True, hide_index=True)
        return

    # ── Results ───────────────────────────────────────────────────────────
    st.markdown(f'<div class="fp-section-header">Top {len(similar)} Matched Cases</div>',
                unsafe_allow_html=True)

    for rank, case in enumerate(similar, 1):
        meta   = case["metadata"]
        struct = case.get("structured", {})
        score  = case.get("score_pct", 0)
        reasons = case.get("reasons", [])
        score_color = "#3fb950" if score >= 70 else "#d29922" if score >= 45 else "#8b949e"

        with st.expander(
            f"#{rank} — {meta.get('title', case['case_id'])} · Match: {score:.0f}%",
            expanded=(rank == 1)
        ):
            # Header row
            st.markdown(f"""
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">
  <span style="background:{score_color}22;border:1px solid {score_color};color:{score_color};
    border-radius:5px;padding:3px 10px;font-size:12px;font-weight:700;">
    {score:.0f}% similarity
  </span>
  <span style="background:#2d333b;color:#c9d1d9;border-radius:5px;padding:3px 10px;font-size:12px;">
    {meta.get('household_type','').replace('_',' ').title()}
  </span>
  <span style="background:#2d333b;color:#c9d1d9;border-radius:5px;padding:3px 10px;font-size:12px;">
    {meta.get('life_stage','').replace('_',' ').title()}
  </span>
  <span style="background:#2d333b;color:#8b949e;border-radius:5px;padding:3px 10px;font-size:12px;">
    {meta.get('source','')}
  </span>
</div>""", unsafe_allow_html=True)

            # Why matched
            st.markdown('<div class="fp-section-header" style="font-size:12px;">Why This Case Matches</div>',
                        unsafe_allow_html=True)
            reason_html = "".join(
                f'<span style="background:#1f6feb22;border:1px solid #1f6feb;color:#58a6ff;'
                f'border-radius:4px;padding:2px 8px;font-size:11px;margin:2px;display:inline-block;">'
                f'✓ {r}</span>'
                for r in reasons
            )
            st.markdown(f'<div style="margin-bottom:12px;">{reason_html}</div>', unsafe_allow_html=True)

            # Key insights from structured summary
            col_a, col_b = st.columns(2)
            if struct.get("planning_issues"):
                with col_a:
                    st.markdown("**Key Planning Issues in This Case**")
                    for iss in struct["planning_issues"][:4]:
                        st.markdown(f"- {iss}")
            if struct.get("candidate_recommendations"):
                with col_b:
                    st.markdown("**Recommendations That Worked**")
                    for rec in struct["candidate_recommendations"][:4]:
                        st.markdown(f"- {rec}")

            # Outcome
            outcome = meta.get("outcome", "")
            if outcome:
                st.markdown(f"""
<div style="background:#0d1117;border-left:3px solid #3fb950;padding:10px 14px;border-radius:0 6px 6px 0;
  font-size:13px;color:#c9d1d9;margin-top:10px;">
  <strong style="color:#3fb950;">Case Outcome:</strong> {outcome}
</div>""", unsafe_allow_html=True)

            # Follow-up questions from this case
            if struct.get("follow_up_questions"):
                with st.expander("💬 Follow-up questions adapted from this case", expanded=False):
                    for q in struct["follow_up_questions"][:4]:
                        st.markdown(f"- {q}")

            # Raw excerpt
            raw = case.get("raw_excerpt", "")
            if raw:
                with st.expander("📄 Case Narrative Excerpt", expanded=False):
                    st.markdown(f"""
<div style="font-size:12px;color:#8b949e;background:#161b22;padding:12px;border-radius:6px;
  line-height:1.7;white-space:pre-wrap;">{raw[:600]}…</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOMMENDATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _tab_recommendation_report() -> None:
    from fp_schemas import ClientProfile
    import fp_report as rpt
    import fp_retriever as retriever

    st.markdown("#### 📄 Recommendation Report")

    d = st.session_state.get("fp_profile_dict")
    if not d:
        st.warning("Load a client profile first.")
        return
    if not st.session_state.get("fp_issues"):
        st.warning("Run Planning Analysis first (tab 3).")
        return

    client = _get_openai_client()
    if not client:
        st.error("**OPENAI_API_KEY** not set.")
        return

    profile  = ClientProfile.from_dict(d)
    issues   = st.session_state.fp_issues
    quant    = st.session_state.fp_quant_checks
    scenarios= st.session_state.fp_scenarios

    similar_cases = st.session_state.get("fp_similar_cases") or []

    st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px 16px;margin-bottom:12px;font-size:12px;color:#8b949e;">
  📊 <strong style="color:#c9d1d9;">{len(issues)} issues</strong> identified &nbsp;·&nbsp;
  🏛️ <strong style="color:#c9d1d9;">{len(similar_cases)} similar cases</strong> {'loaded' if similar_cases else '— visit Case Insights tab to load'} &nbsp;·&nbsp;
  📚 <strong style="color:#c9d1d9;">{retriever.store_count()} doc chunks</strong> in knowledge base
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="fp-cta-btn">', unsafe_allow_html=True)
    gen_clicked = st.button(
        "🤖 Generate AI Planning Report", key="fp_gen_report",
        type="primary", use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if gen_clicked:
        with st.spinner("Retrieving sources and generating report…"):
            query = f"Financial planning for {profile.name}, age {profile.age}: " + \
                    "; ".join(iss.title for iss in issues[:4])
            retrieved = retriever.retrieve(
                query, client,
                top_k=st.session_state.fp_top_k,
                topic_filter=None,
            )
            st.session_state.fp_retrieved_docs = retrieved
            report = rpt.generate_report(
                profile, issues, quant, scenarios, retrieved, client,
                model=st.session_state.fp_model,
                similar_cases=similar_cases if st.session_state.get("fp_use_cases", True) else None,
            )
            st.session_state.fp_report = report
        st.success("✅ Report generated — see below.")

    report = st.session_state.get("fp_report")
    if not report:
        st.info("Click **Generate Planning Report** to produce the full analysis.")
        return

    # ── Header card ─────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;margin-bottom:16px;">
  <div style="font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px;">Planning Report</div>
  <div style="font-size:20px;font-weight:700;color:#e6edf3;">{report.client_name}</div>
  <div style="font-size:13px;color:#6e7681;">{report.client_snapshot}</div>
  <div style="font-size:11px;color:#6e7681;margin-top:6px;">Generated: {report.generated_at}</div>
</div>""", unsafe_allow_html=True)

    # ── Executive summary ────────────────────────────────────────────────────
    if report.executive_summary:
        st.markdown("##### Executive Summary")
        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px 18px;font-size:14px;color:#c9d1d9;line-height:1.8;">
{report.executive_summary}
</div>""", unsafe_allow_html=True)
        st.markdown("")

    # ── Prioritized recommendations ──────────────────────────────────────────
    st.markdown("##### Prioritized Action Plan")
    timeline_order = {"0–30 days": 0, "1–3 months": 1, "3–6 months": 2, "6–24 months": 3}
    for rec in sorted(report.recommendations, key=lambda r: (timeline_order.get(r.timeline,9), r.priority)):
        tl_color = {"0–30 days": "#ff6b6b", "1–3 months": "#f0883e", "3–6 months": "#d29922", "6–24 months": "#58a6ff"}.get(rec.timeline, "#8b949e")
        with st.expander(f"Priority {rec.priority}: {rec.action}", expanded=(rec.priority <= 3)):
            cols = st.columns([3,1])
            cols[0].markdown(f"**Why:** {rec.reason}")
            cols[0].markdown(f"**Expected benefit:** {rec.expected_benefit}")
            cols[0].markdown(f"**Trade-off:** {rec.tradeoff}")
            cols[1].markdown(f"""
<div style="background:{tl_color}22;border:1px solid {tl_color};border-radius:6px;padding:8px;text-align:center;">
  <div style="font-size:11px;color:#8b949e;">Timeline</div>
  <div style="font-size:13px;font-weight:700;color:{tl_color};">{rec.timeline}</div>
</div>""", unsafe_allow_html=True)

    # ── Follow-up questions ──────────────────────────────────────────────────
    if report.follow_up_questions:
        st.markdown("---")
        st.markdown("##### Follow-up Questions for Client")
        for q in report.follow_up_questions:
            if q:
                st.markdown(f"- {q}")

    if report.missing_info:
        st.markdown("##### Missing Information Required")
        for m in report.missing_info:
            if m:
                st.warning(f"⚠ {m}")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(report.DISCLAIMER)

    # ── Download ─────────────────────────────────────────────────────────────
    report_md = _report_to_markdown(report)
    st.download_button("⬇️ Download Report (Markdown)", data=report_md,
                       file_name=f"planning_report_{report.client_name.replace(' ','_')}.md",
                       mime="text/markdown", key="fp_dl_report")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def _tab_explainability() -> None:
    st.markdown("#### 🔍 Explainability — Source-Based Reasoning")
    st.caption("Shows which documents were retrieved, why they were relevant, and how they informed the recommendations.")

    report = st.session_state.get("fp_report")
    if not report:
        st.info("Generate a report first (tab 4) to see source reasoning.")
        return

    # Case reasoning narrative
    st.markdown("##### Case Reasoning")
    st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px 18px;font-size:14px;color:#c9d1d9;line-height:1.8;">
{report.case_reasoning}
</div>""", unsafe_allow_html=True)

    # Retrieved sources
    st.markdown("---")
    st.markdown(f"##### Retrieved Sources ({len(report.retrieved_sources)} chunks)")
    if not report.retrieved_sources:
        st.info("No documents were uploaded — recommendations are based on built-in rules only.")
        return

    for i, src in enumerate(report.retrieved_sources):
        m = src["metadata"]
        with st.expander(
            f"[{i+1}] {m.get('source','?')} · page {m.get('page','?')} · score {src['score']:.3f} · {m.get('topic','?')}",
            expanded=(i < 3)
        ):
            st.markdown(f"""
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
  <span style="background:#1f6feb22;border:1px solid #1f6feb;color:#58a6ff;border-radius:4px;padding:2px 8px;font-size:11px;">{m.get('source_type','?')}</span>
  <span style="background:#2d333b;color:#8b949e;border-radius:4px;padding:2px 8px;font-size:11px;">{m.get('reliability_level','?')}</span>
  <span style="background:#2d333b;color:#8b949e;border-radius:4px;padding:2px 8px;font-size:11px;">Similarity: {src['score']:.3f}</span>
</div>
<div style="font-size:13px;color:#c9d1d9;background:#161b22;padding:12px;border-radius:6px;line-height:1.7;">{src['text']}</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def _tab_settings() -> None:
    import fp_case_retriever as case_ret
    st.markdown("#### ⚙️ Settings")

    # ── LLM & Retrieval ──────────────────────────────────────────────────
    st.markdown('<div class="fp-section-header">LLM & Retrieval</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("LLM Model",
                          ["gpt-4o-mini","gpt-4o","gpt-4-turbo"],
                          index=["gpt-4o-mini","gpt-4o","gpt-4-turbo"].index(
                              st.session_state.get("fp_model","gpt-4o-mini")),
                          key="fp_set_model")
    top_k = c2.number_input("Doc Retrieval Top-k", value=st.session_state.get("fp_top_k",8),
                             min_value=1, max_value=20, key="fp_set_topk")
    use_cases = c3.checkbox("Use case library in report", value=st.session_state.get("fp_use_cases", True),
                             key="fp_set_use_cases",
                             help="Include similar cases in LLM report prompt for analogical reasoning")

    if st.button("💾 Apply Settings", key="fp_apply_settings"):
        st.session_state.fp_model     = model
        st.session_state.fp_top_k     = top_k
        st.session_state.fp_use_cases = use_cases
        st.success("Settings saved.")

    # ── Case Library Management ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="fp-section-header">Built-in Case Library</div>', unsafe_allow_html=True)

    index = case_ret.load_case_index()
    is_emb = case_ret.is_indexed()
    st.markdown(f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">
  <div class="fp-metric" style="flex:1;min-width:120px;">
    <div style="font-size:24px;font-weight:700;color:#58a6ff;">{len(index)}</div>
    <div style="font-size:11px;color:#8b949e;">Total Cases</div>
  </div>
  <div class="fp-metric" style="flex:1;min-width:120px;">
    <div style="font-size:24px;font-weight:700;color:{'#3fb950' if is_emb else '#d29922'};">{'Yes' if is_emb else 'No'}</div>
    <div style="font-size:11px;color:#8b949e;">Embeddings Cached</div>
  </div>
</div>""", unsafe_allow_html=True)

    col_re, col_cl = st.columns(2)
    with col_re:
        client = _get_openai_client()
        if st.button("🔄 Re-index Case Library", key="fp_reindex_cases", disabled=(not client)):
            case_ret.clear_case_store()
            with st.spinner("Re-embedding all cases…"):
                store = case_ret.build_case_embeddings(client)
            if store.get("has_embeddings"):
                st.success(f"Re-indexed {store['count']} cases.")
            else:
                st.error(f"Failed: {store.get('error','unknown')}")
    with col_cl:
        if st.button("🗑️ Clear Case Cache", key="fp_clear_case_cache"):
            case_ret.clear_case_store()
            st.session_state.fp_similar_cases = None
            st.success("Case cache cleared.")

    if index:
        with st.expander("View all cases in library", expanded=False):
            import pandas as pd
            df = pd.DataFrame([{
                "ID": m["case_id"],
                "Title": m["title"],
                "Type": m["household_type"],
                "Stage": m["life_stage"],
                "Topics": ", ".join(m["major_topics"][:3]),
                "Source": m.get("source",""),
            } for m in index])
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Rule Configuration Reference ─────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="fp-section-header">Rule Thresholds</div>', unsafe_allow_html=True)
    rules_path = os.path.join(os.path.dirname(__file__), "data", "rule_configs", "planning_rules.json")
    if os.path.exists(rules_path):
        with open(rules_path) as f:
            rules = json.load(f)
        st.json(rules, expanded=False)
    else:
        st.warning("planning_rules.json not found.")

    st.markdown("---")
    st.caption(
        "⚠️ This application is built for educational purposes only. "
        "All outputs should be reviewed by a licensed CFP, CPA, or attorney before any action."
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def show_financial_planner() -> None:
    _init_state()
    _fp_inject_css()

    col_back, col_title = st.columns([1, 11])
    with col_back:
        if st.button("← Back", key="back_fp"):
            st.session_state.current_view = "home"
            st.rerun()
    with col_title:
        d = st.session_state.get("fp_profile_dict")
        name_str = f" — {d['name']}" if d else ""
        st.markdown(f"""
<div style="padding:6px 0 4px 0;">
  <span style="font-size:22px;font-weight:700;color:#e6edf3;">🧭 Financial Planning Assistant{name_str}</span>
  <span style="font-size:12px;color:#8b949e;margin-left:12px;">
  Hybrid rules engine · Built-in case library · Scenario projections · LLM narrative
  </span>
</div>""", unsafe_allow_html=True)

    tabs = st.tabs([
        "📋 Client Input",
        "📊 Analysis",
        "🏛️ Case Insights",
        "📄 Report",
        "📚 Knowledge Base",
        "🔍 Explainability",
        "⚙️ Settings",
    ])

    with tabs[0]: _tab_client_input()
    with tabs[1]: _tab_planning_analysis()
    with tabs[2]: _tab_case_insights()
    with tabs[3]: _tab_recommendation_report()
    with tabs[4]: _tab_case_library()
    with tabs[5]: _tab_explainability()
    with tabs[6]: _tab_settings()


# ── Markdown export helper ────────────────────────────────────────────────────

def _report_to_markdown(report) -> str:
    lines = [
        f"# Financial Planning Report — {report.client_name}",
        f"Generated: {report.generated_at}",
        f"\n**Client:** {report.client_snapshot}",
        "\n---",
        "\n## Executive Summary",
        report.executive_summary or "_Not available_",
        "\n## Key Planning Issues",
    ]
    for iss in report.issues:
        if iss.severity != "INFO":
            lines.append(f"- **[{iss.severity}] {iss.category}** — {iss.title}: {iss.detail}")
    lines += ["\n## Quantitative Checks"]
    for q in report.quant_checks:
        lines.append(f"- {q.label}: **{q.value}** (benchmark: {q.benchmark}) [{q.status}]")
    lines += ["\n## Prioritized Recommendations"]
    for rec in report.recommendations:
        lines.append(f"\n### Priority {rec.priority}: {rec.action}")
        lines.append(f"- **Timeline:** {rec.timeline}")
        lines.append(f"- **Why:** {rec.reason}")
        lines.append(f"- **Trade-off:** {rec.tradeoff}")
    lines += ["\n## Retirement Scenarios"]
    for s in report.scenarios:
        lines.append(f"\n### {s.name}")
        lines.append(s.summary)
    if report.follow_up_questions:
        lines += ["\n## Follow-up Questions"]
        for q in report.follow_up_questions:
            if q: lines.append(f"- {q}")
    if report.missing_info:
        lines += ["\n## Missing Information"]
        for m in report.missing_info:
            if m: lines.append(f"- {m}")
    lines += ["\n---", f"\n_{report.DISCLAIMER}_"]
    return "\n".join(lines)
