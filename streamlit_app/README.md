# AI Financial Advisor — Streamlit Dashboard

A custom Streamlit web application built on top of [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund).
The original repo provides the multi-agent CLI framework (LangGraph + Financial Datasets API).
Everything in `streamlit_app/` was written entirely from scratch and does not exist in the original repo.

**Live app:** https://oaowouo--ai-financial-advisor-streamlit-appapp-qwt5aa.streamlit.app/

---

## Tools

| Tool | File | What it does |
|---|---|---|
| **Stock Analyzer** | `stock_analyzer.py` | CFA-style technical analysis (RSI, MACD, Bollinger Bands, ADX), fundamental scoring (valuation / profitability / growth / financial health), multi-model valuation (P/E, DCF, analyst consensus), BUY / HOLD / SELL recommendation |
| **Portfolio Allocator** | `portfolio_allocator.py` | Multi-stock signal analysis, position sizing, risk metrics (Sharpe, Beta, VaR), S&P 500 benchmark comparison, dividend tracking, one-click rebalancing |
| **Financial Planner** | `financial_planner.py` | Hybrid rules engine + RAG + LLM narrative: 6-tab UI covering client profile, document library, gap analysis (emergency fund / DTI / cash flow / insurance / retirement), Conservative/Balanced/Aggressive scenario projections, recommendation report with source citations, and explainability layer |

---

## File Map

### UI Layer

| File | Role |
|---|---|
| `app.py` | Dark-themed home page with card navigation to the three tools |
| `stock_analyzer.py` | Stock Analyzer tool — standalone, no external state |
| `portfolio_allocator.py` | Portfolio Allocator tool — standalone, no external state |
| `financial_planner.py` | Financial Planner tool — 6-tab Streamlit UI that wires all `fp_*` modules together |

### Financial Planner Backend

| File | Role |
|---|---|
| `fp_schemas.py` | Pydantic data models: `ClientProfile`, `PlanningIssue`, `Recommendation`, `QuantCheck`, `ScenarioProjection`, `PlanningReport` |
| `fp_calculators.py` | All deterministic financial math: DTI, emergency fund months, net worth benchmark (Stanley-Danko 1996), retirement future-value projection, goal savings rate, SWR corpus |
| `fp_rules.py` | Rules engine: 8 check categories (emergency fund, debt, cash flow, insurance, retirement match, savings trajectory, net worth, goals) — thresholds loaded from `data/rule_configs/planning_rules.json`, not hardcoded |
| `fp_scenarios.py` | Retirement scenario engine: Conservative (5% return / 3.5% SWR), Balanced (7% / 4%), Aggressive (9% / 4.5%) — calculates projected corpus, gap, and required monthly savings |
| `fp_retriever.py` | In-memory NumPy cosine-similarity RAG: ingest PDF/MD/TXT/HTML uploads, embed with OpenAI `text-embedding-3-small`, retrieve top-k chunks for report grounding |
| `fp_case_retriever.py` | Case-based reasoning retriever: 12 built-in reference cases matched by client demographics and issue type |
| `fp_report.py` | LLM narrative layer: takes rules issues + quant checks → calls GPT to write Executive Summary, Case Reasoning, Follow-up Questions, and Missing Information sections |

### Config & Secrets

| Path | Role |
|---|---|
| `.streamlit/secrets.toml.example` | Template for Streamlit secrets (copy to `secrets.toml` and add `OPENAI_API_KEY`) |
| `data/rule_configs/planning_rules.json` | All financial planning thresholds — edit here to change what triggers a warning without touching Python |

---

## Architecture: Why Hybrid Rules Engine + LLM?

A pure-LLM financial planner hallucinates thresholds and cannot be audited. The strict layer separation is:

```
ClientProfile
    ├── fp_calculators.py  →  raw numbers (DTI, emergency fund months, FV)
    ├── fp_rules.py        →  pass / warn / fail decisions  (reads planning_rules.json)
    ├── fp_scenarios.py    →  Conservative / Balanced / Aggressive projections
    └── fp_report.py
            ├── fp_retriever.py     →  relevant document chunks (RAG)
            ├── fp_case_retriever.py →  similar past case (CBR)
            └── GPT                 →  plain-language narrative only
```

**The LLM never decides whether a metric is a problem. The rules engine does. The LLM only explains it.**

This makes every recommendation traceable to a formula, threshold, and data source.

---

## Data Sources & Methodology Credits

### Market Data
- **Yahoo Finance** via [yfinance](https://github.com/ranaroussi/yfinance) — real-time quotes, historical prices, fundamentals (15–20 min delayed)

### AI / LLM
- **OpenAI API** — GPT-4o-mini for narrative generation, `text-embedding-3-small` for RAG embeddings

### Financial Planning Methodology
- **Bengen (1994)** — 4% Safe Withdrawal Rate. *Journal of Financial Planning.*
- **Stanley & Danko (1996)** — Net worth benchmarks by age. *The Millionaire Next Door.*
- **CFPB** — DTI standards: ≤36% recommended, ≤43% qualified mortgage limit.
- **SSA** — Social Security replacement rate estimates by income tier (ssa.gov).
- **IRS** — 401(k) limits: $23,000 standard / $30,500 catch-up (age ≥ 50) for 2024. IRS Publication 560.
- **IRS** — Roth IRA income phase-out: $146k–$161k (single) / $230k–$240k (MFJ) for 2024. IRS Publication 590-A.

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml: OPENAI_API_KEY = "sk-..."

# 3. Run
streamlit run app.py
```

No API key is required for Stock Analyzer or Portfolio Allocator. The Financial Planner requires OpenAI for RAG embeddings and LLM narrative.

---

## Disclaimer

This application is for **educational purposes only**.
It does not constitute legal, tax, or investment advice.
Always consult a licensed CFP, CPA, or attorney before making financial decisions.
Market data is provided by Yahoo Finance and is not guaranteed to be accurate or complete.

---

> Built for **MGMT 690 — Mastering AI for Finance**, Purdue University (Spring 2026).
> Base framework: [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) by [@virattt](https://github.com/virattt).
