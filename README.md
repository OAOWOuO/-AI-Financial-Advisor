[![CI](https://github.com/OAOWOuO/-AI-Financial-Advisor/actions/workflows/ci.yml/badge.svg)](https://github.com/OAOWOuO/-AI-Financial-Advisor/actions/workflows/ci.yml)

# AI Financial Advisor

**Course:** MGMT 690 — Mastering AI for Finance, Purdue University (Spring 2026)
**Student:** YuanTeng Fan
**Live app:** https://oaowouo--ai-financial-advisor-streamlit-appapp-qwt5aa.streamlit.app/

---

This repo is a fork of [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) — a multi-agent CLI that simulates famous investors making trading decisions.

On top of that framework, I built a **custom Streamlit web application from scratch** in `streamlit_app/`:

| My Custom Work | Original Framework |
|---|---|
| `streamlit_app/` — written entirely from scratch | `src/` — unchanged fork of virattt/ai-hedge-fund |
| yfinance + OpenAI API + NumPy RAG | Financial Datasets API + LangGraph |
| Runs at the live Streamlit URL above | Runs via `poetry run python src/main.py` |

---

## Table of Contents

- [Quick Start — Streamlit App](#quick-start--streamlit-app)
- [Tool 1: Stock Analyzer](#tool-1-stock-analyzer)
- [Tool 2: Portfolio Allocator](#tool-2-portfolio-allocator)
- [Tool 3: Financial Planner](#tool-3-financial-planner)
- [File Map](#file-map)
- [Architecture & Design Decisions](#architecture--design-decisions)
- [Tests](#tests)
- [AI Collaboration Log](#ai-collaboration-log)
- [Original Framework (src/)](#original-framework-src)
- [Disclaimer](#disclaimer)

---

## Quick Start — Streamlit App

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key (choose one method)
cp .env.example .env
# then edit .env: OPENAI_API_KEY=sk-...

# OR use Streamlit secrets:
cp streamlit_app/.streamlit/secrets.toml.example streamlit_app/.streamlit/secrets.toml
# then edit secrets.toml: OPENAI_API_KEY = "sk-..."

# 3. Launch
streamlit run streamlit_app/app.py
```

> No API key is needed for Stock Analyzer or Portfolio Allocator (yfinance only).
> The Financial Planner requires OpenAI for RAG embeddings and LLM narrative generation.

---

## Tool 1: Stock Analyzer

**File:** `streamlit_app/stock_analyzer.py`

Institutional-grade single-stock deep-dive in one page.

**What it does:**
- **Technical analysis** — RSI, MACD, Bollinger Bands, ADX (CFA-style scoring and interpretation)
- **Fundamental scoring** — four sub-scores: valuation, profitability, growth, financial health
- **Multi-model valuation** — P/E relative, DCF, and analyst consensus combined into one signal
- **Final recommendation** — BUY / HOLD / SELL with a sanity-checked Conclusion and one-year Forecast narrative

**Data source:** Yahoo Finance via yfinance (15–20 min delayed)
**Tested with:** AAPL, MSFT, NVDA

---

## Tool 2: Portfolio Allocator

**File:** `streamlit_app/portfolio_allocator.py`

Multi-stock portfolio optimization and risk dashboard.

**What it does:**
- Enter any list of tickers and a portfolio dollar value
- Signal analysis per stock (momentum, trend, mean reversion)
- Optimal position sizing based on signals and risk limits
- Risk metrics: Sharpe ratio, Beta vs S&P 500, Value at Risk (VaR)
- Dividend tracking and yield contribution per position
- One-click rebalancing view — current vs target allocation

**Data source:** Yahoo Finance via yfinance

---

## Tool 3: Financial Planner

**Files:** `streamlit_app/financial_planner.py` + supporting modules (`fp_*.py`)

AI financial planning assistant with a hybrid rules engine + RAG + LLM narrative architecture.

**What it does — 6-tab UI:**

| Tab | What it shows |
|---|---|
| **Client Profile** | Input form for all financial data; AI auto-fill from an uploaded document |
| **Document Library** | Upload PDF / MD / TXT / HTML reference documents for RAG grounding |
| **Gap Analysis** | Deterministic checks: emergency fund, DTI, cash flow, insurance, retirement match |
| **Scenario Projections** | Conservative / Balanced / Aggressive retirement outcomes with gap and monthly savings targets |
| **Recommendation Report** | Prioritized action plan with source citations and GPT-written narrative |
| **Explainability** | Shows exactly which rules fired, which thresholds were used, and why |

**Key design principle:** The LLM never decides whether a metric is a problem — the rules engine does. The LLM only writes plain-language explanations of what the rules already determined. See [Architecture](#architecture--design-decisions).

---

## File Map

### `streamlit_app/` — Custom Streamlit App (100% original)

| File | Role |
|---|---|
| `app.py` | Dark-themed home page with card navigation to the three tools |
| `stock_analyzer.py` | Stock Analyzer — technical + fundamental analysis, multi-model valuation, recommendation |
| `portfolio_allocator.py` | Portfolio Allocator — signal analysis, position sizing, risk metrics, rebalancing |
| `financial_planner.py` | Financial Planner — 6-tab Streamlit UI wiring all fp_* modules together |
| `fp_schemas.py` | Data models: `ClientProfile`, `PlanningIssue`, `Recommendation`, `QuantCheck`, `ScenarioProjection`, `PlanningReport` |
| `fp_calculators.py` | All financial math: DTI, emergency fund months, FV projection, SWR corpus, net worth benchmark (Stanley-Danko) |
| `fp_rules.py` | Rules engine: 8 check categories with thresholds loaded from `data/rule_configs/planning_rules.json` |
| `fp_scenarios.py` | Retirement scenario engine: Conservative (5% return / 3.5% SWR), Balanced (7% / 4%), Aggressive (9% / 4.5%) |
| `fp_retriever.py` | In-memory NumPy cosine-similarity RAG: ingest uploaded documents, retrieve top-k chunks for report grounding |
| `fp_case_retriever.py` | Case-based reasoning: 12 built-in reference cases matched by client demographics and issue patterns |
| `fp_report.py` | LLM narrative layer: quant checks + rules issues → GPT writes Executive Summary, Case Reasoning, Follow-up Questions |

### `scripts/` — Data Pipeline Scripts (original)

| File | Role |
|---|---|
| `build_index.py` | RAG ingestion: chunk PDFs/markdown, embed with `text-embedding-3-small`, store in ChromaDB. Output: `data/processed/chunks.json` |
| `download_fpa_cases.py` | Download financial planning case studies from the FPA |
| `ingest_fpa_cases.py` | Parse and ingest FPA cases into the vector store |

### `tests/` — Unit Tests (147 total, all passing)

| File | Tests | Covers |
|---|---|---|
| `test_fp_calculators.py` | 51 | All financial math functions |
| `test_fp_schemas.py` | 26 | Data model validation and edge cases |
| `test_fp_rules.py` | 25 | Rules engine thresholds and boundary values |
| `test_fp_report.py` | 21 | Report generation and LLM output parsing |
| `test_fp_scenarios.py` | 13 | Retirement projection engine |
| `test_build_index.py` | 11 | RAG ingestion pipeline |

No LLM, Streamlit, or external API required to run any test.

### `data/` — Reference Data

| Path | Contents |
|---|---|
| `data/rule_configs/planning_rules.json` | All financial planning thresholds (DTI, emergency fund, savings rate, etc.) — editable without touching code |
| `data/raw/` | Raw source documents for RAG ingestion |
| `data/processed/chunks.json` | Reproducible output of `scripts/build_index.py` |

### `product/` — Planning Artifacts

| File | Contents |
|---|---|
| `product/sections/run_2026-02-02.md` | Full saved CLI output from running the multi-agent system on AAPL, MSFT, NVDA (Feb 2, 2026) |
| `product/product-overview.md` | Future feature design: AI-Powered Options Wheel System (AI stock selection → systematic options income) |
| `product/product-roadmap.md` | Development roadmap for future work |

### `src/` — Original Framework (unchanged fork of virattt/ai-hedge-fund)

17 AI investment agents + LangGraph multi-agent orchestration + backtester. See [Original Framework](#original-framework-src).

---

## Architecture & Design Decisions

### 1. Why Hybrid Rules Engine + LLM, Not Pure LLM?

A pure-LLM financial planner has two critical problems:
- **Hallucination risk** — LLMs invent plausible-sounding but wrong numbers (e.g., wrong DTI thresholds)
- **Non-auditability** — you cannot explain *why* a recommendation was made or trace it back to a formula

The solution is a strict separation of responsibilities:

| Layer | Component | Responsibility |
|---|---|---|
| **Deterministic** | `fp_calculators.py` | All math — DTI, emergency fund months, FV projections, SWR corpus |
| **Rule-based** | `fp_rules.py` + `planning_rules.json` | Threshold decisions — pass / warn / fail, severity classification |
| **Narrative only** | `fp_report.py` → GPT | Writes English explanation of numbers the rules engine already produced |

The LLM never decides whether a DTI of 38% is a problem — the rules engine does. The LLM only explains it in plain language. This makes the system auditable, reproducible, and safe.

### 2. Data Flow — Financial Planner

```
ClientProfile (fp_schemas.py)
        │
        ├── RulesEngine (fp_rules.py)  ──→  List[PlanningIssue]
        │         └── fp_calculators.py
        │
        ├── ScenarioEngine (fp_scenarios.py) ──→  List[ScenarioProjection]
        │         └── fp_calculators.py
        │
        ├── build_quant_checks (fp_report.py) ──→  List[QuantCheck]
        │
        └── generate_report (fp_report.py)
                  ├── Retrieved docs  (fp_retriever.py — cosine-similarity RAG)
                  ├── Similar cases   (fp_case_retriever.py — CBR)
                  └── GPT narrative   ──→  PlanningReport
```

### 3. Why In-Memory NumPy RAG Instead of ChromaDB?

`scripts/build_index.py` uses ChromaDB for batch ingestion. For the live Streamlit session, I switched to an in-memory NumPy cosine-similarity store for three reasons:

1. **Streamlit Cloud statelessness** — ChromaDB requires persistent disk storage; session uploads disappear on page refresh regardless
2. **No cold-start** — in-memory initialises instantly with no file I/O
3. **Sufficient scale** — a planner session uploads fewer than 20 documents; NumPy is faster than ChromaDB at this scale

Trade-off: uploaded documents do not persist across browser sessions. Acceptable for a classroom demo tool.

### 4. Why External Thresholds in `planning_rules.json`?

All financial planning thresholds live in `data/rule_configs/planning_rules.json` rather than being hardcoded in Python. This means:

- A professor or planner can **change a threshold without touching code**
- The rules are **transparent and auditable** — anyone can read the JSON to see what triggers a warning
- Unit tests can override the rules file to test edge cases independently

### 5. Why Three Retirement Scenarios?

Single-point projections give false precision. Three scenarios (Conservative / Balanced / Aggressive) with different return, inflation, and SWR assumptions:

- Show the **range of outcomes** a client should plan for
- Make savings rate sensitivity concrete (e.g., "+2% savings rate closes a $180k shortfall")
- Mirror how professional CFPs present retirement plans (Monte Carlo is the gold standard; three scenarios is a practical classroom approximation)

### 6. Case-Based Reasoning (CBR)

The 12 built-in reference cases in `fp_case_retriever.py` enable analogical reasoning — finding a similar past case and explaining what worked. This supplements document RAG (content retrieval) with demographic and issue pattern matching. The LLM is prompted to cite which case is analogous and why, making the reasoning transparent.

### 7. AI Auto-Fill from Document

Upload a case study → GPT extracts structured data → auto-populates the client profile form. GPT is prompted with the exact JSON schema at `temperature=0` to minimise hallucination. The user sees a preview before data is applied — GPT output is never silently trusted.

---

## Tests

```bash
pytest tests/
```

```
tests/test_fp_calculators.py   51 tests   ← financial math
tests/test_fp_rules.py         25 tests   ← rules engine thresholds
tests/test_fp_schemas.py       26 tests   ← data models
tests/test_fp_scenarios.py     13 tests   ← retirement projections
tests/test_fp_report.py        21 tests   ← report generation
tests/test_build_index.py      11 tests   ← RAG pipeline
─────────────────────────────────────────
Total: 147 tests, all passing (CI green)
```

CI (`.github/workflows/ci.yml`) runs flake8 lint + pytest on every push. Lint is scoped to `streamlit_app/` and `scripts/` only — the original `src/` is excluded to avoid modifying the upstream framework.

---

## AI Collaboration Log

See [`AI_LOG.md`](AI_LOG.md) for a session-by-session record of:

- What Claude Code generated vs. what the student wrote
- Where the student modified, rejected, or redirected AI output
- Every key architectural decision and why the student made it

Summary of what the student decided (not AI): the hybrid rules-vs-LLM architecture, the NumPy-over-ChromaDB choice, the lint scope, the test strategy (pure functions only), all UI/UX decisions, and all financial threshold values (reviewed against CFP standards).

---

## Original Framework (src/)

This repo forks [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund). The `src/` directory is unchanged from the original.

**What it does:** 17 AI agents simulate famous investors making trading decisions via LangGraph multi-agent orchestration. Educational proof of concept — no real trades are made.

**Agents:**
Aswath Damodaran, Ben Graham, Bill Ackman, Cathie Wood, Charlie Munger, Michael Burry, Mohnish Pabrai, Peter Lynch, Phil Fisher, Rakesh Jhunjhunwala, Stanley Druckenmiller, Warren Buffett, plus Valuation, Sentiment, Fundamentals, Technicals, Risk Manager, and Portfolio Manager agents.

### Install

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

### Configure API Keys

```bash
cp .env.example .env
# Required: at least one LLM key
OPENAI_API_KEY=your-openai-api-key

# Required for tickers other than AAPL, GOOGL, MSFT, NVDA, TSLA
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

### Run

```bash
# Run hedge fund (interactive CLI)
poetry run python src/main.py --tickers AAPL,MSFT,NVDA

# With date range
poetry run python src/main.py --tickers AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01

# With local LLMs (Ollama)
poetry run python src/main.py --tickers AAPL,MSFT,NVDA --ollama

# Run backtester
poetry run python src/backtester.py --tickers AAPL,MSFT,NVDA
```

Saved output from my Feb 2, 2026 run: [`product/sections/run_2026-02-02.md`](product/sections/run_2026-02-02.md)

---

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment decisions
- No investment advice or guarantees provided
- Consult a licensed CFP, CPA, or financial advisor before making real financial decisions
- Market data from Yahoo Finance is 15–20 min delayed and not guaranteed to be accurate
- Past performance does not indicate future results

---

> Built for **MGMT 690 — Mastering AI for Finance**, Purdue University (Spring 2026).
> Base framework: [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) by [@virattt](https://github.com/virattt).
> Licensed under MIT.
