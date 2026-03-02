[![CI](https://github.com/OAOWOuO/-AI-Financial-Advisor/actions/workflows/ci.yml/badge.svg)](https://github.com/OAOWOuO/-AI-Financial-Advisor/actions/workflows/ci.yml)

## 🆕 What I Added (My Contributions)

> **This repo is a course project built on top of [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund).**
> The original repo provides the multi-agent CLI framework (LangGraph + Financial Datasets API).
> Everything in `streamlit_app/` was written entirely from scratch by me and does **not** exist in the original repo.

### My Original Work

#### `streamlit_app/` — Custom Streamlit Web Application (100% original)

None of these files existed in virattt/ai-hedge-fund. Built independently using a completely different tech stack:
**yfinance** (market data) + **OpenAI API** (LLM + embeddings) + **ChromaDB** (vector store).
The original repo uses Financial Datasets API + LangGraph; my app uses none of those.

| File | What it does |
|---|---|
| `app.py` | Dark-themed home page with card-based navigation between all three tools |
| `portfolio_allocator.py` | Portfolio optimization UI: multi-stock signal analysis, position sizing, risk metrics (Sharpe, Beta, VaR), S&P 500 benchmark comparison, dividend tracking, one-click rebalancing |
| `stock_analyzer.py` | Individual stock deep-dive: CFA-style technical analysis (RSI, MACD, Bollinger Bands, ADX), fundamental scoring (valuation / profitability / growth / financial health), multi-model valuation (P/E, DCF, analyst consensus), BUY / HOLD / SELL recommendation with sanity-checked Conclusion & Forecast |
| `financial_planner.py` | AI Financial Planning Assistant (replaces Case Q&A): hybrid rules engine + RAG + LLM narrative. 6-tab UI — client profile input, document library, gap analysis (emergency fund / DTI / cash flow / insurance / retirement), Conservative/Balanced/Aggressive scenario projections, prioritized recommendation report with source citations, and explainability layer |
| `fp_schemas.py` | Data models: `ClientProfile`, `PlanningIssue`, `Recommendation`, `QuantCheck`, `ScenarioProjection`, `PlanningReport` |
| `fp_calculators.py` | Deterministic financial math: emergency fund months, DTI, net worth benchmark (Stanley-Danko), retirement FV projection, goal savings |
| `fp_rules.py` | Rules engine: 8 check categories (emergency fund, debt, cash flow, insurance, retirement match capture, trajectory, net worth, goals) with thresholds from `data/rule_configs/planning_rules.json` |
| `fp_scenarios.py` | Retirement scenario engine: Conservative (5%/3.5% SWR), Balanced (7%/4%), Aggressive (9%/4.5%) projections with gap and monthly savings targets |
| `fp_retriever.py` | In-memory numpy cosine-similarity RAG: ingest PDF/MD/TXT/HTML with topic + reliability metadata, retrieve top-k chunks for report grounding |
| `fp_report.py` | LLM narrative layer: deterministic quant checks + rules issues → GPT writes Executive Summary, Case Reasoning, Follow-up Questions, Missing Information |

#### `scripts/build_index.py` — RAG Ingestion Pipeline (original)

One command to ingest course PDFs / markdown files, chunk them, embed with
`text-embedding-3-small`, and store in ChromaDB. Outputs are reproducible via
`data/processed/chunks.json`.

#### `product/sections/` — Reproducible Run Outputs

Saved CLI outputs from running the multi-agent system on **AAPL, MSFT, NVDA**
(Feb 2, 2026). Demonstrates the system works end-to-end with real tickers.

#### `.github/workflows/ci.yml` — CI Pipeline

GitHub Actions workflow: flake8 lint + pytest on every push.

---

### How to Run My Streamlit App

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key (choose one method)
#    Option A — .env file (local dev):
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...

#    Option B — Streamlit secrets (local or Streamlit Cloud):
cp streamlit_app/.streamlit/secrets.toml.example streamlit_app/.streamlit/secrets.toml
# Edit secrets.toml and add: OPENAI_API_KEY = "sk-..."

# 3. Launch
streamlit run streamlit_app/app.py
```

The **Financial Planner** tab requires an OpenAI API key for:
- Embedding uploaded reference documents (RAG ingestion)
- Generating the LLM narrative sections of the planning report

No API key is needed to view client profiles, run the rules engine, or generate scenario projections (all deterministic).

Live deployment: <https://oaowouo--ai-financial-advisor-streamlit-appapp-qwt5aa.streamlit.app/>

> This project builds upon [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund).
> Original multi-agent framework by [@virattt](https://github.com/virattt).

---

## Architecture & Design Decisions

> This section explains the **why** behind key technical choices — not just what was built.

### 1. Why a Hybrid Rules Engine + LLM, Not Pure LLM?

A pure-LLM approach to financial planning has two critical problems:
- **Hallucination risk**: LLMs invent plausible-sounding but wrong numbers (e.g., wrong DTI thresholds)
- **Non-auditability**: you cannot explain *why* a recommendation was made or trace it to a formula

The solution is a strict separation of responsibilities:

| Layer | Component | Responsibility |
|---|---|---|
| **Deterministic** | `fp_calculators.py` | All math — DTI, emergency fund months, FV projections, SWR corpus |
| **Rule-based** | `fp_rules.py` + `planning_rules.json` | Threshold decisions — pass/warn/fail, severity classification |
| **Narrative only** | `fp_report.py` → GPT | Writes English explanation of the numbers the rules engine already produced |

The LLM **never decides** whether a DTI of 38% is a problem — the rules engine does. The LLM only explains it in plain language. This makes the system auditable, reproducible, and safe.

### 2. Data Flow

```
ClientProfile (fp_schemas.py)
        │
        ├── RulesEngine (fp_rules.py)  ──→  List[PlanningIssue]
        │         │                               │
        │         └── fp_calculators.py           │
        │                                         │
        ├── ScenarioEngine (fp_scenarios.py) ──→  List[ScenarioProjection]
        │         │
        │         └── fp_calculators.py
        │
        ├── build_quant_checks (fp_report.py) ──→ List[QuantCheck]
        │
        └── generate_report (fp_report.py)
                  │
                  ├── Retrieved docs (fp_retriever.py, cosine-similarity RAG)
                  ├── Similar cases  (fp_case_retriever.py, CBR)
                  └── GPT narrative  ──→ PlanningReport
```

### 3. Why In-Memory NumPy RAG Instead of ChromaDB?

The original `scripts/build_index.py` uses ChromaDB for batch ingestion. For the **live Streamlit session**, I switched to an in-memory numpy cosine-similarity store for three reasons:

1. **Streamlit Cloud statelessness** — ChromaDB requires persistent disk storage; session uploads disappear on page refresh regardless
2. **No cold-start** — in-memory store initialises instantly with no file I/O
3. **Sufficient scale** — a planner session typically uploads < 20 documents; numpy is faster than ChromaDB at this scale

The trade-off is that uploaded documents do not persist across browser sessions. For a classroom demo tool, this is acceptable.

### 4. Why External Thresholds in `planning_rules.json`?

All financial planning thresholds (emergency fund months, DTI limits, replacement ratios, etc.) live in `data/rule_configs/planning_rules.json` rather than being hardcoded in Python. This means:

- A professor or planner can **change a threshold without touching code**
- The rules are **transparent and auditable** — anyone can read the JSON to understand what triggers a warning
- Unit tests can override the rules file to test edge cases

### 5. Why Three Retirement Scenarios Instead of One?

Single-point projections give a **false precision**. Three scenarios (Conservative / Balanced / Aggressive) with different return, inflation, and savings rate assumptions:
- Show the **range of outcomes** the client should plan for
- Make the sensitivity to savings rate increases concrete (e.g., "+2% saves $X shortfall")
- Mirror how professional CFPs present retirement plans (Monte Carlo is the gold standard; three scenarios is a simplified version)

### 6. Case-Based Reasoning (CBR) Layer

The 12 built-in reference cases enable **analogical reasoning** — finding a similar past case and explaining what worked. This supplements RAG (document retrieval) with pattern matching on client demographics and issues. The LLM is explicitly instructed to cite which case is analogous and why, making the reasoning transparent.

### 7. AI Auto-Fill from Document

A document upload → GPT extraction pipeline (`_ai_extract_profile()`) allows a planner to paste in a case study or financial summary and have the structured form auto-populated. GPT is prompted with the exact JSON schema and `temperature=0` to minimise hallucination. The user sees a preview before the data is applied — GPT's output is never silently trusted.

---

## MGMT 690 – Project 1 (Feb 2, 2026)

**Student:** YuanTeng Fan | **Course:** MGMT 690, Purdue University | **Semester:** Spring 2026

I set up the AI Hedge Fund repo locally, configured `.env` safely (gitignored), and ran the CLI on **AAPL, MSFT, NVDA**.

- **Run command:** `poetry run python src/main.py --tickers AAPL,MSFT,NVDA`
- **Saved output:** `product/sections/run_2026-02-02.md`
- **What I learned:** interactive prompts must run in a real terminal; GitHub pushes require PAT/SSH (password won't work).
- **Feature added next:** auto-save run outputs to `product/sections/` (so results are reproducible and easy to review).

---

# AI Hedge Fund (Original Framework)

This is a proof of concept for an AI-powered hedge fund.  The goal of this project is to explore the use of AI to make trading decisions.  This project is for **educational** purposes only and is not intended for real trading or investment.

This system employs several agents working together:

1. Aswath Damodaran Agent - The Dean of Valuation, focuses on story, numbers, and disciplined valuation
2. Ben Graham Agent - The godfather of value investing, only buys hidden gems with a margin of safety
3. Bill Ackman Agent - An activist investor, takes bold positions and pushes for change
4. Cathie Wood Agent - The queen of growth investing, believes in the power of innovation and disruption
5. Charlie Munger Agent - Warren Buffett's partner, only buys wonderful businesses at fair prices
6. Michael Burry Agent - The Big Short contrarian who hunts for deep value
7. Mohnish Pabrai Agent - The Dhandho investor, who looks for doubles at low risk
8. Peter Lynch Agent - Practical investor who seeks "ten-baggers" in everyday businesses
9. Phil Fisher Agent - Meticulous growth investor who uses deep "scuttlebutt" research
10. Rakesh Jhunjhunwala Agent - The Big Bull of India
11. Stanley Druckenmiller Agent - Macro legend who hunts for asymmetric opportunities with growth potential
12. Warren Buffett Agent - The oracle of Omaha, seeks wonderful companies at a fair price
13. Valuation Agent - Calculates the intrinsic value of a stock and generates trading signals
14. Sentiment Agent - Analyzes market sentiment and generates trading signals
15. Fundamentals Agent - Analyzes fundamental data and generates trading signals
16. Technicals Agent - Analyzes technical indicators and generates trading signals
17. Risk Manager - Calculates risk metrics and sets position limits
18. Portfolio Manager - Makes final trading decisions and generates orders

[Screenshot of the original multi-agent CLI framework — see virattt/ai-hedge-fund]

Note: the system does not actually make any trades.

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions
- Past performance does not indicate future results

By using this software, you agree to use it solely for learning purposes.

## Table of Contents
- [How to Install](#how-to-install)
- [How to Run](#how-to-run)
  - [⌨️ Command Line Interface](#️-command-line-interface)
  - [🖥️ Web Application](#️-web-application)
- [License](#license)

## How to Install

Before you can run the AI Hedge Fund, you'll need to install it and set up your API keys. These steps are common to both the full-stack web application and command line interface.

### 1. Clone the Repository

```bash
git clone https://github.com/OAOWOuO/-AI-Financial-Advisor.git
cd -AI-Financial-Advisor
```

### 2. Set up API keys

Create a `.env` file for your API keys:
```bash
# Create .env file for your API keys (in the root directory)
cp .env.example .env
```

Open and edit the `.env` file to add your API keys:
```bash
# For running LLMs hosted by openai (gpt-4o, gpt-4o-mini, etc.)
OPENAI_API_KEY=your-openai-api-key

# For getting financial data to power the hedge fund
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

**Important**: You must set at least one LLM API key (e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`) for the hedge fund to work.

**Financial Data**: Data for AAPL, GOOGL, MSFT, NVDA, and TSLA is free and does not require an API key. For any other ticker, you will need to set the `FINANCIAL_DATASETS_API_KEY` in the .env file.

## How to Run

### ⌨️ Command Line Interface

You can run the AI Hedge Fund directly via terminal. This approach offers more granular control and is useful for automation, scripting, and integration purposes.

[Screenshot of the original multi-agent CLI framework — see virattt/ai-hedge-fund]

#### Quick Start

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

#### Run the AI Hedge Fund
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

You can also specify a `--ollama` flag to run the AI hedge fund using local LLMs.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama
```

You can optionally specify the start and end dates to make decisions over a specific time period.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

#### Run the Backtester
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

Note: The `--ollama`, `--start-date`, and `--end-date` flags work for the backtester, as well!

### 🖥️ Web Application

Run my custom Streamlit app (built from scratch — see [My Contributions](#-what-i-added-my-contributions) above):

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

Live: <https://oaowouo--ai-financial-advisor-streamlit-appapp-qwt5aa.streamlit.app/>

## License

This project is licensed under the MIT License - see the LICENSE file for details.
