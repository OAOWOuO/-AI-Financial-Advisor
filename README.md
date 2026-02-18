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
| `stock_analyzer.py` | Individual stock deep-dive: CFA-style technical analysis (RSI, MACD, Bollinger Bands, ADX), fundamental scoring (valuation / profitability / growth / financial health), multi-model valuation (P/E, DCF, analyst consensus), BUY / HOLD / SELL recommendation |
| `case_qa.py` | RAG-powered Case Q&A chat tab: answers questions using **only** documents in `data/raw/`, shows citations (file + page + chunk ID), refuses unsupported questions explicitly |

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
pip install -r streamlit_app/requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add OPENAI_API_KEY=...

# 3. (Optional) Build Case Q&A index
# Drop PDFs into data/raw/ then:
python scripts/build_index.py

# 4. Launch
streamlit run streamlit_app/app.py
```

Live deployment: <https://ai-hedge-fund-project1-ezfbsk5cgwj62swdmzwe38.streamlit.app/>

> This project builds upon [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund).
> Original multi-agent framework by [@virattt](https://github.com/virattt).

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
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Live: <https://ai-hedge-fund-project1-ezfbsk5cgwj62swdmzwe38.streamlit.app/>

## License

This project is licensed under the MIT License - see the LICENSE file for details.
