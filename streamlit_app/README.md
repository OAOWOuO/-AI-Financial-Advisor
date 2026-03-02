# AI Financial Advisor — Streamlit Dashboard

A custom Streamlit web application built on top of [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund).
The original repo provides the multi-agent CLI framework (LangGraph + Financial Datasets API).
Everything in `streamlit_app/` was written entirely from scratch and does not exist in the original repo.

**Live app:** https://oaowouo--ai-financial-advisor-streamlit-appapp-qwt5aa.streamlit.app/

---

## Tools

| Tool | Description |
|---|---|
| **Portfolio Allocator** | Multi-stock signal analysis, position sizing, risk metrics (Sharpe, Beta, VaR), S&P 500 benchmark comparison, dividend tracking, rebalancing |
| **Stock Analyzer** | CFA-style technical analysis (RSI, MACD, Bollinger Bands, ADX), fundamental scoring (valuation / profitability / growth / financial health), multi-model valuation (P/E, DCF, analyst consensus), BUY / HOLD / SELL recommendation |
| **Financial Planner** | Hybrid rules engine + RAG + LLM narrative: 6-tab UI — client profile input, document library, gap analysis (emergency fund / DTI / cash flow / insurance / retirement), Conservative/Balanced/Aggressive scenario projections, prioritized recommendation report with source citations |

---

## Data Sources & Methodology Credits

### Market Data
- **Yahoo Finance** via [yfinance](https://github.com/ranaroussi/yfinance) — real-time quotes, historical prices, fundamentals, SEC filings (15–20 min delayed)

### AI / LLM
- **OpenAI API** (GPT-4o, GPT-4o-mini, text-embedding-3-small) — analyst narrative, report generation, RAG embeddings

### Financial Planning Methodology
- **Bengen (1994)** — 4% Safe Withdrawal Rate rule. *Bengen, W.P. (1994). Determining Withdrawal Rates Using Historical Data. Journal of Financial Planning.*
- **Stanley & Danko (1996)** — Net worth benchmarks by age. *Stanley, T.J. & Danko, W.D. (1996). The Millionaire Next Door.*
- **CFPB** — Debt-to-Income (DTI) ratio standards (≤36% recommended, ≤43% qualified mortgage limit). Consumer Financial Protection Bureau.
- **SSA** — Social Security replacement rate estimates based on average SSA replacement rates by income tier (NOT the AIME/PIA formula). Social Security Administration, ssa.gov.
- **IRS** — 401(k) contribution limits: $23,000 standard / $30,500 catch-up (age ≥ 50) for 2024. IRS Publication 560.
- **Roth IRA income limits** — 2024 phase-out: $146,000–$161,000 (single) / $230,000–$240,000 (married filing jointly). IRS Publication 590-A.

### Vector Store
- **ChromaDB** — local vector database for RAG document retrieval

---

## Quick Start (Local)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API key:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and add: OPENAI_API_KEY = "sk-..."
```

3. Run:
```bash
streamlit run app.py
```

---

## Disclaimer

This application is for **educational purposes only**.
It does not constitute legal, tax, or investment advice.
Always consult a licensed CFP, CPA, or attorney before making financial decisions.
Market data is provided by Yahoo Finance and is not guaranteed to be accurate or complete.

---

> Built for **MGMT 690 — Mastering AI for Finance**, Purdue University (Spring 2026).
> Base framework: [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) by [@virattt](https://github.com/virattt).
