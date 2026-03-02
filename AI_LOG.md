# AI Collaboration Log

**Student:** YuanTeng Fan | **Course:** MGMT 690, Purdue University | **Spring 2026**

This log documents how AI tools (Claude Code via Anthropic API) were used to build this project —
what was generated, what was reviewed, what was modified, and what decisions the student made.

---

## How AI Was Used

This project used **Claude Code** (Claude Sonnet 4.6) as a coding assistant throughout development.
The workflow was:
1. Student defines the requirement or problem
2. Claude generates code or proposes a solution
3. Student reviews, tests, and decides whether to accept, modify, or reject
4. Student directs the next step

> The student operated the AI — the AI did not operate independently.

---

## Session Log

### Project 1 (Feb 2, 2026) — Initial Setup
**Student action:** Cloned virattt/ai-hedge-fund, configured `.env`, ran CLI on AAPL/MSFT/NVDA.
**AI role:** None for initial setup. Student ran CLI manually.
**Output saved:** `product/sections/run_2026-02-02.md`

---

### Financial Planner — Architecture Design
**Student prompt (paraphrased):** "Build an AI financial planning assistant with a rules engine, RAG, and LLM narrative. It should never let the LLM make threshold decisions."

**Key design decision by student:** Enforce strict separation — deterministic math in `fp_calculators.py`, threshold rules in `fp_rules.py`, LLM only writes narrative in `fp_report.py`. Student explicitly required this because pure-LLM financial advice is unreliable.

**AI generated:**
- `fp_schemas.py` — data models (`ClientProfile`, `PlanningIssue`, `Recommendation`, `QuantCheck`, `ScenarioProjection`, `PlanningReport`)
- `fp_calculators.py` — all financial math functions (DTI, emergency fund months, FV projection, SWR corpus, net worth benchmark)
- `fp_rules.py` — `RulesEngine` with 8 check categories driven by `data/rule_configs/planning_rules.json`
- `fp_scenarios.py` — Conservative / Balanced / Aggressive retirement projection engine
- `fp_report.py` — `build_quant_checks`, `build_recommendations`, `generate_report`, `_parse_llm_sections`
- `fp_retriever.py` — in-memory NumPy cosine-similarity RAG (student chose this over ChromaDB for Streamlit Cloud statelessness)
- `fp_case_retriever.py` — case-based reasoning retriever with 12 built-in reference cases
- `financial_planner.py` — 6-tab Streamlit UI

**Student modifications:**
- Switched RAG from ChromaDB to in-memory NumPy after testing on Streamlit Cloud — ChromaDB required persistent disk storage that cloud deployment doesn't support
- Added explainability tab after reviewing the initial UI and finding it lacked transparency
- Tuned all scoring thresholds (emergency fund: 3/6 months; DTI: 36%/43%; savings rate: 10%/15%) after reviewing against CFP standards
- Added data source attribution display after reviewing the initial report output

---

### Stock Analyzer
**Student prompt (paraphrased):** "Build an institutional-grade stock analyzer with CFA-style technical and fundamental analysis."

**AI generated:** `stock_analyzer.py` — RSI, MACD, Bollinger Bands, ADX technical indicators; P/E, DCF, analyst consensus fundamental scoring; BUY/HOLD/SELL recommendation engine.

**Student review and modifications:**
- Reviewed all scoring thresholds and adjusted fundamental score weights
- Identified that dollar amounts in `st.markdown()` were rendering as LaTeX (green math font) — directed AI to fix all instances by escaping `$` → `\\$`
- Approved final output format after testing with real tickers (AAPL, MSFT, NVDA)

---

### Portfolio Allocator
**Student prompt (paraphrased):** "Build a portfolio optimization tool with position sizing, risk metrics, and rebalancing."

**AI generated:** `portfolio_allocator.py` — multi-stock signal analysis, Sharpe ratio, Beta, VaR, S&P 500 benchmark comparison, dividend tracking.

---

### CI Pipeline
**Student prompt:** "Set up GitHub Actions CI with flake8 lint and pytest."

**AI generated:** `.github/workflows/ci.yml`, `.flake8` config.

**Student decision:** After seeing CI fail due to lint errors in the existing `src/` directory (original repo code), student directed AI to scope lint only to `streamlit_app/` and `scripts/` — the student's own code. Left `src/` untouched to avoid modifying the original framework.

---

### Unit Tests (147 tests)
**Student requirement:** "Add unit tests for all pure functions — no LLM, no Streamlit, no API required."

**AI generated:**
- `tests/test_fp_calculators.py` — 51 tests
- `tests/test_fp_rules.py` — 25 tests
- `tests/test_fp_schemas.py` — 26 tests
- `tests/test_fp_scenarios.py` — 13 tests
- `tests/test_fp_report.py` — 21 tests
- `tests/test_build_index.py` — 11 tests

**Student review:** Ran all tests locally before committing. Verified edge cases (e.g., emergency fund with 1.5 months coverage, DTI > 43%, negative cash flow) matched expected behavior.

---

### Bug Fixes and Code Quality

| Issue | Found by | Student action |
|-------|----------|----------------|
| `$` in `st.markdown()` renders as LaTeX (green math font) | Student — noticed during live demo | Directed AI to fix all affected lines |
| Dead `_load_rules()` function in `fp_rules.py` and `fp_scenarios.py` | AI audit | Student approved removal |
| Misleading default `effective_tax_rate=0.25` in `calc_monthly_cash_flow` | AI audit | Student approved removal — makes parameter required |
| `import re` inside function body in `fp_report.py` | AI audit | Student approved moving to module level |
| 14 bare `except:` in `portfolio_allocator.py` | AI audit | Student approved changing to `except Exception:` |
| `\$` SyntaxWarning in `stock_analyzer.py` (Python 3.14 will break) | AI audit | Student approved changing to `\\$` |

---

## What the Student Decided (Not AI)

- **Architecture**: Hybrid rules + LLM design — student's requirement that LLM never makes threshold decisions
- **Tech stack**: yfinance + OpenAI + NumPy RAG — student chose these after comparing alternatives
- **Scope of lint**: Only student-written code (`streamlit_app/`, `scripts/`) — original repo code excluded
- **Test strategy**: Pure functions only, no mocking of Streamlit or LLM — student's requirement for reliable CI
- **UI/UX**: 6-tab layout, dark theme, card navigation — student reviewed and directed all UI decisions
- **Threshold values**: All financial thresholds (DTI, emergency fund, savings rate, net worth benchmark) reviewed against CFP standards by student

---

## AI Tools Used

| Tool | Purpose |
|------|---------|
| Claude Code (claude-sonnet-4-6) | Primary coding assistant — architecture, implementation, debugging, code review |
| OpenAI GPT-4o-mini | Runtime LLM inside the app — financial plan narrative generation |
| OpenAI text-embedding-3-small | Runtime embeddings — RAG document ingestion |

---

## Validation Summary

All student-written pure functions are covered by automated tests:

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
