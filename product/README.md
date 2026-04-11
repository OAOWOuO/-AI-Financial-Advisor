# product/

This folder contains planning artifacts, saved run outputs, and future feature designs for the AI Financial Advisor project.

---

## Contents

### `sections/run_2026-02-02.md` — Saved Multi-Agent CLI Output

Full terminal output from running the original virattt/ai-hedge-fund CLI on **AAPL, MSFT, NVDA** on February 2, 2026.

- **Run command:** `poetry run python src/main.py --tickers AAPL,MSFT,NVDA`
- **Purpose:** Demonstrates the 17-agent system working end-to-end with real tickers. Saved so the result is reproducible and reviewable without re-running the CLI.
- **What it contains:** Each agent's analysis (Buffett, Munger, Graham, etc.), risk manager position limits, and portfolio manager final trading decisions.

---

### `product-overview.md` — Future Feature: AI-Powered Options Wheel System

Design document for a future extension of this project: combining the existing AI agent stock-picking signals with systematic options income generation (the "wheel strategy").

**The idea in one sentence:** Use the 17 AI agents to identify high-conviction bullish stocks, then sell cash-secured puts on those picks to collect 1–3% monthly premium instead of buying shares outright.

**Key components designed:**
- AI → Options bridge: map agent BUY signals and confidence scores to put-selling opportunities
- Greeks-optimized strike selection: auto-select strikes at delta 0.20–0.30, accounting for IV percentile and 30–45 DTE sweet spot
- Wheel state machine: track position lifecycle (CASH → PUT_SOLD → ASSIGNED → CALL_SOLD → CALLED_AWAY)
- Income dashboard: premium collected, annualized yield, Greeks exposure, upcoming expirations

**Status:** Design only — not yet implemented.

---

### `product-roadmap.md` — Development Roadmap

Phased build plan for the Options Wheel System, including open questions on paper vs. live trading, agent subset selection, position sizing (fixed vs. Kelly criterion), and rolling rules.

---

## Context

The `product/` folder was used during development to:
1. Store evidence that the original framework runs correctly (saved CLI output)
2. Plan future extensions beyond the course deliverable scope
