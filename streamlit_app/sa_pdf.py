import datetime
from typing import Dict, Optional

from fpdf import FPDF

_UNICODE_MAP = {
    "—": "-",  # em dash
    "–": "-",  # en dash
    "‘": "'",  # left single quote
    "’": "'",  # right single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "•": "*",  # bullet
    "…": "...",  # ellipsis
    " ": " ",  # non-breaking space
    "·": "*",  # middle dot
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "−": "-",  # minus sign
    "≥": ">=",  # ≥
    "≤": "<=",  # ≤
    "×": "x",  # ×
    "°": " deg",  # °
    "®": "(R)",  # ®
    "™": "(TM)",  # ™
    "\\$": "$",  # escaped dollar (from Streamlit LaTeX prevention)
}


def _s(text: str) -> str:
    """Sanitize text to Latin-1 safe for core PDF fonts."""
    if not text:
        return ""
    for ch, rep in _UNICODE_MAP.items():
        text = text.replace(ch, rep)
    # Strip remaining markdown bold/italic markers
    text = text.replace("**", "").replace("__", "").replace("*", "").replace("_", " ")
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _s_bullets(text: str) -> str:
    """Sanitize and convert markdown list lines to plain bullet lines."""
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            lines.append("  * " + stripped[2:])
        else:
            lines.append(stripped)
    return _s("\n".join(lines))


class _PDF(FPDF):
    def __init__(self, ticker: str, date_str: str):
        super().__init__(orientation="P", unit="mm", format="Letter")
        self._ticker = ticker
        self._date_str = date_str
        self.set_margins(left=18, top=18, right=18)
        self.set_auto_page_break(auto=True, margin=18)
        self.alias_nb_pages()

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(
            0,
            5,
            f"AI Financial Advisor  |  {self._ticker}  |  {self._date_str}  |  Page {self.page_no()}/{{nb}}  |  Educational use only",
            align="C",
        )
        self.set_text_color(0, 0, 0)


def _fmt(val, fmt: str = "{:.1f}", suffix: str = "", prefix: str = "", scale: float = 1.0) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{prefix}{fmt.format(float(val) * scale)}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _market_cap_str(val) -> str:
    if not val:
        return "N/A"
    if val >= 1e12:
        return f"${val / 1e12:.2f}T"
    return f"${val / 1e9:.1f}B"


def _section_heading(pdf: FPDF, title: str) -> None:
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(26, 58, 92)
    pdf.cell(0, 6, _s(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(26, 58, 92)
    pdf.set_line_width(0.5)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + pdf.epw, pdf.get_y())
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _two_col_table(pdf: FPDF, rows: list, label_w: int = 65) -> None:
    val_w = pdf.epw - label_w
    for i, (label, value) in enumerate(rows):
        fill_color = (245, 247, 250) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill_color)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(label_w, 6, _s(str(label)), fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(val_w, 6, _s(str(value)), fill=True, new_x="LMARGIN", new_y="NEXT")


def _body_text(pdf: FPDF, text: str, size: int = 9) -> None:
    pdf.set_font("Helvetica", "", size)
    pdf.multi_cell(0, 5, _s(text))
    pdf.ln(1)


def _rating_color(action: str):
    m = {
        "STRONG BUY": (0, 180, 0),
        "BUY": (63, 185, 80),
        "HOLD": (210, 153, 34),
        "UNDERPERFORM": (248, 81, 73),
        "SELL": (220, 0, 0),
    }
    return m.get((action or "").upper(), (140, 140, 140))


def generate_pdf(
    data: Dict,
    valuation: Dict,
    tech_analysis: Optional[Dict] = None,
    fund_analysis: Optional[Dict] = None,
    forecasts: Optional[Dict] = None,
    recommendation: Optional[Dict] = None,
) -> bytes:
    """Generate a multi-page PDF financial analysis report. Returns raw PDF bytes."""
    ticker = data.get("ticker", "N/A")
    name = data.get("name", ticker)
    date_str = datetime.date.today().isoformat()

    pdf = _PDF(ticker, date_str)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 1 — OVERVIEW & INVESTMENT RATING
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # ── Header bar ────────────────────────────────────────────────────────
    half_w = pdf.epw / 2
    pdf.set_fill_color(26, 58, 92)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(half_w, 11, _s(ticker), fill=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(half_w, 11, "AI Financial Analysis Report", fill=True, align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(half_w, 7, _s(name), fill=True)
    pdf.cell(half_w, 7, date_str, fill=True, align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # ── Key stats bar ─────────────────────────────────────────────────────
    price = _fmt(data.get("price"), "{:.2f}", prefix="$")
    mcap = _market_cap_str(data.get("market_cap"))
    sector = _s(data.get("sector") or "N/A")
    pe = _fmt(data.get("pe_ratio"), "{:.1f}", suffix="x")
    beta = _fmt(data.get("beta"), "{:.2f}")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(
        0,
        5,
        f"Price: {price}   |   Market Cap: {mcap}   |   Sector: {sector}   |   P/E: {pe}   |   Beta: {beta}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    # ── Investment Rating card (if recommendation available) ───────────────
    if recommendation:
        action = _s(recommendation.get("action", "N/A"))
        rc = _rating_color(action)
        score = recommendation.get("combined_score", 50)
        target = recommendation.get("target_price")
        upside = recommendation.get("upside", 0)
        cur_price = data.get("price")
        upside_label = f"{abs(upside):.1f}% potential downside" if upside < 0 else f"{upside:.1f}% potential upside"

        # Rating box
        pdf.set_fill_color(245, 247, 250)
        pdf.set_draw_color(*rc)
        pdf.set_line_width(0.8)
        box_top = pdf.get_y()
        box_h = 26
        pdf.rect(pdf.l_margin, box_top, pdf.epw, box_h, style="DF")
        pdf.set_y(box_top + 3)

        third = pdf.epw / 3
        # Rating column
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(third, 5, "INVESTMENT RATING")
        pdf.cell(third, 5, "12-MONTH PRICE TARGET", align="C")
        pdf.cell(third, 5, "COMPOSITE SCORE (50=NEUTRAL)", align="R", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(*rc)
        pdf.cell(third, 12, action)
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(30, 30, 30)
        t_str = f"${target:.2f}" if target else "N/A"
        pdf.cell(third, 12, t_str, align="C")
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(third, 12, f"{score:.0f} / 100", align="R", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(80, 80, 80)
        trade = _s(recommendation.get("trade_decision", ""))
        cur_str = f"Current: ${cur_price:.2f}" if cur_price else ""
        pdf.cell(third, 5, trade)
        pdf.cell(third, 5, upside_label, align="C")
        pdf.cell(third, 5, cur_str, align="R", new_x="LMARGIN", new_y="NEXT")

        pdf.set_text_color(0, 0, 0)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(0.2)
        pdf.ln(2)

    # ── AI Investment Thesis ───────────────────────────────────────────────
    _section_heading(pdf, "AI Investment Thesis")
    thesis = (data.get("_orchestrator_thesis") or "").strip() or "Analysis not available."
    _body_text(pdf, thesis, size=9)

    # ── Fundamental Metrics ────────────────────────────────────────────────
    _section_heading(pdf, "Fundamental Metrics")
    _two_col_table(
        pdf,
        [
            ("Current Price", _fmt(data.get("price"), "{:.2f}", prefix="$")),
            ("Market Cap", _market_cap_str(data.get("market_cap"))),
            (
                "52-Week Range",
                f"{_fmt(data.get('low_52w'), '{:.2f}', prefix='$')} - {_fmt(data.get('high_52w'), '{:.2f}', prefix='$')}",
            ),
            ("P/E (TTM)", _fmt(data.get("pe_ratio"), "{:.1f}", suffix="x")),
            ("Forward P/E", _fmt(data.get("forward_pe"), "{:.1f}", suffix="x")),
            ("EPS (TTM)", _fmt(data.get("eps"), "{:.2f}", prefix="$")),
            ("Revenue (TTM)", _fmt(data.get("revenue"), "{:.1f}", prefix="$", suffix="B", scale=1e-9)),
            ("Revenue Growth", _fmt(data.get("revenue_growth"), "{:+.1f}", suffix="%", scale=100)),
            ("Gross Margin", _fmt(data.get("gross_margin"), "{:.1f}", suffix="%", scale=100)),
            ("Operating Margin", _fmt(data.get("op_margin"), "{:.1f}", suffix="%", scale=100)),
            ("Profit Margin", _fmt(data.get("profit_margin"), "{:.1f}", suffix="%", scale=100)),
            ("FCF (TTM)", _fmt(data.get("free_cash_flow"), "{:.1f}", prefix="$", suffix="B", scale=1e-9)),
            ("ROE", _fmt(data.get("roe"), "{:.1f}", suffix="%", scale=100)),
            ("Debt / Equity", _fmt(data.get("debt_to_equity"), "{:.2f}", suffix="x")),
            ("Current Ratio", _fmt(data.get("current_ratio"), "{:.2f}")),
            ("Beta (5Y)", _fmt(data.get("beta"), "{:.2f}")),
            ("Dividend Yield", _fmt(data.get("dividend_yield"), "{:.2f}", suffix="%", scale=100)),
        ],
    )

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 2 — TECHNICAL ANALYSIS & VALUATION
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # ── Technical Analysis ─────────────────────────────────────────────────
    _section_heading(pdf, "Technical Analysis")
    if tech_analysis:
        score_pct = tech_analysis.get("score_pct", 0)
        trend_label = "Bullish" if score_pct > 20 else "Bearish" if score_pct < -20 else "Neutral"
        _two_col_table(
            pdf,
            [
                ("Overall Technical Signal", f"{trend_label}  ({score_pct:+.0f} / 100)"),
            ],
            label_w=70,
        )
        pdf.ln(2)

        signals = tech_analysis.get("signals", [])
        if signals:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(26, 58, 92)
            pdf.set_text_color(255, 255, 255)
            col_w = [45, 30, pdf.epw - 75]
            for hdr, w in zip(["Indicator", "Signal", "Detail"], col_w):
                pdf.cell(w, 5, hdr, fill=True)
            pdf.ln()
            pdf.set_text_color(0, 0, 0)

            for i, sig in enumerate(signals[:12]):
                fill = (245, 247, 250) if i % 2 == 0 else (255, 255, 255)
                pdf.set_fill_color(*fill)
                s_label = _s(sig.get("indicator") or sig.get("metric", ""))
                s_signal = _s(sig.get("signal", ""))
                s_detail = _s(sig.get("detail", ""))
                sig_color_map = {
                    "BULLISH": (63, 185, 80),
                    "STRONG": (63, 185, 80),
                    "BEARISH": (248, 81, 73),
                    "WEAK": (248, 81, 73),
                    "NEUTRAL": (140, 140, 140),
                    "OVERBOUGHT": (210, 153, 34),
                    "OVERSOLD": (210, 153, 34),
                }
                sc = sig_color_map.get(s_signal.upper(), (80, 80, 80))
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(col_w[0], 5, s_label, fill=True)
                pdf.set_text_color(*sc)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(col_w[1], 5, s_signal, fill=True)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(col_w[2], 5, s_detail[:55], fill=True, new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Technical analysis data not available.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    # ── Fundamental Score Breakdown ────────────────────────────────────────
    _section_heading(pdf, "Fundamental Score Breakdown")
    if fund_analysis:
        disp = fund_analysis.get("display_score", 50)
        breakdown = fund_analysis.get("breakdown") or {}
        summary_rows = [("Overall Fundamental Score", f"{disp:.0f} / 100  (50 = neutral)")]
        for cat, sub in breakdown.items():
            if isinstance(sub, dict):
                s = sub.get("score", 0)
                m = sub.get("max", 0)
                summary_rows.append((f"  {cat.replace('_', ' ').title()}", f"{s:.1f} / {m:.1f}"))
        _two_col_table(pdf, summary_rows, label_w=80)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Fundamental analysis data not available.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    # ── Valuation Summary ──────────────────────────────────────────────────
    _section_heading(pdf, "Valuation Summary")
    _val = data.get("_valuation") or {}
    intrinsic = _val.get("intrinsic_value")
    ev_ebitda_v = _val.get("ev_ebitda")
    cur_price = data.get("price")

    if intrinsic and cur_price:
        margin = (intrinsic - cur_price) / intrinsic * 100
        margin_str = f"{margin:+.1f}%  ({'Undervalued' if margin > 0 else 'Overvalued'})"
    else:
        margin_str = "N/A"

    dcf_rows = []
    if intrinsic:
        dcf_rows += [
            ("DCF Intrinsic Value", _fmt(intrinsic, "{:.2f}", prefix="$") + " / share"),
            ("Current Price", _fmt(cur_price, "{:.2f}", prefix="$")),
            ("Margin of Safety", margin_str),
        ]
    if ev_ebitda_v:
        dcf_rows.append(("EV / EBITDA", _fmt(ev_ebitda_v, "{:.1f}", suffix="x")))

    pe_val = (valuation or {}).get("pe_valuation")
    fpe_val = (valuation or {}).get("forward_pe_valuation")
    if pe_val:
        l, m, h = pe_val.get("low"), pe_val.get("mid"), pe_val.get("high")
        if None not in (l, m, h):
            dcf_rows.append(("P/E  Bear / Base / Bull", f"${l:.2f} / ${m:.2f} / ${h:.2f}"))
    if fpe_val:
        l, m, h = fpe_val.get("low"), fpe_val.get("mid"), fpe_val.get("high")
        if None not in (l, m, h):
            dcf_rows.append(("Fwd P/E  Bear / Base / Bull", f"${l:.2f} / ${m:.2f} / ${h:.2f}"))

    if dcf_rows:
        _two_col_table(pdf, dcf_rows)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Valuation data unavailable (requires OpenAI API key for DCF).", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    narrative = (_val.get("summary") or "").strip()
    if narrative:
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 4, _s(narrative))
        pdf.set_text_color(0, 0, 0)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 3 — FORECAST & CATALYSTS
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # ── Price Target Scenarios ─────────────────────────────────────────────
    _section_heading(pdf, "Price Target Scenarios")
    if recommendation:
        t_low = recommendation.get("target_low", cur_price)
        t_mid = recommendation.get("target_price", cur_price)
        t_high = recommendation.get("target_high", cur_price)
        u_low = (t_low - cur_price) / cur_price * 100 if cur_price else 0
        u_mid = recommendation.get("upside", 0)
        u_high = (t_high - cur_price) / cur_price * 100 if cur_price else 0
        scen_rows = [
            ("Bear Case (12-month)", f"${t_low:.2f}  ({u_low:+.1f}%)"),
            ("Base Case (12-month)", f"${t_mid:.2f}  ({u_mid:+.1f}%)"),
            ("Bull Case (12-month)", f"${t_high:.2f}  ({u_high:+.1f}%)"),
        ]
        _two_col_table(pdf, scen_rows)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Price targets not available.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    # ── Expected Returns by Time Horizon ───────────────────────────────────
    _section_heading(pdf, "Expected Returns by Time Horizon")
    if forecasts:
        # Header
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(26, 58, 92)
        pdf.set_text_color(255, 255, 255)
        fc_cols = [
            ("Period", 28),
            ("Exp. Return", 28),
            ("80% Range", 42),
            ("Price Target", 30),
            ("Confidence", 28),
            ("Probability", 28),
        ]
        for hdr, w in fc_cols:
            pdf.cell(w, 5, hdr, fill=True)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        for i, (period, fc) in enumerate(forecasts.items()):
            fill = (245, 247, 250) if i % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*fill)
            pdf.set_font("Helvetica", "", 8)
            ret = fc.get("point_estimate", 0)
            ret_c = (63, 185, 80) if ret > 5 else (210, 153, 34) if ret > 0 else (248, 81, 73)
            rlo = fc.get("range_low", 0)
            rhi = fc.get("range_high", 0)
            pt = fc.get("price_target", 0)
            conf = fc.get("confidence", "")
            prob = fc.get("probability", "")

            pdf.cell(28, 5, _s(str(period)), fill=True)
            pdf.set_text_color(*ret_c)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(28, 5, f"{ret:+.1f}%", fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(42, 5, f"{rlo:+.1f}% to {rhi:+.1f}%", fill=True)
            pdf.cell(30, 5, f"${pt:.2f}", fill=True)
            pdf.cell(28, 5, _s(str(conf)), fill=True)
            pdf.cell(28, 5, _s(str(prob)), fill=True, new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Forecast data not available.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    # ── Catalysts & Risks ──────────────────────────────────────────────────
    _section_heading(pdf, "Catalysts & Key Risks")
    left_w = (pdf.epw - 6) / 2
    right_w = pdf.epw - left_w - 6

    if recommendation:
        drivers = recommendation.get("bullish_drivers") or []
        risks = recommendation.get("bearish_risks") or []

        # Sub-headers
        sub_y = pdf.get_y()
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(63, 185, 80)
        pdf.cell(left_w, 5, "Bullish Catalysts")
        pdf.cell(6, 5, "")
        pdf.set_text_color(248, 81, 73)
        pdf.cell(right_w, 5, "Key Risks", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        max_items = max(len(drivers), len(risks), 1)
        for idx in range(max_items):
            fill = (245, 247, 250) if idx % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*fill)
            pdf.set_font("Helvetica", "", 8)
            d_text = _s("+ " + drivers[idx]) if idx < len(drivers) else ""
            r_text = _s("- " + risks[idx]) if idx < len(risks) else ""
            # Use multi_cell trick: set position manually
            row_y = pdf.get_y()
            pdf.set_xy(pdf.l_margin, row_y)
            pdf.multi_cell(left_w, 5, d_text, fill=True)
            end_y_left = pdf.get_y()
            pdf.set_xy(pdf.l_margin + left_w + 6, row_y)
            pdf.multi_cell(right_w, 5, r_text, fill=True)
            end_y_right = pdf.get_y()
            pdf.set_y(max(end_y_left, end_y_right))
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Catalyst data not available.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    # ── Investment Rationale ───────────────────────────────────────────────
    if recommendation and recommendation.get("rationale"):
        _section_heading(pdf, "Investment Rationale")
        rationale_clean = recommendation["rationale"].replace("\\$", "$").replace("- ", "  * ")
        _body_text(pdf, rationale_clean, size=8)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 4 — PEER COMPARISON & INSIDER ACTIVITY
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # ── Peer Comparison ────────────────────────────────────────────────────
    _section_heading(pdf, "Peer Comparison")
    peer_list = data.get("_peer_data") or []
    if not peer_list:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, "No peer data. Run analysis to populate peers.", new_x="LMARGIN", new_y="NEXT")
    else:
        from sa_peers import build_peer_table

        peer_df = build_peer_table(ticker, data, peer_list)

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(80, 80, 80)
        peer_src = "Yahoo Finance"
        if data.get("_peer_tickers"):
            peer_src += " · GPT-4o" if len(peer_list) > 3 else " · built-in"
        pdf.cell(0, 4, f"Source: {peer_src}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        col_names = list(peer_df.columns)
        col_widths = [20, 20, 16, 19, 22, 20, 22, 18, 18]

        pdf.set_font("Helvetica", "B", 7)
        pdf.set_fill_color(26, 58, 92)
        pdf.set_text_color(255, 255, 255)
        for col, w in zip(col_names, col_widths):
            pdf.cell(w, 5, _s(col), fill=True)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        for i, (_, row) in enumerate(peer_df.iterrows()):
            is_main = str(row["Ticker"]) == str(ticker)
            if is_main:
                pdf.set_fill_color(221, 238, 255)
            elif i % 2 == 0:
                pdf.set_fill_color(245, 247, 250)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "B" if is_main else "", 7)
            for col, w in zip(col_names, col_widths):
                val = row[col]
                if val is None:
                    cell_text = "N/A"
                elif col == "Ticker":
                    cell_text = str(val)
                else:
                    try:
                        cell_text = f"{float(val):.1f}"
                    except (TypeError, ValueError):
                        cell_text = str(val)
                pdf.cell(w, 5, _s(cell_text), fill=True)
            pdf.ln()

    # ── Insider Activity ───────────────────────────────────────────────────
    _section_heading(pdf, "Insider Activity")
    insider_text = (data.get("_insider_signal_text") or "").strip() or "No insider data available."
    _body_text(pdf, insider_text, size=9)

    # ── Trade Invalidation ─────────────────────────────────────────────────
    if recommendation and recommendation.get("invalidation"):
        _section_heading(pdf, "Trade Invalidation Criteria")
        inv = recommendation["invalidation"].replace("\\$", "$")
        _body_text(pdf, inv, size=9)

    # ── Disclaimer ─────────────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_draw_color(180, 180, 180)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + pdf.epw, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(120, 120, 120)
    disclaimer = (
        "DISCLAIMER: This report is generated by an AI system for educational and research purposes only. "
        "It does not constitute professional financial, investment, legal, or tax advice and should not be "
        "used as the sole basis for any investment decision. All analysis, scores, valuations, and forecasts "
        "are algorithmically generated and may contain errors. Market data is sourced from Yahoo Finance "
        "(15-20 min delayed), SEC EDGAR, and public web searches. Past performance is not indicative of "
        "future results. Always consult a licensed financial professional before making investment decisions."
    )
    pdf.multi_cell(0, 4, disclaimer)
    pdf.set_text_color(0, 0, 0)

    return bytes(pdf.output())
