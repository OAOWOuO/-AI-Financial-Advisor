import datetime
from typing import Dict, Optional

from fpdf import FPDF


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
            0, 5,
            f"AI Financial Advisor - {self._ticker} - {self._date_str}  |  Page {self.page_no()}/{{nb}}",
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
    pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(26, 58, 92)
    pdf.set_line_width(0.5)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + pdf.epw, pdf.get_y())
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _two_col_table(pdf: FPDF, rows: list) -> None:
    label_w = 65
    val_w = pdf.epw - label_w
    for i, (label, value) in enumerate(rows):
        fill_color = (245, 247, 250) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill_color)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(label_w, 6, label, fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(val_w, 6, value, fill=True, new_x="LMARGIN", new_y="NEXT")


def generate_pdf(data: Dict, valuation: Dict) -> bytes:
    """Generate a PDF financial analysis report. Returns raw PDF bytes."""
    ticker = data.get("ticker", "N/A")
    name = data.get("name", ticker)
    date_str = datetime.date.today().isoformat()

    pdf = _PDF(ticker, date_str)
    pdf.add_page()

    # -- Section 1: Header --
    half_w = pdf.epw / 2
    pdf.set_fill_color(26, 58, 92)
    pdf.set_text_color(255, 255, 255)

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(half_w, 10, ticker, fill=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(half_w, 10, "AI Financial Analysis Report", fill=True, align="R",
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(half_w, 7, name, fill=True)
    pdf.cell(half_w, 7, date_str, fill=True, align="R",
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    price = _fmt(data.get("price"), "{:.2f}", prefix="$")
    mcap = _market_cap_str(data.get("market_cap"))
    sector = data.get("sector") or "N/A"
    pe = _fmt(data.get("pe_ratio"), "{:.1f}", suffix="x")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, f"Price: {price}   |   Market Cap: {mcap}   |   Sector: {sector}   |   P/E: {pe}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    # ── Section 2: AI Investment Thesis ──────────────────────────────────────
    _section_heading(pdf, "AI Investment Thesis")
    thesis = (data.get("_orchestrator_thesis") or "").strip() or "Analysis not available."
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, thesis)

    # ── Section 3: Fundamental Metrics ───────────────────────────────────────
    _section_heading(pdf, "Fundamental Metrics")
    _two_col_table(pdf, [
        ("Current Price",   _fmt(data.get("price"), "{:.2f}", prefix="$")),
        ("Market Cap",      _market_cap_str(data.get("market_cap"))),
        ("P/E (TTM)",       _fmt(data.get("pe_ratio"), "{:.1f}", suffix="x")),
        ("Forward P/E",     _fmt(data.get("forward_pe"), "{:.1f}", suffix="x")),
        ("EPS (TTM)",       _fmt(data.get("eps"), "{:.2f}", prefix="$")),
        ("Revenue Growth",  _fmt(data.get("revenue_growth"), "{:+.1f}", suffix="%", scale=100)),
        ("Profit Margin",   _fmt(data.get("profit_margin"), "{:.1f}", suffix="%", scale=100)),
        ("Debt / Equity",   _fmt(data.get("debt_to_equity"), "{:.2f}", suffix="x")),
    ])

    # ── Section 4: Valuation Summary ─────────────────────────────────────────
    _section_heading(pdf, "Valuation Summary")
    _val = data.get("_valuation") or {}
    intrinsic = _val.get("intrinsic_value")
    ev_ebitda_v = _val.get("ev_ebitda")
    narrative = (_val.get("summary") or "").strip()
    cur_price = data.get("price")

    if intrinsic and cur_price:
        margin = (intrinsic - cur_price) / intrinsic * 100
        margin_str = f"{margin:+.1f}% ({'Undervalued' if margin > 0 else 'Overvalued'})"
    else:
        margin_str = "N/A"

    if intrinsic or ev_ebitda_v:
        _two_col_table(pdf, [
            ("DCF Intrinsic Value", _fmt(intrinsic, "{:.2f}", prefix="$") + " / share"),
            ("Current Price",       _fmt(cur_price, "{:.2f}", prefix="$")),
            ("Margin of Safety",    margin_str),
            ("EV / EBITDA",         _fmt(ev_ebitda_v, "{:.1f}", suffix="x")),
        ])
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Valuation agent data unavailable.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    if narrative:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 4, narrative)
        pdf.set_text_color(0, 0, 0)

    pe_val = (valuation or {}).get("pe_valuation")
    fpe_val = (valuation or {}).get("forward_pe_valuation")
    if pe_val or fpe_val:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "P/E-Based Price Targets:", new_x="LMARGIN", new_y="NEXT")
        target_rows = []
        if pe_val:
            target_rows.append((
                "P/E  Bear / Base / Bull",
                f"${pe_val['low']:.2f} / ${pe_val['mid']:.2f} / ${pe_val['high']:.2f}",
            ))
        if fpe_val:
            target_rows.append((
                "Fwd P/E  Bear / Base / Bull",
                f"${fpe_val['low']:.2f} / ${fpe_val['mid']:.2f} / ${fpe_val['high']:.2f}",
            ))
        _two_col_table(pdf, target_rows)
    else:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Insufficient data for P/E target calculation.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    # ── Section 5: Peer Comparison ────────────────────────────────────────────
    _section_heading(pdf, "Peer Comparison")
    peer_list = data.get("_peer_data") or []
    if not peer_list:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 5, "No peer data. Run analysis with an API key to populate peers.",
                 new_x="LMARGIN", new_y="NEXT")
    else:
        from sa_peers import build_peer_table
        peer_df = build_peer_table(ticker, data, peer_list)

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 4, "Source: Yahoo Finance · GPT-4o", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        col_names = list(peer_df.columns)
        col_widths = [20, 20, 16, 19, 22, 20, 22, 18, 18]

        # Header row
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_fill_color(26, 58, 92)
        pdf.set_text_color(255, 255, 255)
        for col, w in zip(col_names, col_widths):
            pdf.cell(w, 5, col, fill=True)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        # Data rows
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
                pdf.cell(w, 5, cell_text, fill=True)
            pdf.ln()

    # ── Section 6: Insider Activity ───────────────────────────────────────────
    _section_heading(pdf, "Insider Activity")
    insider_text = (data.get("_insider_signal_text") or "").strip() or "No insider data available."
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, insider_text)

    return bytes(pdf.output())
