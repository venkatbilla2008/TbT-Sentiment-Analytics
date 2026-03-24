"""
tbt_ui.py
=========
Reusable UI component renderers for the TbT Sentiment Analytics Streamlit app.

Each function either:
- Renders HTML/CSS via ``st.markdown(html, unsafe_allow_html=True)``, or
- Returns an HTML string for the caller to embed.

Keeping rendering helpers here means:
- ``tbt_app.py`` reads like a clean routing layer.
- CSS strings are colocated with the components that use them.
- Helpers can be unit-tested without a live Streamlit session.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

PHASE_ICONS: Dict[str, str] = {"start": "🚀", "middle": "🔄", "end": "🏁"}


def _score_color(score: float) -> str:
    """Map a compound score to a CSS colour string."""
    if score >= 0.1:
        return "#2ecc71"
    if score <= -0.1:
        return "#e74c3c"
    return "#f39c12"


def _badge_html(label: str) -> str:
    """Return a coloured HTML badge for a sentiment label."""
    if label == "positive":
        return '<span class="badge-csat">✅ Positive</span>'
    if label == "negative":
        return '<span class="badge-dsat">⛔ Negative</span>'
    return '<span class="badge-neutral">➖ Neutral</span>'


def _score_bar_html(score: float) -> str:
    """Return an HTML progress-bar + score label for a compound score."""
    pct   = int((score + 1) / 2 * 100)
    color = _score_color(score)
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div>'
        f'<span style="color:{color};font-size:.8rem;font-weight:600">{score:+.3f}</span>'
        f'</div>'
    )


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


# ---------------------------------------------------------------------------
# Global CSS (injected once at app start)
# ---------------------------------------------------------------------------

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---- Header ---- */
.app-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2.4rem 2rem 1.8rem;
    margin-bottom: 1.8rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.app-header h1 { color:#fff; font-size:2.1rem; font-weight:700; margin:0 0 .4rem; letter-spacing:-.5px; }
.app-header p  { color:#bbb; font-size:1rem; margin:0; }

/* ---- Metric cards ---- */
.metric-grid { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.4rem; }
.metric-card {
    flex:1; min-width:130px;
    background: linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
    border-radius:14px; padding:1.1rem 1.2rem;
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0 4px 20px rgba(0,0,0,0.25);
    text-align:center;
}
.metric-card .m-label { color:#888; font-size:.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:.3rem; }
.metric-card .m-value { color:#fff; font-size:1.9rem; font-weight:700; line-height:1.1; }
.metric-card .m-sub   { color:#aaa; font-size:.8rem;  margin-top:.25rem; }

/* ---- Phase table ---- */
.phase-table { width:100%; border-collapse:collapse; font-size:.88rem; }
.phase-table th { background:#1e1e3f; color:#ccc; padding:.6rem .9rem; text-align:left; font-weight:600; }
.phase-table td { padding:.55rem .9rem; border-bottom:1px solid rgba(255,255,255,0.05); color:#eee; }
.phase-table tr:hover td { background:rgba(255,255,255,0.03); }

/* ---- Sentiment badges ---- */
.badge-csat    { background:#1a6640; color:#7fff9e; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:600; }
.badge-dsat    { background:#6b1a1a; color:#ff9e9e; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:600; }
.badge-neutral { background:#3a3a5c; color:#c7c7ff; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:600; }

/* ---- Turn viewer cards ---- */
.turn-card {
    border-radius:10px; padding:.8rem 1rem; margin-bottom:.5rem;
    border-left:4px solid transparent;
}
.turn-customer { background:rgba(255,107,107,0.08); border-color:#ff6b6b; }
.turn-agent    { background:rgba(78,205,196,0.08);  border-color:#4ecdc4; }
.turn-header   { font-size:.75rem; color:#888; margin-bottom:.2rem; }
.turn-text     { font-size:.93rem; color:#eee; }
.turn-meta     { font-size:.72rem; color:#666; margin-top:.3rem; }

/* ---- Recommendation cards ---- */
.rec-card {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:10px; padding:.75rem 1rem; margin-bottom:.5rem;
    font-size:.88rem; color:#ddd;
}

/* ---- Score bar ---- */
.score-bar-wrap  { display:flex; align-items:center; gap:.5rem; }
.score-bar-track { flex:1; height:6px; background:#333; border-radius:999px; overflow:hidden; }
.score-bar-fill  { height:100%; border-radius:999px; transition:width .4s; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d0d1a 0%,#12122b 100%);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stFileUploader label { color:#ccc !important; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] {
    background:rgba(255,255,255,0.04); border-radius:8px 8px 0 0;
    color:#aaa; font-size:.87rem; padding:.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#6c63ff22,#4ecdc422) !important;
    color:#fff !important;
    border-bottom:2px solid #6c63ff;
}
</style>
"""


def inject_css() -> None:
    """Inject global CSS once at app start. Call from top of main()."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

def render_header() -> None:
    """Render the gradient app header banner."""
    st.markdown(
        """
        <div class="app-header">
            <h1>🎭 Domain Agnostic — TbT Sentiment Analytics</h1>
            <p>Granular Turn-by-Turn Analysis &nbsp;·&nbsp; Start → Middle → End &nbsp;·&nbsp; CSAT / DSAT per Phase</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# KPI metrics row
# ---------------------------------------------------------------------------

def render_kpi_row(insights: Dict[str, Any]) -> None:
    """Render the top-level KPI metric card strip."""
    cs  = insights["customer_satisfaction"]
    ap  = insights["agent_performance"]
    cp  = insights["conversation_patterns"]

    def card(label: str, value: str, sub: str = "") -> str:
        return (
            f'<div class="metric-card">'
            f'<div class="m-label">{label}</div>'
            f'<div class="m-value">{value}</div>'
            f'<div class="m-sub">{sub}</div>'
            f'</div>'
        )

    overall = insights["overall_sentiment"]["average"]
    oc = _score_color(overall)

    esc_color = (
        "#e74c3c" if cs["escalation_rate"] > 0.15
        else "#f39c12" if cs["escalation_rate"] > 0.10
        else "#2ecc71"
    )
    res_color = (
        "#2ecc71" if cs["resolution_rate"] > 0.6
        else "#f39c12" if cs["resolution_rate"] > 0.4
        else "#e74c3c"
    )

    html = '<div class="metric-grid">'
    html += card("Conversations",   f"{insights['total_conversations']:,}")
    html += card("Total Turns",     f"{insights['total_turns']:,}",
                 f"avg {insights['avg_turns_per_conversation']:.1f}/conv")
    html += card("Overall Sentiment",
                 f'<span style="color:{oc}">{overall:+.3f}</span>')
    html += card("Customer Avg",
                 f'<span style="color:{_score_color(cs["average_sentiment"])}">'
                 f'{cs["average_sentiment"]:+.3f}</span>')
    html += card("Agent Avg",
                 f'<span style="color:{_score_color(ap["average_sentiment"])}">'
                 f'{ap["average_sentiment"]:+.3f}</span>')
    html += card("Escalation Rate",
                 f'<span style="color:{esc_color}">{_fmt_pct(cs["escalation_rate"])}</span>')
    html += card("Resolution Rate",
                 f'<span style="color:{res_color}">{_fmt_pct(cs["resolution_rate"])}</span>')
    html += card("Sentiment Trend",
                 f'<span style="color:{_score_color(cp["sentiment_improvement"])}">'
                 f'{cp["sentiment_improvement"]:+.3f}</span>',
                 "end − start")
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Phase table
# ---------------------------------------------------------------------------

def render_phase_table(insights: Dict[str, Any]) -> None:
    """Render the CSAT / DSAT breakdown table for Start / Middle / End phases."""
    pcd = insights.get("phase_csat_dsat", {})
    cp  = insights.get("conversation_patterns", {})

    rows_html = ""
    for pn in ["start", "middle", "end"]:
        p    = pcd.get(pn, {})
        icon = PHASE_ICONS[pn]
        csat = p.get("csat_pct",     0)
        dsat = p.get("dsat_pct",     0)
        cnt  = p.get("count",        0)
        avg  = cp.get(f"avg_sentiment_{pn}", 0)
        ind  = "✅" if csat >= 0.6 else "⚠️" if csat >= 0.4 else "🔴"
        rows_html += (
            f"<tr>"
            f"<td>{icon} <strong>{pn.capitalize()}</strong></td>"
            f"<td>{ind}</td>"
            f"<td>{cnt:,}</td>"
            f"<td><span class='badge-csat'>{_fmt_pct(csat)} CSAT</span></td>"
            f"<td><span class='badge-dsat'>{_fmt_pct(dsat)} DSAT</span></td>"
            f"<td>{_score_bar_html(avg)}</td>"
            f"</tr>"
        )

    html = (
        "<table class='phase-table'><thead><tr>"
        "<th>Phase</th><th>Status</th><th>Customer Turns</th>"
        "<th>CSAT</th><th>DSAT</th><th>Avg Score</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Turn-by-turn viewer
# ---------------------------------------------------------------------------

def render_turn_viewer(df: pd.DataFrame, conv_id: str) -> None:
    """
    Render a conversation as a series of colour-coded turn cards.

    Customer turns use a warm red accent; agent turns use a teal accent.
    Each card shows: speaker, timestamp, phase, message, sentiment badge + bar.
    """
    sub = df[df["conversation_id"] == conv_id].sort_values("turn_sequence")
    if sub.empty:
        st.info("No turns found for this conversation.")
        return

    for _, r in sub.iterrows():
        spk   = str(r["speaker"]).upper()
        css   = "turn-customer" if spk == "CUSTOMER" else "turn-agent"
        icon  = "👤" if spk == "CUSTOMER" else "🎧"
        ts    = (f" · {r['timestamp']}"
                 if r.get("timestamp") and str(r["timestamp"]) not in ("nan", "None", "")
                 else "")
        phase_icon = PHASE_ICONS.get(str(r.get("phase", "middle")), "🔄")
        score = float(r["compound"])
        lbl   = str(r.get("sentiment_label", "neutral"))

        html = (
            f'<div class="turn-card {css}">'
            f'<div class="turn-header">'
            f'{icon} {spk}{ts} &nbsp; {phase_icon} {str(r.get("phase","")).capitalize()}'
            f" &nbsp; Turn #{int(r['turn_sequence'])}"
            f'</div>'
            f'<div class="turn-text">{r["message"]}</div>'
            f'<div class="turn-meta">'
            f'{_badge_html(lbl)} &nbsp; {_score_bar_html(score)}'
            f' &nbsp; Confidence: {float(r.get("sentiment_confidence", 0)):.0%}'
            f'</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def render_recommendations(insights: Dict[str, Any]) -> None:
    """Render recommendation strings as styled cards."""
    for rec in insights.get("recommendations", []):
        st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Welcome / landing screen
# ---------------------------------------------------------------------------

def render_landing() -> None:
    """Render the welcome screen shown before any data is loaded."""
    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown(
            """
### Welcome to TbT Sentiment Analytics

This app performs **granular, turn-by-turn sentiment analysis** on conversation
transcripts and customer feedback across six domain formats.

**Supported formats:**

| Icon | Domain | Data Type |
|------|--------|-----------|
| 🎵 | Media / Entertainment A | ISO-timestamp transcripts |
| 🎬 | Media / Entertainment B | Bracket `[HH:MM:SS]` transcripts |
| 🏥 | Healthcare A | Call-centre transcripts `[MM:SS]` |
| 🩼 | Healthcare B | Chat & SMS logs |
| 🚗 | Transportation | Customer verbatim / feedback |
| 🏨 | Travel | Guest feedback |

**Getting started:**
1. Select your domain in the sidebar
2. Upload a CSV / Excel file to begin analysis
"""
        )
    with col_r:
        st.markdown("#### 🔍 What you'll see")
        for item in [
            "📊 KPI dashboard (conversations, turns, sentiment)",
            "📈 CSAT / DSAT breakdown per phase (Start → Middle → End)",
            "🔄 Interactive turn-by-turn flow chart + momentum chart",
            "🗣️ Per-turn detail viewer with sentiment badges",
            "💡 Automated business recommendations",
            "⬇️ One-click export (Excel + CSV + JSON ZIP)",
        ]:
            st.markdown(f"- {item}")
