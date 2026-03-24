"""
tbt_app.py
==========
Streamlit entrypoint for the Domain Agnostic Turn-by-Turn Sentiment Analytics app.

Run with:
    streamlit run tbt_app.py

Project structure
-----------------
tbt_app.py      ← you are here  (UI routing & session state)
tbt_engine.py   ← parsing, scoring, analytics pipeline
tbt_charts.py   ← Plotly chart factories
tbt_ui.py       ← reusable HTML/CSS component renderers
tbt_export.py   ← Excel / ZIP export helpers
requirements.txt
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

# Internal modules
from tbt_charts import (
    chart_conversation_heatmap,
    chart_escalation_resolution,
    chart_phase_comparison,
    chart_sentiment_dist,
    chart_sentiment_momentum,
    chart_sentiment_progression,
    chart_speaker_box,
    chart_speaker_phase_heatmap,
    chart_tbt_flow,
)
from tbt_engine import FORMAT_LABELS, run_pipeline
from tbt_export import to_excel, to_zip
from tbt_ui import (
    inject_css,
    render_header,
    render_kpi_row,
    render_landing,
    render_phase_table,
    render_recommendations,
    render_turn_viewer,
)

# ============================================================================
# Page config — MUST be the first Streamlit call in the script
# ============================================================================

st.set_page_config(
    page_title="TbT Sentiment Analytics",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


# ============================================================================
# Session-state helpers
# ============================================================================

def _ss_get(key: str, default=None):
    """Thin wrapper around st.session_state.get for readability."""
    return st.session_state.get(key, default)


def _ss_set(key: str, value) -> None:
    st.session_state[key] = value


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar() -> tuple[str, Optional[object]]:
    """
    Render the sidebar and return (dataset_type, uploaded_file).

    ``uploaded_file`` is None if no file has been uploaded yet.
    """
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:.6rem 0 1.2rem">'
            '<span style="font-size:2rem">🎭</span>'
            '<div style="color:#fff;font-weight:700;font-size:1.05rem;margin-top:.3rem">TbT Analytics</div>'
            '<div style="color:#888;font-size:.78rem">Turn-by-Turn Sentiment</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        # ---- Domain / format selector ----
        st.markdown("### ⚙️ Configuration")
        domain_keys   = list(FORMAT_LABELS.keys())
        domain_labels = [FORMAT_LABELS[k] for k in domain_keys]
        sel_idx = st.selectbox(
            "Domain / Format",
            options=range(len(domain_keys)),
            format_func=lambda i: domain_labels[i],
            index=6,   # defaults to "auto"
            help=(
                "Select the transcript format that matches your data, "
                "or leave as Auto-Detect."
            ),
        )
        dataset_type = domain_keys[sel_idx]

        # ---- File uploader ----
        st.markdown("---")
        st.markdown("### 📂 Upload Data")
        uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            help="Upload a file containing conversation transcripts or customer feedback.",
        )

        st.markdown("---")
        st.markdown(
            '<div style="color:#555;font-size:.72rem;text-align:center">'
            "Domain Agnostic · TbT Granular Sentiment v2.1"
            "</div>",
            unsafe_allow_html=True,
        )

    return dataset_type, uploaded


# ============================================================================
# Data loading
# ============================================================================

def load_data(
    uploaded,
    dataset_type: str,
) -> tuple[Optional[pd.DataFrame], str, str]:
    """
    Load data from the uploaded file.

    Returns (df_raw, source_label, dataset_type), or
    (None, "", dataset_type) if no file has been uploaded.
    """
    if uploaded is not None:
        try:
            df_raw = (
                pd.read_csv(uploaded)
                if uploaded.name.endswith(".csv")
                else pd.read_excel(uploaded)
            )
            return df_raw, f"📂 {uploaded.name}", dataset_type
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            return None, "", dataset_type

    return None, "", dataset_type


# ============================================================================
# Pipeline (cached per data fingerprint)
# ============================================================================

def _run_cached_pipeline(df_raw: pd.DataFrame, dataset_type: str):
    """
    Run the analysis pipeline and cache results in st.session_state.

    The cache key is derived from the raw data bytes + dataset_type so that
    re-uploading the same file does not re-run the (potentially slow) pipeline,
    but changing the domain selector does.
    """
    cache_key = f"results_{hash(df_raw.values.tobytes())}_{dataset_type}"
    if cache_key not in st.session_state:
        with st.spinner("🔄 Running TbT analysis pipeline…"):
            try:
                result = run_pipeline(df_raw, dataset_type)
                _ss_set(cache_key, result)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)
                return None, None, None, None

    return (cache_key,) + st.session_state[cache_key]


# ============================================================================
# Tab renderers
# ============================================================================

def _tab_overview(df_results: pd.DataFrame, insights: dict) -> None:
    """Tab 1 — summary metrics, phase table, and overview charts."""
    st.markdown("#### 📊 Phase-Level CSAT / DSAT")
    render_phase_table(insights)

    st.markdown("#### 📈 Visual Overview")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_sentiment_dist(df_results),  width='stretch')
    with c2:
        st.plotly_chart(chart_speaker_box(df_results),     width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_phase_comparison(insights),       width='stretch')
    with c4:
        st.plotly_chart(chart_escalation_resolution(df_results), width='stretch')

    st.plotly_chart(chart_sentiment_progression(df_results), width='stretch')
    st.plotly_chart(chart_conversation_heatmap(df_results),  width='stretch')


def _tab_tbt_flow(df_results: pd.DataFrame) -> None:
    """Tab 2 — per-conversation turn-by-turn flow and momentum charts."""
    st.markdown("#### 🔄 Turn-by-Turn Sentiment Flow")
    conv_ids = sorted(df_results["conversation_id"].unique().tolist())

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sel_conv     = st.selectbox("Select Conversation", conv_ids, key="flow_conv")
    with col2:
        speaker_flt  = st.selectbox("Speaker",  ["All", "CUSTOMER", "AGENT"], key="flow_spk")
    with col3:
        phase_flt    = st.selectbox("Phase",    ["All", "start", "middle", "end"], key="flow_phase")

    # Filtered view for mini-metrics
    df_view = df_results[df_results["conversation_id"] == sel_conv].copy()
    if speaker_flt != "All":
        df_view = df_view[df_view["speaker"] == speaker_flt]
    if phase_flt != "All":
        df_view = df_view[df_view["phase"] == phase_flt]

    # Flow chart (always shows full conversation, unfiltered)
    st.plotly_chart(chart_tbt_flow(df_results, sel_conv), width='stretch')

    # Momentum chart
    st.plotly_chart(chart_sentiment_momentum(df_results, sel_conv), width='stretch')

    # Speaker × phase heatmap
    st.plotly_chart(chart_speaker_phase_heatmap(df_results, sel_conv), width='stretch')

    # Mini metrics for selected conversation / filter
    cu_sub = df_view[df_view["speaker"] == "CUSTOMER"]
    ag_sub = df_view[df_view["speaker"] == "AGENT"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Turns (filtered)", len(df_view))
    m2.metric("Customer Avg",
              f"{cu_sub['compound'].mean():+.3f}" if not cu_sub.empty else "—")
    m3.metric("Agent Avg",
              f"{ag_sub['compound'].mean():+.3f}" if not ag_sub.empty else "—")
    m4.metric("Escalation turns",
              int(df_view["potential_escalation"].sum())
              if "potential_escalation" in df_view.columns else 0)


def _tab_conversation_explorer(df_results: pd.DataFrame) -> None:
    """Tab 3 — conversation transcript viewer with per-turn sentiment cards."""
    st.markdown("#### 🗣️ Conversation Explorer")
    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        conv_ids_exp  = sorted(df_results["conversation_id"].unique().tolist())
        sel_conv_exp  = st.selectbox("Choose conversation", conv_ids_exp, key="exp_conv")
    with col_r:
        phase_exp = st.selectbox("Filter by phase", ["All", "start", "middle", "end"], key="exp_phase")

    df_exp = df_results[df_results["conversation_id"] == sel_conv_exp]
    if phase_exp != "All":
        df_exp = df_exp[df_exp["phase"] == phase_exp]

    render_turn_viewer(df_exp, sel_conv_exp)


def _tab_data_table(df_results: pd.DataFrame) -> None:
    """Tab 4 — filterable full results table with CSV download."""
    st.markdown("#### 📋 Full Results Table")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        f_speaker   = st.selectbox("Speaker",   ["All", "CUSTOMER", "AGENT"], key="tbl_spk")
    with col_f2:
        f_sentiment = st.selectbox("Sentiment", ["All", "positive", "neutral", "negative"], key="tbl_sent")
    with col_f3:
        f_phase     = st.selectbox("Phase",     ["All", "start", "middle", "end"], key="tbl_phase")

    df_tbl = df_results.copy()
    if f_speaker   != "All": df_tbl = df_tbl[df_tbl["speaker"]         == f_speaker]
    if f_sentiment != "All": df_tbl = df_tbl[df_tbl["sentiment_label"] == f_sentiment]
    if f_phase     != "All": df_tbl = df_tbl[df_tbl["phase"]           == f_phase]

    display_cols = [
        "conversation_id", "turn_sequence", "phase", "speaker",
        "timestamp", "message", "sentiment_label", "compound",
        "sentiment_confidence", "potential_escalation", "potential_resolution",
    ]
    display_cols = [c for c in display_cols if c in df_tbl.columns]

    st.markdown(f"**{len(df_tbl):,} rows** after filters")
    st.dataframe(
        df_tbl[display_cols].reset_index(drop=True),
        width='stretch',
        height=450,
    )

    csv_buf = io.StringIO()
    df_tbl[display_cols].to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv_buf.getvalue(),
        file_name=f"tbt_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def _tab_recommendations(df_results: pd.DataFrame, insights: dict) -> None:
    """Tab 5 — recommendations, raw insights JSON, and bulk export."""
    st.markdown("#### 💡 Automated Business Recommendations")
    render_recommendations(insights)

    st.markdown("---")
    st.markdown("#### 📊 Raw Insights (JSON)")
    with st.expander("View full insights object"):
        st.json(insights)

    st.markdown("---")
    st.markdown("#### ⬇️ Export Results")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Download Excel (all sheets)",
            data=to_excel(df_results, insights),
            file_name=f"tbt_results_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch',
        )
    with col_dl2:
        st.download_button(
            "📥 Download ZIP (CSV + Excel + JSON)",
            data=to_zip(df_results, insights),
            file_name=f"tbt_complete_{ts}.zip",
            mime="application/zip",
            width='stretch',
        )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """
    App entry point.

    Flow
    ----
    1. Render header + sidebar → collect user inputs.
    2. Load data from uploaded file.
    3. If no data → show landing / welcome screen.
    4. Run the analysis pipeline (cached by data fingerprint).
    5. Render status bar + KPI row.
    6. Render five tabs: Overview / Flow / Explorer / Data / Recommendations.
    """
    render_header()
    dataset_type, uploaded = render_sidebar()

    # -- Resolve data source --------------------------------------------------
    df_raw, source_label, dataset_type = load_data(uploaded, dataset_type)

    if df_raw is None:
        render_landing()
        return

    # -- Run pipeline (cached) ------------------------------------------------
    cache_key, df_results, insights, detected = _run_cached_pipeline(df_raw, dataset_type)
    if df_results is None:
        return   # error already shown inside _run_cached_pipeline

    # -- Status bar -----------------------------------------------------------
    col_a, col_b, col_c = st.columns([3, 2, 1])
    with col_a:
        st.markdown(
            f'<div style="color:#888;font-size:.82rem">'
            f'{source_label} &nbsp;·&nbsp; '
            f'Format: <span style="color:#6c63ff;font-weight:600">{detected}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_c:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇️ Export ZIP",
            data=to_zip(df_results, insights),
            file_name=f"tbt_analysis_{ts}.zip",
            mime="application/zip",
            width='stretch',
        )

    st.markdown("---")

    # -- KPI row --------------------------------------------------------------
    render_kpi_row(insights)

    # -- Tabs -----------------------------------------------------------------
    tabs = st.tabs([
        "📊 Overview",
        "🔄 Turn-by-Turn Flow",
        "🗣️ Conversation Explorer",
        "📋 Data Table",
        "💡 Recommendations",
    ])

    with tabs[0]: _tab_overview(df_results, insights)
    with tabs[1]: _tab_tbt_flow(df_results)
    with tabs[2]: _tab_conversation_explorer(df_results)
    with tabs[3]: _tab_data_table(df_results)
    with tabs[4]: _tab_recommendations(df_results, insights)


if __name__ == "__main__":
    main()
