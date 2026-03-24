"""
tbt_charts.py
=============
Plotly chart factories for the TbT Sentiment Analytics Streamlit app.

All functions accept a ``pd.DataFrame`` (df_results) or insights dict and
return a ``plotly.graph_objects.Figure`` ready to pass to
``st.plotly_chart()``.

Keeping chart logic isolated here means:
- tbt_app.py stays lean and focused on UI routing.
- Individual charts can be tested and iterated independently.
- A future migration to another charting library touches only this file.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Shared theme applied to every chart
# ---------------------------------------------------------------------------

CHART_THEME: Dict[str, Any] = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ccc", family="Inter, sans-serif"),
    margin=dict(l=30, r=20, t=45, b=30),
)

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral":  "#4682b4",
    "negative": "#e74c3c",
}

PHASE_ICONS = {"start": "🚀", "middle": "🔄", "end": "🏁"}


# ---------------------------------------------------------------------------
# Overview charts
# ---------------------------------------------------------------------------

def chart_sentiment_dist(df: pd.DataFrame) -> go.Figure:
    """Vertical bar chart of overall positive / neutral / negative counts."""
    counts = df["sentiment_label"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.bar(
        counts, x="sentiment", y="count",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        labels={"sentiment": "Sentiment", "count": "Turns"},
        title="Overall Sentiment Distribution",
    )
    fig.update_layout(**CHART_THEME, title_font_size=14, showlegend=False)
    fig.update_traces(marker_line_width=0)
    return fig


def chart_speaker_box(df: pd.DataFrame) -> go.Figure:
    """Box plots comparing compound scores for CUSTOMER vs AGENT."""
    fig = go.Figure()
    for role, color in [("CUSTOMER", "#ff6b6b"), ("AGENT", "#4ecdc4")]:
        sub = df[df["speaker"] == role]["compound"]
        if not sub.empty:
            fig.add_trace(go.Box(
                y=sub, name=role.capitalize(),
                marker_color=color, boxpoints="outliers",
                line_color=color,
            ))
    fig.update_layout(
        **CHART_THEME,
        title="Customer vs Agent Sentiment Distribution",
        title_font_size=14,
    )
    return fig


def chart_phase_comparison(insights: Dict) -> go.Figure:
    """Grouped bar chart — CSAT % vs DSAT % for Start / Middle / End phases."""
    pcd = insights.get("phase_csat_dsat", {})
    phases = ["Start", "Middle", "End"]
    csat_vals = [pcd.get(p.lower(), {}).get("csat_pct", 0) * 100 for p in phases]
    dsat_vals = [pcd.get(p.lower(), {}).get("dsat_pct", 0) * 100 for p in phases]

    fig = go.Figure(data=[
        go.Bar(name="CSAT %", x=phases, y=csat_vals, marker_color="#2ecc71"),
        go.Bar(name="DSAT %", x=phases, y=dsat_vals, marker_color="#e74c3c"),
    ])
    fig.update_layout(
        **CHART_THEME,
        barmode="group",
        title="CSAT vs DSAT by Conversation Phase",
        title_font_size=14,
        yaxis=dict(title="% Customer Turns", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def chart_sentiment_progression(df: pd.DataFrame) -> go.Figure:
    """Area line chart of average sentiment per turn number (all conversations)."""
    tp = df.groupby("turn_sequence")["compound"].mean().reset_index()
    tp = tp[tp["turn_sequence"] <= 30]   # cap at 30 for readability
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
    fig.add_trace(go.Scatter(
        x=tp["turn_sequence"], y=tp["compound"],
        mode="lines+markers",
        line=dict(color="#4ecdc4", width=2.5),
        marker=dict(size=6, color="#4ecdc4"),
        fill="tozeroy",
        fillcolor="rgba(78,205,196,0.08)",
        name="Avg Sentiment",
    ))
    fig.update_layout(
        **CHART_THEME,
        title="Avg Sentiment by Turn Position (all conversations, first 30 turns)",
        title_font_size=14,
        xaxis=dict(title="Turn Number", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Avg Compound Score", gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def chart_escalation_resolution(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing escalation and resolution event counts."""
    esc   = int(df["potential_escalation"].sum())
    res   = int(df["potential_resolution"].sum())
    total = max(df["conversation_id"].nunique(), 1)
    fig = go.Figure(go.Bar(
        x=["Escalations", "Resolutions"],
        y=[esc, res],
        marker_color=["#e74c3c", "#2ecc71"],
        text=[f"{esc} ({esc/total:.0%})", f"{res} ({res/total:.0%})"],
        textposition="auto",
    ))
    fig.update_layout(
        **CHART_THEME,
        title="Escalation & Resolution Events",
        title_font_size=14,
        yaxis=dict(title="Event Count", gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False,
    )
    return fig


def chart_conversation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: conversation length vs average sentiment, coloured by sentiment."""
    conv_metrics = (
        df.groupby("conversation_id")
        .agg(avg_sentiment=("compound", "mean"), turns=("turn_sequence", "max"))
        .reset_index()
    )
    fig = px.scatter(
        conv_metrics, x="turns", y="avg_sentiment",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
        range_color=[-1, 1],
        hover_name="conversation_id",
        labels={
            "turns":         "Conversation Length (turns)",
            "avg_sentiment": "Avg Sentiment",
        },
        title="Conversation Performance Map",
    )
    fig.update_layout(**CHART_THEME, title_font_size=14)
    fig.update_coloraxes(colorbar=dict(thickness=10))
    return fig


# ---------------------------------------------------------------------------
# Turn-by-Turn Flow chart (per conversation)
# ---------------------------------------------------------------------------

def chart_tbt_flow(df: pd.DataFrame, conv_id: str) -> go.Figure:
    """
    Line + marker chart showing the sentiment trajectory of one conversation.

    Phase bands (start / middle / end) are rendered as translucent vrects.
    Markers are coloured on a Red→Yellow→Green scale by compound score.
    """
    sub = df[df["conversation_id"] == conv_id].sort_values("turn_sequence")
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")

    fig.add_trace(go.Scatter(
        x=sub["turn_sequence"],
        y=sub["compound"],
        mode="lines+markers",
        line=dict(color="#6c63ff", width=2.5),
        marker=dict(
            size=9,
            color=sub["compound"],
            colorscale="RdYlGn",
            cmin=-1, cmax=1,
            showscale=True,
            colorbar=dict(thickness=10, title="Score"),
        ),
        text=[
            (
                f"Turn {r.turn_sequence}<br>{r.speaker}<br>{r.message[:60]}..."
                if len(r.message) > 60
                else f"Turn {r.turn_sequence}<br>{r.speaker}<br>{r.message}"
            )
            for _, r in sub.iterrows()
        ],
        hovertemplate="%{text}<br>Score: %{y:.3f}<extra></extra>",
    ))

    # Shaded phase bands
    mt = int(sub["turn_sequence"].max()) if not sub.empty else 1
    phase_bands = {
        "start":  (1, 3,           "rgba(108,99,255,0.08)"),
        "middle": (4, max(4, mt-3), "rgba(78,205,196,0.06)"),
        "end":    (max(4, mt-2), mt, "rgba(255,107,107,0.08)"),
    }
    for pn, (start, end, color) in phase_bands.items():
        if start <= end:
            fig.add_vrect(
                x0=start - 0.5, x1=end + 0.5,
                fillcolor=color, line_width=0,
                annotation_text=PHASE_ICONS[pn],
                annotation_position="top left",
            )

    fig.update_layout(
        **CHART_THEME,
        title=f"Turn-by-Turn Sentiment Flow — {conv_id}",
        title_font_size=14,
        xaxis=dict(title="Turn Sequence",   gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Sentiment Score", range=[-1.1, 1.1],
                   gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Speaker-level sentiment heatmap (per conversation)
# ---------------------------------------------------------------------------

def chart_speaker_phase_heatmap(df: pd.DataFrame, conv_id: str) -> go.Figure:
    """
    Heatmap of average sentiment per speaker × phase for one conversation.
    Useful for quickly spotting which role drives negativity in which phase.
    """
    sub = df[df["conversation_id"] == conv_id]
    pivot = (
        sub.groupby(["speaker", "phase"])["compound"]
        .mean()
        .unstack(fill_value=0)
        .reindex(columns=["start", "middle", "end"], fill_value=0)
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=["Start", "Middle", "End"],
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=[[f"{v:+.2f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(thickness=10),
    ))
    fig.update_layout(
        **CHART_THEME,
        title=f"Sentiment by Speaker × Phase — {conv_id}",
        title_font_size=14,
        xaxis=dict(title="Phase"),
        yaxis=dict(title="Speaker"),
    )
    return fig


# ---------------------------------------------------------------------------
# Momentum chart (per conversation)
# ---------------------------------------------------------------------------

def chart_sentiment_momentum(df: pd.DataFrame, conv_id: str) -> go.Figure:
    """
    Bar chart of turn-by-turn sentiment momentum (3-turn rolling change).
    Positive bars indicate improving sentiment, negative bars indicate decline.
    """
    sub = df[df["conversation_id"] == conv_id].sort_values("turn_sequence")
    colors = ["#2ecc71" if v >= 0 else "#e74c3c"
              for v in sub["sentiment_momentum"]]
    fig = go.Figure(go.Bar(
        x=sub["turn_sequence"],
        y=sub["sentiment_momentum"],
        marker_color=colors,
        hovertemplate="Turn %{x}<br>Momentum: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(
        **CHART_THEME,
        title=f"Sentiment Momentum — {conv_id}",
        title_font_size=14,
        xaxis=dict(title="Turn Sequence",  gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="3-Turn Momentum", gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False,
    )
    return fig
