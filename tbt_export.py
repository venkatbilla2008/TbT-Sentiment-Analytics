"""
tbt_export.py
=============
Export helpers for the TbT Sentiment Analytics Streamlit app.

Functions return raw bytes suitable for ``st.download_button(data=...)``.

Keeping export logic here means:
- tbt_app.py never imports openpyxl or zipfile directly.
- Export logic can be tested independently of the UI.
- Adding a new export format (e.g. Parquet) is a single, isolated change.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def to_excel(df: pd.DataFrame, insights: Dict[str, Any]) -> bytes:
    """
    Build a multi-sheet Excel workbook and return its bytes.

    Sheets
    ------
    All Turns       — complete df_results
    Customer Turns  — CUSTOMER-speaker rows only
    Agent Turns     — AGENT-speaker rows only
    Summary         — flat table of key insight metrics
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Turns", index=False)

        cu = df[df["speaker"] == "CUSTOMER"]
        if not cu.empty:
            cu.to_excel(writer, sheet_name="Customer Turns", index=False)

        ag = df[df["speaker"] == "AGENT"]
        if not ag.empty:
            ag.to_excel(writer, sheet_name="Agent Turns", index=False)

        # Flat summary sheet
        pcd = insights.get("phase_csat_dsat", {})
        summary_rows = [
            {"Metric": "Total Conversations",       "Value": insights["total_conversations"]},
            {"Metric": "Total Turns",               "Value": insights["total_turns"]},
            {"Metric": "Avg Turns / Conversation",  "Value": f"{insights['avg_turns_per_conversation']:.1f}"},
            {"Metric": "Overall Avg Sentiment",     "Value": f"{insights['overall_sentiment']['average']:.3f}"},
            {"Metric": "Customer Avg Sentiment",    "Value": f"{insights['customer_satisfaction']['average_sentiment']:.3f}"},
            {"Metric": "Agent Avg Sentiment",       "Value": f"{insights['agent_performance']['average_sentiment']:.3f}"},
            {"Metric": "Escalation Rate",           "Value": f"{insights['customer_satisfaction']['escalation_rate']:.1%}"},
            {"Metric": "Resolution Rate",           "Value": f"{insights['customer_satisfaction']['resolution_rate']:.1%}"},
            {"Metric": "Sentiment Improvement",     "Value": f"{insights['conversation_patterns']['sentiment_improvement']:.3f}"},
        ]
        for pn in ["start", "middle", "end"]:
            p = pcd.get(pn, {})
            summary_rows += [
                {"Metric": f"{pn.capitalize()} CSAT %",    "Value": f"{p.get('csat_pct', 0):.1%}"},
                {"Metric": f"{pn.capitalize()} DSAT %",    "Value": f"{p.get('dsat_pct', 0):.1%}"},
                {"Metric": f"{pn.capitalize()} Avg Score", "Value": f"{p.get('avg_sentiment', 0):.3f}"},
            ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# ZIP export (CSV + Excel + JSON)
# ---------------------------------------------------------------------------

def to_zip(df: pd.DataFrame, insights: Dict[str, Any]) -> bytes:
    """
    Bundle CSV, Excel, and JSON insights into a single ZIP archive.

    Returns bytes suitable for ``st.download_button``.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # CSV
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        zf.writestr("tbt_results.csv", csv_buf.getvalue())

        # JSON insights
        zf.writestr(
            "tbt_insights.json",
            json.dumps(insights, indent=2, default=str),
        )

        # Excel (reuse the existing builder)
        zf.writestr("tbt_results.xlsx", to_excel(df, insights))

    return buf.getvalue()
