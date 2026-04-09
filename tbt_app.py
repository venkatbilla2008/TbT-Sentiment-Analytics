"""
tbt_app.py  —  Conversation Turn-by-Turn Sentiment Analytics  v5.1
======================================================================
Supported transcript formats (conversation-based only)
  ✓ netflix  — Bracket [HH:MM:SS SPEAKER]: transcripts
  ✓ spotify  — ISO-timestamp Speaker: transcripts
  ✓ humana   — Call-centre [MM:SS] Speaker: transcripts
  ✓ ppt      — HTML <b>HH:MM:SS name:</b> or SMS chat logs

Removed in v5.1
  ✗ lyft  (customer verbatim / single-turn feedback) — not conversation-based
  ✗ hilton (guest feedback / single-turn feedback)   — not conversation-based

Performance
  ✓ Parallel VADER  — ThreadPoolExecutor(4 workers)
  ✓ @st.cache_data on ALL 5 pipeline stages
  ✓ Polars everywhere — all groupbys / aggregations
  ✓ Vectorised label assignment — numpy np.where
  ✓ Scatter / sunburst subsampled ≤2 000 points
  ✓ Paginated data table — 200 rows per page

Run:  streamlit run tbt_app.py
"""

from __future__ import annotations

import gc, hashlib, io, json, os, re, warnings, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Suppress noisy but harmless warnings from third-party libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*streamlit.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*[Kk]aleido.*")
warnings.filterwarnings("ignore", message=".*write_image.*")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TbT Sentiment Analytics",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    teal="#2D5F6E", teal_l="#3A7A8C", slate="#6B8A99", steel="#A8BCC8",
    warm="#D1CFC4", warm_l="#E8E6DD", gold="#D4B94E", gold_l="#E8D97A",
    bg="#F5F4F0", card="#FFFFFF", border="#D1CFC4",
    text="#1E2D33", text2="#3D5A66", muted="#6B8A99",
    ok="#3D7A5F", warn="#B8963E", err="#A04040",
    pos="#2ecc71", neg="#e74c3c", neu="#4682b4",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{{
  --teal:{C['teal']};--teal-l:{C['teal_l']};--slate:{C['slate']};--steel:{C['steel']};
  --warm:{C['warm']};--warm-l:{C['warm_l']};--gold:{C['gold']};--gold-l:{C['gold_l']};
  --bg:{C['bg']};--card:{C['card']};--border:{C['border']};
  --text:{C['text']};--text2:{C['text2']};--muted:{C['muted']};
  --ok:{C['ok']};--warn:{C['warn']};--err:{C['err']};
}}
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;}}
.stApp{{background:var(--bg);}}
section[data-testid="stSidebar"]{{background:var(--warm-l)!important;border-right:1px solid var(--warm)!important;}}
section[data-testid="stSidebar"] *{{color:var(--text)!important;}}
section[data-testid="stSidebar"] .stButton>button{{
  justify-content:flex-start!important;text-align:left!important;
  padding:10px 16px!important;border-radius:8px!important;width:100%!important;
  margin-bottom:3px!important;font-weight:500!important;font-size:13px!important;
  background:transparent!important;border:1px solid transparent!important;
  color:var(--text2)!important;transition:all .15s!important;}}
section[data-testid="stSidebar"] .stButton>button:hover{{
  background:var(--warm)!important;border-color:var(--teal)!important;color:var(--teal)!important;}}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{{
  background:var(--teal)!important;color:#fff!important;border-color:var(--teal)!important;font-weight:600!important;}}
.mc{{background:var(--card);border:1px solid var(--border);border-radius:10px;
    padding:16px 14px;text-align:center;border-top:3px solid var(--teal);
    box-shadow:0 1px 4px rgba(45,95,110,0.06);transition:all .2s;}}
.mc:hover{{box-shadow:0 4px 16px rgba(45,95,110,0.1);transform:translateY(-1px);}}
.mv{{font-size:22px;font-weight:700;color:var(--text);margin:0;line-height:1.2;}}
.ml{{font-size:10px;font-weight:600;color:var(--muted);margin:5px 0 0;text-transform:uppercase;letter-spacing:.7px;}}
.sh{{display:flex;align-items:center;gap:8px;margin:24px 0 12px;font-size:15px;
    font-weight:600;color:var(--text);padding-bottom:8px;border-bottom:2px solid var(--warm);}}
.pt{{width:100%;border-collapse:separate;border-spacing:0;font-size:13px;
    border-radius:8px;overflow:hidden;border:1px solid var(--border);}}
.pt th{{background:var(--teal);color:#fff;font-weight:600;padding:10px 14px;
       text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;}}
.pt td{{padding:8px 14px;border-bottom:1px solid var(--warm-l);color:var(--text);}}
.pt tr:nth-child(even){{background:var(--warm-l);}}.pt tr:hover td{{background:#D6E8EE;}}
.badge{{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:5px;font-size:11px;font-weight:600;}}
.b-ok{{background:#D4E8DC;color:var(--ok);}}.b-warn{{background:#F0E6C8;color:#7A6620;}}
.b-err{{background:#F2D6D6;color:var(--err);}}.b-info{{background:#D6E8EE;color:var(--teal);}}
.tc{{border-radius:10px;padding:10px 14px;margin-bottom:6px;
    border-left:4px solid transparent;background:var(--card);
    box-shadow:0 1px 3px rgba(45,95,110,0.06);}}
.tc-cu{{border-color:{C['neg']};background:#FEF5F5;}}
.tc-ag{{border-color:{C['teal']};background:#F0F7FA;}}
.tc-hdr{{font-size:11px;color:var(--muted);margin-bottom:4px;font-weight:500;}}
.tc-txt{{font-size:13px;color:var(--text);line-height:1.6;}}
.tc-meta{{font-size:11px;color:var(--slate);margin-top:5px;}}
.sbar{{display:flex;align-items:center;gap:6px;}}
.sbar-t{{flex:1;height:5px;background:var(--warm);border-radius:999px;overflow:hidden;}}
.sbar-f{{height:100%;border-radius:999px;}}
.rc{{background:var(--warm-l);border:1px solid var(--border);border-radius:8px;
    padding:10px 14px;margin-bottom:6px;font-size:13px;color:var(--text);line-height:1.6;}}
.ex-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;
         padding:16px;border-top:3px solid var(--gold);}}
.ex-title{{font-size:14px;font-weight:600;color:var(--text);margin-bottom:4px;}}
.ex-desc{{font-size:12px;color:var(--muted);margin-bottom:12px;line-height:1.5;}}
.stButton>button[kind="primary"]{{background:var(--teal)!important;border-color:var(--teal)!important;color:#fff!important;font-weight:600!important;}}
.stButton>button[kind="primary"]:hover{{background:var(--teal-l)!important;}}
.stTabs [data-baseweb="tab"]{{font-weight:500;color:var(--muted);font-size:13px;}}
.stTabs [aria-selected="true"]{{color:var(--teal)!important;border-bottom-color:var(--teal)!important;font-weight:600;}}
footer,.stDeployButton{{display:none!important;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FORMAT_LABELS: Dict[str, str] = {
    "auto":    "🔍 Auto-Detect",
    "netflix": "🎬 Media / Entertainment  (Bracket [HH:MM:SS])",
    "spotify": "🎵 Media / Entertainment  (Timestamp)",
    "humana":  "🏥 Healthcare A  (Call transcript [MM:SS])",
    "ppt":     "🩼 Healthcare B  (Chat / SMS)",
}
PHASE_ICONS   = {"start": "🚀", "middle": "🔄", "end": "🏁"}

# ── Environment-aware performance limits ──────────────────────────────────────
# Auto-detects Cloud (1 GB RAM) vs local machine and sets limits accordingly.
# Local machines get generous limits so large datasets (1L+ turns) work fine.
# Override any value via environment variable, e.g.:
#   MAX_TURNS=500000 streamlit run tbt_app.py          (Mac/Linux)
#   set MAX_TURNS=500000 && streamlit run tbt_app.py   (Windows)
def _detect_env_limits() -> dict:
    import os
    try:
        import psutil
        _ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        _cpu    = psutil.cpu_count(logical=False) or 1
    except Exception:
        try:
            with open("/proc/meminfo") as _f:
                _ram_gb = next(
                    int(l.split()[1]) / (1024 ** 2)
                    for l in _f if "MemTotal" in l
                )
        except Exception:
            _ram_gb = 2.0
        _cpu = os.cpu_count() or 1

    _is_cloud = bool(
        os.environ.get("STREAMLIT_SHARING_MODE") or
        os.environ.get("IS_STREAMLIT_CLOUD") or
        _ram_gb < 1.5
    )

    if _is_cloud:
        _lim = dict(MAX_TURNS=100_000, CHUNK_TURNS=5_000,
                    CHART_SAMPLE=5_000, VADER_WORKERS=2)
    elif _ram_gb >= 32:
        _lim = dict(MAX_TURNS=1_000_000, CHUNK_TURNS=50_000,
                    CHART_SAMPLE=25_000, VADER_WORKERS=min(_cpu, 8))
    elif _ram_gb >= 16:
        _lim = dict(MAX_TURNS=500_000,   CHUNK_TURNS=25_000,
                    CHART_SAMPLE=15_000, VADER_WORKERS=min(_cpu, 6))
    else:
        _lim = dict(MAX_TURNS=250_000,   CHUNK_TURNS=10_000,
                    CHART_SAMPLE=10_000, VADER_WORKERS=min(_cpu, 4))

    for _k in _lim:
        if os.environ.get(_k):
            try:
                _lim[_k] = int(os.environ[_k])
            except ValueError:
                pass
    return _lim

_ENV_LIMITS   = _detect_env_limits()
MAX_TURNS     = _ENV_LIMITS["MAX_TURNS"]    # hard cap on turns processed
CHUNK_TURNS   = _ENV_LIMITS["CHUNK_TURNS"]  # VADER batch size per chunk
CHART_SAMPLE  = _ENV_LIMITS["CHART_SAMPLE"] # max points sent to browser
VADER_WORKERS = _ENV_LIMITS["VADER_WORKERS"]# parallel VADER threads
CHART_LAYOUT  = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C['text'], family="DM Sans"),
    margin=dict(l=10, r=20, t=40, b=10),
    hoverlabel=dict(bgcolor=C['text'], font_size=12, font_color=C['warm_l']),
)

# ─────────────────────────────────────────────────────────────────────────────
# POLARS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _safe_collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Version-safe Polars collect — tries streaming engine first (lower RAM),
    falls back to standard collect() on API mismatch or unsupported plan.
    Polars changed streaming API: <1.25 uses streaming=True, >=1.25 uses engine='streaming'.
    Window functions (over()) may not be supported in streaming mode — fallback handles this."""
    try:
        _parts = pl.__version__.split('.')
        _ver = (int(_parts[0]), int(''.join(c for c in _parts[1] if c.isdigit())))
    except Exception:
        _ver = (0, 0)
    try:
        if _ver >= (1, 25):
            return lf.collect(engine='streaming')
        else:
            return lf.collect(streaming=True)
    except Exception:
        return lf.collect()


# ─────────────────────────────────────────────────────────────────────────────
# PII REDACTION  — 8 pattern types
# ─────────────────────────────────────────────────────────────────────────────
class PIIRedactor:
    """
    Redacts 8 PII pattern types from text using Polars vectorised regex.

    Patterns
    --------
    EMAIL  — standard e-mail addresses
    CARD   — Visa / MC / Amex / Discover credit-card numbers
    SSN    — US Social-Security Numbers  (###-##-####)
    MRN    — Medical Record Numbers  (MRN / Patient ID prefix)
    DOB    — Dates of birth  (MM/DD/YYYY or MM-DD-YYYY)
    IP     — IPv4 addresses
    PHONE  — North-American phone numbers (various separators)
    ADDR   — Street addresses  (number + street type keyword)

    Modes
    -----
    token   → [EMAIL]            (type tag only)
    mask    → [EMAIL:REDACTED]   (type + REDACTED label)
    remove  → ""                 (blank — PII stripped entirely)
    """

    PATS: Dict[str, str] = {
        "EMAIL": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
        "CARD":  r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b",
        "SSN":   r"\b\d{3}-\d{2}-\d{4}\b",
        "MRN":   r"\b(?:MRN|Medical\s*Record|Patient\s*ID)[:\s#]+[A-Z0-9]{5,12}\b",
        "DOB":   r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b",
        "IP":    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        "PHONE": r"(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b",
        "ADDR":  r"\b\d{1,5}\s+[A-Za-z0-9\s,.\-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Circle|Cir|Apt|Suite|Ste|Unit)\b",
    }

    @classmethod
    def redact_series(cls, col: pl.Series, mode: str = "mask") -> pl.Series:
        """
        Apply all PII patterns to a Polars Series of strings.
        Returns a new Series with PII replaced according to *mode*.
        Silently skips any pattern that raises a regex error.
        """
        result = col.cast(pl.Utf8).fill_null("")
        for ptype, pat in cls.PATS.items():
            if mode == "token":
                replacement = f"[{ptype}]"
            elif mode == "remove":
                replacement = ""
            else:  # default: "mask"
                replacement = f"[{ptype}:REDACTED]"
            try:
                result = result.str.replace_all(pat, replacement)
            except Exception:
                pass  # skip patterns that fail on edge-case data
        return result

    @classmethod
    def redact_dataframe(
        cls,
        df: pd.DataFrame,
        columns: List[str],
        mode: str = "mask",
    ) -> Tuple[pd.DataFrame, int]:
        """
        Redact PII from *columns* in a Pandas DataFrame.
        Returns (redacted_df, n_rows_with_any_redaction).
        """
        df_out = df.copy()
        redacted_rows: set = set()
        for col in columns:
            if col not in df_out.columns:
                continue
            original  = pl.Series(df_out[col].astype(str).fillna(""))
            redacted  = cls.redact_series(original, mode=mode)
            changed   = (original != redacted).to_numpy()
            redacted_rows.update(int(i) for i in np.where(changed)[0])
            df_out[col] = redacted.to_list()
        return df_out, len(redacted_rows)

    @classmethod
    def count_pii(cls, series: pd.Series) -> Dict[str, int]:
        """Return per-type hit counts for an audit badge."""
        counts: Dict[str, int] = {}
        col = pl.Series(series.astype(str).fillna(""))
        for ptype, pat in cls.PATS.items():
            try:
                hits = int(col.str.count_matches(pat).sum())
                if hits > 0:
                    counts[ptype] = hits
            except Exception:
                pass
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _score_color(s: float) -> str:
    return C['pos'] if s >= 0.1 else C['neg'] if s <= -0.1 else C['gold']

def _badge(lbl: str) -> str:
    if lbl == "positive": return '<span class="badge b-ok">✅ Positive</span>'
    if lbl == "negative": return '<span class="badge b-err">⛔ Negative</span>'
    return '<span class="badge b-info">➖ Neutral</span>'

def _sbar(s: float) -> str:
    pct = int((s + 1) / 2 * 100); col = _score_color(s)
    return (f'<div class="sbar"><div class="sbar-t">'
            f'<div class="sbar-f" style="width:{pct}%;background:{col}"></div></div>'
            f'<span style="color:{col};font-size:11px;font-weight:600;'
            f'font-family:\'JetBrains Mono\',monospace">{s:+.3f}</span></div>')

def _pct(v: float) -> str: return f"{v:.1%}"

def mc(label: str, value: str, color: str = "var(--teal)") -> str:
    return (f'<div class="mc" style="border-top-color:{color}">'
            f'<p class="mv">{value}</p><p class="ml">{label}</p></div>')

def mc2(label_a: str, value_a: str, label_b: str, value_b: str, color: str = "var(--teal)") -> str:
    """Merged dual-metric card — two related KPIs in one card to save horizontal space."""
    return (
        f'<div class="mc" style="border-top-color:{color}">'
        f'<p class="mv">{value_a}</p>'
        f'<p class="ml">{label_a}</p>'
        f'<p style="margin:4px 0 0;font-size:11px;color:var(--muted);border-top:1px solid rgba(168,188,200,0.2);padding-top:4px">'
        f'<span style="font-weight:600;color:{color}">{value_b}</span> {label_b}</p>'
        f'</div>'
    )

def sh(icon: str, text: str) -> None:
    st.markdown(f'<div class="sh">{icon}&nbsp;{text}</div>', unsafe_allow_html=True)

def apply_chart(fig: go.Figure, h: int = None) -> go.Figure:
    fig.update_layout(**CHART_LAYOUT)
    if h: fig.update_layout(height=h)
    return fig

def _to_pd(lf) -> pd.DataFrame:
    """Collect Polars LazyFrame or DataFrame → Pandas."""
    if isinstance(lf, pl.LazyFrame): return lf.collect().to_pandas()
    if isinstance(lf, pl.DataFrame): return lf.to_pandas()
    return lf  # already pandas


# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS  (kept for reference; diff/rolling now handled by Polars window)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class ConversationProcessor:
    _PRIO = [
        "Comments","comments","COMMENTS",
        "Conversation","conversation","CONVERSATION",
        "transcripts","transcript","Transcripts","Transcript",
        "messages","message","Message Text (Translate/Original)",
        "text","chat",
    ]
    def __init__(self, dataset_type: str = "auto"):
        self.dataset_type = dataset_type.lower()
        self._pt  = re.compile(r"^\|?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\s+(Consumer|Customer|Agent|Advisor|Support):\s*(.*)$", re.I|re.MULTILINE)
        self._pb  = re.compile(r"^\[(\d{1,2}:\d{2}:\d{2})\s+(AGENT|CUSTOMER|CONSUMER|ADVISOR|SUPPORT)\]:\s*(.*)$", re.I|re.MULTILINE)
        self._ph  = re.compile(r"\[(\d{1,3}:\d{2})\]\s+([^:]+?):\s*([^\[]+?)(?=\[|$)", re.I|re.DOTALL|re.MULTILINE)
        self._pph = re.compile(r"<b>(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*</b>([^<]+?)(?:<br\s*/?>|$)", re.I|re.DOTALL)
        self._pps = re.compile(
            r"(?<!\d{4}-\d{2}-\d{2} )"   # NOT preceded by ISO date (avoids stealing Spotify rows)
            r"(\d{2}:\d{2}:\d{2})"        # HH:MM:SS timestamp
            r"\s+([^:]+?)\s*:\s*(.+?)(?=\d{2}:\d{2}:\d{2}\s+|$)",
            re.DOTALL|re.MULTILINE
        )

    # Known column names that hold a conversation / session identifier
    # Exact-match candidates — checked first (most specific → least specific).
    # Add any new domain-specific ID column names here.
    _ID_COLS = [
        # ── Conversation / Chat variants ──
        "Conversation Id", "Conversation ID", "conversation_id", "ConversationId",
        "conversation id", "CONVERSATION_ID", "Conversation_Id", "Conversation_ID",
        # ── Ticket / Support variants ──
        "CS Ticket ID", "CS Ticket Id", "cs_ticket_id", "Ticket ID", "Ticket Id",
        "ticket_id", "TicketId", "ticket id", "TICKET_ID",
        # ── Chat / Interaction variants ──
        "Chat No", "Chat No.", "chat_no", "ChatNo", "CHAT_NO",
        "Chat ID", "Chat Id", "chat_id", "ChatId", "chat id",
        "Interaction ID", "Interaction Id", "interaction_id", "InteractionId",
        # ── Session variants ──
        "Session Id", "Session ID", "session_id", "SessionId",
        "session id", "SESSION_ID",
        # ── Call variants ──
        "Call Id", "Call ID", "call_id", "CallId", "call id", "CALL_ID",
        "Call No", "Call No.", "call_no", "CallNo",
        # ── Integer / Reference ID variants ──
        "Int ID", "Int Id", "int_id", "IntId", "INT_ID",
        "Ref ID", "Ref Id", "ref_id", "RefId", "Reference ID", "Reference Id",
        "Case ID", "Case Id", "case_id", "CaseId", "case id",
        # ── Generic ID (checked last — most likely to conflict) ──
        "ID", "Id", "id",
    ]

    def _find_id_col(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the conversation ID column in *df*.

        Strategy (in order):
        1. Exact match against _ID_COLS list (most-specific first).
        2. Case-insensitive substring match: any column whose lowercase name
           contains "id", "no", "ticket", "chat", "session", "call", "ref",
           "case", or "interaction" — but only if it looks like an ID column
           (high cardinality: unique values >= 80% of rows).
        """
        # ── Pass 1: exact match ───────────────────────────────────────────────
        for name in self._ID_COLS:
            if name in df.columns:
                return name

        # ── Pass 2: fuzzy match on column name with cardinality guard ─────────
        ID_KEYWORDS = ("ticket", "chat", "session", "call", "conversation",
                       "interaction", "case", "reference", "ref", "int id",
                       "chat no", "call no")
        n = max(len(df), 1)
        for col in df.columns:
            col_lower = col.lower().strip()
            # Must contain at least one ID keyword
            if not any(kw in col_lower for kw in ID_KEYWORDS):
                # Also accept bare "id" / "no" columns if high cardinality
                if not (col_lower in ("id", "no", "no.") or
                        col_lower.endswith(" id") or col_lower.endswith("_id") or
                        col_lower.endswith(" no") or col_lower.endswith("_no")):
                    continue
            # Cardinality guard: genuine ID columns are highly unique
            try:
                n_unique = df[col].nunique()
                if n_unique / n >= 0.5:   # at least 50% unique values
                    return col
            except Exception:
                pass

        return None

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self._find_col(df)
        if not col:
            raise ValueError("No transcript column found. Expected: Conversation, Transcripts, Comments, message, chat, etc.")
        if self.dataset_type == "auto":
            self.dataset_type = self._detect_from_df(df, col)

        # ── Detect real conversation ID column ───────────────────────────
        # If the source file has a Conversation Id column, use those values.
        # Otherwise fall back to generating CONV_XXXX from the row index.
        id_col  = self._find_id_col(df)
        id_arr  = df[id_col].astype(str).values if id_col else None

        # ── Parallel parse via ThreadPoolExecutor — Cloud safe ────────────
        texts_arr = df[col].astype(str).values
        n_rows    = len(texts_arr)
        dispatch  = self._dispatch

        def _parse_cell(args):
            row_idx, text = args
            if not text or text == "nan" or len(text) < 5:
                return []
            real_id = str(id_arr[row_idx]) if id_arr is not None else ""
            return dispatch(text, row_idx, conv_id=real_id) or []

        # Use min(4, cpu) workers — enough parallelism without memory spike
        n_workers = min(4, max(1, n_rows // 500))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_parse_cell, enumerate(texts_arr)))

        # Flatten results
        rows: List[Dict] = []
        for cell in results:
            rows.extend(cell)

        if not rows:
            raise ValueError("No turns parsed. Check the domain selector matches your file format.")

        # Count how many source rows produced zero turns (skipped)
        skipped = sum(1 for cell in results if not cell)

        # Build output DataFrame — use Polars for the string ops (faster than pandas .str)
        out_pl = pl.DataFrame(rows)
        out_pl = out_pl.with_columns([
            pl.Series("turn_id", np.arange(1, len(out_pl) + 1, dtype=np.int32)),
            pl.col("message").str.to_lowercase().str.strip_chars().alias("cleaned_message"),
        ])
        result = out_pl.to_pandas()
        result.attrs["skipped_rows"] = skipped
        return result

    @property
    def detected_format(self) -> str:
        return FORMAT_LABELS.get(self.dataset_type, self.dataset_type.upper())

    def _detect_format(self, s: str) -> str:
        """Detect format from a single text sample. Priority order is
        most-specific first so generic patterns never shadow specific ones."""
        if self._pb.search(s):  return "netflix"
        if self._pt.search(s):  return "spotify"
        if self._pph.search(s): return "ppt"
        if self._ph.search(s):  return "humana"
        if self._pps.search(s): return "ppt"
        return "ppt"

    def _detect_from_df(self, df: pd.DataFrame, col: str) -> str:
        """Vote across up to 10 spread samples for robust format detection."""
        non_null = df[col].dropna()
        n = min(10, len(non_null))
        if n == 0: return "ppt"
        indices = np.linspace(0, len(non_null) - 1, n, dtype=int)
        votes: Dict[str, int] = {}
        for i in indices:
            fmt = self._detect_format(str(non_null.iloc[i]))
            votes[fmt] = votes.get(fmt, 0) + 1
        priority = ["netflix", "spotify", "humana", "ppt"]
        return max(priority, key=lambda f: (votes.get(f, 0), -priority.index(f)))

    def _detect(self, s, col):
        """Back-compat wrapper — delegates to _detect_format."""
        return self._detect_format(s)

    def _dispatch(self, text, idx, conv_id: str = ""):
        if self.dataset_type == "netflix":  return self._parse_bracket(text, idx, conv_id)
        if self.dataset_type == "humana":   return self._parse_humana(text, idx, conv_id)
        if self.dataset_type == "ppt":      return self._parse_ppt(text, idx, conv_id)
        return self._parse_spotify(text, idx, conv_id)

    def _find_col(self, df):
        for n in self._PRIO:
            if n in df.columns: return n
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                s = str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else ""
                if any(p.search(s) for p in [self._pb,self._pt,self._ph,self._pph,self._pps]): return col
        for col in df.columns:
            if df[col].dtype == object and df[col].dropna().astype(str).str.len().mean() > 20: return col
        return None

    def _norm(self, s):
        u = s.upper().strip()
        if u in {"AGENT","ADVISOR","SUPPORT","REP","REPRESENTATIVE","SPECIALIST"}: return "AGENT"
        if u in {"CUSTOMER","CONSUMER","CLIENT","USER","MEMBER","PATIENT","CALLER"}: return "CUSTOMER"
        return u

    def _row(self, idx, seq, ts, spk, msg, conv_id: str = ""):
        # Use real conversation ID from source data when available;
        # fall back to sequential CONV_XXXX only when no ID column exists.
        cid = conv_id if conv_id and str(conv_id) not in ("", "nan", "None") else f"CONV_{idx+1:04d}"
        return {"conversation_id": cid, "turn_sequence": seq,
                "timestamp": ts, "speaker": spk, "message": msg}

    def _parse_bracket(self, text, idx, conv_id: str = ""):
        lines=text.split("\n"); turns=[]; tn=1; cs=ct=None; cm=[]
        def flush():
            nonlocal tn
            if cs:
                msg=" ".join(cm).strip()
                if msg: turns.append(self._row(idx,tn,ct,self._norm(cs),msg,conv_id)); tn+=1
        for line in lines:
            ls=line.strip(); m=self._pb.match(ls)
            if m:
                flush(); ct,cs,cm=m.group(1),m.group(2),[]
                r=m.group(3).strip()
                if r: cm.append(r)
            elif cs and ls: cm.append(ls)
        flush(); return turns

    def _parse_spotify(self, text, idx, conv_id: str = ""):
        lines=text.split("\n"); turns=[]; tn=1; cs=ct=None; cm=[]
        def flush():
            nonlocal tn
            if cs:
                msg=" ".join(cm).strip()
                if msg: turns.append(self._row(idx,tn,ct,self._norm(cs),msg,conv_id)); tn+=1
        for line in lines:
            ls=line.strip().lstrip("|").strip()  # strip leading pipe used by some Spotify exports
            m=self._pt.search(ls)                # search() handles (?:^|\n) anchor in pattern
            if m:
                flush(); ct,cs=m.group(1),m.group(2); cm=[m.group(3).strip()] if m.group(3).strip() else []
            elif cs and ls: cm.append(ls)
        flush(); return turns

    def _parse_humana(self, text, idx, conv_id: str = ""):
        turns=[]; tn=1
        for ts,spk,msg in self._ph.findall(text):
            sl=spk.strip().lower()
            if sl in {"system","automated","ivr"}: continue
            m=msg.strip()
            if not m or len(m)<3: continue
            ns=("CUSTOMER" if any(k in sl for k in ["member","customer","patient","caller"])
                else "AGENT" if any(k in sl for k in ["agent","representative","rep","advisor","specialist"])
                else spk.strip().upper())
            turns.append(self._row(idx,tn,ts,ns,m,conv_id)); tn+=1
        return turns

    def _parse_ppt(self, text, idx, conv_id: str = ""):
        hm=self._pph.findall(text)
        if hm: return self._ppt_turns(hm,idx,False,conv_id)
        sm=self._pps.findall(text)
        if sm: return self._ppt_turns(sm,idx,True,conv_id)
        return []

    def _ppt_turns(self, matches, idx, is_sms, conv_id: str = ""):
        spk_msgs={}; ordered=[]
        for ts,spk,msg in matches:
            sl=spk.strip().lower()
            if sl=="system": continue
            if is_sms:
                for pat in [r"\d{4}-\d{2}-\d{2}T[\d:.]+Z\w*$",
                            r"Looks up (?:Phone|SSN).*?digits-\d+",
                            r"(?:Phone Numbers|SSN) rule for Chat"]:
                    msg=re.sub(pat,"",msg)
            m=msg.strip()
            if not m: continue
            if sl not in spk_msgs: spk_msgs[sl]=[]; ordered.append(sl)
            spk_msgs[sl].append((ts,m))
        if not spk_msgs: return []
        roles={}
        if is_sms:
            for s in ordered: roles[s]="CUSTOMER" if re.match(r"^\d+$",s) else "AGENT"
        else:
            cnts={s:len(v) for s,v in spk_msgs.items()}
            if len(cnts) == 1:
                # Only one speaker — treat as CUSTOMER
                cust = ordered[0]
            elif len(cnts) == 2:
                # Two speakers: the one who speaks less is CUSTOMER (agent drives more turns).
                # Use first-speaker as tiebreaker when counts are equal.
                spk_a, spk_b = ordered[0], ordered[1]
                cust = spk_a if cnts[spk_a] <= cnts[spk_b] else spk_b
            else:
                # 3+ speakers: fewest-turns speaker is CUSTOMER; first speaker breaks ties
                min_count = min(cnts.values())
                candidates = [s for s in ordered if cnts[s] == min_count]
                cust = candidates[0]
            for s in ordered: roles[s]="CUSTOMER" if s==cust else "AGENT"
        all_m=sorted([(ts,s,m) for s in ordered for ts,m in spk_msgs[s]],key=lambda x:x[0])
        return [self._row(idx,i,ts,roles.get(s,"CUSTOMER"),m,conv_id) for i,(ts,s,m) in enumerate(all_m,1)]


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT ENGINE  — parallel VADER + vectorised label assignment
# ─────────────────────────────────────────────────────────────────────────────
def _score_chunk(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Score one chunk of messages in a worker thread.
    Each thread owns its own SentimentIntensityAnalyzer instance
    (VADER is not thread-safe if shared).

    Optimisation over the original:
      - Pre-filter short messages with a numpy boolean mask (no Python if inside loop)
      - Store results directly into a pre-allocated array — no append overhead
      - Skip polarity_scores() call entirely for blank/short strings

    Returns (start_index, array of shape [chunk_size, 4])
    columns: compound, pos, neg, neu
    """
    start, msgs = args
    vader = SentimentIntensityAnalyzer()
    n     = len(msgs)
    out   = np.zeros((n, 4), dtype=np.float32)

    # Batch-filter: only score messages with >= 5 chars
    msg_arr  = np.array([str(m) for m in msgs], dtype=object)
    len_arr  = np.vectorize(len)(msg_arr)
    valid    = np.where(len_arr >= 5)[0]          # indices worth scoring

    for j in valid:
        sc = vader.polarity_scores(msg_arr[j])
        out[j, 0] = sc["compound"]
        out[j, 1] = sc["pos"]
        out[j, 2] = sc["neg"]
        out[j, 3] = sc["neu"]
    return start, out


class SentimentEngine:
    """
    Parallel VADER scoring with ThreadPoolExecutor.

    Strategy
    --------
    1. Split messages into VADER_WORKERS equal chunks.
    2. Score all chunks concurrently (one SentimentIntensityAnalyzer per thread).
    3. Reassemble results into numpy arrays.
    4. Apply label + confidence assignment with vectorised numpy — no Python loop.

    Speed gain: ~VADER_WORKERS× on multi-core hosts (Streamlit Cloud = 2 vCPUs → ~2×).
    """
    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()   # used only for calibration
        self.thr = {"pos": 0.05, "neg": -0.05, "nr": 0.10}

    def calibrate(self, df: pd.DataFrame):
        """Set adaptive thresholds from 30th / 70th percentile of a sample.
        Uses _score_chunk in a single thread to reuse the batch numpy path."""
        n    = min(1000, len(df))
        msgs = df["cleaned_message"].fillna("").sample(n=n, random_state=42).tolist()
        # Reuse _score_chunk (numpy pre-filter, no Python loop per message)
        _, arr = _score_chunk((0, msgs))
        scores = arr[:, 0].astype(np.float64)          # compound column
        valid  = scores[scores != 0.0]                 # exclude un-scored blanks
        if len(valid) == 0: return
        pos = max(float(np.percentile(valid, 70)), 0.10)
        neg = min(float(np.percentile(valid, 30)), -0.10)
        self.thr = {"pos": pos, "neg": neg, "nr": pos - neg}

    def score(self, df: pd.DataFrame, progress_cb=None) -> pd.DataFrame:
        """
        Score all messages in parallel, then assign labels vectorised.
        progress_cb(fraction) is called once per completed worker chunk.
        """
        msgs   = df["cleaned_message"].fillna("").tolist()
        n      = len(msgs)
        n_jobs = max(1, min(VADER_WORKERS, n))
        chunk  = max(1, (n + n_jobs - 1) // n_jobs)

        # Build chunk arguments: (start_index, messages_list)
        chunks = [
            (i, msgs[i: i + chunk])
            for i in range(0, n, chunk)
        ]

        # Pre-allocate result arrays
        compound = np.zeros(n, dtype=np.float32)
        positive = np.zeros(n, dtype=np.float32)
        negative = np.zeros(n, dtype=np.float32)
        neutral  = np.zeros(n, dtype=np.float32)

        completed = 0
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_score_chunk, arg): arg[0] for arg in chunks}
            for fut in as_completed(futures):
                start, arr = fut.result()
                end = min(start + arr.shape[0], n)
                compound[start:end] = arr[:end-start, 0]
                positive[start:end] = arr[:end-start, 1]
                negative[start:end] = arr[:end-start, 2]
                neutral [start:end] = arr[:end-start, 3]
                completed += 1
                if progress_cb: progress_cb(min(completed / len(chunks), 1.0))

        # Vectorised label + confidence — single numpy pass, no Python loop
        pos_mask = compound >= self.thr["pos"]
        neg_mask = compound <= self.thr["neg"]

        labels = np.where(pos_mask, "positive",
                 np.where(neg_mask, "negative", "neutral"))

        conf = np.where(
            pos_mask, np.clip(compound / self.thr["pos"], 0, 1),
            np.where(
                neg_mask, np.clip(np.abs(compound) / abs(self.thr["neg"]), 0, 1),
                np.clip(1.0 - np.abs(compound) / (self.thr["nr"] / 2 + 1e-9), 0, 1)
            )
        ).astype(np.float32)

        out = df.copy()
        out["compound"]             = compound.astype(float)
        out["positive"]             = positive.astype(float)
        out["negative"]             = negative.astype(float)
        out["neutral"]              = neutral.astype(float)
        out["sentiment_label"]      = labels
        out["sentiment_confidence"] = conf.astype(float)
        gc.collect()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS ENGINE  — Polars for all groupbys / aggregations
# ─────────────────────────────────────────────────────────────────────────────
class AnalyticsEngine:
    """
    Uses Polars for all aggregation-heavy operations:
      - turn metrics: sort, group_by, join
      - insights:     group_by, mean, count via Polars expressions
    Returns a Pandas DataFrame for Streamlit compatibility.
    """

    def compute_turn_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert to Polars for fast groupby/sort
        lf = pl.from_pandas(df).lazy()

        # Sort
        lf = lf.sort(["conversation_id", "turn_sequence"])

        # Max turn per conversation (for turn_position and phase boundaries)
        max_turn = (
            lf.group_by("conversation_id")
              .agg(pl.col("turn_sequence").max().alias("max_turn"))
        )
        lf = lf.join(max_turn, on="conversation_id", how="left")

        lf = lf.with_columns([
            (pl.col("turn_sequence") / pl.col("max_turn")).alias("turn_position"),
            (pl.col("turn_sequence") <= 3).alias("is_conversation_start"),
            (pl.col("turn_sequence") > (pl.col("max_turn") - 3)).alias("is_conversation_end"),
        ]).with_columns([
            (~pl.col("is_conversation_start") & ~pl.col("is_conversation_end")).alias("is_conversation_middle"),
        ]).with_columns([
            pl.when(pl.col("is_conversation_start")).then(pl.lit("start"))
              .when(pl.col("is_conversation_end")).then(pl.lit("end"))
              .otherwise(pl.lit("middle")).alias("phase"),
            (pl.col("compound") >= 0).alias("is_csat"),
            (pl.col("compound") <  0).alias("is_dsat"),
        ])

        # ── Polars window: sentiment change + momentum + speaker tracking ──────
        lf = lf.with_columns([
            (pl.col("compound") - pl.col("compound").shift(1).over("conversation_id"))
              .fill_null(0.0).alias("sentiment_change"),
        ]).with_columns([
            pl.col("sentiment_change")
              .rolling_mean(window_size=3, min_periods=1)
              .over("conversation_id")
              .alias("sentiment_momentum"),
            pl.col("speaker").shift(1).over("conversation_id").alias("prev_speaker"),
        ]).with_columns([
            (pl.col("speaker") != pl.col("prev_speaker").fill_null("")).alias("speaker_changed"),
        ])

        # consecutive_turns run-length
        lf = lf.with_columns([
            pl.col("speaker_changed").cast(pl.Int32)
              .cum_sum().over("conversation_id").alias("_streak_id"),
        ]).with_columns([
            (pl.col("turn_sequence").rank(method="ordinal").over(["conversation_id","_streak_id"]))
              .alias("consecutive_turns"),
        ]).drop("_streak_id")

        d = _safe_collect(lf).to_pandas()
        d = d.sort_values(["conversation_id","turn_sequence"]).reset_index(drop=True)

        # ── Multi-signal escalation detection — vectorised ───────────────────────
        # Signal A (25%): sentiment drop > 0.2 from previous customer turn
        # Signal B (40%): explicit escalation trigger language
        # Signal C (20%): repetition — same complaint bigrams repeated
        # Signal D (15%): frustration build — 3+ consecutive negative customer turns
        ESC_TRIGGERS = [
            "speak to manager","supervisor","escalate","unacceptable","not good enough",
            "complained before","file a complaint","this is ridiculous","how many times",
            "nobody helps","waste of time","cancel everything","worst service",
            "third time","been waiting","still not resolved","no one helped",
            "calling back","not working","sick of this","completely useless",
        ]

        is_cust = (d["speaker"] == "CUSTOMER").values

        # ── Signal A — pure numpy, no loop ───────────────────────────────────
        sc_arr = d["sentiment_change"].fillna(0.0).values.astype(np.float32)
        ts_arr = d["turn_sequence"].values
        sig_a  = np.where(
            is_cust & (sc_arr < -0.2) & (ts_arr > 2),
            np.clip(np.abs(sc_arr) / 0.6, 0.0, 1.0),
            0.0,
        ).astype(np.float32)

        # ── Signal B — single Polars regex pass over all messages ────────────
        _trigger_pat = "|".join(re.escape(t) for t in ESC_TRIGGERS)
        _msg_pl      = pl.Series(d["cleaned_message"].fillna("").astype(str))
        sig_b = (
            pl.Series(is_cust) & _msg_pl.str.contains(_trigger_pat)
        ).cast(pl.Float32).to_numpy()

        # ── Signal D — Polars run-length streak, joined back by position ─────
        # d is already sorted by [conversation_id, turn_sequence] and reset_index(drop=True)
        # so positional order after Polars sort matches d's row order.
        _streak_lf = (
            pl.from_pandas(
                d[["conversation_id", "turn_sequence", "speaker", "compound"]]
                .assign(_pos=np.arange(len(d)))
            ).lazy()
            .sort(["conversation_id", "turn_sequence"])
            .with_columns(
                ((pl.col("speaker") == "CUSTOMER") & (pl.col("compound") < -0.1))
                .alias("_neg_cust")
            )
            .with_columns(
                (pl.col("_neg_cust") != pl.col("_neg_cust").shift(1).fill_null(False))
                .cast(pl.Int32).cum_sum().over("conversation_id").alias("_rle")
            )
            .with_columns(
                # cum_sum of _neg_cust (0/1) within each run-length partition:
                # True-runs → 1,2,3…  False-runs → 0,0,0…
                pl.col("_neg_cust").cast(pl.Int32)
                  .cum_sum().over(["conversation_id", "_rle"])
                  .alias("frustration_streak")
            )
            .select(["_pos", "frustration_streak"])
            .sort("_pos")
        )
        _streak_pd   = _safe_collect(_streak_lf).to_pandas()
        frust_streak = _streak_pd["frustration_streak"].values.astype(np.int32)
        sig_d        = np.where(is_cust & (frust_streak >= 3), 1.0, 0.0).astype(np.float32)

        # ── Signal C — pre-compute bigram sets once; single numpy mask write ─
        # Bigrams computed upfront for all rows (no per-row DF mutation inside loop).
        # Inner loop only touches customer rows and writes to a plain numpy bool array.
        _msg_arr = d["cleaned_message"].fillna("").values
        _spk_arr = d["speaker"].values

        def _bg(text: str) -> frozenset:
            w = re.findall(r"[a-z]{3,}", str(text).lower())
            return frozenset(f"{w[i]} {w[i+1]}" for i in range(len(w) - 1))

        _bigram_sets = np.array([_bg(m) for m in _msg_arr], dtype=object)

        rep_mask = np.zeros(len(d), dtype=bool)
        for _cid, _grp in d.groupby("conversation_id", sort=False):
            _cust_pos = _grp.index[_grp["speaker"] == "CUSTOMER"].tolist()
            _seen: set = set()
            for _pos in _cust_pos:
                _bg_set = _bigram_sets[_pos]
                if len(_bg_set & _seen) >= 2:
                    rep_mask[_pos] = True
                _seen |= _bg_set
        sig_c = rep_mask.astype(np.float32)

        # ── Final score — numpy weighted sum, no Python loop ─────────────────
        esc_score = np.where(
            is_cust,
            0.25 * sig_a + 0.40 * sig_b + 0.20 * sig_c + 0.15 * sig_d,
            0.0,
        ).astype(np.float32)

        # Signal type strings — built with numpy char ops, no Python loop
        _t = np.char.add
        sig_types = np.char.rstrip(
            _t(_t(_t(
                np.where(sig_a > 0, "sentiment_drop|", ""),
                np.where(sig_b > 0, "language|",       "")),
                np.where(sig_c > 0, "repetition|",     "")),
                np.where(sig_d > 0, "frustration_build|", "")),
            "|",
        )

        d["escalation_score"]       = esc_score
        d["escalation_signal_type"] = sig_types
        d["frustration_streak"]     = frust_streak
        d["potential_escalation"]   = esc_score >= 0.40

        # ── CHANGE 1: Hybrid resolution detection ────────────────────────────
        # Per conversation: compute resolution_status (4-way) and resolution_score
        RESOLVED_PHRASES = [
            "thank you","that worked","resolved","sorted","all set","appreciate",
            "great help","got it","perfect","that's all i needed","you've fixed",
            "problem solved","issue resolved","taken care","wonderful","excellent",
            "fixed it","glad we sorted","happy with","that helps","thank you so much",
            # From actual transcripts
            "glad to assist","glad i was able to help","glad i could help",
            "have a nice day","have a great day","have a wonderful day",
            "you're welcome","you are welcome","you're most welcome",
            "successfully cancelled","successfully updated","successfully processed",
            "successfully resolved","feel free to contact","will be all",
            "nothing else needed","no further questions","thanks a lot",
            "thanks very much","many thanks","appreciate your time",
            "all sorted out","everything is sorted","happy to help",
            "issue has been resolved","issue has been fixed","account is updated",
            "got that sorted","payment processed","appointment cancelled",
            "appointment rescheduled","subscription cancelled",
        ]
        UNRESOLVED_PHRASES = [
            "still not","hasn't been","not fixed","same problem","not resolved",
            "wait again","calling back","not working","still waiting","no one helped",
            "not happy","this is unacceptable","not satisfied","doesn't work",
            "still having","keep getting","continues to","nothing changed",
            "never resolved","same issue","back again","problem persists",
            # From actual transcripts
            "need a refund","want a refund","i want my money back","money back",
            "request a refund","want my money back","speak to a supervisor",
            "speak to supervisor","want to speak to manager","speak with a manager",
            "wasted my time","waste of time","wasting my time","kept me waiting",
            "on hold again","transferred again","keep transferring",
            "wrong information","misinformed","misleading information",
            "third time contacting","called again","no response","no reply",
            "still waiting for","extremely frustrated","very frustrated",
            "really frustrated","not getting anywhere","filing a complaint",
            "raised a complaint","going to complain",
        ]

        # Per-conversation resolution computation
        res_status_map: Dict[str, str]  = {}
        res_score_map:  Dict[str, float] = {}

        for cid, grp in d.groupby("conversation_id", sort=False):
            cust  = grp[grp["speaker"] == "CUSTOMER"]
            last3 = cust.tail(3)
            if last3.empty:
                res_status_map[cid] = "Unresolved"; res_score_map[cid] = 0.0; continue

            # A: Sentiment signal (30%) — avg compound in last 3 customer turns
            sent_avg  = float(last3["compound"].mean())
            sig_sent  = max(0.0, min(1.0, (sent_avg + 0.5)))   # map [-0.5,0.5] → [0,1]

            # B: Resolution language (50%)
            end_text  = " ".join(last3["cleaned_message"].fillna("").tolist()).lower()
            pos_hits  = sum(1 for p in RESOLVED_PHRASES   if p in end_text)
            neg_hits  = sum(1 for p in UNRESOLVED_PHRASES if p in end_text)
            sig_lang  = max(0.0, min(1.0, (pos_hits*0.3 - neg_hits*0.4 + 0.5)))

            # C: Outcome signal (20%) — no escalation in final 3 turns + no negative drop
            last3_esc = grp.tail(3)["potential_escalation"].any()
            final_drop = float(grp.tail(1)["sentiment_change"].values[0]) if len(grp) > 0 else 0
            sig_out   = 0.0 if last3_esc else (1.0 if final_drop >= -0.1 else 0.5)

            score = 0.30*sig_sent + 0.50*sig_lang + 0.20*sig_out

            # Has explicit unresolved language → cap score
            if neg_hits > 0 and neg_hits >= pos_hits:
                score = min(score, 0.29)

            res_score_map[cid] = round(score, 3)

            # 4-way status
            has_esc_end = grp.tail(3)["potential_escalation"].any()
            if has_esc_end and score < 0.35:
                res_status_map[cid] = "Escalated/Unrecovered"
            elif score >= 0.60:
                res_status_map[cid] = "Truly Resolved"
            elif score >= 0.30:
                res_status_map[cid] = "Partially Resolved"
            else:
                res_status_map[cid] = "Unresolved"

        d["resolution_score"]  = d["conversation_id"].map(res_score_map).fillna(0.0)
        d["resolution_status"] = d["conversation_id"].map(res_status_map).fillna("Unresolved")
        # Keep backward-compat potential_resolution flag (True = Truly Resolved)
        d["potential_resolution"] = d["resolution_status"] == "Truly Resolved"

        # ── CHANGE 2: Agent effectiveness (per-conversation) ─────────────────
        # For each AGENT turn: measure customer compound change in next 3 customer turns
        # Bucket per conversation: Improver / Stabiliser / Worsener
        agent_eff_map: Dict[str, str]   = {}
        agent_delta_map: Dict[str, float] = {}

        # Vectorised agent-effectiveness via sorted merge + rolling window.
        # For each agent turn: baseline = last customer compound before it,
        # after = mean of next 3 customer compounds.  No per-row Python loop.
        _cust_all  = d[d["speaker"] == "CUSTOMER"][["conversation_id", "turn_sequence", "compound"]].copy()
        _agent_all = d[d["speaker"] == "AGENT"][["conversation_id", "turn_sequence"]].copy()

        if not _agent_all.empty and not _cust_all.empty:
            # merge_asof requires the `on` key (turn_sequence) to be sorted globally
            _cust_all  = _cust_all.sort_values("turn_sequence")
            _agent_all = _agent_all.sort_values("turn_sequence")

            # baseline: last customer turn strictly before each agent turn
            _base = pd.merge_asof(
                _agent_all, _cust_all,
                on="turn_sequence", by="conversation_id",
                direction="backward",
                suffixes=("", "_cust"),
            ).rename(columns={"compound": "baseline"})

            # For the "after" side we need up to 3 customer turns *after* each
            # agent turn.  Expand customer turns with a rank within each group,
            # then merge_asof forward and keep rank ≤ 3.
            _cust_fwd = _cust_all.copy()
            _cust_fwd["_rank"] = _cust_fwd.groupby("conversation_id").cumcount()

            _after = pd.merge_asof(
                _agent_all, _cust_fwd,
                on="turn_sequence", by="conversation_id",
                direction="forward",
                suffixes=("", "_cust"),
            )
            # _after gives rank of the *first* forward customer turn per agent turn.
            # Expand: for each agent turn include ranks [first_rank, first_rank+2].
            _after = _after.dropna(subset=["_rank"])
            _after["_rank"] = _after["_rank"].astype(int)

            # Join back all customer rows whose rank falls in [first_rank, first_rank+2]
            _expanded = _after.merge(
                _cust_fwd[["conversation_id", "_rank", "compound"]].rename(
                    columns={"_rank": "_crank", "compound": "cust_compound"}),
                on="conversation_id",
            )
            _expanded = _expanded[
                (_expanded["_crank"] >= _expanded["_rank"]) &
                (_expanded["_crank"] <  _expanded["_rank"] + 3)
            ]
            _after_mean = (
                _expanded.groupby(["conversation_id", "turn_sequence"])["cust_compound"]
                .mean()
                .reset_index()
                .rename(columns={"cust_compound": "after_mean"})
            )

            # Combine baseline + after_mean
            _merged = _base.merge(_after_mean, on=["conversation_id", "turn_sequence"], how="inner")
            _merged = _merged.dropna(subset=["baseline", "after_mean"])
            _merged["delta"] = _merged["after_mean"] - _merged["baseline"]

            _conv_delta = _merged.groupby("conversation_id")["delta"].mean()
            for cid, avg_delta in _conv_delta.items():
                avg_delta = float(avg_delta)
                agent_delta_map[cid] = round(avg_delta, 3)
                if avg_delta > 0.10:
                    agent_eff_map[cid] = "Improver"
                elif avg_delta < -0.05:
                    agent_eff_map[cid] = "Worsener"
                else:
                    agent_eff_map[cid] = "Stabiliser"

        d["agent_effectiveness"]       = d["conversation_id"].map(agent_eff_map).fillna("Stabiliser")
        d["agent_customer_delta"]      = d["conversation_id"].map(agent_delta_map).fillna(0.0)

        gc.collect()
        return d

    def compute_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """All aggregations run through Polars expressions."""
        lf = pl.from_pandas(df).lazy()

        ins: Dict[str, Any] = {}
        ins["total_conversations"]        = int(df["conversation_id"].nunique())
        ins["total_turns"]                = int(len(df))
        ins["avg_turns_per_conversation"] = float(
            lf.group_by("conversation_id").agg(pl.len().alias("n"))
              .select(pl.col("n").mean()).collect().item()
        )

        overall = (
            lf.select([
                pl.col("compound").mean().alias("avg"),
                pl.col("compound").median().alias("med"),
                pl.col("compound").std().alias("std"),
            ]).collect().to_dicts()[0]
        )
        ins["overall_sentiment"] = {"average": overall["avg"] or 0.0,
                                     "median":  overall["med"] or 0.0,
                                     "std":     overall["std"] or 0.0}

        def _agg(mask_col, mask_val, cols):
            """Aggregate a filtered subset via Polars."""
            sub = lf.filter(pl.col(mask_col) == mask_val)
            result = sub.select([pl.col(c).mean().alias(c) for c in cols]).collect()
            if result.is_empty(): return {c: 0.0 for c in cols}
            row = result.to_dicts()[0]
            return {c: (float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0)
                    for c, v in row.items()}

        cu = _agg("speaker", "CUSTOMER", ["compound", "potential_escalation", "potential_resolution"])
        ag = _agg("speaker", "AGENT",    ["compound", "sentiment_change"])

        pos_pct = float(
            lf.filter(pl.col("speaker") == "CUSTOMER")
              .select((pl.col("sentiment_label") == "positive").mean())
              .collect().item() or 0.0
        )

        # ── Resolution rate from hybrid 4-way status ──────────────────────────
        n_convs = int(df["conversation_id"].nunique())
        res_counts = df.drop_duplicates("conversation_id")["resolution_status"].value_counts()                      if "resolution_status" in df.columns else pd.Series(dtype=int)
        truly_resolved   = int(res_counts.get("Truly Resolved",       0))
        partially_res    = int(res_counts.get("Partially Resolved",    0))
        unresolved_n     = int(res_counts.get("Unresolved",            0))
        esc_unrecovered  = int(res_counts.get("Escalated/Unrecovered", 0))
        # resolution_rate = Truly Resolved / total (strict) for KPI card
        hybrid_res_rate  = truly_resolved / max(n_convs, 1)

        ins["customer_satisfaction"] = {
            "average_sentiment":      cu.get("compound", 0.0),
            "positive_pct":           pos_pct,
            "escalation_rate":        cu.get("potential_escalation", 0.0),
            "resolution_rate":        hybrid_res_rate,
            # 4-way breakdown (used by new charts/tables)
            "truly_resolved":         truly_resolved,
            "partially_resolved":     partially_res,
            "unresolved":             unresolved_n,
            "escalated_unrecovered":  esc_unrecovered,
        }

        # ── Agent effectiveness from per-conversation buckets ─────────────────
        ag_std = float(
            lf.filter(pl.col("speaker") == "AGENT")
              .select(pl.col("compound").std()).collect().item() or 0.0
        )
        eff_counts = df.drop_duplicates("conversation_id")["agent_effectiveness"].value_counts()                      if "agent_effectiveness" in df.columns else pd.Series(dtype=int)
        n_c_eff = max(n_convs, 1)
        avg_delta = float(df.drop_duplicates("conversation_id")["agent_customer_delta"].mean())                     if "agent_customer_delta" in df.columns else 0.0

        ins["agent_performance"] = {
            "average_sentiment":      ag.get("compound", 0.0),
            "response_effectiveness": ag.get("sentiment_change", 0.0),
            "consistency_score":      max(0.0, 1.0 - ag_std),
            # New effectiveness distribution
            "improver_pct":           int(eff_counts.get("Improver",   0)) / n_c_eff,
            "stabiliser_pct":         int(eff_counts.get("Stabiliser", 0)) / n_c_eff,
            "worsener_pct":           int(eff_counts.get("Worsener",   0)) / n_c_eff,
            "avg_customer_delta":     round(avg_delta, 3),
        }

        phase_avgs = (
            lf.group_by("phase")
              .agg(pl.col("compound").mean().alias("avg"))
              .collect().to_dicts()
        )
        pa = {r["phase"]: r["avg"] for r in phase_avgs}
        ins["conversation_patterns"] = {
            "avg_sentiment_start":   pa.get("start",  0.0) or 0.0,
            "avg_sentiment_middle":  pa.get("middle", 0.0) or 0.0,
            "avg_sentiment_end":     pa.get("end",    0.0) or 0.0,
            "sentiment_improvement": (pa.get("end", 0.0) or 0.0) - (pa.get("start", 0.0) or 0.0),
        }

        # Phase CSAT/DSAT — customer turns only
        cust_lf = lf.filter(pl.col("speaker") == "CUSTOMER")
        phase_stats = (
            cust_lf.group_by("phase")
                   .agg([
                       pl.len().alias("count"),
                       pl.col("compound").mean().alias("avg_sentiment"),
                       (pl.col("compound") >= 0).sum().cast(pl.Float64).alias("csat_n"),
                       (pl.col("compound") <  0).sum().cast(pl.Float64).alias("dsat_n"),
                   ])
        )
        phase_stats = _safe_collect(phase_stats).to_dicts()
        pcd: Dict[str, Dict] = {}
        for row in phase_stats:
            t = row["count"] or 1
            pcd[row["phase"]] = {
                "csat_pct":      row["csat_n"] / t,
                "dsat_pct":      row["dsat_n"] / t,
                "avg_sentiment": row["avg_sentiment"] or 0.0,
                "count":         int(t),
            }
        for pn in ("start","middle","end"):
            pcd.setdefault(pn, {"csat_pct":0.0,"dsat_pct":0.0,"avg_sentiment":0.0,"count":0})
        ins["phase_csat_dsat"]  = pcd
        ins["recommendations"]  = self._recs(ins)
        return ins

    def _recs(self, ins):
        r=[]; cs=ins["customer_satisfaction"]; ap=ins["agent_performance"]
        cp=ins["conversation_patterns"]; pcd=ins.get("phase_csat_dsat",{})
        if cs["average_sentiment"]       < 0:    r.append("🔴 Customer sentiment below neutral — review agent training & scripts.")
        if cs["escalation_rate"]         > 0.15: r.append(f"⚠️ High escalation rate ({cs['escalation_rate']:.1%}) — train de-escalation techniques.")
        elif cs["escalation_rate"]       > 0.10: r.append(f"⚠️ Moderate escalation rate ({cs['escalation_rate']:.1%}) — monitor closely.")
        # Resolution: use hybrid rate
        if cs["resolution_rate"]         < 0.4:  r.append(f"🔴 True resolution rate only {cs['resolution_rate']:.1%} — review closing language & outcome verification.")
        esc_unr = cs.get("escalated_unrecovered", 0)
        if esc_unr > 0: r.append(f"🔴 {esc_unr} conversations escalated and never recovered — priority coaching needed.")
        # Agent effectiveness
        if ap.get("worsener_pct", 0) > 0.20: r.append(f"⚠️ {ap['worsener_pct']:.0%} of agents worsen customer sentiment — review response scripts.")
        if ap.get("improver_pct", 0) > 0.50: r.append(f"✅ {ap['improver_pct']:.0%} of agents improve customer sentiment — document best practices.")
        if ap["average_sentiment"]       < 0.1:  r.append("📚 Agent sentiment low — tone coaching recommended.")
        if cp["sentiment_improvement"]   < 0:    r.append("📉 Conversations end worse than they start — review resolution processes.")
        elif cp["sentiment_improvement"] > 0.2:  r.append("📈 Strong positive improvement — document & replicate best-practice behaviours.")
        mid=pcd.get("middle",{}); end=pcd.get("end",{}); start=pcd.get("start",{})
        if mid.get("dsat_pct",0) > 0.5:  r.append(f"⚠️ Mid-conversation DSAT {mid['dsat_pct']:.1%} — reduce handle time.")
        if end.get("dsat_pct",0) > 0.4:  r.append(f"🔴 End DSAT {end['dsat_pct']:.1%} — fix wrap-up process.")
        if start.get("csat_pct",0)>0.7 and end.get("dsat_pct",0)>0.3:
            r.append("📉 CRITICAL: Customers start satisfied but end dissatisfied.")
        if not r: r.append("✅ All key metrics within healthy ranges — maintain current practices.")
        return r


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE  — cached so UI interactions never recompute

def _file_checksum(file_bytes: bytes) -> str:
    """SHA-256 of file bytes — used as a stable, unique cache key."""
    return hashlib.sha256(file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def _cached_parse(checksum: str, fname: str, dataset_type: str,
                  file_bytes: bytes,
                  pii_enabled: bool = False,
                  pii_mode: str = "mask",
                  excel_sheet=0) -> pd.DataFrame:
    """
    Parse raw bytes → turns DataFrame.
    Cache key = SHA-256 checksum + filename + domain + PII settings.
    Using checksum (not raw bytes) as first arg guarantees Streamlit hashes
    a short string rather than a potentially huge bytes object — preventing
    false cache hits when the internal hash truncates large files.
    """
    # ── File reading — robust dtype handling ─────────────────────────────────
    # Excel: always use pandas + openpyxl with dtype=str so every column
    # (transcript text, IDs, timestamps) is read as a plain string.
    # This prevents Polars schema inference from truncating rows, mistyping
    # ISO timestamps as dates, or dropping multi-line cells.
    # CSV: Polars read_csv with infer_schema_length=0 (all cols as Utf8).
    if fname.endswith(".csv"):
        df_raw = pl.read_csv(
            io.BytesIO(file_bytes),
            infer_schema_length=0,          # all columns read as strings
            null_values=["","NA","N/A","null","NULL","None"],
            truncate_ragged_lines=True,     # tolerate jagged rows
        ).to_pandas()
    else:
        # pandas + openpyxl is the most reliable Excel reader for:
        #   • merged cells  • multi-line cells  • mixed-type columns
        #   • UUID / long-string ID columns  • large files
        # dtype=str prevents openpyxl from coercing timestamps or numbers.
        # keep_default_na=False stops it turning "NA" strings into NaN.
        df_raw = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=excel_sheet,     # user-selected sheet (default: first)
            dtype=str,                  # every column as string — no type coercion
            keep_default_na=False,      # preserve "NA", "null", etc. as strings
            engine="openpyxl",
        )
        # Strip whitespace from all string columns to avoid hidden parse failures
        for _c in df_raw.select_dtypes(include="object").columns:
            df_raw[_c] = df_raw[_c].str.strip()
    # ── Sanity check: warn if suspiciously few rows loaded ───────────────────
    # Catches silent truncation from schema mismatches or wrong sheet.
    if len(df_raw) < 2:
        raise ValueError(
            f"Only {len(df_raw)} row(s) were read from '{fname}'. "
            "The file may be empty, on the wrong sheet, or in an unsupported format. "
            "Check that data starts on row 1 with a header row."
        )

    proc  = ConversationProcessor(dataset_type=dataset_type)
    df_p  = proc.parse(df_raw)
    df_p.attrs["detected_format"] = proc.detected_format

    # ── PII redaction applied right after parsing, before scoring ──
    if pii_enabled:
        redact_cols = [c for c in ("message", "cleaned_message") if c in df_p.columns]
        df_p, n_redacted = PIIRedactor.redact_dataframe(df_p, redact_cols, mode=pii_mode)
        df_p.attrs["pii_redacted_rows"] = n_redacted
        df_p.attrs["pii_mode"]          = pii_mode
    else:
        df_p.attrs["pii_redacted_rows"] = 0
        df_p.attrs["pii_mode"]          = "off"

    return df_p


@st.cache_data(show_spinner=False)
def _cached_score(df_p: pd.DataFrame) -> pd.DataFrame:
    """
    Score turns with VADER — streaming chunk approach for OOM safety.

    For datasets > CHUNK_TURNS rows the DataFrame is split into batches
    of CHUNK_TURNS, scored independently, then concatenated.  Each batch
    is released from RAM before the next is allocated, capping peak memory
    to roughly 2× the size of one chunk rather than 2× the full dataset.
    """
    sent = SentimentEngine()
    sent.calibrate(df_p)

    n = len(df_p)
    if n <= CHUNK_TURNS:
        # Small dataset — score in one shot (original fast path)
        result = sent.score(df_p)
        gc.collect()
        return result

    # Large dataset — stream through CHUNK_TURNS batches
    parts: List[pd.DataFrame] = []
    for start in range(0, n, CHUNK_TURNS):
        chunk  = df_p.iloc[start : start + CHUNK_TURNS].copy()
        scored = sent.score(chunk)
        parts.append(scored)
        del chunk                  # free input chunk immediately
        gc.collect()               # force GC between batches

    result = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return result


@st.cache_data(show_spinner=False)
def _cached_analytics(df_s: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute turn metrics + insights — OOM-safe collect() strategy.

    Polars LazyFrame.collect() materialises the entire result at once.
    For 120k+ turns that can spike RAM beyond 2GB.  Solution: collect()
    conversation-level aggregates first (tiny), join them back lazily,
    then collect the enriched frame in streaming mode using
    _safe_collect() which uses Polars streaming engine row-by-row in Rust.
    """
    anal = AnalyticsEngine()
    df_r = anal.compute_turn_metrics(df_s)   # now uses streaming collect internally
    ins  = anal.compute_insights(df_r)
    gc.collect()
    return df_r, ins


def run_pipeline(
    file_bytes: bytes,
    fname: str,
    dataset_type: str,
    progress_bar=None,
    pii_enabled: bool = False,
    pii_mode: str = "mask",
    excel_sheet=0,
) -> Tuple[pd.DataFrame, Dict, str, Dict]:
    """
    Full pipeline — every stage independently cached.

    Cache behaviour
    ---------------
    _cached_parse    : busted when SHA-256 checksum OR domain OR PII settings change
    _cached_score    : busted when parsed DataFrame changes
    _cached_analytics: busted when scored DataFrame changes
    _precompute_aggs : busted when results DataFrame changes
    UI interactions  : recompute NOTHING (all stages already cached)
    """
    checksum = _file_checksum(file_bytes)

    # ── Stage 1: Parse ────────────────────────────────────────────────────────
    if progress_bar: progress_bar.progress(0.10, text="Parsing transcripts…")
    try:
        df_p     = _cached_parse(checksum, fname, dataset_type, file_bytes,
                                 pii_enabled=pii_enabled, pii_mode=pii_mode,
                                 excel_sheet=excel_sheet)
    except MemoryError:
        gc.collect()
        raise MemoryError(
            f"Out of memory while parsing '{fname}'. "
            "Try splitting the file into smaller batches (≤ 5,000 rows each)."
        )
    # ── Extract all attrs from df_p NOW before it is ever deleted ───────────
    detected        = df_p.attrs.get("detected_format", "—")
    _pii_redacted   = df_p.attrs.get("pii_redacted_rows", 0)
    _pii_mode_used  = df_p.attrs.get("pii_mode", "off")
    _skipped_rows   = df_p.attrs.get("skipped_rows", 0)

    if _skipped_rows > 0:
        _total_rows  = len(df_p) + _skipped_rows  # approximate source row count
        _skip_pct    = _skipped_rows / max(_total_rows, 1)
        if _skip_pct >= 0.10:
            st.warning(
                f"⚠️ {_skipped_rows:,} source rows ({_skip_pct:.0%}) produced no turns and were skipped. "
                "This usually means those rows are blank, too short, or don't match the selected format. "
                "Try switching the domain selector if you expect more data."
            )

    # ── Hard cap: enforce MAX_TURNS before any heavy computation ─────────────
    n_raw = len(df_p)
    if n_raw > MAX_TURNS:
        df_p = df_p.iloc[:MAX_TURNS].copy()
        gc.collect()
        _is_cloud = bool(os.environ.get("STREAMLIT_SHARING_MODE") or
                         os.environ.get("IS_STREAMLIT_CLOUD"))
        if _is_cloud:
            st.warning(
                f"⚠️ Dataset capped at {MAX_TURNS:,} turns (your file had {n_raw:,}). "
                "Streamlit Cloud is limited to 1 GB RAM. "
                "Run locally with `streamlit run tbt_app.py` to process the full dataset, "
                "or set the MAX_TURNS env variable to increase the cap."
            )
        else:
            st.info(
                f"ℹ️ Dataset capped at {MAX_TURNS:,} turns (your file had {n_raw:,}). "
                f"To raise the cap, run: MAX_TURNS={n_raw} streamlit run tbt_app.py"
            )

    # ── Soft warning: large dataset — tell user it may take a moment ──────────
    n_turns = len(df_p)
    if n_turns > 50_000:
        st.info(
            f"📊 Large dataset: {n_turns:,} turns. "
            "Scoring in streaming batches — this may take 1–2 minutes."
        )

    # ── Stage 2: Score (chunked streaming — OOM safe) ─────────────────────────
    if progress_bar: progress_bar.progress(0.35, text=f"Scoring {n_turns:,} turns in {max(1, n_turns//CHUNK_TURNS)} batch(es)…")
    try:
        df_s = _cached_score(df_p)
        del df_p          # safe to delete — all attrs already extracted above
        gc.collect()
    except MemoryError:
        gc.collect()
        raise MemoryError(
            f"Out of memory while scoring {n_turns:,} turns. "
            f"The dataset is too large for available RAM. "
            f"Try uploading ≤ {CHUNK_TURNS:,} rows at a time."
        )

    # ── Stage 3: Analytics (Polars streaming collect — OOM safe) ─────────────
    if progress_bar: progress_bar.progress(0.75, text="Computing analytics with Polars streaming…")
    try:
        df_r, ins = _cached_analytics(df_s)
        del df_s          # release scored frame after analytics
        gc.collect()
    except MemoryError:
        gc.collect()
        raise MemoryError(
            "Out of memory during analytics. "
            "Try reducing the dataset size or restarting the app."
        )

    # Carry PII audit metadata forward — uses pre-extracted values, not df_p
    pii_meta = {
        "enabled":       pii_enabled,
        "mode":          _pii_mode_used if pii_enabled else "off",
        "redacted_rows": _pii_redacted,
    }

    if progress_bar: progress_bar.progress(1.0, text="Done ✓")
    return df_r, ins, detected, pii_meta


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _to_excel(df: pd.DataFrame, ins_json: str) -> bytes:
    """Cached Excel build — ins_json is a stable string key."""
    ins = json.loads(ins_json)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="All Turns", index=False)
        cu=df[df["speaker"]=="CUSTOMER"]; ag=df[df["speaker"]=="AGENT"]
        if not cu.empty: cu.to_excel(w, sheet_name="Customer Turns", index=False)
        if not ag.empty: ag.to_excel(w, sheet_name="Agent Turns",   index=False)
        pcd=ins.get("phase_csat_dsat",{})
        rows=[
            {"Metric":"Total Conversations","Value":ins["total_conversations"]},
            {"Metric":"Total Turns",        "Value":ins["total_turns"]},
            {"Metric":"Avg Turns/Conv",     "Value":f"{ins['avg_turns_per_conversation']:.1f}"},
            {"Metric":"Overall Sentiment",  "Value":f"{ins['overall_sentiment']['average']:.3f}"},
            {"Metric":"Customer Avg",       "Value":f"{ins['customer_satisfaction']['average_sentiment']:.3f}"},
            {"Metric":"Agent Avg",          "Value":f"{ins['agent_performance']['average_sentiment']:.3f}"},
            {"Metric":"Escalation Rate",    "Value":_pct(ins["customer_satisfaction"]["escalation_rate"])},
            {"Metric":"Resolution Rate",    "Value":_pct(ins["customer_satisfaction"]["resolution_rate"])},
            {"Metric":"Sentiment Trend",    "Value":f"{ins['conversation_patterns']['sentiment_improvement']:.3f}"},
        ]
        for pn in ["start","middle","end"]:
            p=pcd.get(pn,{})
            rows+=[{"Metric":f"{pn.capitalize()} CSAT %","Value":_pct(p.get("csat_pct",0))},
                   {"Metric":f"{pn.capitalize()} DSAT %","Value":_pct(p.get("dsat_pct",0))},
                   {"Metric":f"{pn.capitalize()} Avg",   "Value":f"{p.get('avg_sentiment',0):.3f}"}]
        pd.DataFrame(rows).to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame(ins.get("recommendations",[])).to_excel(w, sheet_name="Recommendations",
                                                              index=False, header=False)
    return buf.getvalue()

def _to_csv(df: pd.DataFrame) -> str:
    buf=io.StringIO(); df.to_csv(buf,index=False); return buf.getvalue()

def _to_zip(df: pd.DataFrame, ins: Dict) -> bytes:
    ins_json = json.dumps(ins, indent=2, default=str)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("tbt_results.csv",   _to_csv(df))
        zf.writestr("tbt_insights.json", ins_json)
        zf.writestr("tbt_results.xlsx",  _to_excel(df, ins_json))
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTED AGGREGATES  — computed once, reused by all charts
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _precompute_aggs(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Run all heavy groupbys once via Polars and cache the results.
    Charts call these pre-aggregated frames instead of re-aggregating.
    """
    lf = pl.from_pandas(df).lazy()

    sent_dist = (
        lf.group_by("sentiment_label")
          .agg(pl.len().alias("count"))
          .collect().to_pandas()
    )

    conv_map = (
        lf.group_by("conversation_id")
          .agg([pl.col("compound").mean().alias("avg_sentiment"),
                pl.col("turn_sequence").max().alias("turns")])
          .collect().to_pandas()
    )

    phase_speaker = (
        lf.group_by(["speaker","phase","sentiment_label"])
          .agg(pl.len().alias("count"))
          .collect().to_pandas()
    )

    turn_prog = (
        lf.filter(pl.col("turn_sequence") <= 30)
          .group_by("turn_sequence")
          .agg(pl.col("compound").mean().alias("compound"))
          .sort("turn_sequence")
          .collect().to_pandas()
    )

    esc_res = pl.DataFrame({
        "event":  ["Escalations","Resolutions"],
        "count":  [int(df["potential_escalation"].sum()), int(df["potential_resolution"].sum())],
        "convs":  [int(df["conversation_id"].nunique())] * 2,
    }).to_pandas()

    # Subsample for scatter / sunburst to keep browser fast
    n = len(df)
    sample = df if n <= CHART_SAMPLE else df.sample(n=CHART_SAMPLE, random_state=42)

    # ── Sankey pre-aggregations ───────────────────────────────────────────────
    # 1. Phase-to-phase sentiment flow  (Start → Middle → End)
    #    For each conversation: dominant sentiment label per phase
    #    Then count conversation-level transitions between phases
    cust_df = pl.from_pandas(df[df["speaker"] == "CUSTOMER"].copy()).lazy()

    phase_dom = (
        cust_df
        .group_by(["conversation_id", "phase", "sentiment_label"])
        .agg(pl.len().alias("n"))
        .sort(["conversation_id", "phase", "n"], descending=[False, False, True])
        .group_by(["conversation_id", "phase"])
        .agg(pl.col("sentiment_label").first().alias("dominant"))
        .collect().to_pandas()
    )

    # Pivot: one row per conversation, columns = start/middle/end dominant
    phase_pivot = phase_dom.pivot_table(
        index="conversation_id", columns="phase",
        values="dominant", aggfunc="first"
    ).reset_index()
    for ph in ["start", "middle", "end"]:
        if ph not in phase_pivot.columns:
            phase_pivot[ph] = "neutral"
    phase_pivot = phase_pivot.fillna("neutral")

    # Start → Middle flow
    sm_flow = (
        phase_pivot.groupby(["start", "middle"])
        .size().reset_index(name="count")
        .rename(columns={"start": "source", "middle": "target"})
    )
    # Middle → End flow
    me_flow = (
        phase_pivot.groupby(["middle", "end"])
        .size().reset_index(name="count")
        .rename(columns={"middle": "source", "end": "target"})
    )
    # Start → End (direct arc — skip middle)
    se_flow = (
        phase_pivot.groupby(["start", "end"])
        .size().reset_index(name="count")
        .rename(columns={"start": "source", "end": "target"})
    )

    # 2. Outcome Flow Sankey: Start Sentiment → Resolution Status → End Sentiment
    #    Uses hybrid resolution_status column from compute_turn_metrics
    if "resolution_status" in df.columns:
        conv_summary = df.drop_duplicates("conversation_id").copy()

        # Start sentiment: dominant customer sentiment in start phase
        start_sent_df = (
            cust_df.filter(pl.col("phase") == "start")
              .group_by(["conversation_id","sentiment_label"])
              .agg(pl.len().alias("n"))
              .sort(["conversation_id","n"], descending=[False,True])
              .group_by("conversation_id")
              .agg(pl.col("sentiment_label").first().alias("start_sent"))
              .collect().to_pandas()
        )
        # End sentiment: dominant customer sentiment in end phase
        end_sent_df = (
            cust_df.filter(pl.col("phase") == "end")
              .group_by(["conversation_id","sentiment_label"])
              .agg(pl.len().alias("n"))
              .sort(["conversation_id","n"], descending=[False,True])
              .group_by("conversation_id")
              .agg(pl.col("sentiment_label").first().alias("end_sent"))
              .collect().to_pandas()
        )

        outcome_flow_df = (
            conv_summary[["conversation_id","resolution_status"]]
            .merge(start_sent_df, on="conversation_id", how="left")
            .merge(end_sent_df,   on="conversation_id", how="left")
            .fillna({"start_sent": "neutral", "end_sent": "neutral",
                     "resolution_status": "Unresolved"})
        )
    else:
        outcome_flow_df = pd.DataFrame(
            columns=["conversation_id","resolution_status","start_sent","end_sent"]
        )

    # Speaker-sentiment Sankey: Speaker → Phase → Sentiment (kept for internal use)
    spk_phase_sent = (
        lf.group_by(["speaker", "phase", "sentiment_label"])
          .agg(pl.len().alias("count"))
          .collect().to_pandas()
    )

    # 3. Turn-by-Turn Sentiment Transitions: consecutive sentiment label pairs
    #    For each turn, pair its label with the next turn's label within the same conversation
    tf_pl = (
        lf.sort(["conversation_id", "turn_sequence"])
          .with_columns([
              pl.col("sentiment_label").shift(-1).over("conversation_id").alias("_next_label"),
              pl.col("speaker").alias("_spk"),
          ])
          .filter(pl.col("_next_label").is_not_null())
          .group_by(["_spk", "sentiment_label", "_next_label"])
          .agg(pl.len().alias("count"))
          .rename({"sentiment_label": "source", "_next_label": "target", "_spk": "speaker"})
          .collect().to_pandas()
    )

    return {
        "sent_dist":       sent_dist,
        "conv_map":        conv_map,
        "phase_speaker":   phase_speaker,
        "turn_prog":       turn_prog,
        "esc_res":         esc_res,
        "sample":          sample,
        # Phase Sankey data (retained)
        "sm_flow":         sm_flow,
        "me_flow":         me_flow,
        "se_flow":         se_flow,
        "phase_pivot":     phase_pivot,
        "spk_phase_sent":  spk_phase_sent,
        # Turn-by-Turn Transitions Sankey
        "turn_flow":       tf_pl,
        # Outcome Flow Sankey (new)
        "outcome_flow_df": outcome_flow_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART FACTORIES  — all use pre-aggregated data
# ─────────────────────────────────────────────────────────────────────────────
def _chart_sentiment_dist(aggs):
    cnt=aggs["sent_dist"]
    fig=px.bar(cnt,x="sentiment_label",y="count",color="sentiment_label",
               color_discrete_map={"positive":C['pos'],"neutral":C['neu'],"negative":C['neg']},
               title="Sentiment Distribution",labels={"sentiment_label":"","count":"Turns"})
    fig.update_traces(marker_line_width=0)
    return apply_chart(fig.update_layout(showlegend=False,title_font_size=14))

def _chart_speaker_box(df):
    fig=go.Figure()
    for role,color in [("CUSTOMER",C['neg']),("AGENT",C['teal'])]:
        sub=df[df["speaker"]==role]["compound"]
        if not sub.empty:
            fig.add_trace(go.Box(y=sub,name=role.capitalize(),
                                 marker_color=color,boxpoints="outliers",line_color=color))
    return apply_chart(fig.update_layout(title="Customer vs Agent Sentiment",title_font_size=14))

def _chart_phase_comparison(ins):
    pcd=ins.get("phase_csat_dsat",{}); phases=["Start","Middle","End"]
    fig=go.Figure(data=[
        go.Bar(name="CSAT %",x=phases,
               y=[pcd.get(p.lower(),{}).get("csat_pct",0)*100 for p in phases],
               marker_color=C['pos'],marker_line_width=0),
        go.Bar(name="DSAT %",x=phases,
               y=[pcd.get(p.lower(),{}).get("dsat_pct",0)*100 for p in phases],
               marker_color=C['neg'],marker_line_width=0),
    ])
    return apply_chart(fig.update_layout(barmode="group",title="CSAT vs DSAT by Phase",
        title_font_size=14,yaxis=dict(title="% Customer Turns"),
        margin=dict(l=10,r=20,t=60,b=10),
        legend=dict(orientation="h",yanchor="top",y=1.0,xanchor="right",x=1,
                    bgcolor="rgba(0,0,0,0)")))

def _chart_sentiment_progression(aggs):
    tp=aggs["turn_prog"]
    fig=go.Figure()
    fig.add_hline(y=0,line_dash="dash",line_color=C['warm'])
    fig.add_trace(go.Scatter(x=tp["turn_sequence"],y=tp["compound"],mode="lines+markers",
        line=dict(color=C['teal'],width=2.5),marker=dict(size=6,color=C['teal']),
        fill="tozeroy",fillcolor="rgba(45,95,110,0.1)"))
    return apply_chart(fig.update_layout(title="Avg Sentiment by Turn (first 30)",
        title_font_size=14,xaxis=dict(title="Turn"),yaxis=dict(title="Avg Score")))

# ─────────────────────────────────────────────────────────────────────────────
# SANKEY CHART FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette for sentiment labels — shared across all Sankey charts
_SENT_HEX = {
    "positive": C['pos'],   # green
    "neutral":  C['neu'],   # blue
    "negative": C['neg'],   # red
}
_PHASE_HEX = {
    "start":  "#2D5F6E",    # teal
    "middle": "#D4B94E",    # gold
    "end":    "#A04040",    # red-brown
}

def _sankey_node_color(label: str) -> str:
    label_l = label.lower()
    for k, v in _SENT_HEX.items():
        if k in label_l: return v
    for k, v in _PHASE_HEX.items():
        if k in label_l: return v
    return C['slate']

def _build_sankey(labels, sources, targets, values, colors_node,
                  colors_link, title: str, height: int = 560) -> go.Figure:
    """Core Sankey builder — all 4 charts call this."""
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=30, thickness=26,
            line=dict(color=C['border'], width=0.5),
            label=labels,
            color=colors_node,
            hovertemplate="<b>%{label}</b><br>Total flow: %{value:,}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors_link,
            hovertemplate=(
                "<b>%{source.label}</b> → <b>%{target.label}</b><br>"
                "Conversations: <b>%{value:,}</b><extra></extra>"
            ),
        ),
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, family="DM Sans", color=C['text']),
            x=0,
        ),
        height=height,
        margin=dict(l=20, r=20, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=C['text'], size=14),
    )
    return fig


@st.cache_data(show_spinner=False)
def _chart_sankey_phase_flow(aggs: dict) -> go.Figure:
    """
    Sankey 1 — Phase Sentiment Flow  (Start → Middle → End)
    --------------------------------------------------------
    Each conversation is classified by its dominant customer sentiment
    per phase (positive / neutral / negative).
    Flows show how many conversations moved through each sentiment
    state as the call progressed.
    """
    sm  = aggs["sm_flow"]   # start→middle
    me  = aggs["me_flow"]   # middle→end

    PHASES  = ["start",  "middle", "end"]
    SENTIMS = ["positive", "neutral", "negative"]

    # Build unique node list:  "Start Positive", "Middle Neutral", etc.
    node_labels = []
    node_idx    = {}
    for ph in PHASES:
        for s in SENTIMS:
            lbl = f"{ph.capitalize()} · {s.capitalize()}"
            node_idx[(ph, s)] = len(node_labels)
            node_labels.append(lbl)

    node_colors = [_sankey_node_color(lbl) for lbl in node_labels]

    srcs, tgts, vals, link_colors = [], [], [], []

    def _add_flow(flow_df, src_phase, tgt_phase):
        for _, row in flow_df.iterrows():
            s = str(row["source"]).lower()
            t = str(row["target"]).lower()
            if s not in SENTIMS: s = "neutral"
            if t not in SENTIMS: t = "neutral"
            v = int(row["count"])
            if v <= 0: continue
            srcs.append(node_idx[(src_phase, s)])
            tgts.append(node_idx[(tgt_phase, t)])
            vals.append(v)
            # Link colour = source node colour at 40% opacity
            hex_c = _SENT_HEX.get(s, C['slate'])
            r,g,b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
            link_colors.append(f"rgba({r},{g},{b},0.35)")

    _add_flow(sm, "start", "middle")
    _add_flow(me, "middle", "end")

    if not vals:
        fig = go.Figure()
        fig.add_annotation(text="No data — need Start, Middle & End phases",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    return _build_sankey(node_labels, srcs, tgts, vals, node_colors, link_colors,
                         title="🌊 Sentiment Flow: Start → Middle → End  (customer dominant sentiment per phase)")


@st.cache_data(show_spinner=False)
def _chart_sankey_turn_transitions(aggs: dict, speaker_filter: str = "ALL") -> go.Figure:
    """
    Sankey 2 — Turn-by-Turn Sentiment Transitions
    -----------------------------------------------
    Shows how many turns moved between each pair of sentiment labels
    (positive↔neutral↔negative) on consecutive turns within a conversation.
    Filterable by speaker (ALL / CUSTOMER / AGENT).
    """
    tf = aggs["turn_flow"].copy()
    if speaker_filter != "ALL":
        tf = tf[tf["speaker"] == speaker_filter]

    tf = tf.groupby(["source", "target"], as_index=False)["count"].sum()
    tf = tf[tf["count"] > 0]

    SENTIMS = ["positive", "neutral", "negative"]
    # Source nodes on left, target nodes on right — prefix to keep them separate
    src_labels = [f"{s.capitalize()} (from)" for s in SENTIMS]
    tgt_labels = [f"{s.capitalize()} (to)"   for s in SENTIMS]
    all_labels = src_labels + tgt_labels

    node_colors = [_SENT_HEX.get(s, C['slate']) for s in SENTIMS] * 2

    srcs, tgts, vals, link_colors = [], [], [], []
    for _, row in tf.iterrows():
        s = str(row["source"]).lower()
        t = str(row["target"]).lower()
        if s not in SENTIMS or t not in SENTIMS: continue
        v = int(row["count"])
        si = SENTIMS.index(s)
        ti = len(SENTIMS) + SENTIMS.index(t)
        srcs.append(si); tgts.append(ti); vals.append(v)
        hex_c = _SENT_HEX.get(s, C['slate'])
        r,g,b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
        link_colors.append(f"rgba({r},{g},{b},0.35)")

    if not vals:
        fig = go.Figure()
        fig.add_annotation(text="No transition data available",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    spk_label = f"({speaker_filter})" if speaker_filter != "ALL" else "(All Speakers)"
    return _build_sankey(all_labels, srcs, tgts, vals, node_colors, link_colors,
                         title=f"🔀 Turn-by-Turn Sentiment Transitions {spk_label}  (consecutive turns within conversations)")


@st.cache_data(show_spinner=False)
def _chart_sankey_speaker_journey(aggs: dict) -> go.Figure:
    """
    Sankey 3 — Speaker → Phase → Sentiment
    ----------------------------------------
    Three-level hierarchy: who spoke → which phase → what sentiment.
    Reveals whether AGENT or CUSTOMER drives negativity at each phase.
    """
    sps = aggs["spk_phase_sent"].copy()
    sps = sps[sps["count"] > 0]

    speakers = ["CUSTOMER", "AGENT"]
    phases   = ["start", "middle", "end"]
    sentims  = ["positive", "neutral", "negative"]

    # Node layout: [speakers] → [phase buckets per speaker] → [sentiment sinks]
    labels = []
    idx    = {}
    # Layer 1: speakers
    for spk in speakers:
        idx[("spk", spk)] = len(labels)
        labels.append(spk.capitalize())
    # Layer 2: speaker × phase
    for spk in speakers:
        for ph in phases:
            idx[("spk_ph", spk, ph)] = len(labels)
            labels.append(f"{spk.capitalize()} · {ph.capitalize()}")
    # Layer 3: sentiment sinks
    for s in sentims:
        idx[("sent", s)] = len(labels)
        labels.append(s.capitalize())

    node_colors = (
        [_PHASE_HEX.get("start", C['teal'])] * len(speakers) +
        [_sankey_node_color(f"{ph}") for spk in speakers for ph in phases] +
        [_SENT_HEX.get(s, C['slate']) for s in sentims]
    )

    srcs, tgts, vals, link_colors = [], [], [], []

    for _, row in sps.iterrows():
        spk = str(row["speaker"]).upper()
        ph  = str(row["phase"]).lower()
        s   = str(row["sentiment_label"]).lower()
        v   = int(row["count"])
        if spk not in speakers or ph not in phases or s not in sentims: continue

        # Link 1: speaker → speaker×phase
        srcs.append(idx[("spk",    spk)])
        tgts.append(idx[("spk_ph", spk, ph)])
        vals.append(v)
        link_colors.append("rgba(45,95,110,0.25)")

        # Link 2: speaker×phase → sentiment
        srcs.append(idx[("spk_ph", spk, ph)])
        tgts.append(idx[("sent",   s)])
        vals.append(v)
        hex_c = _SENT_HEX.get(s, C['slate'])
        r,g,b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
        link_colors.append(f"rgba({r},{g},{b},0.30)")

    if not vals:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    return _build_sankey(labels, srcs, tgts, vals, node_colors, link_colors,
                         title="👥 Speaker → Phase → Sentiment  (who drives what sentiment, when)",
                         height=640)


@st.cache_data(show_spinner=False)
def _chart_sankey_start_to_end(aggs: dict) -> go.Figure:
    """
    Sankey 4 — Direct Start → End Sentiment  (skip middle)
    --------------------------------------------------------
    Answers the key business question:
    "Of conversations that started positive, how many ended negative?"
    Shows every Start sentiment → End sentiment pairing.
    """
    se = aggs["se_flow"].copy()
    se = se[se["count"] > 0]

    SENTIMS = ["positive", "neutral", "negative"]
    src_labels = [f"Start {s.capitalize()}" for s in SENTIMS]
    tgt_labels = [f"End {s.capitalize()}"   for s in SENTIMS]
    all_labels = src_labels + tgt_labels

    node_colors = [_SENT_HEX.get(s, C['slate']) for s in SENTIMS] * 2

    srcs, tgts, vals, link_colors = [], [], [], []
    for _, row in se.iterrows():
        s = str(row["source"]).lower()
        t = str(row["target"]).lower()
        if s not in SENTIMS or t not in SENTIMS: continue
        v = int(row["count"])
        si = SENTIMS.index(s)
        ti = len(SENTIMS) + SENTIMS.index(t)
        srcs.append(si); tgts.append(ti); vals.append(v)
        hex_c = _SENT_HEX.get(s, C['slate'])
        r,g,b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
        link_colors.append(f"rgba({r},{g},{b},0.38)")

    if not vals:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    return _build_sankey(all_labels, srcs, tgts, vals, node_colors, link_colors,
                         title="🎯 Start → End Sentiment  (how conversations open vs. close — skips middle)")

@st.cache_data(show_spinner=False)
def _chart_sankey_outcome_flow(aggs: dict) -> go.Figure:
    """
    Sankey 3 — Outcome Flow: Start Sentiment → Resolution Status → End Sentiment
    ---------------------------------------------------------------------------
    Shows the real business outcome journey for each conversation.
    Resolution Status uses the hybrid 4-way classification:
      Truly Resolved / Partially Resolved / Unresolved / Escalated+Unrecovered
    """
    df = aggs.get("outcome_flow_df", pd.DataFrame())
    if df.empty or "resolution_status" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No resolution status data — re-run analysis",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    SENTIMS = ["positive", "neutral", "negative"]
    STATUSES = ["Truly Resolved", "Partially Resolved", "Unresolved", "Escalated/Unrecovered"]

    # Node layout: [Start Sentiment] → [Resolution Status] → [End Sentiment]
    labels, idx = [], {}
    for s in SENTIMS:
        idx[("start", s)] = len(labels)
        labels.append(f"Start {s.capitalize()}")
    for st in STATUSES:
        idx[("res", st)] = len(labels)
        labels.append(st)
    for s in SENTIMS:
        idx[("end", s)] = len(labels)
        labels.append(f"End {s.capitalize()}")

    STATUS_COLORS = {
        "Truly Resolved":         C["pos"],
        "Partially Resolved":     C["gold"],
        "Unresolved":             C["warn"],
        "Escalated/Unrecovered":  C["neg"],
    }
    SENT_COLORS = {
        "positive": C["pos"],
        "neutral":  C["neu"],
        "negative": C["neg"],
    }
    node_colors = (
        [SENT_COLORS.get(s, C["slate"]) for s in SENTIMS] +
        [STATUS_COLORS.get(st, C["slate"]) for st in STATUSES] +
        [SENT_COLORS.get(s, C["slate"]) for s in SENTIMS]
    )

    srcs, tgts, vals, link_colors = [], [], [], []

    # Layer 1: Start → Resolution Status
    for (start_s, res_s), grp in df.groupby(["start_sent", "resolution_status"]):
        start_s = str(start_s).lower()
        if start_s not in SENTIMS or res_s not in STATUSES: continue
        v = len(grp)
        if v <= 0: continue
        srcs.append(idx[("start", start_s)])
        tgts.append(idx[("res",   res_s)])
        vals.append(v)
        hx = SENT_COLORS.get(start_s, C["slate"])
        r,g,b = int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16)
        link_colors.append(f"rgba({r},{g},{b},0.30)")

    # Layer 2: Resolution Status → End Sentiment
    for (res_s, end_s), grp in df.groupby(["resolution_status", "end_sent"]):
        end_s = str(end_s).lower()
        if res_s not in STATUSES or end_s not in SENTIMS: continue
        v = len(grp)
        if v <= 0: continue
        srcs.append(idx[("res", res_s)])
        tgts.append(idx[("end", end_s)])
        vals.append(v)
        hx = STATUS_COLORS.get(res_s, C["slate"])
        r,g,b = int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16)
        link_colors.append(f"rgba({r},{g},{b},0.28)")

    if not vals:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for Outcome Flow chart",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=32, thickness=26,
            line=dict(color=C["border"], width=0.5),
            label=labels,
            color=node_colors,
            hovertemplate="<b>%{label}</b><br>Conversations: %{value:,}<extra></extra>",
        ),
        link=dict(
            source=srcs, target=tgts, value=vals, color=link_colors,
            hovertemplate=(
                "<b>%{source.label}</b> → <b>%{target.label}</b><br>"
                "Conversations: <b>%{value:,}</b><extra></extra>"
            ),
        ),
    ))
    fig.update_layout(
        title=dict(
            text="<b>🎯 Outcome Flow: Start Sentiment → Resolution Status → End Sentiment</b>",
            font=dict(size=15, family="DM Sans", color=C["text"]), x=0,
        ),
        height=640,
        margin=dict(l=20, r=20, t=58, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=C["text"], size=14),
    )
    return fig


def _chart_escalation_resolution(aggs):
    er=aggs["esc_res"]; tot=er["convs"].iloc[0]
    fig=go.Figure(go.Bar(x=er["event"],y=er["count"],
        marker_color=[C['neg'],C['pos']],marker_line_width=0,
        text=[f"{int(v)} ({int(v)/tot:.0%})" for v in er["count"]],textposition="auto"))
    return apply_chart(fig.update_layout(title="Escalation & Resolution Events",
        title_font_size=14,showlegend=False))

def _chart_conv_scatter(aggs):
    """Scatter on pre-sampled conv_map — ≤2k points."""
    cm=aggs["conv_map"]
    n=len(cm); sample=cm if n<=CHART_SAMPLE else cm.sample(n=CHART_SAMPLE,random_state=42)
    fig=px.scatter(sample,x="turns",y="avg_sentiment",color="avg_sentiment",
        color_continuous_scale="RdYlGn",range_color=[-1,1],
        hover_name="conversation_id",
        title=f"Conversation Map  (showing {len(sample):,} of {n:,})",
        labels={"turns":"Turns","avg_sentiment":"Avg Sentiment"})
    fig.update_coloraxes(colorbar=dict(thickness=10))
    return apply_chart(fig.update_layout(title_font_size=14))

def _chart_sunburst(aggs):
    """Sunburst on pre-aggregated phase_speaker counts — no raw data needed."""
    grp=aggs["phase_speaker"]
    fig=px.sunburst(grp,path=["speaker","phase","sentiment_label"],values="count",
        color="sentiment_label",
        color_discrete_map={"positive":C['pos'],"negative":C['neg'],"neutral":C['neu']},
        title="Conversation Breakdown  (Speaker → Phase → Sentiment)")
    fig.update_traces(textfont=dict(family="DM Sans",size=11),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percentParent:.1%}<extra></extra>")
    return apply_chart(fig.update_layout(title_font_size=14,margin=dict(l=10,r=10,t=45,b=10)))

def _chart_waterfall(ins):
    cp=ins["conversation_patterns"]
    vals=[cp["avg_sentiment_start"],
          cp["avg_sentiment_middle"]-cp["avg_sentiment_start"],
          cp["avg_sentiment_end"]  -cp["avg_sentiment_middle"]]
    fig=go.Figure(go.Waterfall(
        orientation="v",measure=["absolute","relative","relative"],
        x=["Start","Middle → Δ","End → Δ"],y=vals,
        connector=dict(line=dict(color=C['border'],width=1)),
        increasing=dict(marker_color=C['pos']),
        decreasing=dict(marker_color=C['neg']),
        text=[f"{v:+.3f}" for v in vals],textposition="outside"))
    return apply_chart(fig.update_layout(title="Sentiment Journey  (Start → Middle → End)",
        title_font_size=14,showlegend=False,yaxis=dict(title="Avg Score")))

def _chart_escalation_timeline(aggs):
    """Scatter of escalation events — uses pre-sampled df."""
    esc=aggs["sample"]
    esc=esc[esc["potential_escalation"]] if "potential_escalation" in esc.columns else pd.DataFrame()
    if esc.empty: return None
    fig=px.strip(esc,x="turn_position",y="conversation_id",color="speaker",
        color_discrete_map={"CUSTOMER":C['neg'],"AGENT":C['teal']},
        hover_data={"message":True,"compound":True},
        title="Escalation Event Map  (turn position within conversation)")
    fig.update_traces(jitter=0.4,marker_size=8)
    return apply_chart(fig.update_layout(title_font_size=14,
        xaxis=dict(title="Position (0=start, 1=end)"),yaxis=dict(title="")))

def _chart_tbt_flow(df, conv_id, show_speaker_lines: bool = True):
    """
    Turn-by-Turn flow chart.

    When show_speaker_lines=True: stacked lines per speaker (Customer teal,
    Agent red) instead of a single combined line — reveals divergence clearly.
    Escalation turns get vertical red dashed marker lines.
    Phase bands always rendered.
    """
    sub = df[df["conversation_id"] == conv_id].sort_values("turn_sequence")
    if sub.empty: return go.Figure()

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color=C['warm'], line_width=1)

    if show_speaker_lines:
        # ── Stacked speaker lines ──
        for role, color, dash in [
            ("CUSTOMER", C['neg'],  "solid"),
            ("AGENT",    C['teal'], "dot"),
        ]:
            rs = sub[sub["speaker"] == role]
            if rs.empty: continue
            fig.add_trace(go.Scatter(
                x=rs["turn_sequence"], y=rs["compound"],
                mode="lines+markers",
                name=role.capitalize(),
                line=dict(color=color, width=2.5, dash=dash),
                marker=dict(size=8, color=color,
                            line=dict(width=1.5, color="white")),
                text=[f"Turn {r.turn_sequence}<br>{r.speaker}<br>"
                      f"{r.message[:55]}…" if len(r.message)>55
                      else f"Turn {r.turn_sequence}<br>{r.speaker}<br>{r.message}"
                      for _, r in rs.iterrows()],
                hovertemplate="%{text}<br>Score: %{y:.3f}<extra></extra>",
            ))
    else:
        # ── Single combined line (original style) ──
        fig.add_trace(go.Scatter(
            x=sub["turn_sequence"], y=sub["compound"],
            mode="lines+markers", name="All turns",
            line=dict(color=C['teal'], width=2.5),
            marker=dict(size=9, color=sub["compound"],
                        colorscale="RdYlGn", cmin=-1, cmax=1,
                        showscale=True, colorbar=dict(thickness=10, title="Score")),
            text=[f"Turn {r.turn_sequence}<br>{r.speaker}<br>"
                  f"{r.message[:55]}…" if len(r.message)>55
                  else f"Turn {r.turn_sequence}<br>{r.speaker}<br>{r.message}"
                  for _, r in sub.iterrows()],
            hovertemplate="%{text}<br>Score: %{y:.3f}<extra></extra>",
        ))

    # ── Phase bands ──
    mt = int(sub["turn_sequence"].max())
    for pn, (s, e, col) in {
        "start":  (1, 3,            "rgba(45,95,110,0.08)"),
        "middle": (4, max(4, mt-3), "rgba(212,185,78,0.06)"),
        "end":    (max(4, mt-2), mt, "rgba(160,64,64,0.08)"),
    }.items():
        if s <= e:
            fig.add_vrect(x0=s-.5, x1=e+.5, fillcolor=col, line_width=0,
                          annotation_text=PHASE_ICONS[pn],
                          annotation_position="top left")

    # ── Escalation markers — vertical red dashed lines ──
    esc_turns = sub[sub["potential_escalation"] == True]["turn_sequence"].tolist()
    for t in esc_turns:
        fig.add_vline(
            x=t, line_dash="dash", line_color=C['neg'], line_width=1.5,
            annotation_text="⚠️", annotation_position="top right",
            annotation_font_size=11,
        )

    # ── Resolution markers — vertical green dashed lines ──
    res_turns = sub[sub["potential_resolution"] == True]["turn_sequence"].tolist()
    for t in res_turns:
        fig.add_vline(
            x=t, line_dash="dot", line_color=C['pos'], line_width=1.5,
            annotation_text="✅", annotation_position="bottom right",
            annotation_font_size=11,
        )

    return apply_chart(fig.update_layout(
        title=dict(
            text=f"Turn-by-Turn Flow — {conv_id}",
            font=dict(size=14),
            x=0, y=0.98, xanchor="left", yanchor="top",
        ),
        margin=dict(l=10, r=20, t=55, b=10),
        xaxis=dict(title="Turn"),
        yaxis=dict(title="Sentiment Score", range=[-1.1, 1.1]),
        legend=dict(
            orientation="v",              # vertical list — sits in corner, never near title
            yanchor="top",   y=0.98,      # top-right inside plot area
            xanchor="right", x=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor=C['border'],
            borderwidth=1,
            font=dict(size=11),
        ),
    ))

def _chart_momentum(df, conv_id):
    sub=df[df["conversation_id"]==conv_id].sort_values("turn_sequence")
    colors=[C['pos'] if v>=0 else C['neg'] for v in sub["sentiment_momentum"]]
    fig=go.Figure(go.Bar(x=sub["turn_sequence"],y=sub["sentiment_momentum"],
        marker_color=colors,marker_line_width=0,
        hovertemplate="Turn %{x}<br>Momentum: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0,line_dash="dash",line_color=C['warm'])
    return apply_chart(fig.update_layout(title=f"Sentiment Momentum — {conv_id}",
        title_font_size=14,showlegend=False,
        xaxis=dict(title="Turn"),yaxis=dict(title="Momentum")))

def _chart_speaker_heatmap(df, conv_id):
    """Speaker × Phase heatmap — avg sentiment + turn count overlay."""
    sub   = df[df["conversation_id"] == conv_id]
    grp   = sub.groupby(["speaker", "phase"])
    avg   = grp["compound"].mean().unstack(fill_value=0).reindex(columns=["start","middle","end"], fill_value=0)
    cnt   = grp.size().unstack(fill_value=0).reindex(columns=["start","middle","end"], fill_value=0)

    # Text: sentiment avg + turn count underneath
    text = [[f"{avg.loc[spk, ph]:+.2f}<br><span style='font-size:10px'>({int(cnt.loc[spk, ph])} turns)</span>"
             if spk in avg.index and ph in avg.columns else ""
             for ph in ["start","middle","end"]]
            for spk in avg.index]

    fig = go.Figure(go.Heatmap(
        z=avg.values, x=["Start","Middle","End"], y=avg.index.tolist(),
        colorscale="RdYlGn", zmin=-1, zmax=1,
        text=[[f"{avg.values[i][j]:+.2f} ({int(cnt.values[i][j])} turns)"
               for j in range(3)] for i in range(len(avg))],
        texttemplate="%{text}",
        showscale=True, colorbar=dict(thickness=10), xgap=3, ygap=3,
    ))
    return apply_chart(fig.update_layout(
        title=f"Speaker × Phase Heatmap — {conv_id}  (avg score · turn count)",
        title_font_size=14, xaxis=dict(title="Phase"), yaxis=dict(title=""),
    ))


def _chart_compare_two(df: pd.DataFrame, conv_a: str, conv_b: str) -> go.Figure:
    """
    Side-by-side sentiment trajectory for two conversations on the same axes.
    Uses turn_position (0–1) so different-length conversations align.
    """
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color=C['warm'], line_width=1)

    palette = [(conv_a, C['teal'], "solid"), (conv_b, C['gold'], "dash")]
    for cid, color, dash in palette:
        sub = df[df["conversation_id"] == cid].sort_values("turn_sequence")
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["turn_position"], y=sub["compound"],
            mode="lines+markers", name=cid,
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=7, color=color),
            hovertemplate=f"<b>{cid}</b><br>Position: %{{x:.2f}}<br>Score: %{{y:.3f}}<extra></extra>",
        ))

    return apply_chart(fig.update_layout(
        title=dict(
            text=f"Conversation Comparison — {conv_a} vs {conv_b}",
            font=dict(size=14),
            x=0, y=0.98, xanchor="left", yanchor="top",
        ),
        margin=dict(l=10, r=20, t=55, b=10),
        xaxis=dict(title="Turn Position (0=start, 1=end)"),
        yaxis=dict(title="Sentiment Score", range=[-1.1, 1.1]),
        legend=dict(
            orientation="v",              # vertical list inside plot — never overlaps title
            yanchor="top",   y=0.98,      # top-right corner of plot area
            xanchor="right", x=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor=C['border'],
            borderwidth=1,
            font=dict(size=11),
        ),
    ))


def _chart_replay_animation(df: pd.DataFrame, conv_id: str) -> go.Figure:
    """
    Animated 'replay' of a conversation turn by turn using Plotly frames.
    Each frame adds one more turn to a cumulative line so users can watch
    sentiment evolve step by step with the Play button.
    """
    sub = df[df["conversation_id"] == conv_id].sort_values("turn_sequence").reset_index(drop=True)
    if sub.empty or len(sub) < 2: return go.Figure()

    turns = sub["turn_sequence"].tolist()
    scores = sub["compound"].tolist()

    # Build one frame per turn (cumulative)
    frames = []
    for i in range(1, len(turns) + 1):
        frames.append(go.Frame(
            data=[go.Scatter(
                x=turns[:i], y=scores[:i],
                mode="lines+markers",
                line=dict(color=C['teal'], width=2.5),
                marker=dict(
                    size=9,
                    color=scores[:i],
                    colorscale="RdYlGn", cmin=-1, cmax=1,
                ),
            )],
            name=str(turns[i-1]),
        ))

    fig = go.Figure(
        data=[go.Scatter(x=turns[:1], y=scores[:1], mode="lines+markers",
                         line=dict(color=C['teal'], width=2.5),
                         marker=dict(size=9, color=scores[:1],
                                     colorscale="RdYlGn", cmin=-1, cmax=1,
                                     showscale=True, colorbar=dict(thickness=10)))],
        frames=frames,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=C['warm'], line_width=1)

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.15, x=0, xanchor="left",
            buttons=[
                dict(label="▶ Play",  method="animate",
                     args=[None, {"frame":{"duration":400,"redraw":True},
                                  "fromcurrent":True,"transition":{"duration":200}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame":{"duration":0,"redraw":False},
                                    "mode":"immediate","transition":{"duration":0}}]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[f.name],
                        {"mode":"immediate","frame":{"duration":300,"redraw":True},
                         "transition":{"duration":100}}],
                        label=f"Turn {f.name}") for f in frames],
            active=0, y=0, x=0, len=1.0,
            currentvalue=dict(prefix="Turn: ", visible=True, xanchor="center"),
            transition=dict(duration=200),
        )],
        title=f"▶ Replay — {conv_id}",
        title_font_size=14,
        xaxis=dict(title="Turn", range=[turns[0]-.5, turns[-1]+.5]),
        yaxis=dict(title="Score", range=[-1.1, 1.1]),
    )
    return apply_chart(fig)


@st.cache_data(show_spinner=False)
def _get_smart_conv_lists(df: pd.DataFrame) -> Dict[str, list]:
    """
    Pre-compute curated conversation lists — cached per DataFrame.

    Returns
    -------
    worst20  : 20 conversations with lowest avg customer sentiment
    best20   : 20 with highest avg customer sentiment
    longest20: 20 longest by turn count
    all_ids  : all conversation IDs sorted
    """
    lf = pl.from_pandas(df).lazy()

    conv_stats = (
        lf.group_by("conversation_id")
          .agg([
              pl.col("compound").mean().alias("avg_sentiment"),
              pl.len().alias("n_turns"),
              (pl.col("speaker") == "CUSTOMER").sum().alias("cu_turns"),
          ])
          .collect()
    )
    cu_stats = (
        lf.filter(pl.col("speaker") == "CUSTOMER")
          .group_by("conversation_id")
          .agg(pl.col("compound").mean().alias("cu_avg"))
          .collect()
    )
    merged = conv_stats.join(cu_stats, on="conversation_id", how="left")

    worst20  = (merged.sort("cu_avg").head(20)["conversation_id"].to_list())
    best20   = (merged.sort("cu_avg", descending=True).head(20)["conversation_id"].to_list())
    longest20= (merged.sort("n_turns", descending=True).head(20)["conversation_id"].to_list())
    all_ids  = sorted(merged["conversation_id"].to_list())

    return {"worst20": worst20, "best20": best20, "longest20": longest20, "all_ids": all_ids}


# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE  — premium full-page design (adapted from reference app)
# ─────────────────────────────────────────────────────────────────────────────

LANDING_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
header[data-testid="stHeader"],footer,.stDeployButton,section[data-testid="stSidebar"]{display:none!important}
.block-container{padding:0!important;max-width:100%!important}

/* ── ANIMATIONS ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(40px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes gradSweep{0%{background-position:0% center}100%{background-position:200% center}}
@keyframes pulse1{0%,100%{transform:translate(-50%,-50%) scale(1);opacity:.35}50%{transform:translate(-45%,-55%) scale(1.2);opacity:.55}}
@keyframes pulse2{0%,100%{transform:translate(-50%,-50%) scale(1);opacity:.2}50%{transform:translate(-55%,-45%) scale(1.25);opacity:.4}}
@keyframes pulse3{0%,100%{transform:translate(-50%,-50%) scale(1);opacity:.15}50%{transform:translate(-48%,-52%) scale(1.15);opacity:.3}}
@keyframes float3d{0%,100%{transform:perspective(1000px) rotateX(2deg) rotateY(-1deg) translateY(0)}50%{transform:perspective(1000px) rotateX(-1deg) rotateY(1deg) translateY(-14px)}}
@keyframes barG1{0%{width:0}100%{width:78%}}@keyframes barG2{0%{width:0}100%{width:62%}}
@keyframes barG3{0%{width:0}100%{width:45%}}@keyframes barG4{0%{width:0}100%{width:30%}}
@keyframes barG5{0%{width:0}100%{width:20%}}@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
@keyframes gridP{0%,100%{opacity:.04}50%{opacity:.08}}
@keyframes typeLoop{0%,28%{content:"Start → Middle → End."}33%,61%{content:"CSAT / DSAT per Phase."}66%,94%{content:"Turn-by-Turn Insight."}100%{content:"Start → Middle → End."}}
@keyframes nodeFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
@keyframes flowDot{0%{left:0;opacity:0}10%{opacity:1}90%{opacity:1}100%{left:100%;opacity:0}}
@keyframes countUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

/* ── BASE ── */
.lp *{margin:0;padding:0;box-sizing:border-box;font-family:'DM Sans',sans-serif}

/* ── HERO ── */
.lp-hero{position:relative;min-height:100vh;background:#0C1418;overflow:hidden;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 24px}
.lp-grd{position:absolute;inset:0;background-image:linear-gradient(rgba(168,188,200,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(168,188,200,.05) 1px,transparent 1px);background-size:52px 52px;animation:gridP 8s ease-in-out infinite;z-index:1;pointer-events:none}
.lp-o1{position:absolute;width:800px;height:800px;border-radius:50%;background:radial-gradient(circle,rgba(45,95,110,.45) 0%,transparent 70%);top:15%;left:20%;transform:translate(-50%,-50%);filter:blur(100px);animation:pulse1 10s ease-in-out infinite;z-index:0}
.lp-o2{position:absolute;width:600px;height:600px;border-radius:50%;background:radial-gradient(circle,rgba(212,185,78,.3) 0%,transparent 70%);top:65%;left:75%;transform:translate(-50%,-50%);filter:blur(80px);animation:pulse2 12s ease-in-out infinite;z-index:0}
.lp-o3{position:absolute;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(61,122,95,.25) 0%,transparent 70%);top:80%;left:30%;transform:translate(-50%,-50%);filter:blur(90px);animation:pulse3 14s ease-in-out infinite;z-index:0}
.lp-bdg{position:relative;z-index:2;display:inline-flex;align-items:center;gap:6px;background:rgba(212,185,78,.08);color:#D4B94E;padding:8px 22px;border-radius:24px;font-size:11px;font-weight:600;letter-spacing:2px;border:1px solid rgba(212,185,78,.2);margin-bottom:32px;animation:fadeUp .7s ease-out both;backdrop-filter:blur(4px)}
.lp-bdg::before{content:'';width:6px;height:6px;border-radius:50%;background:#D4B94E;box-shadow:0 0 8px rgba(212,185,78,.6)}
.lp-ttl{position:relative;z-index:2;font-size:clamp(40px,7vw,78px);font-weight:700;line-height:1.05;text-align:center;margin-bottom:14px;letter-spacing:-1px;background:linear-gradient(90deg,#6B8A99,#E8E6DD 20%,#D4B94E 40%,#E8E6DD 60%,#A8BCC8 80%,#6B8A99);background-size:200% 100%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:fadeUp .7s ease-out .15s both,gradSweep 5s linear infinite}
.lp-sub{position:relative;z-index:2;font-size:20px;color:#A8BCC8;text-align:center;margin-bottom:8px;animation:fadeUp .7s ease-out .3s both;min-height:30px}
.lp-sub::after{content:"Start → Middle → End.";animation:typeLoop 9s ease-in-out infinite;font-style:italic;color:#D4B94E}
.lp-dsc{position:relative;z-index:2;font-size:15px;color:#4A6B78;text-align:center;max-width:600px;line-height:1.8;margin:8px auto 50px;animation:fadeUp .7s ease-out .45s both}

/* ── GLASS MOCKUP ── */
.lp-mk{position:relative;z-index:2;width:min(720px,92vw);margin:0 auto;animation:fadeUp .8s ease-out .6s both,float3d 7s ease-in-out 2s infinite}
.lp-wn{background:rgba(22,36,42,.5);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);border:1px solid rgba(168,188,200,.12);border-radius:16px;overflow:hidden;box-shadow:0 40px 100px rgba(0,0,0,.5)}
.lp-wh{display:flex;align-items:center;gap:8px;padding:16px 20px;background:rgba(12,20,24,.7);border-bottom:1px solid rgba(168,188,200,.08)}
.lp-dt{width:12px;height:12px;border-radius:50%}.lp-dr{background:#A04040}.lp-dy{background:#D4B94E}.lp-dg{background:#3D7A5F}
.lp-wt{font-size:12px;color:#4A6B78;margin-left:10px;font-family:'JetBrains Mono',monospace}
.lp-wb{padding:24px;font-family:'JetBrains Mono',monospace;font-size:12px;line-height:2;color:#6B8A99}
.ck{color:#D4B94E}.cf{color:#A8BCC8}.cs{color:#3D7A5F}.cm{color:#3D5A66;font-style:italic}
.lp-cur{display:inline-block;width:2px;height:14px;background:#D4B94E;animation:blink 1s step-end infinite;vertical-align:text-bottom;margin-left:2px}
.lp-bars{margin-top:20px;padding-top:16px;border-top:1px solid rgba(168,188,200,.08);display:flex;flex-direction:column;gap:10px}
.lp-br{display:flex;align-items:center;gap:12px}.lp-bl{width:120px;text-align:right;font-size:11px;color:#4A6B78}
.lp-bt{flex:1;height:8px;background:rgba(168,188,200,.08);border-radius:4px;overflow:hidden}
.lp-bf{height:100%;border-radius:4px}
.lb1{background:linear-gradient(90deg,#2ecc71,#3D7A5F);animation:barG1 1.8s cubic-bezier(.4,0,.2,1) 1.4s both}
.lb2{background:linear-gradient(90deg,#4682b4,#2D5F6E);animation:barG2 1.8s cubic-bezier(.4,0,.2,1) 1.6s both}
.lb3{background:linear-gradient(90deg,#D4B94E,#E8D97A);animation:barG3 1.8s cubic-bezier(.4,0,.2,1) 1.8s both}
.lb4{background:linear-gradient(90deg,#e74c3c,#A04040);animation:barG4 1.8s cubic-bezier(.4,0,.2,1) 2.0s both}
.lb5{background:linear-gradient(90deg,#6B8A99,#A8BCC8);animation:barG5 1.8s cubic-bezier(.4,0,.2,1) 2.2s both}
.lp-bp{width:44px;font-size:11px;color:#A8BCC8;font-family:'JetBrains Mono',monospace;text-align:right}

/* ── PHASE BAND ── */
.lp-ph{display:flex;gap:6px;margin-top:14px;padding-top:14px;border-top:1px solid rgba(168,188,200,.06)}
.lp-ph-s{flex:1;border-radius:6px;padding:6px 10px;text-align:center;font-size:10px;font-weight:600;letter-spacing:.5px}
.ph-start{background:rgba(45,95,110,.25);color:#3A7A8C}
.ph-mid{background:rgba(212,185,78,.15);color:#D4B94E}
.ph-end{background:rgba(160,64,64,.2);color:#c0726f}

/* ── STATS ── */
.lp-sts{position:relative;z-index:2;display:flex;justify-content:center;gap:48px;margin-top:56px;flex-wrap:wrap;animation:fadeUp .7s ease-out .8s both}
.lp-st{text-align:center}.lp-sn{font-size:36px;font-weight:700;color:#E8E6DD;font-family:'JetBrains Mono',monospace;animation:countUp .5s ease-out 1.2s both}
.lp-sn span{color:#D4B94E}.lp-sl{font-size:11px;color:#4A6B78;text-transform:uppercase;letter-spacing:1.5px;margin-top:4px}

/* ── FEATURES ── */
.lp-ft{background:#F5F4F0;padding:90px 40px;text-align:center}
.lp-fh{font-size:34px;font-weight:700;color:#1E2D33;margin-bottom:16px}
.lp-fsh{font-size:15px;color:#6B8A99;margin-bottom:52px;max-width:600px;margin-left:auto;margin-right:auto;line-height:1.6}
.lp-fg{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px;max-width:1140px;margin:0 auto}
.lp-fc{background:rgba(255,255,255,.7);backdrop-filter:blur(12px);border:1px solid rgba(209,207,196,.5);border-radius:16px;padding:36px 28px;transition:all .4s cubic-bezier(.25,.46,.45,.94);position:relative;overflow:hidden;text-align:left}
.lp-fc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#2D5F6E,#D4B94E);transform:scaleX(0);transform-origin:left;transition:transform .4s}
.lp-fc:hover{transform:translateY(-8px);box-shadow:0 20px 60px rgba(45,95,110,.12);background:rgba(255,255,255,.95);border-color:#2D5F6E}
.lp-fc:hover::before{transform:scaleX(1)}
.lp-fi{width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;margin-bottom:20px;background:linear-gradient(135deg,#2D5F6E,#3A7A8C);box-shadow:0 6px 20px rgba(45,95,110,.25);transition:transform .3s;font-size:22px}
.lp-fc:hover .lp-fi{transform:scale(1.08) rotate(-3deg)}
.lp-fc h3{font-size:16px;font-weight:600;color:#1E2D33;margin-bottom:10px}.lp-fc p{font-size:13px;color:#6B8A99;line-height:1.7}

/* ── HOW IT WORKS ── */
.lp-hw{background:#0C1418;padding:90px 40px;position:relative;overflow:hidden}
.lp-hw::before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(168,188,200,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(168,188,200,.03) 1px,transparent 1px);background-size:52px 52px;pointer-events:none}
.lp-hwt{text-align:center;font-size:34px;font-weight:700;color:#E8E6DD;margin-bottom:16px;position:relative;z-index:1}
.lp-hwst{text-align:center;font-size:14px;color:#4A6B78;margin-bottom:56px;position:relative;z-index:1}
.lp-hws{display:flex;justify-content:center;gap:0;max-width:1000px;margin:0 auto;flex-wrap:wrap;position:relative;z-index:1;align-items:flex-start}
.lp-stp{text-align:center;flex:1;min-width:220px;padding:36px 24px;background:rgba(22,36,42,.5);backdrop-filter:blur(12px);border:1px solid rgba(168,188,200,.08);border-radius:16px;transition:all .35s}
.lp-stp:hover{border-color:rgba(212,185,78,.3);transform:translateY(-6px)}
.lp-snm{width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#D4B94E,#E8D97A);color:#1E2D33;font-size:24px;font-weight:700;display:flex;align-items:center;justify-content:center;margin:0 auto 20px;box-shadow:0 6px 28px rgba(212,185,78,.3);animation:nodeFloat 4s ease-in-out infinite}
.lp-stp h4{font-size:17px;font-weight:600;color:#E8E6DD;margin-bottom:10px}.lp-stp p{font-size:13px;color:#6B8A99;line-height:1.65}
.lp-conn{display:flex;align-items:center;padding-top:50px;width:60px;position:relative}
.lp-conn::after{content:'';width:100%;height:2px;background:linear-gradient(90deg,rgba(212,185,78,.1),rgba(212,185,78,.4),rgba(212,185,78,.1))}
.lp-conn .lp-dot{position:absolute;width:6px;height:6px;background:#D4B94E;border-radius:50%;animation:flowDot 2.5s ease-in-out infinite;box-shadow:0 0 8px rgba(212,185,78,.5)}

/* ── COMPARISON ── */
.lp-cmp{background:#F5F4F0;padding:90px 40px;text-align:center}
.lp-cmpt{font-size:34px;font-weight:700;color:#1E2D33;margin-bottom:52px}
.lp-cmptbl{max-width:820px;margin:0 auto;border-collapse:separate;border-spacing:0;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(45,95,110,.08)}
.lp-cmptbl th{padding:16px 24px;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px}
.lp-cmptbl th:first-child{background:#0C1418;color:#6B8A99;text-align:left;width:40%}
.lp-cmptbl th:nth-child(2){background:#e8e6dd;color:#6B8A99}
.lp-cmptbl th:nth-child(3){background:#2D5F6E;color:#fff}
.lp-cmptbl td{padding:14px 24px;font-size:14px;border-bottom:1px solid #E8E6DD}
.lp-cmptbl td:first-child{font-weight:500;color:#1E2D33;text-align:left;background:#fafaf8}
.lp-cmptbl td:nth-child(2){color:#6B8A99;background:#fafaf8;text-align:center}
.lp-cmptbl td:nth-child(3){color:#2D5F6E;font-weight:600;background:rgba(45,95,110,.04);text-align:center}
.lp-cmptbl tr:last-child td{border-bottom:none}

/* ── TECH STACK ── */
.lp-tc{background:#0C1418;padding:52px 40px;text-align:center;border-top:1px solid rgba(168,188,200,.06)}
.lp-tl{font-size:11px;color:#4A6B78;text-transform:uppercase;letter-spacing:2px;font-weight:600}
.lp-tr{display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-top:16px}
.lp-tp{background:rgba(22,36,42,.5);border:1px solid rgba(168,188,200,.1);border-radius:8px;padding:9px 20px;font-size:13px;font-weight:500;color:#6B8A99;transition:all .25s;backdrop-filter:blur(4px)}
.lp-tp:hover{border-color:#D4B94E;color:#D4B94E}

/* ── CTA ── */
.lp-cta{background:linear-gradient(135deg,#0C1418 0%,#162A32 100%);padding:80px 40px;text-align:center;position:relative;overflow:hidden}
.lp-cta::before{content:'';position:absolute;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(212,185,78,.12) 0%,transparent 70%);top:50%;left:50%;transform:translate(-50%,-50%);filter:blur(60px);pointer-events:none}
.lp-ctah{font-size:34px;font-weight:700;color:#E8E6DD;margin-bottom:12px;position:relative;z-index:1}
.lp-ctap{font-size:15px;color:#4A6B78;margin-bottom:10px;position:relative;z-index:1}
.lp-ctatrust{font-size:12px;color:#3D5A66;margin-top:20px;position:relative;z-index:1;letter-spacing:.5px}
.lp-cta-krow{display:flex;justify-content:center;gap:16px;flex-wrap:wrap;margin:28px auto 0;max-width:720px;position:relative;z-index:1}
.lp-cta-kc{background:rgba(22,36,42,.7);border:1px solid rgba(168,188,200,.1);border-radius:10px;padding:14px 20px;text-align:center;min-width:130px;backdrop-filter:blur(8px)}
.lp-cta-kv{font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace}
.lp-cta-kl{font-size:9px;color:#4A6B78;text-transform:uppercase;letter-spacing:1px;margin-top:4px}

/* ── BUILT FOR ── */
.lp-bf-strip{background:#0C1418;padding:28px 40px;text-align:center;border-top:1px solid rgba(168,188,200,.06);border-bottom:1px solid rgba(168,188,200,.04)}
.lp-bf-lbl{font-size:10px;color:#3D5A66;text-transform:uppercase;letter-spacing:2px;font-weight:600;margin-bottom:14px}
.lp-bf-row{display:flex;justify-content:center;gap:10px;flex-wrap:wrap}
.lp-bf-pill{background:rgba(22,36,42,.6);border:1px solid rgba(168,188,200,.1);border-radius:20px;padding:7px 18px;font-size:13px;color:#6B8A99;font-weight:500;transition:all .25s;backdrop-filter:blur(4px)}
.lp-bf-pill:hover{border-color:#D4B94E;color:#D4B94E}

/* ── APP PREVIEW MOCKUP ── */
.lp-tabs{display:flex;gap:1px;border-bottom:1px solid rgba(168,188,200,.1);margin-bottom:14px;font-family:'DM Sans',sans-serif}
.lp-tab{padding:5px 11px;font-size:10px;color:#3D5A66;border-bottom:2px solid transparent;letter-spacing:.3px;white-space:nowrap}
.lp-tab-a{color:#D4B94E;border-bottom-color:#D4B94E;font-weight:600}
.lp-krow{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:14px}
.lp-kc{background:rgba(12,20,24,.7);border:1px solid rgba(168,188,200,.08);border-radius:6px;padding:8px 6px;text-align:center;border-top-width:2px;border-top-style:solid}
.lp-kv{font-size:17px;font-weight:700;font-family:'JetBrains Mono',monospace;line-height:1.1}
.lp-kl{font-size:8px;color:#3D5A66;text-transform:uppercase;letter-spacing:.7px;margin-top:3px}
.lp-flow-lbl{font-size:9px;color:#3D5A66;text-transform:uppercase;letter-spacing:1px;margin-bottom:7px}
.lp-snk{display:flex;align-items:center;gap:0;height:72px;margin-bottom:4px}
.lp-sn-nd{width:68px;flex-shrink:0;display:flex;flex-direction:column;align-items:center;justify-content:center;border-radius:6px;border:1px solid;font-size:9px;font-family:'DM Sans',sans-serif;text-align:center;padding:5px 3px;line-height:1.4;font-weight:600;letter-spacing:.3px;height:100%}
.lp-sn-nd span{font-size:13px;font-weight:700;font-family:'JetBrains Mono',monospace;display:block;margin-top:2px}
.lp-sn-cn{flex:1;display:flex;flex-direction:column;justify-content:center;gap:3px;padding:0 3px}
.lp-sn-bnd{border-radius:2px}

/* ── FOOTER ── */
.lp-fo{background:#0C1418;padding:28px;text-align:center;font-size:12px;color:#3D5A66;border-top:1px solid rgba(168,188,200,.06)}
</style>

<div class="lp">

<!-- ═══ HERO ═══ -->
<div class="lp-hero">
<div class="lp-grd"></div><div class="lp-o1"></div><div class="lp-o2"></div><div class="lp-o3"></div>

<div class="lp-bdg">CONVERSATION TRANSCRIPT ANALYTICS</div>
<h1 class="lp-ttl">TbT Sentiment Analytics</h1>
<p class="lp-sub">&nbsp;</p>
<p class="lp-dsc">Transform raw conversation transcripts into granular sentiment intelligence. Phase-level CSAT/DSAT, escalation detection, and executive narratives — powered by Polars and parallel VADER scoring.</p>

<!-- Glass Mockup -->
<div class="lp-mk"><div class="lp-wn">
<div class="lp-wh">
  <div class="lp-dt lp-dr"></div><div class="lp-dt lp-dy"></div><div class="lp-dt lp-dg"></div>
  <span class="lp-wt">pipeline.py — Parallel Sentiment Engine</span>
</div>
<div class="lp-wb" style="padding:16px 20px 14px">
  <!-- Page tabs -->
  <div class="lp-tabs">
    <span class="lp-tab">📊 Overview</span>
    <span class="lp-tab lp-tab-a">🌊 Sankey Flow</span>
    <span class="lp-tab">⚠️ Escalation</span>
    <span class="lp-tab">💡 Narrative</span>
  </div>
  <!-- KPI strip -->
  <div class="lp-krow">
    <div class="lp-kc" style="border-top-color:#3D7A5F">
      <div class="lp-kv" style="color:#3D7A5F">72%</div>
      <div class="lp-kl">CSAT</div>
    </div>
    <div class="lp-kc" style="border-top-color:#A04040">
      <div class="lp-kv" style="color:#A04040">28%</div>
      <div class="lp-kl">DSAT</div>
    </div>
    <div class="lp-kc" style="border-top-color:#D4B94E">
      <div class="lp-kv" style="color:#D4B94E">14%</div>
      <div class="lp-kl">Escalation</div>
    </div>
    <div class="lp-kc" style="border-top-color:#2D5F6E">
      <div class="lp-kv" style="color:#2D5F6E">61%</div>
      <div class="lp-kl">Recovery</div>
    </div>
  </div>
  <!-- CSS Sankey flow preview -->
  <div class="lp-flow-lbl">Sentiment Flow · Start → Resolution → End</div>
  <div class="lp-snk">
    <!-- Start nodes -->
    <div style="display:flex;flex-direction:column;gap:3px;width:68px;height:100%;justify-content:center">
      <div class="lp-sn-nd" style="background:rgba(61,122,95,.25);border-color:rgba(61,122,95,.4);color:#3D7A5F;flex:3">Positive<span>148</span></div>
      <div class="lp-sn-nd" style="background:rgba(45,95,110,.2);border-color:rgba(45,95,110,.35);color:#2D5F6E;flex:2">Neutral<span>86</span></div>
      <div class="lp-sn-nd" style="background:rgba(160,64,64,.2);border-color:rgba(160,64,64,.35);color:#A04040;flex:1">Negative<span>38</span></div>
    </div>
    <!-- Connector bands -->
    <div class="lp-sn-cn">
      <div class="lp-sn-bnd" style="background:rgba(61,122,95,.22);height:18px"></div>
      <div class="lp-sn-bnd" style="background:rgba(61,122,95,.12);height:10px"></div>
      <div class="lp-sn-bnd" style="background:rgba(45,95,110,.15);height:8px"></div>
      <div class="lp-sn-bnd" style="background:rgba(160,64,64,.12);height:6px"></div>
    </div>
    <!-- Resolution nodes -->
    <div style="display:flex;flex-direction:column;gap:3px;width:80px;height:100%;justify-content:center">
      <div class="lp-sn-nd" style="background:rgba(61,122,95,.25);border-color:rgba(61,122,95,.4);color:#3D7A5F;font-size:8px;flex:3">Truly<br>Resolved<span>112</span></div>
      <div class="lp-sn-nd" style="background:rgba(212,185,78,.15);border-color:rgba(212,185,78,.3);color:#D4B94E;font-size:8px;flex:2">Partially<br>Resolved<span>71</span></div>
      <div class="lp-sn-nd" style="background:rgba(160,64,64,.18);border-color:rgba(160,64,64,.32);color:#A04040;font-size:8px;flex:1">Unresolved<span>89</span></div>
    </div>
    <!-- Connector bands -->
    <div class="lp-sn-cn">
      <div class="lp-sn-bnd" style="background:rgba(61,122,95,.22);height:20px"></div>
      <div class="lp-sn-bnd" style="background:rgba(212,185,78,.15);height:10px"></div>
      <div class="lp-sn-bnd" style="background:rgba(160,64,64,.12);height:7px"></div>
      <div class="lp-sn-bnd" style="background:rgba(168,188,200,.08);height:5px"></div>
    </div>
    <!-- End nodes -->
    <div style="display:flex;flex-direction:column;gap:3px;width:68px;height:100%;justify-content:center">
      <div class="lp-sn-nd" style="background:rgba(61,122,95,.25);border-color:rgba(61,122,95,.4);color:#3D7A5F;flex:3">Positive<span>134</span></div>
      <div class="lp-sn-nd" style="background:rgba(45,95,110,.2);border-color:rgba(45,95,110,.35);color:#2D5F6E;flex:2">Neutral<span>68</span></div>
      <div class="lp-sn-nd" style="background:rgba(160,64,64,.2);border-color:rgba(160,64,64,.35);color:#A04040;flex:1">Negative<span>70</span></div>
    </div>
  </div>
  <!-- Phase band -->
  <div class="lp-ph">
    <div class="lp-ph-s ph-start">🚀 START PHASE</div>
    <div class="lp-ph-s ph-mid">🔄 MIDDLE PHASE</div>
    <div class="lp-ph-s ph-end">🏁 END PHASE</div>
  </div>
</div>
</div></div>

<!-- Stats -->
<div class="lp-sts">
<div class="lp-st"><div class="lp-sn">4<span>+</span></div><div class="lp-sl">Transcript Formats</div></div>
<div class="lp-st"><div class="lp-sn">5</div><div class="lp-sl">Escalation Signals</div></div>
<div class="lp-st"><div class="lp-sn">3</div><div class="lp-sl">Phase Analysis</div></div>
<div class="lp-st"><div class="lp-sn">1<span>-click</span></div><div class="lp-sl">Export to Excel</div></div>
</div>
</div>

<!-- ═══ BUILT FOR ═══ -->
<div class="lp-bf-strip">
  <div class="lp-bf-lbl">Built for</div>
  <div class="lp-bf-row">
    <div class="lp-bf-pill">📞 Call Centres</div>
    <div class="lp-bf-pill">🎧 Customer Support</div>
    <div class="lp-bf-pill">🔍 QA &amp; Compliance Teams</div>
    <div class="lp-bf-pill">📈 CX Analytics</div>
    <div class="lp-bf-pill">🏢 Contact Centre Operations</div>
    <div class="lp-bf-pill">💼 Voice of Customer</div>
  </div>
</div>

<!-- ═══ FEATURES ═══ -->
<div class="lp-ft">
<h2 class="lp-fh">Built for Conversation Intelligence</h2>
<p class="lp-fsh">Everything you need to analyse, visualise, and act on conversation data — from turn-level sentiment to executive narratives.</p>
<div class="lp-fg">

<div class="lp-fc">
<div class="lp-fi">📊</div>
<h3>Phase-Level KPIs</h3>
<p>CSAT % and DSAT % independently measured for Start, Middle and End phases. Know exactly when conversations fail.</p>
</div>

<div class="lp-fc">
<div class="lp-fi">🔄</div>
<h3>Turn-by-Turn Flow</h3>
<p>Interactive Plotly chart mapping every conversation turn. Colour-coded by sentiment with phase band overlays and hover detail.</p>
</div>

<div class="lp-fc">
<div class="lp-fi">🌊</div>
<h3>Sankey Flow Analysis</h3>
<p>Interactive Sankey charts map sentiment transitions across phases and resolution outcomes — see exactly where conversations shift and why.</p>
</div>

<div class="lp-fc">
<div class="lp-fi">🌐</div>
<h3>Sunburst Breakdown</h3>
<p>Speaker → Phase → Sentiment hierarchy in a single interactive sunburst. Spot which role drives negativity at a glance.</p>
</div>

<div class="lp-fc">
<div class="lp-fi">⚡</div>
<h3>Escalation Intelligence</h3>
<p>5-signal detection with severity tiers, Quick/Late/Never recovery decomposition, time-to-resolution histograms, and phrase-clustered root causes.</p>
</div>

<div class="lp-fc">
<div class="lp-fi">💡</div>
<h3>Narrative Intelligence</h3>
<p>Auto-generated executive summary with sentiment verdicts, phase breakdowns, and prioritised business recommendations.</p>
</div>

<div class="lp-fc">
<div class="lp-fi">🔀</div>
<h3>Parallel VADER Scoring</h3>
<p>ThreadPoolExecutor with 4 workers, one VADER instance per thread. Vectorised numpy label assignment — no Python loops.</p>
</div>

<div class="lp-fc">
<div class="lp-fi">⚙️</div>
<h3>Polars Analytics</h3>
<p>All groupbys, joins and aggregations run in Polars lazy frames. Rust internals — 5–10× faster than pandas on 50k+ rows.</p>
</div>

<div class="lp-fc">
<div class="lp-fi">⬇️</div>
<h3>Flexible Export</h3>
<p>Download as Excel (5 sheets), flat CSV, or a complete ZIP bundle with insights JSON. Cached — instant re-download.</p>
</div>

</div>
</div>

<!-- ═══ HOW IT WORKS ═══ -->
<div class="lp-hw">
<h2 class="lp-hwt">Three Steps to Insight</h2>
<p class="lp-hwst">From raw transcripts to CSAT/DSAT intelligence in under a minute.</p>
<div class="lp-hws">
<div class="lp-stp">
  <div class="lp-snm">1</div>
  <h4>Upload</h4>
  <p>CSV or Excel with conversation transcripts. Auto-detects format across 4 transcript types — bracket, timestamp, call-centre, and chat/SMS.</p>
</div>
<div class="lp-conn"><div class="lp-dot"></div></div>
<div class="lp-stp">
  <div class="lp-snm">2</div>
  <h4>Analyse</h4>
  <p>Parallel VADER scoring, Polars aggregations, phase classification. Full 50k-turn dataset in seconds.</p>
</div>
<div class="lp-conn"><div class="lp-dot"></div></div>
<div class="lp-stp">
  <div class="lp-snm">3</div>
  <h4>Insight</h4>
  <p>KPI dashboard, flow charts, waterfall, sunburst, narrative summaries, and one-click export.</p>
</div>
</div>
</div>

<!-- ═══ COMPARISON ═══ -->
<div class="lp-cmp">
<h2 class="lp-cmpt">Why TbT Sentiment Analytics?</h2>
<table class="lp-cmptbl">
<thead><tr><th>Capability</th><th>Manual / Spreadsheet</th><th>TbT Sentiment Analytics</th></tr></thead>
<tbody>
<tr><td>Scoring Speed</td><td>Hours per dataset</td><td>Seconds — parallel VADER</td></tr>
<tr><td>Phase Analysis</td><td>Not possible</td><td>Start / Middle / End CSAT & DSAT</td></tr>
<tr><td>Escalation Detection</td><td>Manual review</td><td>Auto-flagged every turn</td></tr>
<tr><td>Domain Support</td><td>One format</td><td>4 transcript formats, auto-detected</td></tr>
<tr><td>Visualisations</td><td>Basic charts</td><td>Sankey Flow, Escalation Drill-down, Sunburst, Phase Charts</td></tr>
<tr><td>Executive Summary</td><td>Written manually</td><td>Auto-generated with recommendations</td></tr>
<tr><td>Recompute on filter</td><td>Always</td><td>Never — 5-stage cache</td></tr>
<tr><td>Setup Time</td><td>Days</td><td>Upload and click Run</td></tr>
</tbody>
</table>
</div>

<!-- ═══ TECH STACK ═══ -->
<div class="lp-tc">
<p class="lp-tl">Powered By</p>
<div class="lp-tr">
  <div class="lp-tp">Polars</div>
  <div class="lp-tp">VADER Sentiment</div>
  <div class="lp-tp">NumPy</div>
  <div class="lp-tp">Plotly</div>
  <div class="lp-tp">Streamlit</div>
  <div class="lp-tp">ThreadPoolExecutor</div>
  <div class="lp-tp">OpenPyXL</div>
</div>
</div>

<!-- ═══ CTA ═══ -->
<div class="lp-cta">
<h2 class="lp-ctah">Ready to Analyse Your Conversations?</h2>
<p class="lp-ctap">Upload your transcripts, click Run Analysis, and get phase-level sentiment intelligence instantly.</p>
<div class="lp-cta-krow">
  <div class="lp-cta-kc">
    <div class="lp-cta-kv" style="color:#3D7A5F">72%</div>
    <div class="lp-cta-kl">CSAT Score</div>
  </div>
  <div class="lp-cta-kc">
    <div class="lp-cta-kv" style="color:#A04040">14%</div>
    <div class="lp-cta-kl">Escalation Rate</div>
  </div>
  <div class="lp-cta-kc">
    <div class="lp-cta-kv" style="color:#D4B94E">61%</div>
    <div class="lp-cta-kl">Recovery Rate</div>
  </div>
  <div class="lp-cta-kc">
    <div class="lp-cta-kv" style="color:#2D5F6E">4 turns</div>
    <div class="lp-cta-kl">Median TTR</div>
  </div>
</div>
<p class="lp-ctatrust" style="margin-top:28px">No cloud dependency · Your data stays on your machine · Instant re-analysis with 5-stage cache</p>
</div>

<div class="lp-fo">TbT Sentiment Analytics v5.1 — Domain Agnostic · Turn-by-Turn Intelligence</div>
</div>
"""


def render_landing():
    """Full-page premium landing — hides sidebar and header, shows on 🏠 Home."""
    st.markdown(LANDING_HTML, unsafe_allow_html=True)
    # Launch button centred below the CTA
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _, cc, _ = st.columns([1, 2, 1])
    with cc:
        if st.button("🚀 Launch Application", type="primary", width="stretch"):
            st.session_state["page"] = "📊 Overview"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
<div style="text-align:center;padding:.8rem 0 1.2rem">
  <div style="font-size:2rem">🎭</div>
  <div style="font-weight:700;font-size:1.05rem;color:{C['text']};margin-top:.3rem">TbT Analytics</div>
  <div style="font-size:.78rem;color:{C['muted']}">Turn-by-Turn Sentiment</div>
</div>""", unsafe_allow_html=True)

        st.markdown("### 🗂 Navigate")
        pages=["🏠 Home","📊 Overview","🌊 Sankey Flow","⚠️ Escalation","🗣️ Explorer","💡 Narrative & Export"]
        for p in pages:
            is_active = st.session_state.get("page") == p
            if st.button(p, key=f"nav_{p}", type="primary" if is_active else "secondary"):
                st.session_state["page"] = p
                st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        domain_keys   = list(FORMAT_LABELS.keys())
        domain_labels = [FORMAT_LABELS[k] for k in domain_keys]

        # After a successful run, sync the selectbox to the auto-detected format
        # so the user always sees what was actually used, not just "Auto-Detect".
        detected_key = st.session_state.get("detected_domain_key", "auto")
        default_idx  = domain_keys.index(detected_key) if detected_key in domain_keys else 0

        sel = st.selectbox("Domain / Format", options=range(len(domain_keys)),
                           format_func=lambda i: domain_labels[i], index=default_idx,
                           key="sb_domain",
                           help="Auto-Detect picks the right format automatically. Override only if needed.")
        dataset_type = domain_keys[sel]

        # If user manually changes the selector, clear the auto-sync so it stays on their choice
        if sel != default_idx:
            st.session_state["detected_domain_key"] = domain_keys[sel]

        st.markdown("---")
        st.markdown("### 🛡️ PII Redaction")
        pii_enabled = st.checkbox(
            "Enable PII Redaction",
            value=False,
            key="sb_pii",
            help="Scrub emails, phones, SSNs, card numbers, IPs, addresses, DOB & MRNs before analysis.",
        )
        pii_mode = "mask"
        if pii_enabled:
            pii_mode = st.selectbox(
                "Redaction Mode",
                options=["mask", "token", "remove"],
                format_func=lambda m: {
                    "mask":   "🔒 Mask  — [EMAIL:REDACTED]",
                    "token":  "🏷️ Token — [EMAIL]",
                    "remove": "🗑️ Remove — blank",
                }[m],
                key="sb_pii_mode",
                help="mask = label+REDACTED tag · token = label only · remove = strip entirely",
            )
            st.markdown(
                f'<div style="background:rgba(45,95,110,0.08);border:1px solid {C["teal"]};'
                f'border-radius:6px;padding:6px 10px;font-size:11px;color:{C["teal"]};margin-top:4px">'
                f'🛡️ 8 pattern types: EMAIL · PHONE · CARD · SSN · MRN · DOB · IP · ADDR'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 📂 Upload Data")
        uploaded = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"],
                                    help="Upload conversation transcripts (CSV or Excel).")
        excel_sheet = 0  # default: first sheet
        if uploaded:
            st.markdown(f"""
<div style="background:rgba(45,95,110,0.1);border:1px solid {C['teal']};
     border-radius:8px;padding:.5rem .8rem;margin-top:.5rem">
  <div style="color:{C['muted']};font-size:.72rem">Loaded:</div>
  <div style="color:{C['text']};font-weight:600;font-size:.88rem">📄 {uploaded.name}</div>
</div>""", unsafe_allow_html=True)
            if uploaded.name.lower().endswith((".xlsx", ".xls")):
                try:
                    import openpyxl as _oxl
                    _wb = _oxl.load_workbook(io.BytesIO(uploaded.read()), read_only=True, data_only=True)
                    _sheet_names = _wb.sheetnames
                    _wb.close()
                    uploaded.seek(0)
                    if len(_sheet_names) > 1:
                        excel_sheet = st.selectbox(
                            "Excel sheet",
                            options=_sheet_names,
                            index=0,
                            help="Select which sheet contains the transcript data.",
                        )
                    else:
                        excel_sheet = _sheet_names[0]
                except Exception:
                    uploaded.seek(0)  # ensure readable even if inspection fails

        st.markdown("---")
        run = st.button("▶ Run Analysis", type="primary", key="run_btn")

        # ── Clear cache button — shown when results exist ──
        if "df_r" in st.session_state:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear & Reset", key="clear_btn", type="secondary"):
                # Wipe all cached data and session results
                st.cache_data.clear()
                gc.collect()
                current_page = st.session_state.get("page", "📊 Overview")
                for k in ("df_r","ins","detected","fname","pii_meta",
                          "_file_checksum","_dataset_type","_pii_key","detected_domain_key","_pipeline_secs"):
                    st.session_state.pop(k, None)
                # Stay on the current page (not Home)
                st.session_state["page"] = current_page
                st.rerun()

            # Show active file + record count as a sanity check
            fname_active = st.session_state.get("fname","")
            n_turns      = len(st.session_state["df_r"])
            n_convs      = st.session_state["ins"]["total_conversations"]
            st.markdown(
                f'<div style="background:rgba(45,95,110,0.08);border:1px solid {C["teal"]};'
                f'border-radius:8px;padding:.5rem .75rem;margin-top:.4rem;font-size:.75rem">'
                f'<div style="color:{C["muted"]}">Active dataset:</div>'
                f'<div style="color:{C["text"]};font-weight:600">📄 {fname_active}</div>'
                f'<div style="color:{C["teal"]}">{n_convs:,} conversations · {n_turns:,} turns</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Show active performance profile
        try:
            import psutil as _ps
            _ram = f"{_ps.virtual_memory().total/(1024**3):.0f} GB RAM"
        except Exception:
            _ram = "RAM unknown"
        _is_cloud = bool(os.environ.get("STREAMLIT_SHARING_MODE") or
                         os.environ.get("IS_STREAMLIT_CLOUD"))
        _env_label = "☁️ Cloud" if _is_cloud else "🖥️ Local"
        st.markdown(
            f'<div style="color:{C["muted"]};font-size:.68rem;text-align:center;'
            f'padding-top:.5rem;line-height:1.6">'
            f'v5.1 — Polars · Parallel VADER · 5-stage Cache<br>'
            f'{_env_label} · {_ram} · '
            f'Cap: <strong>{MAX_TURNS:,}</strong> turns · '
            f'{VADER_WORKERS} VADER threads</div>',
            unsafe_allow_html=True,
        )

    return dataset_type, uploaded, run, pii_enabled, pii_mode, excel_sheet


# ─────────────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def _parse_ts_series(series: pd.Series) -> pd.Series:
    """
    Parse a raw timestamp Series into pd.Timestamp (UTC), per-element.
    Handles ISO dates, HH:MM:SS offsets, and MM:SS offsets.
    Invalid entries become NaT.
    """
    def _parse_one(v):
        v = str(v).strip()
        if not v or v in ("nan", "None", ""):
            return pd.NaT
        # ISO / datetime string
        try:
            return pd.to_datetime(v, utc=True)
        except Exception:
            pass
        # HH:MM:SS  →  treat as seconds offset (call-centre format)
        try:
            parts = v.split(":")
            if len(parts) == 3:
                s = int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
                return pd.Timestamp(s, unit="s", tz="UTC")
        except Exception:
            pass
        # MM:SS
        try:
            parts = v.split(":")
            if len(parts) == 2:
                s = int(parts[0])*60 + float(parts[1])
                return pd.Timestamp(s, unit="s", tz="UTC")
        except Exception:
            pass
        return pd.NaT

    return series.map(_parse_one)


def _fmt_seconds(total_s: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    s = int(total_s)
    if s <= 0:
        return "—"
    days  = s // 86400
    hours = (s % 86400) // 3600
    mins  = (s % 3600) // 60
    secs  = s % 60
    if days > 0:   return f"{days}d {hours}h {mins}m"
    if hours > 0:  return f"{hours}h {mins}m"
    if mins > 0:   return f"{mins}m {secs}s"
    return f"{secs}s"


def _compute_duration_str(df_r: pd.DataFrame) -> str:
    """
    Compute avg call duration per conversation from the timestamp column.

    Correct approach
    ----------------
    For each conversation:  duration = last_turn_timestamp - first_turn_timestamp
    Then return:  avg duration across all conversations  (shown as KPI)
    Also returns total call time as a tooltip-style suffix.

    Why the old approach was wrong
    ------------------------------
    The old code did  max(ALL timestamps) - min(ALL timestamps)  which gives
    the *date range of the whole dataset* (e.g. June 30 → July 1 = 23h 54m),
    not the duration of any individual call.
    """
    ts_col = "timestamp"
    cid_col = "conversation_id"

    if ts_col not in df_r.columns or cid_col not in df_r.columns:
        return "—"

    raw = df_r[[cid_col, ts_col]].copy()
    raw = raw[raw[ts_col].notna()]
    raw = raw[raw[ts_col].astype(str).str.strip().str.len() > 3]
    if raw.empty:
        return "—"

    raw["_ts"] = _parse_ts_series(raw[ts_col].astype(str))
    raw = raw.dropna(subset=["_ts"])
    if raw.empty:
        return "—"

    # Per-conversation: first and last timestamp → duration in seconds
    per_conv = (
        raw.groupby(cid_col)["_ts"]
           .agg(["min", "max"])
    )
    per_conv["dur_s"] = (per_conv["max"] - per_conv["min"]).dt.total_seconds()

    # Only count conversations where we have at least 2 timestamps (dur > 0)
    valid = per_conv[per_conv["dur_s"] > 0]["dur_s"]
    if valid.empty:
        return "—"

    avg_s   = valid.mean()
    total_s = valid.sum()

    avg_str   = _fmt_seconds(avg_s)
    total_str = _fmt_seconds(total_s)

    # Show avg prominently; total in parentheses
    return f"~{avg_str} avg  ({total_str} total)"


def _kpi_row(ins, df_r: pd.DataFrame = None, pipeline_secs: float = None):
    cs      = ins["customer_satisfaction"]
    ap      = ins["agent_performance"]
    cp      = ins["conversation_patterns"]
    overall = ins["overall_sentiment"]["average"]
    total_t = ins["total_turns"]

    # Processing Time (wall-clock pipeline time)
    if pipeline_secs and pipeline_secs > 0:
        ps     = int(pipeline_secs)
        pt_str = f"{ps//60}m {ps%60}s" if ps >= 60 else f"{pipeline_secs:.1f}s"
    else:
        pt_str = "—"

    esc_rate = cs["escalation_rate"]
    res_rate = cs["resolution_rate"]
    esc_c  = C["neg"]  if esc_rate > 0.15 else C["warn"] if esc_rate > 0.10 else C["ok"]
    res_c  = C["ok"]   if res_rate > 0.60  else C["warn"] if res_rate > 0.40  else C["neg"]

    cols = st.columns(8)
    data = [
        ("Conversations",   f"{ins['total_conversations']:,}",           "var(--teal)"),
        ("Total Turns",     f"{total_t:,}",                               "var(--slate)"),
        ("Processing Time", pt_str,                                        C["slate"]),
        ("Overall Sent.",   f"{overall:+.3f}",                            _score_color(overall)),
        ("Customer Avg",    f"{cs['average_sentiment']:+.3f}",            _score_color(cs["average_sentiment"])),
        ("Agent Avg",       f"{ap['average_sentiment']:+.3f}",            _score_color(ap["average_sentiment"])),
        ("Escalation",      f"{esc_rate:.1%}",                            esc_c),
        ("Resolution",      f"{res_rate:.1%}",                            res_c),
    ]
    for col, (lbl, val, color) in zip(cols, data):
        with col:
            st.markdown(
                mc(lbl, f'<span style="color:{color}">{val}</span>', color),
                unsafe_allow_html=True,
            )
def _phase_table(ins):
    pcd=ins.get("phase_csat_dsat",{}); cp=ins.get("conversation_patterns",{})
    rows=""
    for pn in ["start","middle","end"]:
        p=pcd.get(pn,{}); csat=p.get("csat_pct",0); dsat=p.get("dsat_pct",0)
        cnt=p.get("count",0); avg=cp.get(f"avg_sentiment_{pn}",0)
        ind="✅" if csat>=0.6 else "⚠️" if csat>=0.4 else "🔴"
        rows+=(f"<tr><td>{PHASE_ICONS[pn]} <strong>{pn.capitalize()}</strong></td>"
               f"<td>{ind}</td><td>{cnt:,}</td>"
               f"<td><span class='badge b-ok'>{_pct(csat)} CSAT</span></td>"
               f"<td><span class='badge b-err'>{_pct(dsat)} DSAT</span></td>"
               f"<td>{_sbar(avg)}</td></tr>")
    st.markdown(f"<table class='pt'><thead><tr><th>Phase</th><th>Health</th><th>Turns</th>"
                f"<th>CSAT</th><th>DSAT</th><th>Avg Score</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def _turn_viewer(df, conv_id):
    sub = df[df["conversation_id"] == conv_id].sort_values("turn_sequence")
    if sub.empty:
        st.info("No turns for this conversation.")
        return

    # ── pandas apply replaces iterrows — builds all HTML in one vectorised pass ──
    def _build_card(r) -> str:
        spk  = str(r["speaker"]).upper()
        css  = "tc-cu" if spk == "CUSTOMER" else "tc-ag"
        icon = "\U0001f464" if spk == "CUSTOMER" else "\U0001f3a7"   # 👤 🎧
        ts_  = str(r.get("timestamp", ""))
        ts   = f" \u00b7 {ts_}" if ts_ not in ("", "nan", "None") else ""
        pi   = PHASE_ICONS.get(str(r.get("phase", "middle")), "\U0001f504")
        s    = float(r["compound"])
        lbl  = str(r.get("sentiment_label", "neutral"))
        conf = float(r.get("sentiment_confidence", 0))
        phase_cap = str(r.get("phase", "")).capitalize()
        turn_n    = int(r["turn_sequence"])
        msg       = r["message"]
        return (
            f'<div class="tc {css}">' +
            f'<div class="tc-hdr">{icon} {spk}{ts} &nbsp; {pi} {phase_cap} &nbsp; Turn #{turn_n}</div>' +
            f'<div class="tc-txt">{msg}</div>' +
            f'<div class="tc-meta">{_badge(lbl)} &nbsp; {_sbar(s)} &nbsp; Confidence: {conf:.0%}</div>' +
            '</div>'
        )

    cards_html = "".join(sub.apply(_build_card, axis=1).tolist())
    st.markdown(cards_html, unsafe_allow_html=True)


def _export_section(df_r, ins):
    sh("⬇️","Download Results")
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    ins_json=json.dumps(ins,default=str)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown(f'<div class="ex-card"><div class="ex-title">📊 Excel Workbook</div>'
                    f'<div class="ex-desc">All Turns · Customer · Agent · Summary · Recommendations</div></div>',
                    unsafe_allow_html=True)
        st.download_button("📥 Download Excel (.xlsx)", data=_to_excel(df_r, ins_json),
            file_name=f"tbt_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch", type="primary")
    with c2:
        st.markdown(f'<div class="ex-card"><div class="ex-title">📄 CSV File</div>'
                    f'<div class="ex-desc">Flat CSV of all turns — ready for Excel, Power BI or any BI tool</div></div>',
                    unsafe_allow_html=True)
        st.download_button("📥 Download CSV (.csv)", data=_to_csv(df_r),
            file_name=f"tbt_{ts}.csv", mime="text/csv", width="stretch", type="primary")
    with c3:
        st.markdown(f'<div class="ex-card"><div class="ex-title">📦 ZIP Bundle</div>'
                    f'<div class="ex-desc">CSV + Excel + JSON insights in one archive</div></div>',
                    unsafe_allow_html=True)
        st.download_button("📥 Download ZIP (.zip)", data=_to_zip(df_r,ins),
            file_name=f"tbt_{ts}.zip", mime="application/zip", width="stretch", type="primary")


# ─── Sankey Flow ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _build_flow_table(df: pd.DataFrame, aggs: dict) -> pd.DataFrame:
    """
    Build the per-conversation flow summary DataFrame used by the Flow Table tab.
    Cached separately so the pill filters don't re-trigger the heavy computation.
    """
    outcome_df = aggs.get("outcome_flow_df", pd.DataFrame())

    # Per-conversation: avg compound in start / end phase (customer turns only)
    cust = df[df["speaker"] == "CUSTOMER"]
    start_s = (
        cust[cust["phase"] == "start"]
        .groupby("conversation_id")["compound"].mean()
        .rename("start_score")
    )
    end_s = (
        cust[cust["phase"] == "end"]
        .groupby("conversation_id")["compound"].mean()
        .rename("end_score")
    )
    mid_s = (
        cust[cust["phase"] == "middle"]
        .groupby("conversation_id")["compound"].mean()
        .rename("mid_score")
    )
    n_turns = df.groupby("conversation_id").size().rename("turns")
    esc_any = (
        df.groupby("conversation_id")["potential_escalation"].any().rename("escalated")
        if "potential_escalation" in df.columns else pd.Series(dtype=bool)
    )

    base = pd.concat([start_s, mid_s, end_s, n_turns], axis=1).reset_index()
    base.columns = ["conversation_id", "start_score", "mid_score", "end_score", "turns"]
    base["delta"] = (base["end_score"] - base["start_score"]).round(3)

    # Arc label
    def _arc(row):
        s, e = row["start_score"], row["end_score"]
        d    = row["delta"]
        if s <= -0.05 and e >= 0.05:  return "📈 Recovery"
        if s >= 0.05  and e <= -0.05: return "📉 Deterioration"
        if d > 0.10:                  return "↗️ Improvement"
        if d < -0.10:                 return "↘️ Decline"
        if abs(d) <= 0.05:            return "➡️ Stable"
        return "〰️ Volatile"

    base["arc"] = base.apply(_arc, axis=1)

    # Merge resolution status
    if not outcome_df.empty and "resolution_status" in outcome_df.columns:
        res_map = outcome_df.set_index("conversation_id")["resolution_status"].to_dict()
        base["resolution"] = base["conversation_id"].map(res_map).fillna("Unknown")
    elif "resolution_status" in df.columns:
        res_map = df.drop_duplicates("conversation_id").set_index("conversation_id")["resolution_status"].to_dict()
        base["resolution"] = base["conversation_id"].map(res_map).fillna("Unknown")
    else:
        base["resolution"] = "Unknown"

    # Escalation flag
    if not esc_any.empty:
        base["escalated"] = base["conversation_id"].map(esc_any).fillna(False)
    else:
        base["escalated"] = False

    # Sentiment label columns (for pills filter)
    def _lbl(v):
        if pd.isna(v): return "neutral"
        return "positive" if v >= 0.05 else "negative" if v <= -0.05 else "neutral"

    base["start_label"] = base["start_score"].apply(_lbl)
    base["end_label"]   = base["end_score"].apply(_lbl)

    return base.round({"start_score": 3, "mid_score": 3, "end_score": 3})


def _sankey_flow_table(df: pd.DataFrame, aggs: dict, ins: dict) -> None:
    """
    Flow Table tab — native Streamlit pill filters + styled st.dataframe.

    Pills filter by:
      • Arc type   (Recovery / Deterioration / Improvement / Decline / Stable / Volatile)
      • Start sentiment  (positive / neutral / negative)
      • End sentiment    (positive / neutral / negative)
      • Resolution status

    Table uses st.column_config for:
      • ProgressColumn  — start_score, end_score, delta (colour-coded bars)
      • TextColumn      — conversation_id, arc, resolution
      • CheckboxColumn  — escalated
    """
    base = _build_flow_table(df, aggs)
    total = len(base)

    # ── Pill filters ──────────────────────────────────────────────────────────
    st.caption("Filter conversations using the pills below — multiple selections within a group are OR'd.")

    fc1, fc2, fc3, fc4 = st.columns([2, 1.5, 1.5, 2])

    with fc1:
        arc_opts = sorted(base["arc"].unique().tolist())
        arc_sel  = st.pills(
            "Arc type", arc_opts,
            selection_mode="multi", default=None, key="ft_arc",
        )
    with fc2:
        start_sel = st.pills(
            "Start sentiment",
            ["positive", "neutral", "negative"],
            selection_mode="multi", default=None, key="ft_start",
        )
    with fc3:
        end_sel = st.pills(
            "End sentiment",
            ["positive", "neutral", "negative"],
            selection_mode="multi", default=None, key="ft_end",
        )
    with fc4:
        res_opts = sorted(base["resolution"].unique().tolist())
        res_sel  = st.pills(
            "Resolution", res_opts,
            selection_mode="multi", default=None, key="ft_res",
        )

    # Apply filters
    view = base.copy()
    if arc_sel:
        view = view[view["arc"].isin(arc_sel)]
    if start_sel:
        view = view[view["start_label"].isin(start_sel)]
    if end_sel:
        view = view[view["end_label"].isin(end_sel)]
    if res_sel:
        view = view[view["resolution"].isin(res_sel)]

    n_shown = len(view)
    st.caption(f"Showing **{n_shown:,}** of **{total:,}** conversations")

    if view.empty:
        st.info("No conversations match the selected filters.")
        return

    # ── Display columns ───────────────────────────────────────────────────────
    display = view[[
        "conversation_id", "arc", "turns",
        "start_score", "mid_score", "end_score", "delta",
        "resolution", "escalated",
    ]].sort_values("delta").reset_index(drop=True)

    def _score_cell(val) -> str:
        """Render a score value as a colour-coded badge cell."""
        if pd.isna(val):
            return "<td style='text-align:center;color:#aaa'>—</td>"
        v = float(val)
        if v >= 0.05:
            bg, fg, label = "#d4f1dc", "#1a7a3c", f"+{v:.3f}"
        elif v <= -0.05:
            bg, fg, label = "#fde0e0", "#c0392b", f"{v:.3f}"
        else:
            bg, fg, label = "#fff3cd", "#8a6200", f"{v:.3f}"
        bar_pct = int((v + 1) / 2 * 100)
        bar_col = fg
        return (
            f"<td style='padding:4px 8px'>"
            f"<div style='font-size:12px;font-weight:700;color:{fg};background:{bg};"
            f"border-radius:5px;padding:2px 7px;display:inline-block;min-width:54px;"
            f"text-align:center'>{label}</div>"
            f"<div style='margin-top:3px;height:4px;background:#e8e8e8;border-radius:2px;overflow:hidden'>"
            f"<div style='width:{bar_pct}%;height:100%;background:{bar_col};border-radius:2px'></div></div>"
            f"</td>"
        )

    def _delta_cell(val) -> str:
        if pd.isna(val):
            return "<td style='text-align:center;color:#aaa'>—</td>"
        v = float(val)
        if v > 0.01:
            fg, arrow = "#1a7a3c", "▲"
        elif v < -0.01:
            fg, arrow = "#c0392b", "▼"
        else:
            fg, arrow = "#8a6200", "►"
        return f"<td style='text-align:center;font-weight:700;color:{fg};font-size:13px'>{arrow} {v:+.3f}</td>"

    header = (
        "<thead><tr style='background:#f4f4f4;font-size:12px;font-weight:700;color:#555'>"
        "<th style='padding:8px'>Conversation</th>"
        "<th>Arc</th>"
        "<th style='text-align:center'>Turns</th>"
        "<th style='text-align:center'>Start Score</th>"
        "<th style='text-align:center'>Mid Score</th>"
        "<th style='text-align:center'>End Score</th>"
        "<th style='text-align:center'>Δ Start→End</th>"
        "<th>Resolution</th>"
        "<th style='text-align:center'>Escalated</th>"
        "</tr></thead>"
    )
    rows_html = ""
    for i, row in display.iterrows():
        esc_icon = "🚨" if row["escalated"] else "✅"
        bg_row = "#fff" if i % 2 == 0 else "#fafafa"
        rows_html += (
            f"<tr style='background:{bg_row};font-size:12px'>"
            f"<td style='padding:5px 8px;font-family:monospace;font-size:11px;color:#555'>{str(row['conversation_id'])[:28]}</td>"
            f"<td style='padding:5px 8px;white-space:nowrap'>{row['arc']}</td>"
            f"<td style='text-align:center;padding:5px 8px'>{int(row['turns'])}</td>"
            + _score_cell(row["start_score"])
            + _score_cell(row["mid_score"])
            + _score_cell(row["end_score"])
            + _delta_cell(row["delta"])
            + f"<td style='padding:5px 8px;font-size:11px'>{row['resolution']}</td>"
            + f"<td style='text-align:center'>{esc_icon}</td>"
            "</tr>"
        )
    st.markdown(
        f"<div style='overflow-x:auto'>"
        f"<table style='border-collapse:collapse;width:100%;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden'>"
        f"{header}<tbody>{rows_html}</tbody></table></div>",
        unsafe_allow_html=True,
    )

    # ── Quick summary pills below the table ───────────────────────────────────
    arc_counts = view["arc"].value_counts()
    summary_parts = "  ·  ".join(
        f"{arc}  **{cnt}**" for arc, cnt in arc_counts.items()
    )
    st.caption(f"Arc breakdown in current filter: {summary_parts}")


def page_sankey(df_r, ins):
    aggs = _precompute_aggs(df_r)

    sh("🌊", "Sentiment Flow Analysis — Sankey Charts")

    # ── Insight summary strip ──────────────────────────────────────────────────
    pcd  = ins.get("phase_csat_dsat", {})
    s_cs = pcd.get("start",  {}).get("csat_pct", 0)
    e_cs = pcd.get("end",    {}).get("csat_pct", 0)
    m_ds = pcd.get("middle", {}).get("dsat_pct", 0)
    delta = e_cs - s_cs
    d_col = C["pos"] if delta >= 0 else C["neg"]
    d_arrow = "▲" if delta >= 0 else "▼"

    ka, kb, kc, kd = st.columns(4)
    with ka:
        st.markdown(mc("Start CSAT",   f"{s_cs:.0%}", C["pos"]),  unsafe_allow_html=True)
    with kb:
        st.markdown(mc("End CSAT",     f"{e_cs:.0%}", C["ok"]),   unsafe_allow_html=True)
    with kc:
        st.markdown(mc("Middle DSAT",  f"{m_ds:.0%}", C["warn"]), unsafe_allow_html=True)
    with kd:
        st.markdown(mc("Start→End Δ",
                       f'<span style="color:{d_col}">{d_arrow} {abs(delta):.0%}</span>',
                       d_col), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3 Sankey tabs: Phase Flow + Outcome Flow + Flow Table ───────────────
    tab1, tab2, tab3 = st.tabs(["🌊 Phase Flow", "🎯 Outcome Flow", "📋 Flow Table"])

    with tab1:
        st.markdown(
            f'<div style="background:{C["warm_l"]};border-left:3px solid {C["teal"]};'
            f'border-radius:6px;padding:8px 14px;font-size:12px;color:{C["text2"]};margin-bottom:12px">'
            f'📖 <strong>How to read:</strong> Each block is a phase (Start / Middle / End). '
            f'Each colour is a sentiment (🟢 Positive · 🔵 Neutral · 🔴 Negative). '
            f'Band width = number of conversations carrying that sentiment into the next phase.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_chart_sankey_phase_flow(aggs), width="stretch")

    with tab2:
        st.markdown(
            f'<div style="background:{C["warm_l"]};border-left:3px solid {C["teal"]};'
            f'border-radius:6px;padding:8px 14px;font-size:12px;color:{C["text2"]};margin-bottom:12px">'
            f'📖 <strong>How to read:</strong> '
            f'<strong>Left</strong> = how the conversation started (customer sentiment). '
            f'<strong>Middle</strong> = true business outcome (hybrid: sentiment + language + outcome logic). '
            f'<strong>Right</strong> = how it ended. '
            f'Key paths: 🔴 Positive Start → Unresolved → Negative End = deterioration. '
            f'✅ Negative Start → Truly Resolved → Positive End = recovery success.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_chart_sankey_outcome_flow(aggs), width="stretch")

        # Resolution status summary cards
        res_data = [
            ("✅ Truly Resolved",        aggs["outcome_flow_df"]["resolution_status"].eq("Truly Resolved").sum()       if not aggs["outcome_flow_df"].empty else 0, C["pos"]),
            ("🔶 Partially Resolved",    aggs["outcome_flow_df"]["resolution_status"].eq("Partially Resolved").sum()   if not aggs["outcome_flow_df"].empty else 0, C["gold"]),
            ("⛔ Unresolved",            aggs["outcome_flow_df"]["resolution_status"].eq("Unresolved").sum()           if not aggs["outcome_flow_df"].empty else 0, C["warn"]),
            ("🔴 Escalated/Unrecovered", aggs["outcome_flow_df"]["resolution_status"].eq("Escalated/Unrecovered").sum() if not aggs["outcome_flow_df"].empty else 0, C["neg"]),
        ]
        total_r = sum(v for _,v,_ in res_data) or 1
        st.markdown("<br>", unsafe_allow_html=True)
        rc1, rc2, rc3, rc4 = st.columns(4)
        for col, (lbl, cnt, color) in zip([rc1,rc2,rc3,rc4], res_data):
            with col:
                st.markdown(
                    mc(lbl, f'<span style="color:{color}">{int(cnt):,} ({cnt/total_r:.0%})</span>', color),
                    unsafe_allow_html=True,
                )

    with tab3:
        _sankey_flow_table(df_r, aggs, ins)

# ─── Overview ─────────────────────────────────────────────────────────────────
# Category keyword sets — same clusters used in escalation analysis
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Billing / Payment":   ["bill","payment","charge","refund","fee","price","cost","money","paid","invoice","credit","debit"],
    "Wait Time / Delays":  ["wait","waiting","hold","long","hours","days","slow","delay","delayed","response","still","week"],
    "Agent / Service":     ["agent","representative","rep","rude","unhelpful","manager","supervisor","escalate","attitude","service"],
    "Account / Access":    ["account","login","password","access","locked","username","blocked","verify","reset","portal"],
    "Product / Service":   ["product","device","broken","defective","quality","feature","update","software","hardware","app"],
    "Delivery / Order":    ["delivery","order","shipped","tracking","package","arrived","missing","lost","late","dispatch"],
    "Repeat Contact":      ["again","third","second","called back","same issue","already","previous","last time","before","still not"],
    "Dissatisfaction":     ["unacceptable","ridiculous","disgusted","terrible","worst","awful","horrible","disgrace","furious","outraged"],
}


@st.cache_data(show_spinner=False)
def _category_deterioration(df: pd.DataFrame, aggs: dict, threshold: float = 0.45) -> None:
    """
    Category Deterioration Insights.

    For each category (Billing, Wait Time, etc.) find conversations where:
      • The category's keywords appear in ANY customer turn
      • Customer sentiment at START phase was positive (avg compound ≥ 0.05)
      • Customer sentiment at END phase was negative   (avg compound < -0.05)

    Only surfaces categories where ≥ threshold (default 45%) of
    positive-start conversations ended negative.

    This catches the pattern: customer is initially cooperative/polite
    but the interaction itself drives them into a negative state —
    a clear signal of a systemic failure in that topic area.
    """
    outcome_df = aggs.get("outcome_flow_df", pd.DataFrame())

    # Build per-conversation start/end sentiment from outcome_flow_df if available,
    # otherwise compute directly from the turns DataFrame.
    if not outcome_df.empty and "start_sent" in outcome_df.columns:
        conv_arc = outcome_df.set_index("conversation_id")[["start_sent", "end_sent"]]
    else:
        # Fallback: compute from phase averages
        if "phase" not in df.columns:
            st.info("Phase data not available — run the pipeline first.")
            return
        cust = df[df["speaker"] == "CUSTOMER"]
        start_avg = (
            cust[cust["phase"] == "start"]
            .groupby("conversation_id")["compound"].mean()
            .rename("start_avg")
        )
        end_avg = (
            cust[cust["phase"] == "end"]
            .groupby("conversation_id")["compound"].mean()
            .rename("end_avg")
        )
        merged = pd.concat([start_avg, end_avg], axis=1).dropna()
        merged["start_sent"] = merged["start_avg"].apply(
            lambda v: "positive" if v >= 0.05 else "negative" if v <= -0.05 else "neutral"
        )
        merged["end_sent"] = merged["end_avg"].apply(
            lambda v: "positive" if v >= 0.05 else "negative" if v <= -0.05 else "neutral"
        )
        conv_arc = merged[["start_sent", "end_sent"]]

    # Build a lookup: conversation_id → set of lowercase customer message tokens
    cust_msgs = (
        df[df["speaker"] == "CUSTOMER"]
        .groupby("conversation_id")["cleaned_message"]
        .apply(lambda msgs: " ".join(msgs.fillna("").tolist()))
    )

    # ── Early exit: check whether the dataset has any positive-start conversations
    n_total_convs    = len(conv_arc)
    n_positive_start = int((conv_arc["start_sent"] == "positive").sum()) if not conv_arc.empty else 0
    if n_positive_start == 0:
        overall_avg = df[df["speaker"] == "CUSTOMER"]["compound"].mean() if "compound" in df.columns else 0.0
        st.info(
            f"**Category Deterioration requires positive-start conversations — none found in this dataset.**\n\n"
            f"This section tracks customers who began a conversation in a positive or neutral mood and ended "
            f"negative. It cannot run when *all* conversations start with a negative customer sentiment.\n\n"
            f"Your dataset has **{n_total_convs:,} conversations** with an overall customer sentiment average "
            f"of **{overall_avg:+.3f}** — indicating a predominantly negative or already-distressed customer base. "
            f"This is useful context on its own: customers are arriving frustrated before agents even respond."
        )
        return
    elif n_positive_start < 5:
        st.warning(
            f"Only **{n_positive_start}** positive-start conversation(s) found out of {n_total_convs:,}. "
            f"Results below are based on a very small sample and may not be statistically reliable."
        )

    results = []
    for category, keywords in _CATEGORY_KEYWORDS.items():
        # Conversations that contain at least one keyword from this category
        kw_mask = cust_msgs.apply(lambda text: any(kw in text for kw in keywords))
        cat_conv_ids = set(cust_msgs[kw_mask].index)

        if len(cat_conv_ids) < 3:   # skip tiny categories
            continue

        cat_arc = conv_arc[conv_arc.index.isin(cat_conv_ids)]
        started_positive = cat_arc[cat_arc["start_sent"] == "positive"]
        n_started_pos    = len(started_positive)

        if n_started_pos < 2:
            continue

        ended_negative = started_positive[started_positive["end_sent"] == "negative"]
        n_deteriorated = len(ended_negative)
        pct            = n_deteriorated / n_started_pos

        results.append({
            "category":        category,
            "total_convs":     len(cat_conv_ids),
            "started_positive":n_started_pos,
            "ended_negative":  n_deteriorated,
            "deterioration_pct": pct,
            "alert":           pct >= threshold,
        })

    if not results:
        st.info(
            f"No categories had enough positive-start conversations to measure deterioration. "
            f"**{n_positive_start}** positive-start conversation(s) were found but none matched a "
            f"topic category with at least 2 positive-start conversations. "
            f"Try uploading a larger dataset or check that customer messages contain topic keywords "
            f"(billing, account, wait, service, etc.)."
        )
        return

    # Primary sort: volume (ended_negative) desc; secondary: rate desc
    max_det = max(r["ended_negative"] for r in results) if results else 1
    results.sort(key=lambda x: (x["ended_negative"], x["deterioration_pct"]), reverse=True)
    alerts = [r for r in results if r["alert"]]
    others = [r for r in results if not r["alert"]]

    # ── Insight caption ───────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{C["warm_l"]};border-left:3px solid {C["neg"]};'
        f'border-radius:6px;padding:10px 14px;font-size:12px;color:{C["text2"]};margin-bottom:14px">'
        f'For each topic category, shows what % of conversations that <strong>started positive</strong> '
        f'ended <strong>negative</strong>. '
        f'<strong>Sorted by volume</strong> (count of deteriorated conversations) — a category with 40 '
        f'deteriorated conversations ranks above one with 3, even at a higher rate. '
        f'Categories above <strong>{threshold:.0%}</strong> are flagged — the interaction itself drove '
        f'the customer from positive to negative.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Alert cards — categories above threshold ──────────────────────────────
    if alerts:
        st.markdown(
            f'<div style="font-weight:700;font-size:13px;color:{C["err"]};margin-bottom:8px">'
            f'🚨 Above {threshold:.0%} threshold — systemic deterioration ({len(alerts)} categor{"y" if len(alerts)==1 else "ies"})'
            f'</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(alerts), 4))
        for i, r in enumerate(alerts):
            pct     = r["deterioration_pct"]
            bar_w   = int(pct * 100)
            sev_col = C["neg"] if pct >= 0.65 else C["warn"]
            with cols[i % 4]:
                st.markdown(
                    f'<div style="background:#fff;border:1px solid {sev_col};border-top:3px solid {sev_col};'
                    f'border-radius:10px;padding:14px 14px 10px">'
                    f'<div style="font-size:12px;font-weight:700;color:{C["text"]};margin-bottom:6px">{r["category"]}</div>'
                    f'<div style="font-size:28px;font-weight:700;color:{sev_col};line-height:1">{pct:.0%}</div>'
                    f'<div style="font-size:10px;color:{C["muted"]};margin:3px 0 8px">positive start → negative end</div>'
                    f'<div style="background:{C["warm"]};border-radius:4px;height:6px;overflow:hidden">'
                    f'<div style="width:{bar_w}%;height:100%;background:{sev_col};border-radius:4px"></div></div>'
                    f'<div style="font-size:10px;color:{C["muted"]};margin-top:6px">'
                    f'{r["ended_negative"]} of {r["started_positive"]} positive-start convs &nbsp;·&nbsp; '
                    f'{r["total_convs"]} total in category</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── All categories table ───────────────────────────────────────────────────
    with st.expander(f"All categories ({len(results)} total)", expanded=False):
        tbl_rows = ""
        for r in results:
            pct     = r["deterioration_pct"]
            bar_w   = int(pct * 100)
            bar_col = C["neg"] if pct >= threshold else C["ok"]
            flag    = "🚨" if r["alert"] else "✅"
            warm_c = C['warm']
            tbl_rows += (
                f"<tr>"
                f"<td>{flag} <strong>{r['category']}</strong></td>"
                f"<td style='text-align:right'>{r['total_convs']:,}</td>"
                f"<td style='text-align:right'>{r['started_positive']:,}</td>"
                f"<td style='text-align:right'>{r['ended_negative']:,}</td>"
                f"<td style='min-width:160px'>"
                f"<div style='display:flex;align-items:center;gap:6px'>"
                f"<div style='flex:1;background:{warm_c};border-radius:3px;height:7px;overflow:hidden'>"
                f"<div style='width:{bar_w}%;height:100%;background:{bar_col};border-radius:3px'></div></div>"
                f"<span style='font-size:12px;font-weight:700;color:{bar_col};width:40px'>{pct:.0%}</span>"
                f"</div></td>"
                f"</tr>"
            )
        st.markdown(
            f"<table class='pt'><thead><tr>"
            f"<th>Category</th><th>Total Convs</th><th>Started Positive</th>"
            f"<th>Ended Negative</th><th>Deterioration %</th>"
            f"</tr></thead><tbody>{tbl_rows}</tbody></table>",
            unsafe_allow_html=True,
        )


RESOLVED_PHRASES = [
    "thank you","thanks so much","that worked","resolved","sorted","all set",
    "appreciate your help","great help","got it","perfect","that's all i needed",
    "you've fixed","problem solved","issue resolved","taken care","wonderful",
    "excellent service","fixed it","happy with","that helps","thank you so much",
    "much appreciated","issue is fixed","working now","it works","that's great",
    "you've been helpful","fully resolved","completely resolved","no more issues",
    "everything is fine","all good now","looks good","that's sorted","brilliant",
    "that did it","working perfectly","no further issues","satisfied",
    "happy now","great service","you've sorted","glad that's sorted",
    "really appreciate","no other issues","nothing else","that's everything",
    # From actual transcripts (Spotify, PPT, Set11)
    "glad to assist","glad i was able to help","glad i could help",
    "have a nice day","have a great day","have a wonderful day",
    "you're welcome","you are welcome","you're most welcome",
    "successfully cancelled","successfully updated","successfully processed",
    "successfully resolved","feel free to contact","will be all",
    "nothing else needed","no further questions","thanks a lot",
    "thanks very much","many thanks","appreciate your time",
    "all sorted out","everything is sorted","happy to help",
    "issue has been resolved","issue has been fixed","account is updated",
    "got that sorted","payment processed","appointment cancelled",
    "appointment rescheduled","subscription cancelled",
]
UNRESOLVED_PHRASES = [
    "still not working","still not fixed","hasn't been fixed","not fixed",
    "same problem","not resolved","calling back","not working","still waiting",
    "no one helped","not happy","this is unacceptable","not satisfied",
    "doesn't work","still having","keep getting","continues to","nothing changed",
    "never resolved","same issue","back again","problem persists","still broken",
    "still happening","still the same","not been resolved","not been sorted",
    "still an issue","escalate this","speak to manager","speak to supervisor",
    "this is ridiculous","this is a joke","terrible service","awful service",
    "cancel my account","close my account","this is disgraceful",
    "very disappointed","extremely disappointed","absolutely terrible",
    "going nowhere","wasting my time","no progress","still pending",
    "still the problem","hasn't changed","still exists","unresolved",
    "not been dealt","not been addressed","without resolution",
    # From actual transcripts (Spotify, PPT, Set11)
    "need a refund","want a refund","i want my money back","money back",
    "request a refund","want my money back","speak to a supervisor",
    "speak to supervisor","want to speak to manager","speak with a manager",
    "waste of time","kept me waiting","on hold again","transferred again",
    "keep transferring","wrong information","misinformed","misleading information",
    "third time contacting","called again","no response","no reply",
    "still waiting for","extremely frustrated","very frustrated",
    "really frustrated","not getting anywhere","filing a complaint",
    "raised a complaint","going to complain",
]


@st.cache_data(show_spinner=False)
def _compute_resolution_audit(df: pd.DataFrame, last_n_turns: int = 5) -> pd.DataFrame:
    """
    Compute word-signal + sentiment audit per conversation.
    Returns a DataFrame — no widgets, safe to cache.

    Word signals (primary): scan last N customer turns for resolution /
    unresolved phrases → word_verdict.

    Sentiment (secondary): end sentiment acts as tiebreaker when word
    signals are ambiguous or absent, and as a confidence flag when
    word signals and sentiment disagree.

    Combined verdict logic:
      Word=Resolved   + Sent=positive/neutral → Resolved
      Word=Resolved   + Sent=negative         → Resolved ⚠ Sent−
      Word=Unresolved + Sent=negative/neutral → Unresolved
      Word=Unresolved + Sent=positive         → Unresolved ⚠ Sent+
      Word=Ambiguous  + Sent=positive         → Likely Resolved (sentiment breaks tie)
      Word=Ambiguous  + Sent=negative         → Likely Unresolved (sentiment breaks tie)
      Word=Ambiguous  + Sent=neutral          → Ambiguous
      Word=No Signal  + Sent=positive         → Likely Resolved (sentiment only)
      Word=No Signal  + Sent=negative         → Likely Unresolved (sentiment only)
      Word=No Signal  + Sent=neutral          → Inconclusive
    """
    rows = []
    for cid, grp in df.groupby("conversation_id", sort=False):
        cust = grp[grp["speaker"] == "CUSTOMER"].sort_values("turn_sequence")
        if cust.empty:
            continue

        res_status = str(grp["resolution_status"].iloc[0]) if "resolution_status" in grp.columns else "Unresolved"
        res_score  = float(grp["resolution_score"].iloc[0]) if "resolution_score"  in grp.columns else 0.0

        last_turns = cust.tail(last_n_turns)
        scan_text  = " ".join(last_turns["cleaned_message"].fillna("").tolist()).lower()
        end_avg    = float(last_turns["compound"].mean())
        start_avg  = float(cust.head(3)["compound"].mean())
        delta      = end_avg - start_avg

        matched_res = [p for p in RESOLVED_PHRASES   if p in scan_text]
        matched_unr = [p for p in UNRESOLVED_PHRASES if p in scan_text]
        pos_hits    = len(matched_res)
        neg_hits    = len(matched_unr)

        # ── Primary: word-signal verdict ─────────────────────────────────────
        if   pos_hits > 0 and neg_hits == 0: word_verdict = "Resolved"
        elif neg_hits > 0 and pos_hits == 0: word_verdict = "Unresolved"
        elif pos_hits > 0 and neg_hits > 0:  word_verdict = "Ambiguous"
        else:                                 word_verdict = "No Signal"

        # ── Secondary: sentiment direction ───────────────────────────────────
        if   end_avg >= 0.05:  sent_dir = "positive"
        elif end_avg <= -0.05: sent_dir = "negative"
        else:                   sent_dir = "neutral"

        # ── Combined verdict (word primary, sentiment secondary) ─────────────
        if word_verdict == "Resolved":
            combined = "Resolved ⚠ Sent−" if sent_dir == "negative" else "Resolved"
        elif word_verdict == "Unresolved":
            combined = "Unresolved ⚠ Sent+" if sent_dir == "positive" else "Unresolved"
        elif word_verdict == "Ambiguous":
            if   sent_dir == "positive": combined = "Likely Resolved"
            elif sent_dir == "negative": combined = "Likely Unresolved"
            else:                         combined = "Ambiguous"
        else:  # No Signal — sentiment is the only guide
            if   sent_dir == "positive": combined = "Likely Resolved"
            elif sent_dir == "negative": combined = "Likely Unresolved"
            else:                         combined = "Inconclusive"

        # ── Audit result vs pipeline ──────────────────────────────────────────
        is_resolved = res_status == "Truly Resolved"
        combined_says_resolved   = combined in ("Resolved", "Resolved ⚠ Sent−", "Likely Resolved")
        combined_says_unresolved = combined in ("Unresolved", "Unresolved ⚠ Sent+", "Likely Unresolved")

        if   is_resolved     and combined_says_unresolved: audit = "False Positive"
        elif not is_resolved and combined_says_resolved:   audit = "False Negative"
        elif combined in ("Ambiguous", "Inconclusive"):    audit = combined
        else:                                               audit = "Match"

        rows.append({
            "conversation_id":    cid,
            "pipeline_status":    res_status,
            "resolution_score":   round(res_score, 3),
            "word_verdict":       word_verdict,
            "sent_direction":     sent_dir,
            "combined_verdict":   combined,
            "audit_result":       audit,
            "end_sentiment":      round(end_avg, 3),
            "start_sentiment":    round(start_avg, 3),
            "sentiment_delta":    round(delta, 3),
            "matched_resolved":   ", ".join(matched_res) if matched_res else "—",
            "matched_unresolved": ", ".join(matched_unr) if matched_unr else "—",
            "res_hits":           pos_hits,
            "unr_hits":           neg_hits,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _resolution_audit(df: pd.DataFrame, last_n_turns: int = 5) -> None:
    """Render the Resolution Signal Audit — not cached (contains widgets)."""
    audit_df = _compute_resolution_audit(df, last_n_turns)

    if audit_df.empty:
        st.info("No conversations found for audit.")
        return

    # ── Summary KPI strip ────────────────────────────────────────────────────
    n_total = len(audit_df)
    n_match = int((audit_df["audit_result"] == "Match").sum())
    n_fp    = int((audit_df["audit_result"] == "False Positive").sum())
    n_fn    = int((audit_df["audit_result"] == "False Negative").sum())
    n_amb   = int((audit_df["audit_result"].isin(["Ambiguous","Inconclusive"])).sum())
    match_rt = n_match / max(n_total, 1)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(mc("Audited",           f"{n_total:,}",                  C["teal"]),  unsafe_allow_html=True)
    with k2: st.markdown(mc("✅ Match",           f"{n_match:,} ({match_rt:.0%})", C["ok"]),    unsafe_allow_html=True)
    with k3: st.markdown(mc("🔴 False Positive",  f"{n_fp:,}",                    C["neg"]),   unsafe_allow_html=True)
    with k4: st.markdown(mc("🟡 False Negative",  f"{n_fn:,}",                    C["warn"]),  unsafe_allow_html=True)
    with k5: st.markdown(mc("⚪ Ambiguous",        f"{n_amb:,}",                   C["slate"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How it works caption ─────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{C["warm_l"]};border-left:3px solid {C["teal"]};'
        f'border-radius:6px;padding:10px 14px;font-size:12px;color:{C["text2"]};margin-bottom:14px">'
        f'<strong>Word signals (primary)</strong> scanned from last {last_n_turns} customer turns → '
        f'Resolved / Unresolved / Ambiguous / No Signal. &nbsp;'
        f'<strong>Sentiment (secondary)</strong> breaks ties when word signals are ambiguous or absent, '
        f'and flags disagreements (e.g. "Resolved ⚠ Sent−" = phrases say resolved but customer sentiment stayed negative).<br>'
        f'<strong>False Positive</strong> = pipeline says Truly Resolved but combined verdict says Unresolved. &nbsp; '
        f'<strong>False Negative</strong> = pipeline says not resolved but combined verdict says Resolved.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Audit result filter ───────────────────────────────────────────────────
    all_audits  = sorted(audit_df["audit_result"].unique().tolist())
    audit_counts = audit_df["audit_result"].value_counts().to_dict()
    filter_opts  = ["All"] + all_audits
    filter_labels = {"All": f"All ({n_total})"}
    for a in all_audits:
        filter_labels[a] = f"{a} ({audit_counts.get(a, 0)})"

    selected = st.radio(
        "Filter by audit result",
        options=filter_opts,
        format_func=lambda v: filter_labels.get(v, v),
        horizontal=True,
        key="audit_verdict_filter",
    )

    view = audit_df if selected == "All" else audit_df[audit_df["audit_result"] == selected]
    if view.empty:
        st.success(f"No conversations in '{selected}' category.")
        return

    # ── Colour maps ───────────────────────────────────────────────────────────
    AUDIT_COLORS = {
        "False Positive":   C["neg"],
        "False Negative":   C["warn"],
        "Ambiguous":        C["slate"],
        "Inconclusive":     C["muted"],
        "Match":            C["ok"],
    }
    COMBINED_COLORS = {
        "Resolved":              C["ok"],
        "Resolved ⚠ Sent−":     C["warn"],
        "Unresolved":            C["neg"],
        "Unresolved ⚠ Sent+":   C["warn"],
        "Likely Resolved":       "#5A9E6F",
        "Likely Unresolved":     "#C06060",
        "Ambiguous":             C["slate"],
        "Inconclusive":          C["muted"],
    }

    tbl_rows = ""
    for _, r in view.iterrows():
        a_col  = AUDIT_COLORS.get(r["audit_result"],   C["text"])
        cv_col = COMBINED_COLORS.get(r["combined_verdict"], C["text"])
        wv_col = C["ok"] if r["word_verdict"] == "Resolved" else \
                 C["neg"] if r["word_verdict"] == "Unresolved" else C["slate"]
        s_col  = C["ok"] if r["sent_direction"] == "positive" else \
                 C["neg"] if r["sent_direction"] == "negative" else C["neu"]
        res_html = (
            f'<span style="font-size:11px;color:{C["ok"]}">{r["matched_resolved"]}</span>'
            if r["matched_resolved"] != "—"
            else f'<span style="color:{C["muted"]}">—</span>'
        )
        unr_html = (
            f'<span style="font-size:11px;color:{C["neg"]}">{r["matched_unresolved"]}</span>'
            if r["matched_unresolved"] != "—"
            else f'<span style="color:{C["muted"]}">—</span>'
        )
        tbl_rows += (
            f"<tr>"
            f"<td style='font-size:11px'>{r['conversation_id']}</td>"
            f"<td style='font-size:11px'>{r['pipeline_status']}</td>"
            f"<td style='text-align:center'>"
            f"<span style='font-weight:700;color:{wv_col};font-size:11px'>{r['word_verdict']}</span></td>"
            f"<td style='text-align:center;color:{s_col};font-weight:600;font-size:11px'>"
            f"{r['sent_direction']} ({r['end_sentiment']:+.3f})</td>"
            f"<td style='text-align:center'>"
            f"<span style='font-weight:700;color:{cv_col};font-size:11px'>{r['combined_verdict']}</span></td>"
            f"<td style='text-align:center'>"
            f"<span style='font-weight:700;color:{a_col};font-size:11px'>{r['audit_result']}</span></td>"
            f"<td>{res_html}</td>"
            f"<td>{unr_html}</td>"
            f"</tr>"
        )

    st.markdown(
        f"<table class='pt'><thead><tr>"
        f"<th>Conversation</th><th>Pipeline Status</th>"
        f"<th>Word Signal</th><th>Sentiment</th>"
        f"<th>Combined Verdict</th><th>Audit Result</th>"
        f"<th>✅ Resolution Phrases</th><th>🔴 Unresolved Phrases</th>"
        f"</tr></thead><tbody>{tbl_rows}</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Showing {len(view):,} of {n_total:,} conversations · "
        f"Last {last_n_turns} customer turns scanned · "
        f"⚠ suffix = word signal and sentiment disagree — worth manual review."
    )


def page_overview(df_r, ins):
    aggs = _precompute_aggs(df_r)

    # ── 1. Phase CSAT/DSAT health table ──────────────────────────────────────
    sh("📊", "Phase-Level CSAT / DSAT")
    _phase_table(ins)

    # ── 2. CSAT vs DSAT bar + Start→End Sankey side by side ──────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sh("📈", "CSAT vs DSAT by Phase")
        st.plotly_chart(_chart_phase_comparison(ins), width="stretch")
    with c2:
        sh("🎯", "Start → End Sentiment")
        st.plotly_chart(_chart_sankey_start_to_end(aggs), width="stretch")

    # ── 3. Top escalation trigger phrases ────────────────────────────────────
    sh("⚠️", "Top Escalation Trigger Phrases")
    _escalation_triggers_table(df_r, top_n=15)

    # ── 4. Category Deterioration Insights ───────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sh("📉", "Category Deterioration — Positive Start → Negative End")
    _category_deterioration(df_r, aggs, threshold=0.45)

    # ── 5. Resolution Signal Audit ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sh("🔬", "Resolution Signal Audit")
    st.caption(
        "Word-signal driven audit — classifies each conversation using resolution and unresolved "
        "phrase matches, then compares against the pipeline classification to surface mismatches."
    )
    _sa_col, _ = st.columns([1, 3])
    with _sa_col:
        last_n = st.slider(
            "Turns to scan (last N customer turns)",
            min_value=3, max_value=10, value=5, step=1,
            help="How many of the final customer turns to scan for word signals. "
                 "Higher = more context captured but may include mid-conversation noise.",
            key="audit_last_n",
        )
    _resolution_audit(df_r, last_n_turns=last_n)

# ─── Explorer (merged TbT Flow + turn viewer + comparison) ──────────────────
def _progressive_log(df: pd.DataFrame, conv_id: str) -> None:
    """
    Progressive Sentiment Log — tracks the full sentiment arc from Turn 1 to Turn N.

    For each turn, shows:
      • Turn number + phase + speaker
      • Message text
      • Sentiment score with a trend arrow vs the previous turn
      • Running arc state (recovering / declining / stable)

    Arc classification at the top:
      Recovery      — customer sentiment starts negative, ends positive
      Deterioration — starts positive, ends negative
      Improvement   — gradual positive shift throughout
      Decline       — gradual negative shift throughout
      Stable        — minimal net change
      Volatile      — large swings with no clear direction
    """
    sub = df[df["conversation_id"] == conv_id].sort_values("turn_sequence").reset_index(drop=True)
    if sub.empty:
        st.info("No turns for this conversation.")
        return

    # ── Arc summary — use phase groups, fall back to head/tail ────────────────
    def _phase_avg(phase: str) -> float:
        if "phase" in sub.columns:
            vals = sub[sub["phase"] == phase]["compound"]
            return float(vals.mean()) if not vals.empty else 0.0
        return 0.0

    start_avg = _phase_avg("start")  if "phase" in sub.columns else float(sub.head(3)["compound"].mean())
    mid_avg   = _phase_avg("middle") if "phase" in sub.columns else float(sub["compound"].mean())
    end_avg   = _phase_avg("end")    if "phase" in sub.columns else float(sub.tail(3)["compound"].mean())
    delta     = end_avg - start_avg

    # Classify arc
    if start_avg <= -0.05 and end_avg >= 0.05:
        arc_label = "Recovery — Negative Start → Positive End"
        arc_icon  = "📈"
        arc_color = C["pos"]
        arc_bg    = "rgba(46,204,113,0.08)"
        arc_border= C["pos"]
    elif start_avg >= 0.05 and end_avg <= -0.05:
        arc_label = "Deterioration — Positive Start → Negative End"
        arc_icon  = "📉"
        arc_color = C["neg"]
        arc_bg    = "rgba(231,76,60,0.08)"
        arc_border= C["neg"]
    elif delta > 0.10:
        arc_label = "Improvement — Gradual Positive Shift"
        arc_icon  = "↗️"
        arc_color = C["gold"]
        arc_bg    = "rgba(212,185,78,0.08)"
        arc_border= C["gold"]
    elif delta < -0.10:
        arc_label = "Decline — Gradual Negative Shift"
        arc_icon  = "↘️"
        arc_color = C["warn"]
        arc_bg    = "rgba(184,150,62,0.08)"
        arc_border= C["warn"]
    elif abs(delta) <= 0.05:
        arc_label = "Stable — Minimal Net Change"
        arc_icon  = "➡️"
        arc_color = C["neu"]
        arc_bg    = "rgba(70,130,180,0.06)"
        arc_border= C["neu"]
    else:
        # Large swings, no clear direction
        arc_label = "Volatile — Mixed Sentiment Throughout"
        arc_icon  = "〰️"
        arc_color = C["slate"]
        arc_bg    = "rgba(107,138,153,0.06)"
        arc_border= C["slate"]

    # ── Arc header strip ──────────────────────────────────────────────────────
    def _score_node(label: str, val: float, phase_icon: str) -> str:
        col = _score_color(val)
        return (
            f'<div style="text-align:center;flex:1">'
            f'<div style="font-size:18px">{phase_icon}</div>'
            f'<div style="font-size:11px;color:{C["muted"]};font-weight:600;text-transform:uppercase;'
            f'letter-spacing:.6px;margin:2px 0">{label}</div>'
            f'<div style="font-size:20px;font-weight:700;color:{col}">{val:+.3f}</div>'
            f'</div>'
        )

    arrow_col = C["pos"] if delta >= 0 else C["neg"]
    arrow_sym = "▲" if delta >= 0 else "▼"

    st.markdown(
        f'<div style="background:{arc_bg};border:1px solid {arc_border};border-radius:12px;'
        f'padding:16px 20px;margin-bottom:16px">'
        # Arc label
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">'
        f'<span style="font-size:20px">{arc_icon}</span>'
        f'<span style="font-weight:700;font-size:14px;color:{arc_color}">{arc_label}</span>'
        f'<span style="margin-left:auto;font-size:13px;font-weight:600;color:{arrow_col}">'
        f'{arrow_sym} {abs(delta):.3f} net Δ</span>'
        f'</div>'
        # Phase nodes with connector arrows
        f'<div style="display:flex;align-items:center;gap:4px">'
        + _score_node("Start", start_avg, "🚀")
        + f'<div style="font-size:20px;color:{C["warm"]};flex:0 0 auto;text-align:center;padding:0 4px">→</div>'
        + _score_node("Middle", mid_avg, "🔄")
        + f'<div style="font-size:20px;color:{C["warm"]};flex:0 0 auto;text-align:center;padding:0 4px">→</div>'
        + _score_node("End", end_avg, "🏁")
        + f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Running state legend ──────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:11px;color:{C["muted"]};margin-bottom:8px;font-weight:600">'
        f'STATE LEGEND &nbsp; '
        f'<span style="color:{C["pos"]}">▲ Recovering</span> &nbsp;·&nbsp; '
        f'<span style="color:{C["neg"]}">▼ Declining</span> &nbsp;·&nbsp; '
        f'<span style="color:{C["neu"]}">➡ Stable</span> &nbsp;·&nbsp; '
        f'<span style="color:{C["warn"]}">⚠ Escalating</span> &nbsp;·&nbsp; '
        f'<span style="color:{C["ok"]}">✅ Resolving</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Build each turn row ───────────────────────────────────────────────────
    rows_html = []
    prev_compound = None
    running_state = "stable"   # tracks arc state as we walk turns

    for _, r in sub.iterrows():
        spk      = str(r["speaker"]).upper()
        turn_n   = int(r["turn_sequence"])
        phase    = str(r.get("phase", "middle"))
        compound = float(r["compound"])
        lbl      = str(r.get("sentiment_label", "neutral"))
        conf     = float(r.get("sentiment_confidence", 0))
        msg      = str(r["message"])
        is_esc   = bool(r.get("potential_escalation", False))
        is_res   = bool(r.get("potential_resolution", False))
        ts_      = str(r.get("timestamp", ""))
        ts_str   = f" · {ts_}" if ts_ not in ("", "nan", "None") else ""

        # Trend vs previous turn
        if prev_compound is None:
            trend_arrow = "●"
            trend_color = C["muted"]
        else:
            diff = compound - prev_compound
            if diff > 0.05:
                trend_arrow = "▲"
                trend_color = C["pos"]
            elif diff < -0.05:
                trend_arrow = "▼"
                trend_color = C["neg"]
            else:
                trend_arrow = "─"
                trend_color = C["muted"]

        # Running arc state update
        if is_esc:
            running_state = "escalating"
        elif is_res:
            running_state = "resolving"
        elif prev_compound is not None:
            diff = compound - prev_compound
            if diff > 0.05:
                running_state = "recovering"
            elif diff < -0.05:
                running_state = "declining"
            else:
                running_state = "stable"

        state_map = {
            "recovering":  (f'<span style="color:{C["pos"]};font-weight:700">▲ Recovering</span>',  C["pos"]),
            "declining":   (f'<span style="color:{C["neg"]};font-weight:700">▼ Declining</span>',   C["neg"]),
            "stable":      (f'<span style="color:{C["neu"]}">➡ Stable</span>',                       C["neu"]),
            "escalating":  (f'<span style="color:{C["warn"]};font-weight:700">⚠ Escalating</span>', C["warn"]),
            "resolving":   (f'<span style="color:{C["ok"]};font-weight:700">✅ Resolving</span>',    C["ok"]),
        }
        state_html, state_border = state_map.get(running_state, state_map["stable"])

        # Speaker styling
        spk_icon = "👤" if spk == "CUSTOMER" else "🎧"
        row_bg   = "#FEF5F5" if spk == "CUSTOMER" else "#F0F7FA"
        row_border = C["neg"] if spk == "CUSTOMER" else C["teal"]

        # Phase badge
        phase_badge = (
            f'<span style="font-size:10px;padding:1px 7px;border-radius:4px;font-weight:600;'
            f'background:{"rgba(45,95,110,0.12)" if phase=="start" else "rgba(212,185,78,0.15)" if phase=="middle" else "rgba(160,64,64,0.12)"};'
            f'color:{C["teal"] if phase=="start" else C["gold"] if phase=="middle" else C["err"]}">'
            f'{PHASE_ICONS.get(phase, "🔄")} {phase.capitalize()}</span>'
        )

        # Escalation / resolution flags
        flags = ""
        if is_esc: flags += f' <span style="font-size:11px;color:{C["warn"]}">⚠️ ESC</span>'
        if is_res: flags += f' <span style="font-size:11px;color:{C["ok"]}">✅ RES</span>'

        # Score + trend
        score_col = _score_color(compound)
        score_html = (
            f'<span style="color:{trend_color};font-size:13px;font-weight:700">{trend_arrow}</span>'
            f'&nbsp;<span style="color:{score_col};font-family:\'JetBrains Mono\',monospace;'
            f'font-size:13px;font-weight:700">{compound:+.3f}</span>'
        )

        rows_html.append(
            f'<div style="display:flex;align-items:flex-start;gap:0;margin-bottom:5px;'
            f'border-radius:8px;overflow:hidden;border:1px solid {row_border};'
            f'border-left:4px solid {state_border}">'
            # Left: turn meta column
            f'<div style="min-width:100px;max-width:100px;padding:10px 10px;background:rgba(0,0,0,0.03);'
            f'border-right:1px solid {C["border"]};text-align:center">'
            f'<div style="font-size:11px;color:{C["muted"]};font-weight:700">#{turn_n}</div>'
            f'<div style="font-size:10px;margin-top:2px">{phase_badge}</div>'
            f'<div style="font-size:11px;margin-top:4px;font-weight:600;color:{C["text2"]}">{spk_icon} {spk.capitalize()}</div>'
            f'<div style="font-size:9px;color:{C["muted"]};margin-top:2px">{ts_str.strip(" ·")}</div>'
            f'</div>'
            # Middle: message
            f'<div style="flex:1;padding:10px 12px;background:{row_bg}">'
            f'<div style="font-size:13px;color:{C["text"]};line-height:1.6">{msg}</div>'
            f'</div>'
            # Right: sentiment + state
            f'<div style="min-width:140px;max-width:140px;padding:10px 10px;background:rgba(0,0,0,0.02);'
            f'border-left:1px solid {C["border"]};text-align:center">'
            f'<div>{score_html}</div>'
            f'<div style="margin-top:4px">{_badge(lbl)}</div>'
            f'<div style="font-size:10px;color:{C["muted"]};margin-top:3px">conf {conf:.0%}</div>'
            f'<div style="font-size:11px;margin-top:5px">{state_html}{flags}</div>'
            f'</div>'
            f'</div>'
        )

        prev_compound = compound

    st.markdown("".join(rows_html), unsafe_allow_html=True)


def page_explorer(df_r):
    sh("🗣️", "Conversation Explorer")

    lists = _get_smart_conv_lists(df_r)

    # ── Selectors row ─────────────────────────────────────────────────────
    c_mode, c_search, c_view = st.columns([1.6, 2, 1.2])
    with c_mode:
        conv_mode = st.selectbox(
            "Quick filter",
            ["All conversations",
             "😡 Worst 20 (lowest customer sentiment)",
             "😊 Best 20 (highest customer sentiment)",
             "📏 Longest 20 (most turns)"],
            key="exp_mode",
        )
    pool = {
        "😡 Worst 20 (lowest customer sentiment)": lists["worst20"],
        "😊 Best 20 (highest customer sentiment)": lists["best20"],
        "📏 Longest 20 (most turns)":              lists["longest20"],
    }.get(conv_mode, lists["all_ids"])

    with c_search:
        search_txt = st.text_input("🔍 Search conversation ID", value="",
                                   placeholder="e.g. CONV_0042", key="exp_search")
        if search_txt.strip():
            pool = [c for c in lists["all_ids"]
                    if search_txt.strip().upper() in c.upper()] or pool
    with c_view:
        view_mode = st.radio("View", ["Single", "Compare ×2"],
                             horizontal=True, key="exp_viewmode")

    # ── Conversation pickers ───────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([1, 1, 1])
    with fc1:
        sel = st.selectbox("Conversation A", pool, key="exp_conv")
    with fc2:
        spk_toggle = st.toggle("Split by speaker", value=True, key="exp_spk_toggle")
    with fc3:
        pflt = st.selectbox("Phase filter", ["All","start","middle","end"], key="exp_ph")

    sel_b = None
    if view_mode == "Compare ×2":
        sel_b = st.selectbox("Conversation B",
                             [c for c in pool if c != sel], key="exp_conv_b")

    # ── Mini KPI strip ────────────────────────────────────────────────────
    dv  = df_r[df_r["conversation_id"] == sel].copy()
    if pflt != "All": dv = dv[dv["phase"] == pflt]
    cu  = dv[dv["speaker"] == "CUSTOMER"]
    ag  = dv[dv["speaker"] == "AGENT"]
    esc_n = int(dv["potential_escalation"].sum()) if "potential_escalation" in dv.columns else 0
    res_n = int(dv["potential_resolution"].sum()) if "potential_resolution" in dv.columns else 0
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Turns",          len(df_r[df_r["conversation_id"]==sel]))
    k2.metric("Customer Avg",   f"{cu['compound'].mean():+.3f}" if not cu.empty else "—")
    k3.metric("Agent Avg",      f"{ag['compound'].mean():+.3f}" if not ag.empty else "—")
    k4.metric("⚠️ Escalations", esc_n,
              delta="High" if esc_n > 2 else None, delta_color="inverse")
    k5.metric("✅ Resolutions",  res_n)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Compare mode ──────────────────────────────────────────────────────
    if view_mode == "Compare ×2" and sel_b:
        sh("⚖️", f"Comparison: {sel} vs {sel_b}")
        st.plotly_chart(_chart_compare_two(df_r, sel, sel_b), width="stretch")
        return

    # ── Single conversation tabs: Flow / Progressive Log / Turn Viewer / Data ──
    tab_flow, tab_log, tab_turns, tab_data = st.tabs([
        "📈 Sentiment Flow", "📊 Progressive Log", "🗣️ Turn Viewer", "📋 Data Table"
    ])

    with tab_flow:
        st.plotly_chart(
            _chart_tbt_flow(df_r, sel, show_speaker_lines=spk_toggle),
            width="stretch",
        )
        try:
            import plotly.io as pio
            fig_bytes = pio.to_image(
                _chart_tbt_flow(df_r, sel, show_speaker_lines=spk_toggle),
                format="png", width=1200, height=500, scale=2,
            )
            st.download_button(
                "📷 Export chart as PNG", data=fig_bytes,
                file_name=f"flow_{sel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
            )
        except Exception:
            st.caption("PNG export requires `kaleido` — `pip install kaleido`")

    with tab_log:
        _progressive_log(df_r, sel)

    with tab_turns:
        sub = df_r[df_r["conversation_id"] == sel]
        if pflt != "All": sub = sub[sub["phase"] == pflt]
        _turn_viewer(sub, sel)

    with tab_data:
        sub = df_r[df_r["conversation_id"] == sel].copy()
        cols = [c for c in ["turn_sequence","phase","speaker","timestamp","message",
                             "sentiment_label","compound","sentiment_confidence",
                             "potential_escalation","potential_resolution"] if c in sub.columns]
        st.dataframe(sub[cols].reset_index(drop=True), width="stretch", height=420)
        st.download_button(
            "⬇️ Download this conversation (CSV)",
            data=_to_csv(sub[cols]),
            file_name=f"conv_{sel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )



# ─── Escalation helpers ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _compute_escalation_intel(df: pd.DataFrame) -> dict:
    """
    Full escalation intelligence — computed once, cached by DataFrame content.

    Produces:
      trigger_rows      — top 25 trigger words/bigrams with counts + conv coverage
      esc_phase         — escalation count per phase (start/middle/end)
      severity_dist     — conversation-level severity buckets (low/medium/high/severe)
      recovery_intel    — per-escalation recovery within 3 turns + by conv end
      deteriorated      — top 15 conversations that worsened start→end
      top_escalated     — top 15 most escalated conversations with detail
      avg_agent_response— avg agent sentiment_change on turn immediately after esc
      total_convs / n_escalated / n_resolved / n_unresolved
    """
    lf = pl.from_pandas(df).lazy()

    # ── 1. Escalated customer turns ───────────────────────────────────────────
    esc_df = (
        lf.filter(
            (pl.col("potential_escalation") == True) &
            (pl.col("speaker") == "CUSTOMER")
        ).collect().to_pandas()
    )

    total_convs  = int(df["conversation_id"].nunique())
    esc_conv_ids = set(esc_df["conversation_id"].unique()) if not esc_df.empty else set()
    n_escalated  = len(esc_conv_ids)
    res_df       = df[df["potential_resolution"] == True]
    n_resolved   = int(len(set(res_df["conversation_id"]) & esc_conv_ids))
    n_unresolved = n_escalated - n_resolved

    # ── 2. Phrase-level trigger analysis (filtered, not raw frequency) ────────
    # Strategy:
    #   1. Extract bigrams AND trigrams from escalation turns
    #   2. Filter: only phrases appearing in >= 2 distinct conversations (removes noise)
    #   3. Rank by CONVERSATION COVERAGE (not raw count) — more meaningful for management
    #   4. Cluster into semantic themes: billing/payment, wait time, agent issues, etc.
    STOPWORDS = {
        "i","the","a","an","and","or","but","in","on","at","to","for","of","is",
        "was","are","were","be","been","have","has","had","do","did","not","no",
        "my","your","we","they","it","this","that","with","you","me","so","if",
        "up","can","will","just","don","t","s","re","ve","ll","get","got","its",
        "our","he","she","them","their","about","when","what","how","why","all",
        "would","could","should","there","than","then","from","also","more","very",
        "said","say","know","want","need","like","think","going","come","back",
        "yes","okay","ok","right","sure","well","still","now","time","again",
    }
    # Phrase → set of conv_ids that contain it
    phrase_convs: Dict[str, set] = {}
    phrase_count: Dict[str, int] = {}

    for _, row in esc_df.iterrows():
        cid   = str(row.get("conversation_id", ""))
        msg   = str(row.get("message", "")).lower()
        words = [w for w in re.findall(r"[a-z]{3,}", msg) if w not in STOPWORDS]
        # Build bigrams and trigrams
        for n_gram in (2, 3):
            for i in range(len(words) - n_gram + 1):
                phrase = " ".join(words[i:i+n_gram])
                phrase_count[phrase] = phrase_count.get(phrase, 0) + 1
                phrase_convs.setdefault(phrase, set()).add(cid)

    # Filter: must appear in >= 2 conversations (noise guard)
    # Rank by conversation coverage (unique convs), then by count
    qualified = [
        {
            "phrase":    phrase,
            "count":     cnt,
            "convs":     len(phrase_convs[phrase]),
            "conv_pct":  len(phrase_convs[phrase]) / max(n_escalated, 1),
            "is_bigram": len(phrase.split()) == 2,
        }
        for phrase, cnt in phrase_count.items()
        if len(phrase_convs[phrase]) >= 2   # min 2 conversations
    ]
    # Sort: conv_coverage DESC, then count DESC
    trigger_rows = sorted(qualified, key=lambda x: (x["convs"], x["count"]), reverse=True)[:30]

    # ── Semantic clustering of top phrases ────────────────────────────────────
    # Assign each phrase to a business theme based on keyword matching.
    # Themes are ordered by business priority.
    CLUSTER_RULES: List[tuple] = [
        ("Billing / Payment",    ["bill","payment","charge","refund","fee","price","cost","money","paid","invoice","credit","debit"]),
        ("Wait Time / Delays",   ["wait","waiting","hold","long","hours","days","slow","delay","delayed","response","still","week"]),
        ("Agent / Service",      ["agent","representative","rude","help","helpful","spoke","told","promised","manager","supervisor","transfer"]),
        ("Account / Access",     ["account","login","password","access","locked","reset","email","username","profile","verify"]),
        ("Product / Service",    ["product","service","cancel","subscription","plan","upgrade","downgrade","feature","broken","issue","problem","error","bug"]),
        ("Delivery / Order",     ["order","delivery","shipping","package","arrived","missing","lost","track","tracking","return","refund"]),
        ("Repeat Contact",       ["again","third","second","already","before","previous","last time","called before","told me"]),
        ("Dissatisfaction",      ["terrible","horrible","awful","worst","disappointed","never","unacceptable","ridiculous","disgrace","useless"]),
    ]

    def _assign_cluster(phrase: str) -> str:
        pl = phrase.lower()
        for cluster, keywords in CLUSTER_RULES:
            if any(kw in pl for kw in keywords):
                return cluster
        return "Other"

    for r in trigger_rows:
        r["cluster"] = _assign_cluster(r["phrase"])

    # ── Cluster summary: group trigger_rows by cluster, aggregate ─────────────
    cluster_agg: Dict[str, Dict] = {}
    # Also track which conv_ids belong to each cluster (union of phrase_convs)
    cluster_conv_sets: Dict[str, set] = {}
    for r in trigger_rows:
        cl = r["cluster"]
        if cl not in cluster_agg:
            cluster_agg[cl] = {"cluster": cl, "total_convs": 0, "total_count": 0,
                                "top_phrase": r["phrase"], "phrases": []}
        cluster_agg[cl]["total_convs"]  = max(cluster_agg[cl]["total_convs"], r["convs"])
        cluster_agg[cl]["total_count"] += r["count"]
        cluster_agg[cl]["phrases"].append(r["phrase"])
        # Accumulate conv_ids for drill-down
        cluster_conv_sets.setdefault(cl, set()).update(phrase_convs.get(r["phrase"], set()))
    cluster_summary = sorted(
        cluster_agg.values(),
        key=lambda x: x["total_convs"], reverse=True
    )
    for c in cluster_summary:
        c["conv_pct"] = c["total_convs"] / max(n_escalated, 1)
    # Serialise to sorted lists for caching
    cluster_conv_ids: Dict[str, List[str]] = {
        cl: sorted(ids) for cl, ids in cluster_conv_sets.items()
    }

    # ── 3. Escalation by phase ────────────────────────────────────────────────
    esc_phase = (
        lf.filter(pl.col("potential_escalation") == True)
          .group_by("phase")
          .agg(pl.len().alias("escalations"))
          .collect().to_pandas()
    )
    # Add escalation rate per phase (escalations / total turns in that phase)
    phase_totals = (
        lf.group_by("phase")
          .agg(pl.len().alias("total_turns"))
          .collect().to_pandas()
    )
    res_phase = (
        lf.filter(pl.col("potential_resolution") == True)
          .group_by("phase")
          .agg(pl.len().alias("resolutions"))
          .collect().to_pandas()
    )
    if not esc_phase.empty and not phase_totals.empty:
        esc_phase = esc_phase.merge(phase_totals, on="phase", how="left")
        esc_phase["esc_rate"] = esc_phase["escalations"] / esc_phase["total_turns"].clip(lower=1)
        esc_phase = esc_phase.merge(res_phase, on="phase", how="left")
        esc_phase["resolutions"] = esc_phase["resolutions"].fillna(0).astype(int)
        esc_phase["phase_recovery_rate"] = (
            esc_phase["resolutions"] / esc_phase["escalations"].clip(lower=1) * 100
        ).round(1)
    else:
        esc_phase = pd.DataFrame(columns=["phase","escalations","total_turns","esc_rate","resolutions","phase_recovery_rate"])

    # ── 4. Severity distribution ──────────────────────────────────────────────
    # Per conversation: count escalation turns → bucket by count
    #   low=1, medium=2-3, high=4-6, severe=7+
    # Also factor in end sentiment: if end_sentiment < -0.3 → upgrade one level
    conv_esc_counts = (
        lf.filter(pl.col("potential_escalation") == True)
          .group_by("conversation_id")
          .agg(pl.len().alias("esc_count"))
          .collect().to_pandas()
    )
    conv_end_sent = (
        lf.filter(pl.col("is_conversation_end") == True)
          .filter(pl.col("speaker") == "CUSTOMER")
          .group_by("conversation_id")
          .agg(pl.col("compound").mean().alias("end_compound"))
          .collect().to_pandas()
    )
    if not conv_esc_counts.empty:
        sev_df = conv_esc_counts.merge(conv_end_sent, on="conversation_id", how="left")
        sev_df["end_compound"] = sev_df["end_compound"].fillna(0.0)

        def _severity(row):
            n = row["esc_count"]
            e = row["end_compound"]
            if n >= 7 or (n >= 4 and e < -0.3):  return "Severe"
            if n >= 4 or (n >= 2 and e < -0.3):  return "High"
            if n >= 2:                             return "Medium"
            return "Low"

        sev_df["severity"] = sev_df.apply(_severity, axis=1)
        severity_dist = (
            sev_df.groupby("severity")["conversation_id"]
              .count().reset_index(name="count")
        )
        # Ensure all 4 buckets present
        for lvl in ["Low","Medium","High","Severe"]:
            if lvl not in severity_dist["severity"].values:
                severity_dist = pd.concat(
                    [severity_dist, pd.DataFrame([{"severity": lvl, "count": 0}])],
                    ignore_index=True
                )
        sev_order = {"Low": 0, "Medium": 1, "High": 2, "Severe": 3}
        severity_dist["_ord"] = severity_dist["severity"].map(sev_order)
        severity_dist = severity_dist.sort_values("_ord").drop(columns="_ord")

        # Severity trajectory: resolved vs unresolved per severity tier
        resolved_conv_ids = set(res_df["conversation_id"]) & esc_conv_ids
        sev_df["resolved"] = sev_df["conversation_id"].isin(resolved_conv_ids)
        sev_traj = (
            sev_df.groupby("severity")
              .agg(recovered=("resolved", "sum"), total=("resolved", "count"))
              .reset_index()
        )
        sev_traj["unresolved"] = sev_traj["total"] - sev_traj["recovered"]
        sev_traj["recovery_rate"] = (sev_traj["recovered"] / sev_traj["total"].clip(lower=1) * 100).round(1)
        sev_traj["_ord"] = sev_traj["severity"].map(sev_order)
        sev_traj = sev_traj.sort_values("_ord").drop(columns="_ord")
    else:
        severity_dist = pd.DataFrame({"severity": ["Low","Medium","High","Severe"], "count": [0,0,0,0]})
        sev_traj      = pd.DataFrame({"severity": ["Low","Medium","High","Severe"],
                                      "recovered": [0,0,0,0], "unresolved": [0,0,0,0],
                                      "total": [0,0,0,0], "recovery_rate": [0.0,0.0,0.0,0.0]})

    # ── 5. Agent recovery effectiveness ──────────────────────────────────────
    # For every escalation turn classify into one of three buckets:
    #   quick  — customer sentiment positive within next 3 customer turns
    #   late   — not quick, but a resolution turn exists after this esc turn
    #   never  — neither
    df_s = df.sort_values(["conversation_id","turn_sequence"]).reset_index(drop=True)

    recovery_within_3  = 0   # kept for KPI strip (same as quick_recovery)
    recovery_by_end    = 0   # kept for backwards compat
    quick_recovery     = 0
    late_recovery      = 0
    never_recovered    = 0
    n_esc_events       = 0
    agent_resp_vals    = []

    for cid, grp in df_s.groupby("conversation_id", sort=False):
        grp = grp.reset_index(drop=True)
        esc_idx = grp.index[grp["potential_escalation"] == True].tolist()
        if not esc_idx:
            continue
        n_esc_events += len(esc_idx)

        cust_rows = grp[grp["speaker"] == "CUSTOMER"].reset_index(drop=True)
        conv_has_res = (grp["potential_resolution"] == True).any()

        for ei in esc_idx:
            esc_turn = grp.loc[ei, "turn_sequence"]

            # Agent response: first AGENT turn after escalation
            agent_after = grp[(grp["turn_sequence"] > esc_turn) & (grp["speaker"] == "AGENT")]
            if not agent_after.empty:
                agent_resp_vals.append(float(agent_after.iloc[0]["sentiment_change"]))

            # Quick: customer positive within 3 turns after escalation
            cust_after = cust_rows[cust_rows["turn_sequence"] > esc_turn].head(3)
            is_quick   = not cust_after.empty and (cust_after["compound"] > 0.05).any()

            # Late: resolution turn exists strictly after this escalation turn
            res_after  = grp[(grp["potential_resolution"] == True) & (grp["turn_sequence"] > esc_turn)]
            is_late    = (not is_quick) and (not res_after.empty)

            if is_quick:
                quick_recovery    += 1
                recovery_within_3 += 1
            elif is_late:
                late_recovery += 1
            else:
                never_recovered += 1

        if conv_has_res:
            recovery_by_end += len(esc_idx)

    avg_agent_response   = float(np.mean(agent_resp_vals)) if agent_resp_vals else 0.0
    pct_recovery_3turns  = recovery_within_3 / max(n_esc_events, 1)
    pct_recovery_by_end  = recovery_by_end   / max(n_esc_events, 1)

    recovery_intel = {
        "n_esc_events":         n_esc_events,
        "recovery_within_3":    recovery_within_3,
        "recovery_by_end":      recovery_by_end,
        "quick_recovery":       quick_recovery,
        "late_recovery":        late_recovery,
        "never_recovered":      never_recovered,
        "pct_recovery_3turns":  pct_recovery_3turns,
        "pct_recovery_by_end":  pct_recovery_by_end,
        "avg_agent_response":   avg_agent_response,
    }

    # ── 6. Top deteriorated conversations ────────────────────────────────────
    # Conversations where start sentiment > end sentiment (deteriorated)
    # Columns: conv_id, turn_count, first_esc_turn, start_sent, end_sent, delta, trigger_phrase
    phase_sent = (
        lf.filter(pl.col("speaker") == "CUSTOMER")
          .group_by(["conversation_id","phase"])
          .agg(pl.col("compound").mean().alias("avg"))
          .collect().to_pandas()
    )
    start_sent = phase_sent[phase_sent["phase"]=="start"].set_index("conversation_id")["avg"]
    end_sent   = phase_sent[phase_sent["phase"]=="end"].set_index("conversation_id")["avg"]

    first_esc = (
        lf.filter(pl.col("potential_escalation") == True)
          .sort(["conversation_id","turn_sequence"])
          .group_by("conversation_id")
          .agg([
              pl.col("turn_sequence").first().alias("first_esc_turn"),
              pl.col("message").first().alias("trigger_phrase"),
          ])
          .collect().to_pandas()
          .set_index("conversation_id")
    )
    first_res = (
        lf.filter(pl.col("potential_resolution") == True)
          .sort(["conversation_id","turn_sequence"])
          .group_by("conversation_id")
          .agg(pl.col("turn_sequence").first().alias("first_res_turn"))
          .collect().to_pandas()
          .set_index("conversation_id")
    )
    ttr_vals: List[int] = []
    for _cid in esc_conv_ids:
        if _cid in first_esc.index and _cid in first_res.index:
            gap = int(first_res.loc[_cid, "first_res_turn"]) - int(first_esc.loc[_cid, "first_esc_turn"])
            if gap >= 0:
                ttr_vals.append(gap)
    conv_turns = (
        lf.group_by("conversation_id")
          .agg(pl.len().alias("turn_count"))
          .collect().to_pandas()
          .set_index("conversation_id")
    )

    all_convs = set(start_sent.index) & set(end_sent.index)
    det_rows: List[Dict] = []
    for cid in all_convs:
        s = float(start_sent.get(cid, 0.0) or 0.0)
        e = float(end_sent.get(cid,   0.0) or 0.0)
        delta = e - s
        if delta >= 0: continue  # not deteriorated
        fe_row   = first_esc.loc[cid]   if cid in first_esc.index   else None
        tc       = int(conv_turns.loc[cid, "turn_count"]) if cid in conv_turns.index else 0
        fet      = int(fe_row["first_esc_turn"])           if fe_row is not None else None
        phrase   = str(fe_row["trigger_phrase"])[:60]      if fe_row is not None else "—"
        det_rows.append({
            "conversation_id":  cid,
            "turn_count":       tc,
            "first_esc_turn":   fet if fet is not None else "—",
            "start_sentiment":  round(s, 3),
            "end_sentiment":    round(e, 3),
            "delta":            round(delta, 3),
            "trigger_phrase":   phrase,
        })
    det_rows.sort(key=lambda x: x["delta"])
    deteriorated = det_rows[:15]

    # ── 7. Top escalated conversations ───────────────────────────────────────
    top_esc_convs = (
        lf.filter(pl.col("potential_escalation") == True)
          .group_by("conversation_id")
          .agg([
              pl.len().alias("esc_events"),
              pl.col("turn_sequence").max().alias("max_esc_turn"),
          ])
          .sort("esc_events", descending=True)
          .head(15)
          .collect().to_pandas()
    )
    if not top_esc_convs.empty:
        # Add avg customer sentiment + total turns + end sentiment
        cust_avg = (
            lf.filter(pl.col("speaker") == "CUSTOMER")
              .group_by("conversation_id")
              .agg(pl.col("compound").mean().alias("cust_avg"))
              .collect().to_pandas()
        )
        top_esc_convs = top_esc_convs.merge(cust_avg, on="conversation_id", how="left")
        top_esc_convs = top_esc_convs.merge(
            conv_turns.reset_index(), on="conversation_id", how="left"
        )
        # Severity for each
        if not conv_esc_counts.empty:
            top_esc_convs = top_esc_convs.merge(
                sev_df[["conversation_id","severity"]], on="conversation_id", how="left"
            )
            top_esc_convs["severity"] = top_esc_convs["severity"].fillna("Low")
        else:
            top_esc_convs["severity"] = "Low"
        top_esc_convs["cust_avg"]   = top_esc_convs["cust_avg"].round(3)
        top_esc_convs["turn_count"] = top_esc_convs["turn_count"].fillna(0).astype(int)

    return {
        "trigger_rows":       trigger_rows,
        "esc_phase":          esc_phase,
        "severity_dist":      severity_dist,
        "severity_traj":      sev_traj,
        "recovery_intel":     recovery_intel,
        "deteriorated":       deteriorated,
        "top_escalated":      top_esc_convs if not top_esc_convs.empty else pd.DataFrame(),
        "cluster_summary":    cluster_summary,
        "cluster_conv_ids":   cluster_conv_ids,
        "avg_agent_response": avg_agent_response,
        "total_convs":        total_convs,
        "n_escalated":        n_escalated,
        "n_resolved":         n_resolved,
        "n_unresolved":       n_unresolved,
        "esc_conv_ids":       list(esc_conv_ids),
        "first_esc_turns":    first_esc["first_esc_turn"].dropna().astype(int).tolist() if not first_esc.empty else [],
        "ttr_vals":           ttr_vals,
    }

def _escalation_triggers_table(df_r: pd.DataFrame, top_n: int = 15):
    """Overview snippet: top phrase clusters — concise version for the Overview page."""
    intel    = _compute_escalation_intel(df_r)
    clusters = intel.get("cluster_summary", [])
    if not clusters:
        st.info("No escalation turns detected in this dataset.")
        return
    rows = intel["trigger_rows"][:top_n]
    total_esc = intel["n_escalated"]
    # Show cluster summary + top phrases in a compact table
    html_rows = ""
    for i, c in enumerate(clusters[:6]):   # top 6 clusters in Overview
        bar_w    = int(c["conv_pct"] * 100)
        bar_html = (
            f'<div style="display:flex;align-items:center;gap:6px">'
            f'<div style="flex:1;background:{C["warm"]};border-radius:4px;height:7px;overflow:hidden">'
            f'<div style="width:{bar_w}%;height:100%;background:{C["neg"]};border-radius:4px"></div>'
            f'</div><span style="font-size:11px;color:{C["muted"]};width:36px;text-align:right">'
            f'{c["conv_pct"]:.0%}</span></div>'
        )
        sample = ", ".join(c["phrases"][:3])
        html_rows += (
            f"<tr>"
            f"<td><strong>{c['cluster']}</strong><br>"
            f"<span style='font-size:11px;color:{C['muted']}'><em>{sample}</em></span></td>"
            f"<td style='text-align:right;font-weight:600;color:{C['neg']}'>{c['total_convs']:,} / {total_esc:,}</td>"
            f"<td style='min-width:140px'>{bar_html}</td>"
            f"</tr>"
        )
    st.markdown(
        f"<table class='pt'><thead><tr>"
        f"<th>Cluster Theme  <span style='font-weight:400;font-size:11px'>(sample phrases)</span></th>"
        f"<th>Escalated Convs</th><th>Coverage</th>"
        f"</tr></thead><tbody>{html_rows}</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption("See ⚠️ Escalation page for full cluster analysis and filtered phrase detail.")


def _to_escalation_excel(intel: dict) -> bytes:
    """Build an escalation intelligence Excel workbook — 4 sheets."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:

        # Sheet 1 — Cluster Summary
        clusters = intel.get("cluster_summary", [])
        if clusters:
            cl_df = pd.DataFrame([{
                "Cluster Theme":          c["cluster"],
                "Escalated Conversations": c["total_convs"],
                "Phrase Hits":            c["total_count"],
                "% of Escalated Convs":  f"{c['conv_pct']:.1%}",
                "Sample Phrases":        " | ".join(c["phrases"][:5]),
            } for c in clusters])
        else:
            cl_df = pd.DataFrame(columns=["Cluster Theme","Escalated Conversations","Phrase Hits","% of Escalated Convs","Sample Phrases"])
        cl_df.to_excel(w, sheet_name="Clusters", index=False)

        # Sheet 2 — Trigger Phrases
        rows = intel.get("trigger_rows", [])
        if rows:
            tr_df = pd.DataFrame([{
                "Phrase":                  r["phrase"],
                "Cluster":                 r["cluster"],
                "Escalated Conversations": r["convs"],
                "Total Occurrences":       r["count"],
                "Conv Coverage %":         f"{r['conv_pct']:.1%}",
            } for r in rows])
        else:
            tr_df = pd.DataFrame(columns=["Phrase","Cluster","Escalated Conversations","Total Occurrences","Conv Coverage %"])
        tr_df.to_excel(w, sheet_name="Trigger Phrases", index=False)

        # Sheet 3 — Deteriorated Conversations
        det = intel.get("deteriorated", [])
        if det:
            det_df = pd.DataFrame([{
                "Conversation ID":   r["conversation_id"],
                "Turns":             r["turn_count"],
                "First Esc Turn":    r["first_esc_turn"],
                "Start Sentiment":   r["start_sentiment"],
                "End Sentiment":     r["end_sentiment"],
                "Delta":             r["delta"],
                "Trigger Phrase":    r["trigger_phrase"],
            } for r in det])
        else:
            det_df = pd.DataFrame(columns=["Conversation ID","Turns","First Esc Turn","Start Sentiment","End Sentiment","Delta","Trigger Phrase"])
        det_df.to_excel(w, sheet_name="Deteriorated Conversations", index=False)

        # Sheet 4 — Severity Distribution
        sev = intel.get("severity_dist", pd.DataFrame())
        if not sev.empty:
            sev[["severity","count"]].rename(columns={"severity":"Severity","count":"Conversations"}).to_excel(
                w, sheet_name="Severity Distribution", index=False
            )
        else:
            pd.DataFrame(columns=["Severity","Conversations"]).to_excel(w, sheet_name="Severity Distribution", index=False)

        # Sheet 5 — Recovery Summary
        ri = intel.get("recovery_intel", {})
        rec_rows = [
            {"Metric": "Total Escalation Events",        "Value": ri.get("n_esc_events", 0)},
            {"Metric": "Recovery within 3 turns (count)","Value": ri.get("recovery_within_3", 0)},
            {"Metric": "Recovery within 3 turns (%)",    "Value": f"{ri.get('pct_recovery_3turns', 0):.1%}"},
            {"Metric": "Recovery by conv end (count)",   "Value": ri.get("recovery_by_end", 0)},
            {"Metric": "Recovery by conv end (%)",       "Value": f"{ri.get('pct_recovery_by_end', 0):.1%}"},
            {"Metric": "Avg Agent Sentiment Change",     "Value": f"{ri.get('avg_agent_response', 0):+.3f}"},
            {"Metric": "Total Conversations",            "Value": intel.get("total_convs", 0)},
            {"Metric": "Escalated Conversations",        "Value": intel.get("n_escalated", 0)},
            {"Metric": "Resolved Conversations",         "Value": intel.get("n_resolved", 0)},
            {"Metric": "Unresolved Conversations",       "Value": intel.get("n_unresolved", 0)},
        ]
        pd.DataFrame(rec_rows).to_excel(w, sheet_name="Recovery Summary", index=False)

    return buf.getvalue()


# ─── Escalation Page ──────────────────────────────────────────────────────────
def page_escalation(df_r, ins):
    sh("⚠️", "Escalation Analysis")
    intel = _compute_escalation_intel(df_r)
    cs    = ins["customer_satisfaction"]
    ri    = intel["recovery_intel"]

    # ─────────────────────────────────────────────────────────────────────────
    # KPI STRIP  (5 cards — correlated pairs merged to avoid 1080p cramping)
    # ─────────────────────────────────────────────────────────────────────────
    esc_rate  = cs["escalation_rate"]
    res_rate  = cs["resolution_rate"]
    sev_dist  = intel["severity_dist"]
    severe_n  = int(sev_dist.loc[sev_dist["severity"]=="Severe","count"].sum()) if not sev_dist.empty else 0
    severe_rt = severe_n / max(intel["n_escalated"], 1)
    esc_c  = C["neg"]  if esc_rate > 0.15  else C["warn"] if esc_rate > 0.10 else C["ok"]
    sev_c  = C["neg"]  if severe_rt > 0.20 else C["warn"] if severe_rt > 0.10 else C["ok"]
    res_c  = C["ok"]   if res_rate  > 0.60 else C["warn"] if res_rate  > 0.40 else C["neg"]
    rec3_c = C["ok"]   if ri["pct_recovery_3turns"] > 0.40 else C["warn"] if ri["pct_recovery_3turns"] > 0.20 else C["neg"]
    ar     = intel["avg_agent_response"]
    ar_c   = C["ok"]   if ar > 0.05 else C["neg"] if ar < -0.05 else C["gold"]

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(mc("Total Conversations",  f"{intel['total_convs']:,}",  C["teal"]), unsafe_allow_html=True)
    with k2: st.markdown(mc2("Escalated",           f"{intel['n_escalated']:,}",
                              "rate",               f"{esc_rate:.1%}",             esc_c),     unsafe_allow_html=True)
    with k3: st.markdown(mc2("Severe Escalations",  f"{severe_n:,}",
                              "of escalated",       f"{severe_rt:.1%}",            sev_c),     unsafe_allow_html=True)
    with k4: st.markdown(mc2("Resolved",            f"{intel['n_resolved']:,}",
                              "resolution rate",    f"{res_rate:.1%}",             res_c),     unsafe_allow_html=True)
    with k5: st.markdown(mc("Recovery within 3",    f"{ri['pct_recovery_3turns']:.1%}", rec3_c), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 1 — Severity distribution + Phase breakdown
    # ─────────────────────────────────────────────────────────────────────────
    col_sev, col_phase = st.columns(2)

    with col_sev:
        sh("📈", "Severity × Recovery Outcome")
        st.caption("For each severity tier: how many escalated conversations recovered vs stayed unresolved. Higher tiers should recover less — gaps here reveal coaching opportunities.")
        st_traj = intel["severity_traj"].copy()
        SEV_COLORS = {"Low": C["neu"], "Medium": C["warn"], "High": C["neg"], "Severe": "#6B0000"}
        if not st_traj.empty and st_traj["total"].sum() > 0:
            fig_traj = go.Figure()
            fig_traj.add_trace(go.Bar(
                name="Recovered",
                x=st_traj["severity"],
                y=st_traj["recovered"],
                marker_color=C["pos"],
                marker_line_width=0,
                text=st_traj["recovered"].astype(int),
                textposition="auto",
                hovertemplate="<b>%{x}</b> — Recovered<br>%{y:,} conversations<extra></extra>",
            ))
            fig_traj.add_trace(go.Bar(
                name="Unresolved",
                x=st_traj["severity"],
                y=st_traj["unresolved"],
                marker_color=[SEV_COLORS.get(s, C["slate"]) for s in st_traj["severity"]],
                marker_line_width=0,
                text=st_traj["unresolved"].astype(int),
                textposition="auto",
                hovertemplate="<b>%{x}</b> — Unresolved<br>%{y:,} conversations<extra></extra>",
            ))
            # Recovery rate line
            fig_traj.add_trace(go.Scatter(
                name="Recovery %",
                x=st_traj["severity"],
                y=st_traj["recovery_rate"],
                mode="lines+markers",
                line=dict(color=C["gold"], width=2.5, dash="dot"),
                marker=dict(size=8, color=C["gold"]),
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Recovery Rate: %{y:.1f}%<extra></extra>",
            ))
            fig_traj.update_layout(
                height=300, barmode="stack",
                showlegend=True,
                legend=dict(orientation="h", y=1.12, x=0),
                xaxis=dict(categoryorder="array", categoryarray=["Low","Medium","High","Severe"]),
                yaxis=dict(title="Conversations"),
                yaxis2=dict(overlaying="y", side="right", showgrid=False,
                            title="Recovery %", tickformat=".0f",
                            range=[0, 105],
                            titlefont=dict(color=C["gold"]), tickfont=dict(color=C["gold"])),
                margin=dict(l=10, r=40, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color=C["text"]),
            )
            st.plotly_chart(fig_traj, width="stretch")
        else:
            st.info("No severity data available.")

    with col_phase:
        sh("📊", "Escalation vs Resolution by Phase")
        st.caption("Bars show escalation events vs resolution events per phase. The gap is where problems go unaddressed. Recovery rate line shows how well each phase closes that gap.")
        ep = intel["esc_phase"]
        PHASE_ORDER  = {"start": 0, "middle": 1, "end": 2}
        PHASE_COLORS = {"start": C["teal"], "middle": C["warn"], "end": C["neg"]}
        if not ep.empty:
            ep = ep.copy()
            ep["_ord"] = ep["phase"].map(PHASE_ORDER).fillna(9)
            ep = ep.sort_values("_ord")
            phases_cap = ep["phase"].str.capitalize()

            fig_ph = go.Figure()
            fig_ph.add_trace(go.Bar(
                name="Escalations",
                x=phases_cap,
                y=ep["escalations"],
                marker_color=[PHASE_COLORS.get(p, C["slate"]) for p in ep["phase"]],
                marker_line_width=0,
                text=ep["escalations"].astype(int),
                textposition="auto",
                yaxis="y",
                hovertemplate="<b>%{x}</b><br>Escalations: %{y:,}<extra></extra>",
            ))
            if "resolutions" in ep.columns:
                fig_ph.add_trace(go.Bar(
                    name="Resolutions",
                    x=phases_cap,
                    y=ep["resolutions"],
                    marker_color=C["pos"],
                    marker_line_width=0,
                    text=ep["resolutions"].astype(int),
                    textposition="auto",
                    yaxis="y",
                    hovertemplate="<b>%{x}</b><br>Resolutions: %{y:,}<extra></extra>",
                ))
            if "phase_recovery_rate" in ep.columns:
                fig_ph.add_trace(go.Scatter(
                    name="Recovery Rate %",
                    x=phases_cap,
                    y=ep["phase_recovery_rate"],
                    mode="lines+markers",
                    line=dict(color=C["gold"], width=2.5, dash="dot"),
                    marker=dict(size=8, color=C["gold"]),
                    yaxis="y2",
                    hovertemplate="<b>%{x}</b><br>Recovery Rate: %{y:.1f}%<extra></extra>",
                ))
                fig_ph.update_layout(yaxis2=dict(
                    overlaying="y", side="right", showgrid=False,
                    title="Recovery Rate %", tickformat=".0f", range=[0, 110],
                    titlefont=dict(color=C["gold"]), tickfont=dict(color=C["gold"]),
                ))
            fig_ph.update_layout(
                height=300, showlegend=True, barmode="group",
                legend=dict(orientation="h", y=1.12, x=0),
                margin=dict(l=10, r=50, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color=C["text"]),
            )
            st.plotly_chart(fig_ph, width="stretch")
        else:
            st.info("No escalation phase data available.")

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 1b — First escalation turn + Time-to-Resolution histograms
    # ─────────────────────────────────────────────────────────────────────────
    fet_vals = intel.get("first_esc_turns", [])
    ttr_vals = intel.get("ttr_vals", [])
    if fet_vals or ttr_vals:
        col_fet, col_ttr = st.columns(2)

        with col_fet:
            if fet_vals:
                sh("⏱️", "When Does Escalation First Appear?")
                st.caption(
                    "Turn number of each conversation's first escalation. "
                    "Early spikes = agents lose control quickly; late spikes = issues compound over time."
                )
                fet_series = pd.Series(fet_vals, name="first_esc_turn")
                max_turn   = int(fet_series.max())
                bin_width  = 1 if max_turn <= 20 else 2 if max_turn <= 50 else 5
                bins       = list(range(1, max_turn + bin_width + 1, bin_width))
                counts, edges = np.histogram(fet_series, bins=bins)
                bin_labels = [f"{edges[i]:.0f}–{edges[i+1]-1:.0f}" if bin_width > 1
                              else str(int(edges[i])) for i in range(len(counts))]
                thirds    = max_turn / 3
                bar_colors = [
                    C["teal"] if edges[i] <= thirds else
                    C["warn"] if edges[i] <= 2 * thirds else
                    C["neg"]
                    for i in range(len(counts))
                ]
                fig_fet = go.Figure(go.Bar(
                    x=bin_labels, y=counts,
                    marker_color=bar_colors,
                    marker_line_width=0,
                    text=counts,
                    textposition="auto",
                    hovertemplate="<b>Turn %{x}</b><br>Conversations: %{y:,}<extra></extra>",
                ))
                median_turn = float(np.median(fet_series))
                fig_fet.add_vline(
                    x=median_turn - 1,
                    line_dash="dash", line_color=C["gold"], line_width=1.5,
                    annotation_text=f"Median: turn {median_turn:.0f}",
                    annotation_position="top right",
                    annotation_font=dict(color=C["gold"], size=11),
                )
                fig_fet.update_layout(
                    height=280,
                    xaxis=dict(title="First Escalation Turn", tickangle=-45 if max_turn > 20 else 0),
                    yaxis=dict(title="Conversations"),
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color=C["text"]),
                    showlegend=False,
                )
                st.plotly_chart(fig_fet, width="stretch")
                c_leg1, c_leg2, c_leg3 = st.columns(3)
                with c_leg1: st.markdown(f'<span style="color:{C["teal"]};font-size:12px">■</span> <span style="font-size:12px">Early</span>', unsafe_allow_html=True)
                with c_leg2: st.markdown(f'<span style="color:{C["warn"]};font-size:12px">■</span> <span style="font-size:12px">Mid</span>', unsafe_allow_html=True)
                with c_leg3: st.markdown(f'<span style="color:{C["neg"]};font-size:12px">■</span> <span style="font-size:12px">Late</span>', unsafe_allow_html=True)

        with col_ttr:
            if ttr_vals:
                sh("🏁", "Turns to Resolution")
                st.caption(
                    "For recovered conversations: turns between first escalation and first resolution. "
                    "Lower is better — long gaps mean delayed agent response or customer persistence."
                )
                ttr_series = pd.Series(ttr_vals, name="ttr")
                max_ttr    = int(ttr_series.max())
                bw_ttr     = 1 if max_ttr <= 15 else 2 if max_ttr <= 40 else 5
                bins_ttr   = list(range(0, max_ttr + bw_ttr + 1, bw_ttr))
                ttr_counts, ttr_edges = np.histogram(ttr_series, bins=bins_ttr)
                ttr_labels = [f"{ttr_edges[i]:.0f}–{ttr_edges[i+1]-1:.0f}" if bw_ttr > 1
                              else str(int(ttr_edges[i])) for i in range(len(ttr_counts))]
                # Colour: green (fast) → gold (medium) → red (slow)
                ttr_thirds = max_ttr / 3
                ttr_colors = [
                    C["pos"]  if ttr_edges[i] <= ttr_thirds else
                    C["warn"] if ttr_edges[i] <= 2 * ttr_thirds else
                    C["neg"]
                    for i in range(len(ttr_counts))
                ]
                fig_ttr = go.Figure(go.Bar(
                    x=ttr_labels, y=ttr_counts,
                    marker_color=ttr_colors,
                    marker_line_width=0,
                    text=ttr_counts,
                    textposition="auto",
                    hovertemplate="<b>%{x} turns gap</b><br>Conversations: %{y:,}<extra></extra>",
                ))
                median_ttr = float(np.median(ttr_series))
                fig_ttr.add_vline(
                    x=median_ttr,
                    line_dash="dash", line_color=C["gold"], line_width=1.5,
                    annotation_text=f"Median: {median_ttr:.0f} turns",
                    annotation_position="top right",
                    annotation_font=dict(color=C["gold"], size=11),
                )
                fig_ttr.update_layout(
                    height=280,
                    xaxis=dict(title="Turns from Escalation → Resolution", tickangle=-45 if max_ttr > 15 else 0),
                    yaxis=dict(title="Conversations"),
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color=C["text"]),
                    showlegend=False,
                )
                st.plotly_chart(fig_ttr, width="stretch")
                t_leg1, t_leg2, t_leg3 = st.columns(3)
                with t_leg1: st.markdown(f'<span style="color:{C["pos"]};font-size:12px">■</span> <span style="font-size:12px">Fast</span>', unsafe_allow_html=True)
                with t_leg2: st.markdown(f'<span style="color:{C["warn"]};font-size:12px">■</span> <span style="font-size:12px">Medium</span>', unsafe_allow_html=True)
                with t_leg3: st.markdown(f'<span style="color:{C["neg"]};font-size:12px">■</span> <span style="font-size:12px">Slow</span>', unsafe_allow_html=True)
            elif not ttr_vals and fet_vals:
                st.info("No recovered conversations — time-to-resolution unavailable.")

        st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 2 — Recovery funnel + Agent recovery effectiveness
    # ─────────────────────────────────────────────────────────────────────────
    col_funnel, col_recovery = st.columns(2)

    with col_funnel:
        sh("🔻", "Escalation Resolution Breakdown")
        st.caption("Of all conversations, how many escalated — and of those, how many recovered vs stayed unresolved.")
        total  = intel["total_convs"]
        n_esc  = intel["n_escalated"]
        n_res  = intel["n_resolved"]
        n_unr  = intel["n_unresolved"]
        n_no_esc = total - n_esc
        fig_f = go.Figure()
        # Row 1: All conversations split by escalated vs not
        fig_f.add_trace(go.Bar(
            name="No Escalation",
            x=[n_no_esc], y=["All Conversations"],
            orientation="h",
            marker_color=C["pos"], marker_line_width=0,
            text=[f"{n_no_esc:,} ({n_no_esc/max(total,1):.0%})"],
            textposition="auto",
            hovertemplate="<b>No Escalation</b><br>%{x:,} conversations<extra></extra>",
        ))
        fig_f.add_trace(go.Bar(
            name="Escalated",
            x=[n_esc], y=["All Conversations"],
            orientation="h",
            marker_color=C["warn"], marker_line_width=0,
            text=[f"{n_esc:,} ({n_esc/max(total,1):.0%})"],
            textposition="auto",
            hovertemplate="<b>Escalated</b><br>%{x:,} conversations<extra></extra>",
        ))
        # Row 2: Escalated split by recovered vs unresolved
        fig_f.add_trace(go.Bar(
            name="Recovered",
            x=[n_res], y=["Escalated"],
            orientation="h",
            marker_color=C["pos"], marker_line_width=0,
            text=[f"{n_res:,} ({n_res/max(n_esc,1):.0%})"],
            textposition="auto",
            hovertemplate="<b>Recovered</b><br>%{x:,} escalated conversations<extra></extra>",
        ))
        fig_f.add_trace(go.Bar(
            name="Unresolved",
            x=[n_unr], y=["Escalated"],
            orientation="h",
            marker_color=C["neg"], marker_line_width=0,
            text=[f"{n_unr:,} ({n_unr/max(n_esc,1):.0%})"],
            textposition="auto",
            hovertemplate="<b>Unresolved</b><br>%{x:,} escalated conversations<extra></extra>",
        ))
        fig_f.update_layout(
            barmode="stack",
            height=200,
            showlegend=True,
            legend=dict(orientation="h", y=1.18, x=0),
            xaxis=dict(title="Conversations", showgrid=True,
                       gridcolor="rgba(168,188,200,0.2)"),
            yaxis=dict(showgrid=False, autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color=C["text"]),
        )
        st.plotly_chart(fig_f, width="stretch")

    with col_recovery:
        sh("🏃", "Agent Recovery Effectiveness")
        st.caption(f"Each escalation event classified into Quick (≤3 turns), Late (resolved after turn 3), or Never recovered. Total: {ri['n_esc_events']:,} events.")

        n_ev    = ri["n_esc_events"]
        n_quick = ri["quick_recovery"]
        n_late  = ri["late_recovery"]
        n_never = ri["never_recovered"]
        ar_val  = ri["avg_agent_response"]

        pct_quick = n_quick / max(n_ev, 1) * 100
        pct_late  = n_late  / max(n_ev, 1) * 100
        pct_never = n_never / max(n_ev, 1) * 100

        # Single stacked bar decomposing all escalation events
        fig_rec = go.Figure()
        fig_rec.add_trace(go.Bar(
            name="Quick (≤3 turns)",
            x=[n_quick], y=["Recovery"],
            orientation="h",
            marker_color=C["pos"], marker_line_width=0,
            text=[f"{n_quick:,} ({pct_quick:.0f}%)"] if pct_quick >= 8 else [""],
            textposition="inside",
            textfont=dict(color="white", size=12, family="DM Sans"),
            hovertemplate=f"<b>Quick Recovery</b><br>{n_quick:,} events ({pct_quick:.1f}%)<extra></extra>",
        ))
        fig_rec.add_trace(go.Bar(
            name="Late (after turn 3)",
            x=[n_late], y=["Recovery"],
            orientation="h",
            marker_color=C["gold"], marker_line_width=0,
            text=[f"{n_late:,} ({pct_late:.0f}%)"] if pct_late >= 8 else [""],
            textposition="inside",
            textfont=dict(color="white", size=12, family="DM Sans"),
            hovertemplate=f"<b>Late Recovery</b><br>{n_late:,} events ({pct_late:.1f}%)<extra></extra>",
        ))
        fig_rec.add_trace(go.Bar(
            name="Never recovered",
            x=[n_never], y=["Recovery"],
            orientation="h",
            marker_color=C["neg"], marker_line_width=0,
            text=[f"{n_never:,} ({pct_never:.0f}%)"] if pct_never >= 8 else [""],
            textposition="inside",
            textfont=dict(color="white", size=12, family="DM Sans"),
            hovertemplate=f"<b>Never Recovered</b><br>{n_never:,} events ({pct_never:.1f}%)<extra></extra>",
        ))
        fig_rec.update_layout(
            barmode="stack",
            height=110,
            showlegend=True,
            legend=dict(orientation="h", y=1.35, x=0, font=dict(size=11)),
            xaxis=dict(showgrid=True, gridcolor="rgba(168,188,200,0.2)",
                       title="Escalation Events"),
            yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color=C["text"]),
        )
        st.plotly_chart(fig_rec, width="stretch")

        # Avg agent sentiment change — prominent bullet display
        ar_arrow = "▲" if ar_val > 0 else "▼"
        ar_label = "Positive response" if ar_val > 0.05 else "Negative response" if ar_val < -0.05 else "Neutral response"
        # Bullet: background track + filled bar showing magnitude (capped at ±0.3 for display)
        ar_display_pct = min(abs(ar_val) / 0.3 * 100, 100)
        st.markdown(
            f'<div style="background:rgba(168,188,200,0.08);border-radius:8px;padding:12px 14px;margin-top:6px">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px">'
            f'<span style="font-size:12px;color:{C["muted"]}">Avg Agent Sentiment Δ After Escalation</span>'
            f'<span style="font-size:18px;font-weight:700;color:{ar_c}">{ar_arrow} {ar_val:+.3f}</span>'
            f'</div>'
            f'<div style="background:rgba(168,188,200,0.2);border-radius:4px;height:6px;overflow:hidden">'
            f'<div style="width:{ar_display_pct:.0f}%;height:100%;background:{ar_c};border-radius:4px"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;margin-top:4px">'
            f'<span style="font-size:11px;color:{C["muted"]}">0</span>'
            f'<span style="font-size:11px;color:{ar_c}">{ar_label}</span>'
            f'<span style="font-size:11px;color:{C["muted"]}">±0.3</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 3 — Escalation Pattern Clusters  (semantic grouping)
    # ─────────────────────────────────────────────────────────────────────────
    sh("🧩", "Escalation Pattern Clusters")
    st.caption(
        "Phrases from escalation turns grouped into business themes. "
        "Ranked by how many escalated conversations each cluster appears in. "
        "More actionable than raw word frequency — shows the real root causes."
    )
    clusters = intel.get("cluster_summary", [])
    if clusters:
        # ── Cluster summary table ─────────────────────────────────────────────
        cl_rows = ""
        for i, c in enumerate(clusters):
            rank_badge = (
                f'<span style="display:inline-flex;align-items:center;justify-content:center;'
                f'width:22px;height:22px;border-radius:50%;background:{C["neg"] if i==0 else C["warn"] if i<=2 else C["slate"]};'
                f'color:#fff;font-size:11px;font-weight:700">#{i+1}</span>'
            )
            bar_w = int(c["conv_pct"] * 100)
            bar_html = (
                f'<div style="display:flex;align-items:center;gap:6px">'
                f'<div style="flex:1;background:{C["warm"]};border-radius:4px;height:7px;overflow:hidden">'
                f'<div style="width:{bar_w}%;height:100%;background:{C["neg"]};border-radius:4px"></div></div>'
                f'<span style="font-size:11px;color:{C["muted"]};width:36px;text-align:right">{c["conv_pct"]:.0%}</span></div>'
            )
            # Top 3 sample phrases
            sample = " · ".join([f"<em>{p}</em>" for p in c["phrases"][:3]])
            cl_rows += (
                f"<tr>"
                f"<td>{rank_badge}</td>"
                f"<td><strong>{c['cluster']}</strong><br>"
                f"<span style='font-size:11px;color:{C['muted']}'>{sample}</span></td>"
                f"<td style='text-align:right;font-weight:700;color:{C['neg']}'>{c['total_convs']:,}</td>"
                f"<td style='text-align:right'>{c['total_count']:,}</td>"
                f"<td style='min-width:140px'>{bar_html}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table class='pt'><thead><tr>"
            f"<th>#</th><th>Cluster Theme  <span style='font-weight:400;color:var(--muted)'>"
            f"(sample phrases)</span></th>"
            f"<th>Escalated Convs</th><th>Phrase Hits</th><th>% of Escalated Convs</th>"
            f"</tr></thead><tbody>{cl_rows}</tbody></table>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Top filtered phrases bar chart ────────────────────────────────────
        sh("🔑", "Top Trigger Phrases  — filtered & ranked by conversation coverage")
        st.caption(
            "Bigrams and trigrams that appear in ≥ 2 distinct escalated conversations. "
            "Ranked by how many conversations they appear in — not raw frequency. "
            "Coloured by cluster theme."
        )
        rows = intel["trigger_rows"][:20]
        if rows:
            CLUSTER_COLORS = {
                "Billing / Payment":   "#A04040",
                "Wait Time / Delays":  "#B8963E",
                "Agent / Service":     "#2D5F6E",
                "Account / Access":    "#3D7A5F",
                "Product / Service":   "#6B4A8C",
                "Delivery / Order":    "#4682b4",
                "Repeat Contact":      "#C0603C",
                "Dissatisfaction":     "#8B0000",
                "Other":               C["slate"],
            }
            tr_df = pd.DataFrame(rows)
            tr_df["color"] = tr_df["cluster"].map(CLUSTER_COLORS).fillna(C["slate"])
            tr_df["label"] = tr_df["phrase"] + "  [" + tr_df["cluster"] + "]"
            tr_df = tr_df.sort_values("convs")  # ascending for readability

            fig_tr = go.Figure(go.Bar(
                x=tr_df["convs"], y=tr_df["label"],
                orientation="h",
                marker_color=tr_df["color"].tolist(),
                marker_line_width=0,
                text=tr_df["convs"].astype(int),
                textposition="outside",
                customdata=tr_df[["count","conv_pct","cluster"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Escalated conversations: %{x:,}<br>"
                    "Total occurrences: %{customdata[0]:,.0f}<br>"
                    "Conv coverage: %{customdata[1]:.0%}<br>"
                    "Cluster: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            ))
            # Add one invisible scatter trace per cluster present in the chart
            # so Plotly renders a colour legend
            present_clusters = tr_df[["cluster","color"]].drop_duplicates("cluster")
            for _, crow in present_clusters.iterrows():
                fig_tr.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=10, color=crow["color"], symbol="square"),
                    name=crow["cluster"],
                    showlegend=True,
                ))
            fig_tr.update_layout(
                height=max(360, len(rows) * 28),
                margin=dict(l=10, r=60, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color=C["text"], size=12),
                xaxis=dict(title="Distinct Escalated Conversations"),
                yaxis=dict(title="", automargin=True),
                legend=dict(
                    orientation="v", x=1.01, y=1, xanchor="left",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11),
                ),
            )
            st.plotly_chart(fig_tr, width="stretch")
    else:
        st.info("No escalation phrase data found.")

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 4b — Cluster Drill-Down
    # ─────────────────────────────────────────────────────────────────────────
    cluster_conv_ids = intel.get("cluster_conv_ids", {})
    if cluster_conv_ids:
        sh("🔍", "Cluster Drill-Down — Conversations by Theme")
        st.caption("Select a cluster to see every escalated conversation that contains its trigger phrases.")

        cluster_names = [c["cluster"] for c in intel.get("cluster_summary", [])]
        sel_cluster = st.selectbox(
            "Filter by cluster",
            options=cluster_names,
            key="esc_cluster_filter",
        )
        drill_ids = cluster_conv_ids.get(sel_cluster, [])
        if drill_ids:
            drill_df = df_r[df_r["conversation_id"].isin(drill_ids)].copy()

            # Build one summary row per conversation
            cust_drill = drill_df[drill_df["speaker"] == "CUSTOMER"]
            conv_summary = (
                cust_drill.groupby("conversation_id")
                .agg(
                    turns        = ("turn_sequence", "count"),
                    avg_sentiment= ("compound", "mean"),
                    esc_events   = ("potential_escalation", "sum"),
                )
                .reset_index()
            )
            # Resolution status (one value per conv)
            if "resolution_status" in drill_df.columns:
                res_col = drill_df.drop_duplicates("conversation_id")[["conversation_id","resolution_status"]]
                conv_summary = conv_summary.merge(res_col, on="conversation_id", how="left")
            conv_summary["avg_sentiment"] = conv_summary["avg_sentiment"].round(3)
            conv_summary["esc_events"]    = conv_summary["esc_events"].astype(int)
            conv_summary = conv_summary.sort_values("esc_events", ascending=False)

            st.markdown(
                f'<div style="font-size:12px;color:{C["muted"]};margin-bottom:6px">'
                f'<strong style="color:{C["neg"]}">{len(drill_ids)}</strong> conversation(s) '
                f'contain <strong>{sel_cluster}</strong> trigger phrases</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                conv_summary.rename(columns={
                    "conversation_id":  "Conversation",
                    "turns":            "Customer Turns",
                    "avg_sentiment":    "Avg Sentiment",
                    "esc_events":       "Esc Events",
                    "resolution_status":"Resolution",
                }),
                width="stretch",
                height=min(420, 38 + len(conv_summary) * 35),
            )

            # Expandable: show full turn detail for a selected conversation
            with st.expander("View turn detail for a conversation"):
                sel_drill_conv = st.selectbox(
                    "Select conversation",
                    options=sorted(drill_ids),
                    key="esc_drill_conv_sel",
                )
                sub = drill_df[drill_df["conversation_id"] == sel_drill_conv].copy()
                cols = [c for c in ["turn_sequence","phase","speaker","message",
                                    "sentiment_label","compound","potential_escalation"]
                        if c in sub.columns]
                st.dataframe(sub[cols].reset_index(drop=True), width="stretch", height=350)
        else:
            st.info(f"No conversation IDs found for cluster '{sel_cluster}'.")

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 5 — Top deteriorated conversations table
    # ─────────────────────────────────────────────────────────────────────────
    sh("📉", "Top Deteriorated Conversations")
    st.caption("Conversations where end sentiment was significantly worse than start sentiment. Sorted by largest drop.")
    det = intel["deteriorated"]
    if det:
        # Column header row
        h1,h2,h3,h4,h5,h6,h7,h8 = st.columns([2.2, 0.8, 1.0, 1.0, 1.0, 1.2, 2.5, 1.2])
        _muted  = C["muted"]
        _border = C["border"]
        for col, label in zip(
            [h1,h2,h3,h4,h5,h6,h7,h8],
            ["Conversation","Turns","1st Esc","Start","End","Delta","Trigger Phrase",""],
        ):
            col.markdown(f"<div style='font-size:11px;font-weight:700;color:{_muted};padding-bottom:2px'>{label}</div>", unsafe_allow_html=True)
        st.markdown(f"<hr style='margin:2px 0 6px;border-color:{_border}'>", unsafe_allow_html=True)

        for i, r in enumerate(det):
            delta = r["delta"]
            if delta < -0.4:
                delta_html = f'<span style="color:{C["neg"]};font-weight:700">▼ {delta:+.3f} Critical</span>'
            elif delta < -0.2:
                delta_html = f'<span style="color:{C["neg"]};font-weight:600">▼ {delta:+.3f}</span>'
            else:
                delta_html = f'<span style="color:{C["warn"]}">▼ {delta:+.3f}</span>'

            c1,c2,c3,c4,c5,c6,c7,c8 = st.columns([2.2, 0.8, 1.0, 1.0, 1.0, 1.2, 2.5, 1.2])
            c1.markdown(f"<strong style='font-size:12px'>{r['conversation_id']}</strong>", unsafe_allow_html=True)
            c2.markdown(f"<span style='font-family:monospace;font-size:12px'>{r['turn_count']}</span>", unsafe_allow_html=True)
            c3.markdown(f"<span style='font-family:monospace;font-size:12px'>{r['first_esc_turn']}</span>", unsafe_allow_html=True)
            c4.markdown(f"<span style='font-family:monospace;font-size:12px'>{r['start_sentiment']:+.3f}</span>", unsafe_allow_html=True)
            c5.markdown(f"<span style='font-family:monospace;font-size:12px'>{r['end_sentiment']:+.3f}</span>", unsafe_allow_html=True)
            c6.markdown(delta_html, unsafe_allow_html=True)
            c7.markdown(f"<span style='font-size:11px;color:{_muted}'>{r['trigger_phrase'][:45]}{'…' if len(r['trigger_phrase'])>45 else ''}</span>", unsafe_allow_html=True)
            if c8.button("🔍 Explore", key=f"det_explore_{i}", help=f"Open {r['conversation_id']} in Explorer"):
                st.session_state["exp_search"] = r["conversation_id"]
                st.session_state["page"]       = "🗣️ Explorer"
                st.rerun()
    else:
        st.success("✅ No significantly deteriorated conversations detected.")

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    _exp_col, _ = st.columns([1, 3])
    with _exp_col:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download Escalation Report (.xlsx)",
            data=_to_escalation_excel(intel),
            file_name=f"tbt_escalation_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Downloads an Excel workbook with 5 sheets: Clusters · Trigger Phrases · Deteriorated Conversations · Severity Distribution · Recovery Summary",
            type="primary",
            width="stretch",
        )

# ─── Narrative & Export ───────────────────────────────────────────────────────
def page_narrative_export(df_r, ins):
    tab_nar, tab_data, tab_json = st.tabs(["💡 Executive Summary", "📋 Data Table", "🔍 Raw JSON"])

    with tab_nar:
        sh("💡","Narrative Intelligence")
        cs=ins["customer_satisfaction"]; ap=ins["agent_performance"]
        cp=ins["conversation_patterns"]; pcd=ins.get("phase_csat_dsat",{})
        total=ins["total_turns"]; convs=ins["total_conversations"]
        sv=("strongly positive" if ins["overall_sentiment"]["average"]>0.2
            else "positive" if ins["overall_sentiment"]["average"]>0.05
            else "neutral"  if ins["overall_sentiment"]["average"]>-0.05
            else "negative" if ins["overall_sentiment"]["average"]>-0.2
            else "strongly negative")
        tv=("improving" if cp["sentiment_improvement"]>0.05
            else "declining" if cp["sentiment_improvement"]<-0.05 else "stable")
        paras=[
            f"**Dataset overview:** {convs:,} conversations totalling **{total:,} turns** were analysed. Average turns per conversation: **{ins['avg_turns_per_conversation']:.1f}**.",
            f"**Overall sentiment** is **{sv}** (avg {ins['overall_sentiment']['average']:+.3f}). Customer avg {cs['average_sentiment']:+.3f} · Agent avg {ap['average_sentiment']:+.3f}.",
            f"**Trend** is **{tv}** — sentiment shifts {cp['sentiment_improvement']:+.3f} from start to end. Start {cp['avg_sentiment_start']:+.3f} → Middle {cp['avg_sentiment_middle']:+.3f} → End {cp['avg_sentiment_end']:+.3f}.",
            f"**Escalation rate** {_pct(cs['escalation_rate'])} · **Resolution rate** {_pct(cs['resolution_rate'])}. " + ("Escalation above 15% — investigate trigger topics. " if cs['escalation_rate']>0.15 else "") + ("Resolution below 50% — improve closing strategies." if cs['resolution_rate']<0.5 else ""),
        ]
        for pn in ["start","middle","end"]:
            p=pcd.get(pn,{})
            if p.get("count",0)>0:
                paras.append(f"**{pn.capitalize()} phase** ({p['count']:,} customer turns): CSAT {_pct(p['csat_pct'])} · DSAT {_pct(p['dsat_pct'])} · avg {p['avg_sentiment']:+.3f}.")
        card_top = (
            f'<div style="background:{C["card"]};border:1px solid {C["border"]};border-radius:12px;'
            f'padding:1.5rem 1.8rem;margin-bottom:1rem;border-left:4px solid {C["teal"]}">'
            f'<div style="font-size:11px;color:{C["muted"]};text-transform:uppercase;'
            f'letter-spacing:1.2px;margin-bottom:.8rem;font-weight:600">Executive Summary — Auto-Generated</div>'
        )
        st.markdown(card_top, unsafe_allow_html=True)
        for p in paras: st.markdown(p)
        st.markdown("</div>", unsafe_allow_html=True)
        sh("🔔","Recommendations")
        for rec in ins.get("recommendations",[]): st.markdown(f'<div class="rc">{rec}</div>',unsafe_allow_html=True)
        st.markdown("---")
        _export_section(df_r, ins)

    with tab_data:
        sh("📋","Full Results Table")
        c1,c2,c3=st.columns(3)
        with c1: fs  =st.selectbox("Speaker",  ["All","CUSTOMER","AGENT"],key="dt_spk")
        with c2: fsen=st.selectbox("Sentiment",["All","positive","neutral","negative"],key="dt_sen")
        with c3: fp  =st.selectbox("Phase",    ["All","start","middle","end"],key="dt_ph")
        dt=df_r.copy()
        if fs!="All":   dt=dt[dt["speaker"]==fs]
        if fsen!="All": dt=dt[dt["sentiment_label"]==fsen]
        if fp!="All":   dt=dt[dt["phase"]==fp]
        cols=[c for c in ["conversation_id","turn_sequence","phase","speaker","timestamp",
                           "message","sentiment_label","compound","sentiment_confidence",
                           "potential_escalation","potential_resolution"] if c in dt.columns]
        total_r=len(dt); page_size=200
        st.markdown(f"**{total_r:,} rows** after filters &nbsp;·&nbsp; showing {min(page_size,total_r):,} per page")
        page_n=st.number_input("Page",min_value=1,max_value=max(1,(total_r-1)//page_size+1),value=1,step=1,key="dt_page")
        s=(page_n-1)*page_size; e=min(s+page_size,total_r)
        st.dataframe(dt[cols].iloc[s:e].reset_index(drop=True), width="stretch", height=420)
        st.download_button("⬇️ Download filtered CSV",
            data=_to_csv(dt[cols]),
            file_name=f"tbt_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv")

    with tab_json:
        sh("🔍","Raw Insights JSON")
        st.json(ins)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Initialise page state
    if "page" not in st.session_state:
        st.session_state["page"] = "🏠 Home"

    page = st.session_state.get("page", "🏠 Home")

    # ── Landing page — rendered BEFORE sidebar so its CSS (which hides the
    #    sidebar) never interferes with the app layout on inner pages.
    if page == "🏠 Home":
        render_landing()
        return

    # ── Inner app pages — sidebar rendered only here ──────────────────────────
    dataset_type, uploaded, run_clicked, pii_enabled, pii_mode, excel_sheet = render_sidebar()

    # ── Run pipeline when user clicks ▶ Run Analysis ──────────────────────────
    if uploaded is not None and run_clicked:
        try:
            file_bytes = uploaded.read()
            checksum   = _file_checksum(file_bytes)

            # ── Bust stale session results whenever file, domain, or PII settings change ──
            prev_checksum = st.session_state.get("_file_checksum")
            prev_domain   = st.session_state.get("_dataset_type")
            prev_pii      = st.session_state.get("_pii_key")
            pii_key       = f"{pii_enabled}_{pii_mode}"
            if checksum != prev_checksum or dataset_type != prev_domain or pii_key != prev_pii:
                for k in ("df_r", "ins", "detected", "fname", "pii_meta"):
                    st.session_state.pop(k, None)
                st.cache_data.clear()

            pb = st.progress(0, text="Starting pipeline…")
            try:
                _t0 = datetime.now()
                df_r, ins, detected, pii_meta = run_pipeline(
                    file_bytes, uploaded.name, dataset_type, progress_bar=pb,
                    pii_enabled=pii_enabled, pii_mode=pii_mode,
                    excel_sheet=excel_sheet,
                )
                _pipeline_secs = (datetime.now() - _t0).total_seconds()
                pb.empty()
                # Map the human-readable detected label back to a domain key
                # so the sidebar selectbox can sync to it automatically.
                _detected_key = next(
                    (k for k, v in FORMAT_LABELS.items() if v == detected), "auto"
                )
                st.session_state.update({
                    "df_r": df_r, "ins": ins,
                    "detected": detected, "fname": uploaded.name,
                    "pii_meta": pii_meta,
                    "_file_checksum":    checksum,
                    "_dataset_type":     dataset_type,
                    "_pii_key":          pii_key,
                    "detected_domain_key": _detected_key,
                    "_pipeline_secs":    _pipeline_secs,
                })
                if st.session_state.get("page") in ("🏠 Home", "📊 Overview"):
                    st.session_state["page"] = "📊 Overview"
                st.rerun()
            except MemoryError as exc:
                pb.empty()
                gc.collect()
                st.error(
                    f"💾 **Out of Memory — {exc}**\n\n"
                    "**Quick fixes:**\n"
                    "1. Split your file into smaller batches (≤ 5,000 rows / ≤ 50,000 turns)\n"
                    "2. Click **🗑️ Clear & Reset** then re-upload a smaller file\n"
                    "3. Restart the app to free all cached memory"
                )
                return
            except Exception as exc:
                pb.empty()
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)
                return
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            return

    # ── Guard: no results yet → show upload prompt ────────────────────────────
    has_results = "df_r" in st.session_state and "ins" in st.session_state
    if not has_results:
        st.markdown("<br>" * 4, unsafe_allow_html=True)
        st.info("👆 Upload a file in the sidebar and click **▶ Run Analysis** to get started.")
        return

    # ── Result pages ──────────────────────────────────────────────────────────
    df_r     = st.session_state["df_r"]
    ins      = st.session_state["ins"]
    detected = st.session_state.get("detected", "—")
    fname    = st.session_state.get("fname", "")
    pii_meta = st.session_state.get("pii_meta", {"enabled": False, "mode": "off", "redacted_rows": 0})

    # Read pipeline timing from session state (set when Run Analysis was clicked)
    _pipeline_secs = st.session_state.get("_pipeline_secs")

    # Status bar — file info + PII badge only (timing is in the KPI row now)
    ca, cb_col, cc = st.columns([3, 2, 1])
    with ca:
        pii_badge = ""
        if pii_meta.get("enabled"):
            n_r  = pii_meta.get("redacted_rows", 0)
            mode = pii_meta.get("mode", "mask")
            pii_badge = (
                f' &nbsp;·&nbsp; <span style="background:rgba(45,95,110,0.12);' 
                f'border:1px solid {C["teal"]};border-radius:4px;padding:1px 7px;' 
                f'font-size:11px;color:{C["teal"]};font-weight:600">' 
                f'🛡️ PII {mode} · {n_r:,} rows redacted</span>'
            )
        st.markdown(
            f'<div style="color:{C["muted"]};font-size:.82rem">' 
            f'📂 {fname} &nbsp;·&nbsp; ' 
            f'Format: <strong style="color:{C["teal"]}">{detected}</strong>' 
            f'{pii_badge}</div>',
            unsafe_allow_html=True,
        )
    with cc:
        st.download_button(
            "⬇️ Quick ZIP",
            data=_to_zip(df_r, ins),
            file_name=f"tbt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            width="stretch",
        )

    st.markdown("---")
    _kpi_row(ins, df_r=df_r, pipeline_secs=_pipeline_secs)
    st.markdown("---")

    if   page == "📊 Overview":           page_overview(df_r, ins)
    elif page == "🌊 Sankey Flow":         page_sankey(df_r, ins)
    elif page == "⚠️ Escalation":          page_escalation(df_r, ins)
    elif page == "🗣️ Explorer":            page_explorer(df_r)
    elif page == "💡 Narrative & Export":  page_narrative_export(df_r, ins)

    st.markdown(
        f'<div style="text-align:center;color:{C["muted"]};font-size:11px;padding:16px 0">'
        f'TbT Sentiment Analytics v5.1 &nbsp;·&nbsp; Domain Agnostic</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
