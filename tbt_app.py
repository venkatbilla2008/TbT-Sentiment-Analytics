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

import gc, io, json, re, warnings, zipfile
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

warnings.filterwarnings("ignore")

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
MAX_TURNS     = 500_000   # hard safety cap
CHART_SAMPLE  = 25_000    # max points sent to browser for scatter/sunburst
VADER_WORKERS = 8        # threads for parallel VADER scoring
CHART_LAYOUT  = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C['text'], family="DM Sans"),
    margin=dict(l=10, r=20, t=40, b=10),
    hoverlabel=dict(bgcolor=C['text'], font_size=12, font_color=C['warm_l']),
)

# ─────────────────────────────────────────────────────────────────────────────
# PII REDACTION  — 8 pattern types (ported from TextInsightMiner)
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
# MATH HELPERS  (vectorised numpy — no loops, no compile delay)
# ─────────────────────────────────────────────────────────────────────────────
def _rolling_mean3(arr: np.ndarray) -> np.ndarray:
    """Causal 3-point rolling mean, fully vectorised."""
    r = arr.copy().astype(np.float64)
    if len(r) > 1: r[1:] = (arr[:-1] + arr[1:]) / 2   # seed positions 1+
    if len(r) > 2: r[2:] = (arr[:-2] + arr[1:-1] + arr[2:]) / 3
    return r

def _diff(arr: np.ndarray) -> np.ndarray:
    """First-difference, index 0 = 0."""
    r = np.empty_like(arr, dtype=np.float64)
    r[0] = 0.0
    if len(r) > 1: r[1:] = arr[1:] - arr[:-1]
    return r


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
        self._pt  = re.compile(r"^\|?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\s+(Consumer|Customer|Agent|Advisor|Support):\s*(.*)$", re.I)
        self._pb  = re.compile(r"^\[(\d{1,2}:\d{2}:\d{2})\s+(AGENT|CUSTOMER|CONSUMER|ADVISOR|SUPPORT)\]:\s*(.*)$", re.I)
        self._ph  = re.compile(r"\[(\d{1,3}:\d{2})\]\s+([^:]+?):\s*([^\[]+?)(?=\[|$)", re.I|re.DOTALL)
        self._pph = re.compile(r"<b>(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*</b>([^<]+?)(?:<br\s*/?>|$)", re.I|re.DOTALL)
        self._pps = re.compile(
            r"(?<!\d{4}-\d{2}-\d{2} )"   # NOT preceded by ISO date (avoids stealing Spotify rows)
            r"(\d{2}:\d{2}:\d{2})"        # HH:MM:SS timestamp
            r"\s+([^:]+?)\s*:\s*(.+?)(?=\d{2}:\d{2}:\d{2}\s+|$)",
            re.DOTALL
        )

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self._find_col(df)
        if not col: raise ValueError("No transcript column found. Expected: Conversation, Transcripts, Comments, message, chat, etc.")
        if self.dataset_type == "auto":
            # Use multi-row voting for robust detection (not just first row)
            self.dataset_type = self._detect_from_df(df, col)
        rows: List[Dict] = []
        for idx, row in df.iterrows():
            text = str(row[col])
            if not text or text == "nan" or len(text) < 5: continue
            rows.extend(self._dispatch(text, int(idx)))
        if not rows:
            raise ValueError("No turns parsed. Check the domain selector matches your file format.")
        out = pd.DataFrame(rows)
        out["turn_id"]         = range(1, len(out) + 1)
        out["cleaned_message"] = out["message"].str.lower().str.strip()
        return out

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

    def _dispatch(self, text, idx):
        if self.dataset_type == "netflix":  return self._parse_bracket(text, idx)
        if self.dataset_type == "humana":   return self._parse_humana(text, idx)
        if self.dataset_type == "ppt":      return self._parse_ppt(text, idx)
        return self._parse_spotify(text, idx)

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

    def _row(self, idx, seq, ts, spk, msg):
        return {"conversation_id": f"CONV_{idx+1:04d}", "turn_sequence": seq,
                "timestamp": ts, "speaker": spk, "message": msg}

    def _parse_bracket(self, text, idx):
        lines=text.split("\n"); turns=[]; tn=1; cs=ct=None; cm=[]
        def flush():
            nonlocal tn
            if cs:
                msg=" ".join(cm).strip()
                if msg: turns.append(self._row(idx,tn,ct,self._norm(cs),msg)); tn+=1
        for line in lines:
            ls=line.strip(); m=self._pb.match(ls)
            if m:
                flush(); ct,cs,cm=m.group(1),m.group(2),[]
                r=m.group(3).strip()
                if r: cm.append(r)
            elif cs and ls: cm.append(ls)
        flush(); return turns

    def _parse_spotify(self, text, idx):
        lines=text.split("\n"); turns=[]; tn=1; cs=ct=None; cm=[]
        def flush():
            nonlocal tn
            if cs:
                msg=" ".join(cm).strip()
                if msg: turns.append(self._row(idx,tn,ct,self._norm(cs),msg)); tn+=1
        for line in lines:
            ls=line.strip().lstrip("|").strip()  # strip leading pipe used by some Spotify exports
            m=self._pt.search(ls)                # search() handles (?:^|\n) anchor in pattern
            if m:
                flush(); ct,cs=m.group(1),m.group(2); cm=[m.group(3).strip()] if m.group(3).strip() else []
            elif cs and ls: cm.append(ls)
        flush(); return turns

    def _parse_humana(self, text, idx):
        turns=[]; tn=1
        for ts,spk,msg in self._ph.findall(text):
            sl=spk.strip().lower()
            if sl in {"system","automated","ivr"}: continue
            m=msg.strip()
            if not m or len(m)<3: continue
            ns=("CUSTOMER" if any(k in sl for k in ["member","customer","patient","caller"])
                else "AGENT" if any(k in sl for k in ["agent","representative","rep","advisor","specialist"])
                else spk.strip().upper())
            turns.append(self._row(idx,tn,ts,ns,m)); tn+=1
        return turns

    def _parse_ppt(self, text, idx):
        hm=self._pph.findall(text)
        if hm: return self._ppt_turns(hm,idx,False)
        sm=self._pps.findall(text)
        if sm: return self._ppt_turns(sm,idx,True)
        return []

    def _ppt_turns(self, matches, idx, is_sms):
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
            cust=(ordered[0] if len(cnts)==1 or (ordered and cnts.get(ordered[0],999)<=sorted(cnts.values())[0])
                  else min(cnts,key=cnts.get))
            for s in ordered: roles[s]="CUSTOMER" if s==cust else "AGENT"
        all_m=sorted([(ts,s,m) for s in ordered for ts,m in spk_msgs[s]],key=lambda x:x[0])
        return [self._row(idx,i,ts,roles.get(s,"CUSTOMER"),m) for i,(ts,s,m) in enumerate(all_m,1)]


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT ENGINE  — parallel VADER + vectorised label assignment
# ─────────────────────────────────────────────────────────────────────────────
def _score_chunk(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Score one chunk of messages in a worker thread.
    Each thread owns its own SentimentIntensityAnalyzer instance
    (VADER is not thread-safe if shared).

    Returns (start_index, array of shape [chunk_size, 4])
    columns: compound, pos, neg, neu
    """
    start, msgs = args
    vader = SentimentIntensityAnalyzer()
    n     = len(msgs)
    out   = np.zeros((n, 4), dtype=np.float32)   # compound, pos, neg, neu
    for j, m in enumerate(msgs):
        if len(str(m)) < 5: continue
        sc = vader.polarity_scores(str(m))
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
        """Set adaptive thresholds from 30th / 70th percentile of a sample."""
        n    = min(1000, len(df))
        msgs = df["cleaned_message"].fillna("").sample(n=n, random_state=42)
        scores = np.array([
            self._vader.polarity_scores(str(m))["compound"]
            for m in msgs if len(str(m)) > 5
        ], dtype=np.float64)
        if len(scores) == 0: return
        pos = max(float(np.percentile(scores, 70)), 0.10)
        neg = min(float(np.percentile(scores, 30)), -0.10)
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

        # Collect to pandas for per-group diff/rolling (still vectorised per group)
        d = lf.collect().to_pandas()
        d = d.sort_values(["conversation_id","turn_sequence"]).reset_index(drop=True)

        chg=[]; mom=[]
        for _, grp in d.groupby("conversation_id", sort=False):
            s = grp["compound"].to_numpy(dtype=np.float64)
            ch = _diff(s); mo = _rolling_mean3(ch)
            chg.extend(ch.tolist()); mom.extend(mo.tolist())
        d["sentiment_change"]   = chg
        d["sentiment_momentum"] = mom

        prev = d.groupby("conversation_id", sort=False)["speaker"].shift(1)
        d["prev_speaker"]      = prev
        d["speaker_changed"]   = d["speaker"] != prev
        d["consecutive_turns"] = (
            d.groupby(["conversation_id", (d["speaker"] != prev).cumsum()]).cumcount() + 1
        )
        d["potential_escalation"] = (
            (d["sentiment_change"] < -0.3) & (d["speaker"] == "CUSTOMER") & (d["turn_sequence"] > 2)
        )
        d["potential_resolution"] = (
            (d["sentiment_change"] > 0.2) & (d["speaker"] == "CUSTOMER") & d["is_conversation_end"]
        )
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
        ins["customer_satisfaction"] = {
            "average_sentiment": cu.get("compound", 0.0),
            "positive_pct":      pos_pct,
            "escalation_rate":   cu.get("potential_escalation", 0.0),
            "resolution_rate":   cu.get("potential_resolution", 0.0),
        }
        ag_std = float(
            lf.filter(pl.col("speaker") == "AGENT")
              .select(pl.col("compound").std()).collect().item() or 0.0
        )
        ins["agent_performance"] = {
            "average_sentiment":      ag.get("compound", 0.0),
            "response_effectiveness": ag.get("sentiment_change", 0.0),
            "consistency_score":      max(0.0, 1.0 - ag_std),
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
                   ]).collect().to_dicts()
        )
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
        if cs["resolution_rate"]         < 0.5:  r.append(f"🔴 Low resolution rate ({cs['resolution_rate']:.1%}) — improve closing strategies.")
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
# ─────────────────────────────────────────────────────────────────────────────
import hashlib

def _file_checksum(file_bytes: bytes) -> str:
    """SHA-256 of file bytes — used as a stable, unique cache key."""
    return hashlib.sha256(file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def _cached_parse(checksum: str, fname: str, dataset_type: str,
                  file_bytes: bytes,
                  pii_enabled: bool = False,
                  pii_mode: str = "mask") -> pd.DataFrame:
    """
    Parse raw bytes → turns DataFrame.
    Cache key = SHA-256 checksum + filename + domain + PII settings.
    Using checksum (not raw bytes) as first arg guarantees Streamlit hashes
    a short string rather than a potentially huge bytes object — preventing
    false cache hits when the internal hash truncates large files.
    """
    if fname.endswith(".csv"):
        df_raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes))
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
    """Score turns with VADER. Cached by DataFrame content."""
    sent = SentimentEngine()
    sent.calibrate(df_p)
    return sent.score(df_p)


@st.cache_data(show_spinner=False)
def _cached_analytics(df_s: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute turn metrics + insights.  Cached by scored DataFrame content.
    Changing the file or domain key busts the cache automatically.
    """
    anal = AnalyticsEngine()
    df_r = anal.compute_turn_metrics(df_s)
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
) -> Tuple[pd.DataFrame, Dict, str]:
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

    if progress_bar: progress_bar.progress(0.10, text="Parsing transcripts…")
    df_p     = _cached_parse(checksum, fname, dataset_type, file_bytes,
                             pii_enabled=pii_enabled, pii_mode=pii_mode)
    detected = df_p.attrs.get("detected_format", "—")

    # Safety cap
    if len(df_p) > MAX_TURNS:
        df_p = df_p.head(MAX_TURNS)
        st.warning(f"⚠️ Dataset capped at {MAX_TURNS:,} turns for performance. "
                   "Split your file for full analysis.")

    if progress_bar: progress_bar.progress(0.35, text=f"Scoring {len(df_p):,} turns in parallel…")
    df_s = _cached_score(df_p)

    if progress_bar: progress_bar.progress(0.75, text="Computing analytics with Polars…")
    df_r, ins = _cached_analytics(df_s)

    # Carry PII audit metadata forward so the UI can display a badge
    pii_meta = {
        "enabled":       pii_enabled,
        "mode":          pii_mode if pii_enabled else "off",
        "redacted_rows": df_p.attrs.get("pii_redacted_rows", 0),
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

    return {
        "sent_dist":    sent_dist,
        "conv_map":     conv_map,
        "phase_speaker":phase_speaker,
        "turn_prog":    turn_prog,
        "esc_res":      esc_res,
        "sample":       sample,
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
        title=f"Turn-by-Turn Flow — {conv_id}",
        title_font_size=14,
        title_x=0,
        margin=dict(l=10, r=20, t=70, b=10),   # extra top margin to separate title from legend
        xaxis=dict(title="Turn"),
        yaxis=dict(title="Sentiment Score", range=[-1.1, 1.1]),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.08,           # clear of title
            xanchor="left",   x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
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
        title=f"Conversation Comparison — {conv_a} vs {conv_b}",
        title_font_size=14,
        title_x=0,
        margin=dict(l=10, r=20, t=70, b=10),   # extra top margin so title + legend don't collide
        xaxis=dict(title="Turn Position (0=start, 1=end)"),
        yaxis=dict(title="Sentiment Score", range=[-1.1, 1.1]),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.08,           # sits above plot but below title
            xanchor="left",   x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
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
    worst10  : 10 conversations with lowest avg customer sentiment
    best10   : 10 with highest avg customer sentiment
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
<div class="lp-wb">
  <span class="cm"># Parallel VADER · 4 threads · vectorised labels</span><br>
  <span class="ck">with</span> ThreadPoolExecutor(<span class="ck">max_workers</span>=<span class="cs">4</span>) <span class="ck">as</span> ex:<br>
  &nbsp;&nbsp;futures = [ex.submit(_score_chunk, chunk) <span class="ck">for</span> chunk <span class="ck">in</span> chunks]<br>
  &nbsp;&nbsp;labels = np.<span class="cf">where</span>(compound &gt;= thr[<span class="cs">"pos"</span>], <span class="cs">"positive"</span>, ...)<span class="lp-cur"></span>
  <div class="lp-bars">
    <div class="lp-br"><span class="lp-bl">🟢 Positive</span><div class="lp-bt"><div class="lp-bf lb1"></div></div><span class="lp-bp">CSAT</span></div>
    <div class="lp-br"><span class="lp-bl">🔵 Neutral</span><div class="lp-bt"><div class="lp-bf lb2"></div></div><span class="lp-bp">stable</span></div>
    <div class="lp-br"><span class="lp-bl">🟡 Escalation</span><div class="lp-bt"><div class="lp-bf lb3"></div></div><span class="lp-bp">⚠️</span></div>
    <div class="lp-br"><span class="lp-bl">🔴 Negative</span><div class="lp-bt"><div class="lp-bf lb4"></div></div><span class="lp-bp">DSAT</span></div>
    <div class="lp-br"><span class="lp-bl">⬛ Unknown</span><div class="lp-bt"><div class="lp-bf lb5"></div></div><span class="lp-bp">low</span></div>
  </div>
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
<div class="lp-st"><div class="lp-sn">50<span>K</span></div><div class="lp-sl">Turns Supported</div></div>
<div class="lp-st"><div class="lp-sn">4</div><div class="lp-sl">Parallel Threads</div></div>
<div class="lp-st"><div class="lp-sn">5</div><div class="lp-sl">Cached Stages</div></div>
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
<h3>Sentiment Waterfall</h3>
<p>Start → Middle → End delta chart reveals exactly where sentiment improves or drops across the conversation arc.</p>
</div>

<div class="lp-fc">
<div class="lp-fi">🌐</div>
<h3>Sunburst Breakdown</h3>
<p>Speaker → Phase → Sentiment hierarchy in a single interactive sunburst. Spot which role drives negativity at a glance.</p>
</div>

<div class="lp-fc">
<div class="lp-fi">⚡</div>
<h3>Escalation Detection</h3>
<p>Rule-based escalation and resolution flagging. Timeline scatter maps every event across turn positions in all conversations.</p>
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
<tr><td>Visualisations</td><td>Basic charts</td><td>Flow, Waterfall, Sunburst, Heatmap</td></tr>
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
<p class="lp-ctatrust">No cloud dependency. Your data stays on your machine.</p>
</div>

<div class="lp-fo">TbT Sentiment Analytics v5.0 — Domain Agnostic · Turn-by-Turn Intelligence</div>
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
        pages=["🏠 Home","📊 Overview","🔄 TbT Flow","🗣️ Explorer","📋 Data Table","💡 Narrative & Export"]
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
        if uploaded:
            st.markdown(f"""
<div style="background:rgba(45,95,110,0.1);border:1px solid {C['teal']};
     border-radius:8px;padding:.5rem .8rem;margin-top:.5rem">
  <div style="color:{C['muted']};font-size:.72rem">Loaded:</div>
  <div style="color:{C['text']};font-weight:600;font-size:.88rem">📄 {uploaded.name}</div>
</div>""", unsafe_allow_html=True)

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
                          "_file_checksum","_dataset_type","_pii_key","detected_domain_key"):
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

        st.markdown(
            f'<div style="color:{C["muted"]};font-size:.7rem;text-align:center;padding-top:.5rem">'
            f'v5.0 — Polars · Parallel VADER · 5-stage Cache</div>',
            unsafe_allow_html=True,
        )

    return dataset_type, uploaded, run, pii_enabled, pii_mode


# ─────────────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def _kpi_row(ins):
    cs=ins["customer_satisfaction"]; ap=ins["agent_performance"]; cp=ins["conversation_patterns"]
    overall=ins["overall_sentiment"]["average"]
    esc_c=C['neg'] if cs["escalation_rate"]>0.15 else C['gold'] if cs["escalation_rate"]>0.10 else C['ok']
    res_c=C['ok']  if cs["resolution_rate"]>0.6  else C['gold'] if cs["resolution_rate"]>0.4  else C['neg']
    cols=st.columns(8)
    data=[
        ("Conversations",  f"{ins['total_conversations']:,}",         "var(--teal)"),
        ("Total Turns",    f"{ins['total_turns']:,}",                  "var(--slate)"),
        ("Overall Sent.",  f"{overall:+.3f}",                          _score_color(overall)),
        ("Customer Avg",   f"{cs['average_sentiment']:+.3f}",          _score_color(cs["average_sentiment"])),
        ("Agent Avg",      f"{ap['average_sentiment']:+.3f}",          _score_color(ap["average_sentiment"])),
        ("Escalation",     _pct(cs["escalation_rate"]),                esc_c),
        ("Resolution",     _pct(cs["resolution_rate"]),                res_c),
        ("Trend",          f"{cp['sentiment_improvement']:+.3f}",      _score_color(cp["sentiment_improvement"])),
    ]
    for col,(lbl,val,color) in zip(cols,data):
        with col: st.markdown(mc(lbl,f'<span style="color:{color}">{val}</span>',color),
                               unsafe_allow_html=True)

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
    sub=df[df["conversation_id"]==conv_id].sort_values("turn_sequence")
    if sub.empty: st.info("No turns for this conversation."); return
    for _,r in sub.iterrows():
        spk=str(r["speaker"]).upper(); css="tc-cu" if spk=="CUSTOMER" else "tc-ag"
        icon="👤" if spk=="CUSTOMER" else "🎧"
        ts=f" · {r['timestamp']}" if r.get("timestamp") and str(r["timestamp"]) not in ("nan","None","") else ""
        pi=PHASE_ICONS.get(str(r.get("phase","middle")),"🔄")
        s=float(r["compound"]); lbl=str(r.get("sentiment_label","neutral"))
        st.markdown(
            f'<div class="tc {css}">'
            f'<div class="tc-hdr">{icon} {spk}{ts} &nbsp; {pi} {str(r.get("phase","")).capitalize()} &nbsp; Turn #{int(r["turn_sequence"])}</div>'
            f'<div class="tc-txt">{r["message"]}</div>'
            f'<div class="tc-meta">{_badge(lbl)} &nbsp; {_sbar(s)} &nbsp; Confidence: {float(r.get("sentiment_confidence",0)):.0%}</div>'
            f'</div>', unsafe_allow_html=True)

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


# ─── Overview ─────────────────────────────────────────────────────────────────
def page_overview(df_r, ins):
    aggs = _precompute_aggs(df_r)
    sh("📊","Phase-Level CSAT / DSAT"); _phase_table(ins)
    sh("🌊","Sentiment Journey"); st.plotly_chart(_chart_waterfall(ins), width="stretch")
    sh("📈","Visual Overview")
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(_chart_sentiment_dist(aggs),    width="stretch")
    with c2: st.plotly_chart(_chart_speaker_box(df_r),        width="stretch")
    c3,c4=st.columns(2)
    with c3: st.plotly_chart(_chart_phase_comparison(ins),          width="stretch")
    with c4: st.plotly_chart(_chart_escalation_resolution(aggs),    width="stretch")
    sh("🌐","Conversation Sunburst")
    st.plotly_chart(_chart_sunburst(aggs), width="stretch")
    sh("📉","Sentiment Progression")
    st.plotly_chart(_chart_sentiment_progression(aggs), width="stretch")

# ─── TbT Flow ────────────────────────────────────────────────────────────────
def page_tbt_flow(df_r):
    sh("🔄", "Turn-by-Turn Sentiment Flow")

    # ── Cached conversation lists ──────────────────────────────────────────
    lists = _get_smart_conv_lists(df_r)

    # ── Conversation selector row ──────────────────────────────────────────
    c_mode, c_search, c_view = st.columns([1.6, 2, 1.2])

    with c_mode:
        conv_mode = st.selectbox(
            "Quick filter",
            ["All conversations",
             "😡 Worst 20 (lowest customer sentiment)",
             "😊 Best 20 (highest customer sentiment)",
             "📏 Longest 20 (most turns)"],
            key="flow_mode",
        )

    # Pick pool based on mode
    pool = {
        "😡 Worst 20 (lowest customer sentiment)":  lists["worst20"],
        "😊 Best 20 (highest customer sentiment)":  lists["best20"],
        "📏 Longest 20 (most turns)":               lists["longest20"],
    }.get(conv_mode, lists["all_ids"])

    with c_search:
        search_txt = st.text_input("🔍 Search conversation ID", value="",
                                   placeholder="e.g. CONV_0042", key="flow_search")
        if search_txt.strip():
            pool = [c for c in lists["all_ids"]
                    if search_txt.strip().upper() in c.upper()] or pool

    with c_view:
        view_mode = st.radio("View", ["Single", "Compare ×2"],
                             horizontal=True, key="flow_viewmode")

    # ── Speaker toggle + filters ───────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([1, 1, 1])
    with fc1:
        sel = st.selectbox("Conversation A", pool, key="flow_conv")
    with fc2:
        spk_toggle = st.toggle("Split by speaker", value=True, key="flow_spk_toggle")
    with fc3:
        pflt = st.selectbox("Phase filter", ["All","start","middle","end"], key="flow_ph")

    sel_b = None
    if view_mode == "Compare ×2":
        sel_b = st.selectbox("Conversation B", [c for c in pool if c != sel],
                             key="flow_conv_b")

    # ── Mini KPI strip for selected conversation ───────────────────────────
    dv = df_r[df_r["conversation_id"] == sel].copy()
    if pflt != "All": dv = dv[dv["phase"] == pflt]
    cu = dv[dv["speaker"] == "CUSTOMER"]
    ag = dv[dv["speaker"] == "AGENT"]
    esc_n = int(dv["potential_escalation"].sum()) if "potential_escalation" in dv.columns else 0
    res_n = int(dv["potential_resolution"].sum()) if "potential_resolution" in dv.columns else 0

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Turns",          len(df_r[df_r["conversation_id"]==sel]))
    k2.metric("Customer Avg",   f"{cu['compound'].mean():+.3f}" if not cu.empty else "—")
    k3.metric("Agent Avg",      f"{ag['compound'].mean():+.3f}" if not ag.empty else "—")
    k4.metric("⚠️ Escalations", esc_n,
              delta="High" if esc_n > 2 else None,
              delta_color="inverse")
    k5.metric("✅ Resolutions",  res_n)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Main charts ────────────────────────────────────────────────────────
    if view_mode == "Compare ×2" and sel_b:
        sh("⚖️", f"Comparison: {sel} vs {sel_b}")
        st.plotly_chart(_chart_compare_two(df_r, sel, sel_b), width="stretch")

    else:
        # ── Tab set: Flow / Replay / Momentum / Heatmap ──
        tab_flow, tab_replay, tab_mom, tab_heat = st.tabs([
            "📈 Flow Chart", "▶ Replay", "📊 Momentum", "🌡️ Heatmap"
        ])

        with tab_flow:
            st.plotly_chart(
                _chart_tbt_flow(df_r, sel, show_speaker_lines=spk_toggle),
                width="stretch",
            )
            # ── Export chart as PNG ──────────────────────────────────────
            try:
                import plotly.io as pio
                fig_bytes = pio.to_image(
                    _chart_tbt_flow(df_r, sel, show_speaker_lines=spk_toggle),
                    format="png", width=1200, height=500, scale=2,
                )
                st.download_button(
                    "📷 Export chart as PNG",
                    data=fig_bytes,
                    file_name=f"flow_{sel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                )
            except Exception:
                st.caption("PNG export requires `kaleido` — `pip install kaleido`")

        with tab_replay:
            st.caption("Press ▶ Play to watch sentiment evolve turn by turn, "
                       "or drag the slider manually.")
            st.plotly_chart(_chart_replay_animation(df_r, sel), width="stretch")

        with tab_mom:
            st.plotly_chart(_chart_momentum(df_r, sel), width="stretch")

        with tab_heat:
            st.plotly_chart(_chart_speaker_heatmap(df_r, sel), width="stretch")

# ─── Explorer ────────────────────────────────────────────────────────────────
def page_explorer(df_r):
    sh("🗣️","Conversation Explorer")
    c1,c2=st.columns([1.2,1])
    with c1: conv=st.selectbox("Conversation",sorted(df_r["conversation_id"].unique()),key="exp_conv")
    with c2: ph  =st.selectbox("Phase",["All","start","middle","end"],key="exp_ph")
    sub=df_r[df_r["conversation_id"]==conv]
    if ph!="All": sub=sub[sub["phase"]==ph]
    _turn_viewer(sub, conv)

# ─── Data Table ──────────────────────────────────────────────────────────────
def page_data_table(df_r):
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
    total=len(dt); page_size=200
    st.markdown(f"**{total:,} rows** after filters &nbsp;·&nbsp; showing {min(page_size,total):,} per page")
    page_n=st.number_input("Page",min_value=1,max_value=max(1,(total-1)//page_size+1),value=1,step=1,key="dt_page")
    start=(page_n-1)*page_size; end=min(start+page_size,total)
    st.dataframe(dt[cols].iloc[start:end].reset_index(drop=True), width="stretch", height=420)
    st.download_button("⬇️ Download filtered CSV",
        data=_to_csv(dt[cols]),
        file_name=f"tbt_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv")

# ─── Narrative & Export ───────────────────────────────────────────────────────
def page_narrative_export(df_r, ins):
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
        f"**Dataset overview:** {convs:,} conversations totalling **{total:,} turns** were analysed. "
        f"Average turns per conversation: **{ins['avg_turns_per_conversation']:.1f}**.",

        f"**Overall sentiment** is **{sv}** (avg {ins['overall_sentiment']['average']:+.3f}). "
        f"Customer avg {cs['average_sentiment']:+.3f} · Agent avg {ap['average_sentiment']:+.3f}.",

        f"**Trend** is **{tv}** — sentiment shifts {cp['sentiment_improvement']:+.3f} from start to end. "
        f"Start {cp['avg_sentiment_start']:+.3f} → Middle {cp['avg_sentiment_middle']:+.3f} → End {cp['avg_sentiment_end']:+.3f}.",

        f"**Escalation rate** {_pct(cs['escalation_rate'])} · **Resolution rate** {_pct(cs['resolution_rate'])}. "
        + ("Escalation above 15% threshold — investigate trigger topics. " if cs['escalation_rate']>0.15 else "")
        + ("Resolution below 50% — closing strategies need improvement." if cs['resolution_rate']<0.5 else ""),
    ]
    for pn in ["start","middle","end"]:
        p=pcd.get(pn,{})
        if p.get("count",0)>0:
            paras.append(f"**{pn.capitalize()} phase** ({p['count']:,} customer turns): "
                         f"CSAT {_pct(p['csat_pct'])} · DSAT {_pct(p['dsat_pct'])} · avg {p['avg_sentiment']:+.3f}.")

    st.markdown(f"""
<div style="background:{C['card']};border:1px solid {C['border']};border-radius:12px;
     padding:1.5rem 1.8rem;margin-bottom:1rem;border-left:4px solid {C['teal']}">
  <div style="font-size:11px;color:{C['muted']};text-transform:uppercase;letter-spacing:1.2px;
       margin-bottom:.8rem;font-weight:600">Executive Summary — Auto-Generated</div>
""", unsafe_allow_html=True)
    for p in paras: st.markdown(p)
    st.markdown("</div>", unsafe_allow_html=True)

    sh("🔔","Recommendations")
    for rec in ins.get("recommendations",[]): st.markdown(f'<div class="rc">{rec}</div>',unsafe_allow_html=True)

    st.markdown("---")
    _export_section(df_r, ins)
    st.markdown("---")
    sh("🔍","Raw Insights JSON")
    with st.expander("View full insights object"): st.json(ins)


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
    dataset_type, uploaded, run_clicked, pii_enabled, pii_mode = render_sidebar()

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
                df_r, ins, detected, pii_meta = run_pipeline(
                    file_bytes, uploaded.name, dataset_type, progress_bar=pb,
                    pii_enabled=pii_enabled, pii_mode=pii_mode,
                )
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
                })
                if st.session_state.get("page") in ("🏠 Home", "📊 Overview"):
                    st.session_state["page"] = "📊 Overview"
                st.rerun()
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

    # Status bar
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
    _kpi_row(ins)
    st.markdown("---")

    if   page == "📊 Overview":           page_overview(df_r, ins)
    elif page == "🔄 TbT Flow":           page_tbt_flow(df_r)
    elif page == "🗣️ Explorer":           page_explorer(df_r)
    elif page == "📋 Data Table":         page_data_table(df_r)
    elif page == "💡 Narrative & Export": page_narrative_export(df_r, ins)

    st.markdown(
        f'<div style="text-align:center;color:{C["muted"]};font-size:11px;padding:16px 0">'
        f'TbT Sentiment Analytics v5.0 &nbsp;·&nbsp; Domain Agnostic</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
