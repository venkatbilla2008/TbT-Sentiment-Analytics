"""
tbt_app.py
==========
Domain Agnostic — Turn-by-Turn Sentiment Analytics (Streamlit)
Single-file app — all logic self-contained here.

Run with:
    streamlit run tbt_app.py

Fixes in this version
---------------------
- Removed numba dependency (caused Streamlit Cloud health-check crash)
- Fixed sidebar text/label visibility on dark background
- Improved export section with clearly visible download buttons
"""

from __future__ import annotations

# ===========================================================================
# 1. Imports
# ===========================================================================

import gc
import io
import json
import re
import warnings
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ===========================================================================
# Page config — MUST be first Streamlit call
# ===========================================================================

st.set_page_config(
    page_title="TbT Sentiment Analytics",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# 2. Global CSS
# ===========================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---- App header ---- */
.app-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px; padding: 2.4rem 2rem 1.8rem;
    margin-bottom: 1.8rem; text-align: center;
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
    box-shadow:0 4px 20px rgba(0,0,0,0.25); text-align:center;
}
.metric-card .m-label { color:#aaa; font-size:.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:.3rem; }
.metric-card .m-value { color:#fff; font-size:1.9rem; font-weight:700; line-height:1.1; }
.metric-card .m-sub   { color:#aaa; font-size:.8rem; margin-top:.25rem; }

/* ---- Phase table ---- */
.phase-table { width:100%; border-collapse:collapse; font-size:.88rem; }
.phase-table th { background:#1e1e3f; color:#ddd; padding:.6rem .9rem; text-align:left; font-weight:600; }
.phase-table td { padding:.55rem .9rem; border-bottom:1px solid rgba(255,255,255,0.05); color:#eee; }
.phase-table tr:hover td { background:rgba(255,255,255,0.03); }

/* ---- Sentiment badges ---- */
.badge-csat    { background:#1a6640; color:#7fff9e; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:600; }
.badge-dsat    { background:#6b1a1a; color:#ff9e9e; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:600; }
.badge-neutral { background:#3a3a5c; color:#c7c7ff; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:600; }

/* ---- Turn viewer cards ---- */
.turn-card { border-radius:10px; padding:.8rem 1rem; margin-bottom:.5rem; border-left:4px solid transparent; }
.turn-customer { background:rgba(255,107,107,0.08); border-color:#ff6b6b; }
.turn-agent    { background:rgba(78,205,196,0.08);  border-color:#4ecdc4; }
.turn-header   { font-size:.75rem; color:#aaa; margin-bottom:.2rem; }
.turn-text     { font-size:.93rem; color:#eee; }
.turn-meta     { font-size:.72rem; color:#999; margin-top:.3rem; }

/* ---- Recommendation cards ---- */
.rec-card {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
    border-radius:10px; padding:.75rem 1rem; margin-bottom:.5rem;
    font-size:.88rem; color:#ddd;
}

/* ---- Score bar ---- */
.score-bar-wrap  { display:flex; align-items:center; gap:.5rem; }
.score-bar-track { flex:1; height:6px; background:#444; border-radius:999px; overflow:hidden; }
.score-bar-fill  { height:100%; border-radius:999px; transition:width .4s; }

/* ---- Export button cards ---- */
.export-card {
    background: linear-gradient(135deg,#1a1a2e,#16213e);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: .75rem;
}
.export-card .ex-title { color:#fff; font-size:.95rem; font-weight:600; margin-bottom:.25rem; }
.export-card .ex-desc  { color:#aaa; font-size:.8rem; margin-bottom:.75rem; }

/* ---- SIDEBAR — force all text to be visible on dark bg ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #12122b 100%);
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stFileUploader > label {
    color: #e0e0e0 !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stFileUploader [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.2) !important;
    color: #fff !important;
}
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] p {
    color: #ccc !important;
}
/* Uploaded file name visibility */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    color: #fff !important;
    background: rgba(255,255,255,0.08) !important;
    border-radius: 6px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
    color: #fff !important;
    font-weight: 600 !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] {
    background:rgba(255,255,255,0.04); border-radius:8px 8px 0 0;
    color:#aaa; font-size:.87rem; padding:.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#6c63ff22,#4ecdc422) !important;
    color:#fff !important; border-bottom:2px solid #6c63ff;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# 3. Constants & colour helpers
# ===========================================================================

FORMAT_LABELS: Dict[str, str] = {
    "spotify": "Media / Entertainment A  (Timestamp transcript)",
    "netflix": "Media / Entertainment B  (Bracket [HH:MM:SS])",
    "humana":  "Healthcare A  (Call transcript [MM:SS])",
    "ppt":     "Healthcare B  (Chat / SMS)",
    "lyft":    "Transportation  (Customer verbatim)",
    "hilton":  "Travel  (Guest feedback)",
    "auto":    "🔍 Auto-Detect",
}

SENTIMENT_COLORS = {"positive": "#2ecc71", "neutral": "#4682b4", "negative": "#e74c3c"}
PHASE_ICONS      = {"start": "🚀", "middle": "🔄", "end": "🏁"}
CHART_THEME: Dict[str, Any] = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ccc", family="Inter, sans-serif"),
    margin=dict(l=30, r=20, t=45, b=30),
)


def _score_color(score: float) -> str:
    if score >= 0.1:  return "#2ecc71"
    if score <= -0.1: return "#e74c3c"
    return "#f39c12"

def _badge_html(label: str) -> str:
    if label == "positive": return '<span class="badge-csat">✅ Positive</span>'
    if label == "negative": return '<span class="badge-dsat">⛔ Negative</span>'
    return '<span class="badge-neutral">➖ Neutral</span>'

def _score_bar_html(score: float) -> str:
    pct = int((score + 1) / 2 * 100); color = _score_color(score)
    return (f'<div class="score-bar-wrap"><div class="score-bar-track">'
            f'<div class="score-bar-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span style="color:{color};font-size:.8rem;font-weight:600">{score:+.3f}</span></div>')

def _fmt_pct(v: float) -> str: return f"{v:.1%}"


# ===========================================================================
# 4. Pure-numpy math helpers  (replaces numba — no JIT compile delay)
# ===========================================================================

def _fast_rolling_mean_3(arr: np.ndarray) -> np.ndarray:
    """Causal 3-point rolling mean using pure numpy — no compilation needed."""
    result = np.empty(len(arr), dtype=np.float64)
    result[0] = arr[0]
    if len(arr) > 1:
        result[1] = (arr[0] + arr[1]) / 2.0
    for i in range(2, len(arr)):
        result[i] = (arr[i-2] + arr[i-1] + arr[i]) / 3.0
    return result


def _fast_sentiment_change(arr: np.ndarray) -> np.ndarray:
    """First-difference — index 0 is always 0."""
    result    = np.empty(len(arr), dtype=np.float64)
    result[0] = 0.0
    result[1:] = arr[1:] - arr[:-1]
    return result


# ===========================================================================
# 5. ConversationProcessor
# ===========================================================================

class ConversationProcessor:
    """Domain-agnostic parser for six transcript / feedback formats."""

    _PRIORITY_COLS: List[str] = [
        "Comments","comments","COMMENTS",
        "Conversation","conversation","CONVERSATION",
        "Additional Feedback","additional feedback","Additional_Feedback",
        "verbatim","Verbatim","VERBATIM",
        "transcripts","transcript","Transcripts","Transcript",
        "messages","message","Message Text (Translate/Original)",
        "feedback","Feedback","comment","Comment","text","chat",
    ]

    def __init__(self, dataset_type: str = "auto") -> None:
        self.dataset_type = dataset_type.lower()
        self._pat_timestamp = re.compile(
            r"^\|?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})"
            r"\s+(Consumer|Customer|Agent|Advisor|Support):\s*(.*)$", re.IGNORECASE)
        self._pat_bracket = re.compile(
            r"^\[(\d{1,2}:\d{2}:\d{2})\s+(AGENT|CUSTOMER|CONSUMER|ADVISOR|SUPPORT)\]:\s*(.*)$", re.IGNORECASE)
        self._pat_humana = re.compile(
            r"\[(\d{1,3}:\d{2})\]\s+([^:]+?):\s*([^\[]+?)(?=\[|$)", re.IGNORECASE | re.DOTALL)
        self._pat_ppt_html = re.compile(
            r"<b>(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*</b>([^<]+?)(?:<br\s*/?>|$)", re.IGNORECASE | re.DOTALL)
        self._pat_ppt_sms = re.compile(
            r"(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*(.+?)(?=\d{2}:\d{2}:\d{2}\s+|$)", re.DOTALL)

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self._find_col(df)
        if not col:
            raise ValueError("Could not find a transcript/feedback column. "
                             "Expected: Comments, Conversation, Transcripts, verbatim, etc.")
        if self.dataset_type == "auto":
            sample = str(df[col].dropna().iloc[0]) if len(df) > 0 else ""
            self.dataset_type = self._detect(sample, col)

        rows: List[Dict] = []
        for idx, row in df.iterrows():
            text = str(row[col])
            if not text or text == "nan" or len(text) < 5: continue
            rows.extend(self._dispatch(text, int(idx)))

        if not rows and self.dataset_type != "auto":
            sample   = str(df[col].dropna().iloc[0]) if len(df) > 0 else ""
            detected = self._detect(sample, col)
            if detected != self.dataset_type:
                self.dataset_type = detected
                for idx, row in df.iterrows():
                    text = str(row[col])
                    if not text or text == "nan" or len(text) < 5: continue
                    rows.extend(self._dispatch(text, int(idx)))

        if not rows:
            raise ValueError("No turns could be parsed. Verify the file format matches the selected domain.")
        out = pd.DataFrame(rows)
        out["turn_id"]         = range(1, len(out) + 1)
        out["cleaned_message"] = out["message"].str.lower().str.strip()
        return out

    @property
    def detected_format(self) -> str:
        return FORMAT_LABELS.get(self.dataset_type, self.dataset_type.upper())

    def _detect(self, sample: str, col: str) -> str:
        if self._pat_bracket.search(sample):   return "netflix"
        if self._pat_timestamp.search(sample): return "spotify"
        if self._pat_ppt_html.search(sample):  return "ppt"
        if self._pat_humana.search(sample):    return "humana"
        if self._pat_ppt_sms.search(sample):   return "ppt"
        cl = col.lower()
        return "hilton" if ("additional" in cl or "hilton" in cl) else "lyft"

    def _dispatch(self, text: str, idx: int) -> List[Dict]:
        if self.dataset_type == "netflix":         return self._parse_netflix(text, idx)
        if self.dataset_type == "humana":          return self._parse_humana(text, idx)
        if self.dataset_type == "ppt":             return self._parse_ppt(text, idx)
        if self.dataset_type in ("lyft","hilton"): return self._parse_feedback(text, idx)
        return self._parse_spotify(text, idx)

    def _find_col(self, df: pd.DataFrame) -> Optional[str]:
        for name in self._PRIORITY_COLS:
            if name in df.columns: return name
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                s = str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else ""
                if (self._pat_bracket.search(s) or self._pat_timestamp.search(s)
                        or self._pat_humana.search(s) or self._pat_ppt_html.search(s)
                        or self._pat_ppt_sms.search(s)):
                    return col
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                if df[col].dropna().astype(str).str.len().mean() > 20: return col
        return None

    def _norm(self, spk: str) -> str:
        s = spk.upper().strip()
        if s in {"AGENT","ADVISOR","SUPPORT","REP","REPRESENTATIVE","SPECIALIST"}: return "AGENT"
        if s in {"CUSTOMER","CONSUMER","CLIENT","USER","MEMBER","PATIENT","CALLER"}: return "CUSTOMER"
        return s

    def _row(self, idx: int, seq: int, ts: Optional[str], spk: str, msg: str) -> Dict:
        return {"conversation_id": f"CONV_{idx+1:04d}", "turn_sequence": seq,
                "timestamp": ts, "speaker": spk, "message": msg}

    def _parse_netflix(self, text: str, idx: int) -> List[Dict]:
        lines = text.split("\n"); turns: List[Dict] = []
        tn = 1; cs = ct = None; cm: List[str] = []
        def flush():
            nonlocal tn
            if cs:
                msg = " ".join(cm).strip()
                if msg: turns.append(self._row(idx, tn, ct, self._norm(cs), msg)); tn += 1
        for line in lines:
            ls = line.strip(); m = self._pat_bracket.match(ls)
            if m:
                flush(); ct, cs, cm = m.group(1), m.group(2), []
                r = m.group(3).strip()
                if r: cm.append(r)
            elif cs and ls: cm.append(ls)
        flush(); return turns

    def _parse_spotify(self, text: str, idx: int) -> List[Dict]:
        lines = text.split("\n"); turns: List[Dict] = []
        tn = 1; cs = ct = None; cm: List[str] = []
        def flush():
            nonlocal tn
            if cs:
                msg = " ".join(cm).strip()
                if msg: turns.append(self._row(idx, tn, ct, self._norm(cs), msg)); tn += 1
        for line in lines:
            ls = line.strip(); m = self._pat_timestamp.match(ls)
            if m:
                flush(); ct, cs = m.group(1), m.group(2)
                r = m.group(3).strip(); cm = [r] if r else []
            elif cs and ls: cm.append(ls)
        flush(); return turns

    def _parse_humana(self, text: str, idx: int) -> List[Dict]:
        matches = self._pat_humana.findall(text); turns: List[Dict] = []; tn = 1
        for ts, spk, msg in matches:
            sl = spk.strip().lower()
            if sl in {"system","automated","ivr","automated system"}: continue
            m = msg.strip()
            if not m or len(m) < 3: continue
            if any(k in sl for k in ["member","customer","patient","caller"]):             ns = "CUSTOMER"
            elif any(k in sl for k in ["agent","representative","rep","advisor","specialist"]): ns = "AGENT"
            else: ns = spk.strip().upper()
            turns.append(self._row(idx, tn, ts, ns, m)); tn += 1
        return turns

    def _parse_ppt(self, text: str, idx: int) -> List[Dict]:
        html_m = self._pat_ppt_html.findall(text)
        if html_m: return self._ppt_turns(html_m, idx, is_sms=False)
        sms_m  = self._pat_ppt_sms.findall(text)
        if sms_m:  return self._ppt_turns(sms_m,  idx, is_sms=True)
        return []

    def _ppt_turns(self, matches: list, idx: int, is_sms: bool) -> List[Dict]:
        spk_msgs: Dict[str, List] = {}; ordered: List[str] = []
        for ts, spk, msg in matches:
            sl = spk.strip().lower()
            if sl == "system": continue
            if is_sms:
                msg = re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.]+Z\w*$","",msg)
                msg = re.sub(r"Looks up Phone Number.*?digits-\d+","",msg)
                msg = re.sub(r"Looks up SSN number.*?digits-\d+","",msg)
                msg = re.sub(r"Phone Numbers rule for Chat|SSN rule for Chat","",msg)
            m = msg.strip()
            if not m: continue
            if sl not in spk_msgs: spk_msgs[sl] = []; ordered.append(sl)
            spk_msgs[sl].append((ts, m))
        if not spk_msgs: return []
        roles: Dict[str, str] = {}
        if is_sms:
            for s in ordered: roles[s] = "CUSTOMER" if re.match(r"^\d+$",s) else "AGENT"
        else:
            cnts = {s: len(msgs) for s,msgs in spk_msgs.items()}
            if len(cnts) == 1: cust = list(cnts.keys())[0]
            else:
                srt = sorted(cnts.items(), key=lambda x: x[1])
                f = ordered[0] if ordered else None
                cust = f if f and cnts.get(f,999) <= srt[0][1] else srt[0][0]
            for s in ordered: roles[s] = "CUSTOMER" if s == cust else "AGENT"
        all_m = [(ts,s,m) for s in ordered for ts,m in spk_msgs[s]]
        all_m.sort(key=lambda x: x[0])
        return [self._row(idx,i,ts,roles.get(s,"CUSTOMER"),m) for i,(ts,s,m) in enumerate(all_m,1)]

    def _parse_feedback(self, text: str, idx: int) -> List[Dict]:
        m = text.strip()
        return [self._row(idx,1,None,"CUSTOMER",m)] if m else []


# ===========================================================================
# 6. SentimentEngine
# ===========================================================================

class SentimentEngine:
    """VADER scorer with adaptive per-dataset thresholds."""

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()
        self.thresholds = {"positive":0.05,"negative":-0.05,"neutral_range":0.10}

    def calibrate(self, df: pd.DataFrame) -> Dict[str, float]:
        sample = df.sample(n=min(1_000,len(df)), random_state=42)
        scores = [self._vader.polarity_scores(str(m))["compound"]
                  for m in sample["cleaned_message"].fillna("") if len(str(m))>5]
        if not scores: return self.thresholds
        arr = np.array(scores)
        pos = max(float(np.percentile(arr,70)),0.10)
        neg = min(float(np.percentile(arr,30)),-0.10)
        self.thresholds = {"positive":pos,"negative":neg,"neutral_range":pos-neg}
        return self.thresholds

    def score(self, df: pd.DataFrame, chunk_size: int = 500) -> pd.DataFrame:
        out = df.copy()
        for col in ["compound","positive","negative","neutral","sentiment_confidence"]: out[col] = 0.0
        out["sentiment_label"] = "neutral"
        for i in range(0, len(out), chunk_size):
            chunk = out.iloc[i:i+chunk_size]
            for idx in chunk.index:
                msg = out.at[idx,"cleaned_message"]
                if pd.isna(msg) or len(str(msg))<5: continue
                sc = self._vader.polarity_scores(str(msg))
                out.at[idx,"compound"] = sc["compound"]; out.at[idx,"positive"] = sc["pos"]
                out.at[idx,"negative"] = sc["neg"];      out.at[idx,"neutral"]  = sc["neu"]
                c = sc["compound"]
                if   c >= self.thresholds["positive"]: lbl="positive"; conf=min(c/self.thresholds["positive"],1.0)
                elif c <= self.thresholds["negative"]: lbl="negative"; conf=min(abs(c)/abs(self.thresholds["negative"]),1.0)
                else:                                  lbl="neutral";  conf=max(0.0,1.0-abs(c)/(self.thresholds["neutral_range"]/2))
                out.at[idx,"sentiment_label"] = lbl; out.at[idx,"sentiment_confidence"] = conf
            if i % (chunk_size*10) == 0: gc.collect()
        return out


# ===========================================================================
# 7. AnalyticsEngine
# ===========================================================================

class AnalyticsEngine:
    """Computes turn-level metrics and aggregated business insights."""

    def compute_turn_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.sort_values(["conversation_id","turn_sequence"]).reset_index(drop=True)
        changes: List[float] = []; momentum: List[float] = []
        for _, grp in d.groupby("conversation_id"):
            s = grp["compound"].values.astype(np.float64)
            ch = _fast_sentiment_change(s); mo = _fast_rolling_mean_3(ch)
            changes.extend(ch.tolist()); momentum.extend(mo.tolist())
        d["sentiment_change"] = changes; d["sentiment_momentum"] = momentum
        mt = d.groupby("conversation_id")["turn_sequence"].transform("max")
        d["turn_position"]          = d["turn_sequence"] / mt
        d["is_conversation_start"]  = d["turn_sequence"] <= 3
        d["is_conversation_end"]    = d["turn_sequence"] > (mt-3)
        d["is_conversation_middle"] = ~d["is_conversation_start"] & ~d["is_conversation_end"]
        d["phase"] = "middle"
        d.loc[d["is_conversation_start"],"phase"] = "start"
        d.loc[d["is_conversation_end"],  "phase"] = "end"
        d["is_csat"] = d["compound"] >= 0; d["is_dsat"] = d["compound"] < 0
        prev = d.groupby("conversation_id")["speaker"].shift(1)
        d["prev_speaker"]      = prev; d["speaker_changed"] = d["speaker"] != prev
        d["consecutive_turns"] = (d.groupby(["conversation_id",(d["speaker"]!=prev).cumsum()]).cumcount()+1)
        d["potential_escalation"] = ((d["sentiment_change"]<-0.3)&(d["speaker"]=="CUSTOMER")&(d["turn_sequence"]>2))
        d["potential_resolution"] = ((d["sentiment_change"]>0.2) &(d["speaker"]=="CUSTOMER")&(d["is_conversation_end"]))
        return d

    def compute_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        ins: Dict[str, Any] = {}
        ins["total_conversations"]        = int(df["conversation_id"].nunique())
        ins["total_turns"]                = int(len(df))
        ins["avg_turns_per_conversation"] = float(df.groupby("conversation_id").size().mean())
        ins["overall_sentiment"]          = {"average":float(df["compound"].mean()),
                                             "median": float(df["compound"].median()),
                                             "std":    float(df["compound"].std())}
        cu = df[df["speaker"]=="CUSTOMER"]; ag = df[df["speaker"]=="AGENT"]

        def _safe(series, default=0.0):
            v = float(series) if hasattr(series,"__float__") else default
            return default if np.isnan(v) else v

        ins["customer_satisfaction"] = {
            "average_sentiment":     _safe(cu["compound"].mean())                       if not cu.empty else 0.0,
            "positive_interactions": _safe((cu["sentiment_label"]=="positive").mean())  if not cu.empty else 0.0,
            "escalation_rate":       _safe(cu["potential_escalation"].mean())            if not cu.empty else 0.0,
            "resolution_rate":       _safe(cu["potential_resolution"].mean())            if not cu.empty else 0.0,
        }
        ins["agent_performance"] = {
            "average_sentiment":      _safe(ag["compound"].mean())        if not ag.empty else 0.0,
            "response_effectiveness": _safe(ag["sentiment_change"].mean()) if not ag.empty else 0.0,
            "consistency_score":      _safe(1.0-ag["compound"].std())     if not ag.empty else 0.0,
        }
        st_  = df[df["is_conversation_start"]]; mid_ = df[df["is_conversation_middle"]]; en_ = df[df["is_conversation_end"]]
        ins["conversation_patterns"] = {
            "avg_sentiment_start":   _safe(st_["compound"].mean())  if not st_.empty  else 0.0,
            "avg_sentiment_middle":  _safe(mid_["compound"].mean()) if not mid_.empty else 0.0,
            "avg_sentiment_end":     _safe(en_["compound"].mean())  if not en_.empty  else 0.0,
            "sentiment_improvement": (_safe(en_["compound"].mean()-st_["compound"].mean())
                                      if not st_.empty and not en_.empty else 0.0),
        }
        cust = cu if not cu.empty else df

        def _phase_stats(phase_df: pd.DataFrame) -> Dict:
            if phase_df.empty: return {"csat_pct":0.0,"dsat_pct":0.0,"avg_sentiment":0.0,"count":0}
            t = len(phase_df)
            return {"csat_pct":int((phase_df["compound"]>=0).sum())/t,
                    "dsat_pct":int((phase_df["compound"]< 0).sum())/t,
                    "avg_sentiment":float(phase_df["compound"].mean()),"count":t}

        ins["phase_csat_dsat"] = {
            "start":  _phase_stats(cust[cust["phase"]=="start"]),
            "middle": _phase_stats(cust[cust["phase"]=="middle"]),
            "end":    _phase_stats(cust[cust["phase"]=="end"]),
        }
        ins["recommendations"] = self._recommendations(ins)
        return ins

    def _recommendations(self, ins: Dict) -> List[str]:
        r: List[str] = []
        cs=ins["customer_satisfaction"]; ap=ins["agent_performance"]
        cp=ins["conversation_patterns"]; pcd=ins.get("phase_csat_dsat",{})
        if cs["average_sentiment"]      < 0:    r.append("🔴 Customer sentiment is below neutral — review agent training and script quality.")
        if cs["escalation_rate"]        > 0.15: r.append(f"⚠️ High escalation rate ({cs['escalation_rate']:.1%}) — analyse triggers and train de-escalation.")
        elif cs["escalation_rate"]      > 0.10: r.append(f"⚠️ Moderate escalation rate ({cs['escalation_rate']:.1%}) — monitor closely.")
        if cs["resolution_rate"]        < 0.5:  r.append(f"🔴 Low resolution rate ({cs['resolution_rate']:.1%}) — strengthen closing techniques.")
        if ap["average_sentiment"]      < 0.1:  r.append("📚 Agent sentiment is low — consider tone coaching and positive-language training.")
        if cp["sentiment_improvement"]  < 0:    r.append("📉 Conversations end worse than they start — review resolution processes.")
        elif cp["sentiment_improvement"] > 0.2: r.append("📈 Strong positive sentiment improvement — document and replicate best-practice behaviours.")
        if pcd:
            mid=pcd.get("middle",{}); end=pcd.get("end",{}); start=pcd.get("start",{})
            if mid.get("dsat_pct",0)>0.5:  r.append(f"⚠️ Mid-conversation DSAT at {mid['dsat_pct']:.1%} — reduce handle time.")
            if end.get("dsat_pct",0)>0.4:  r.append(f"🔴 End-conversation DSAT at {end['dsat_pct']:.1%} — improve wrap-up and follow-through.")
            if start.get("csat_pct",0)>0.7 and end.get("dsat_pct",0)>0.3:
                r.append("📉 CRITICAL: Customers start satisfied but end dissatisfied — review resolution process.")
        if not r: r.append("✅ All key metrics are within healthy ranges — maintain current practices.")
        return r


# ===========================================================================
# 8. Pipeline entry point
# ===========================================================================

def run_pipeline(df: pd.DataFrame, dataset_type: str = "auto") -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Parse → score → metrics → insights. Returns (df_results, insights, detected_label)."""
    processor  = ConversationProcessor(dataset_type=dataset_type)
    df_parsed  = processor.parse(df)
    detected   = processor.detected_format
    sentiment  = SentimentEngine(); sentiment.calibrate(df_parsed); df_scored = sentiment.score(df_parsed)
    analytics  = AnalyticsEngine(); df_results = analytics.compute_turn_metrics(df_scored)
    insights   = analytics.compute_insights(df_results)
    return df_results, insights, detected


# ===========================================================================
# 9. Export helpers
# ===========================================================================

def _to_excel(df: pd.DataFrame, insights: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Turns", index=False)
        cu = df[df["speaker"]=="CUSTOMER"]
        if not cu.empty: cu.to_excel(writer, sheet_name="Customer Turns", index=False)
        ag = df[df["speaker"]=="AGENT"]
        if not ag.empty: ag.to_excel(writer, sheet_name="Agent Turns", index=False)
        pcd = insights.get("phase_csat_dsat",{})
        rows = [
            {"Metric":"Total Conversations",      "Value":insights["total_conversations"]},
            {"Metric":"Total Turns",              "Value":insights["total_turns"]},
            {"Metric":"Avg Turns / Conversation", "Value":f"{insights['avg_turns_per_conversation']:.1f}"},
            {"Metric":"Overall Avg Sentiment",    "Value":f"{insights['overall_sentiment']['average']:.3f}"},
            {"Metric":"Customer Avg Sentiment",   "Value":f"{insights['customer_satisfaction']['average_sentiment']:.3f}"},
            {"Metric":"Agent Avg Sentiment",      "Value":f"{insights['agent_performance']['average_sentiment']:.3f}"},
            {"Metric":"Escalation Rate",          "Value":_fmt_pct(insights["customer_satisfaction"]["escalation_rate"])},
            {"Metric":"Resolution Rate",          "Value":_fmt_pct(insights["customer_satisfaction"]["resolution_rate"])},
            {"Metric":"Sentiment Improvement",    "Value":f"{insights['conversation_patterns']['sentiment_improvement']:.3f}"},
        ]
        for pn in ["start","middle","end"]:
            p = pcd.get(pn,{})
            rows += [{"Metric":f"{pn.capitalize()} CSAT %",    "Value":_fmt_pct(p.get("csat_pct",0))},
                     {"Metric":f"{pn.capitalize()} DSAT %",    "Value":_fmt_pct(p.get("dsat_pct",0))},
                     {"Metric":f"{pn.capitalize()} Avg Score", "Value":f"{p.get('avg_sentiment',0):.3f}"}]
        pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)
    return buf.getvalue()


def _to_csv(df: pd.DataFrame) -> str:
    buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue()


def _to_zip(df: pd.DataFrame, insights: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("tbt_results.csv",     _to_csv(df))
        zf.writestr("tbt_insights.json",   json.dumps(insights, indent=2, default=str))
        zf.writestr("tbt_results.xlsx",    _to_excel(df, insights))
    return buf.getvalue()


# ===========================================================================
# 10. Chart factories
# ===========================================================================

def _chart_sentiment_dist(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment_label"].value_counts().reset_index(); counts.columns = ["sentiment","count"]
    fig = px.bar(counts, x="sentiment", y="count", color="sentiment",
                 color_discrete_map=SENTIMENT_COLORS, title="Overall Sentiment Distribution",
                 labels={"sentiment":"Sentiment","count":"Turns"})
    fig.update_layout(**CHART_THEME, title_font_size=14, showlegend=False)
    fig.update_traces(marker_line_width=0); return fig

def _chart_speaker_box(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for role, color in [("CUSTOMER","#ff6b6b"),("AGENT","#4ecdc4")]:
        sub = df[df["speaker"]==role]["compound"]
        if not sub.empty:
            fig.add_trace(go.Box(y=sub, name=role.capitalize(), marker_color=color, boxpoints="outliers", line_color=color))
    fig.update_layout(**CHART_THEME, title="Customer vs Agent Sentiment", title_font_size=14); return fig

def _chart_phase_comparison(insights: Dict) -> go.Figure:
    pcd=insights.get("phase_csat_dsat",{}); phases=["Start","Middle","End"]
    fig = go.Figure(data=[
        go.Bar(name="CSAT %", x=phases, y=[pcd.get(p.lower(),{}).get("csat_pct",0)*100 for p in phases], marker_color="#2ecc71"),
        go.Bar(name="DSAT %", x=phases, y=[pcd.get(p.lower(),{}).get("dsat_pct",0)*100 for p in phases], marker_color="#e74c3c"),
    ])
    fig.update_layout(**CHART_THEME, barmode="group", title="CSAT vs DSAT by Phase", title_font_size=14,
                      yaxis=dict(title="% Customer Turns",gridcolor="rgba(255,255,255,0.05)"),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02)); return fig

def _chart_sentiment_progression(df: pd.DataFrame) -> go.Figure:
    tp = df.groupby("turn_sequence")["compound"].mean().reset_index(); tp=tp[tp["turn_sequence"]<=30]
    fig = go.Figure()
    fig.add_hline(y=0,line_dash="dash",line_color="rgba(255,255,255,0.15)")
    fig.add_trace(go.Scatter(x=tp["turn_sequence"],y=tp["compound"],mode="lines+markers",
                             line=dict(color="#4ecdc4",width=2.5),marker=dict(size=6,color="#4ecdc4"),
                             fill="tozeroy",fillcolor="rgba(78,205,196,0.08)"))
    fig.update_layout(**CHART_THEME,title="Avg Sentiment by Turn Position (first 30 turns)",title_font_size=14,
                      xaxis=dict(title="Turn Number",       gridcolor="rgba(255,255,255,0.05)"),
                      yaxis=dict(title="Avg Compound Score",gridcolor="rgba(255,255,255,0.05)")); return fig

def _chart_escalation_resolution(df: pd.DataFrame) -> go.Figure:
    esc=int(df["potential_escalation"].sum()); res=int(df["potential_resolution"].sum())
    total=max(df["conversation_id"].nunique(),1)
    fig=go.Figure(go.Bar(x=["Escalations","Resolutions"],y=[esc,res],marker_color=["#e74c3c","#2ecc71"],
                         text=[f"{esc} ({esc/total:.0%})",f"{res} ({res/total:.0%})"],textposition="auto"))
    fig.update_layout(**CHART_THEME,title="Escalation & Resolution Events",title_font_size=14,showlegend=False,
                      yaxis=dict(gridcolor="rgba(255,255,255,0.05)")); return fig

def _chart_conversation_heatmap(df: pd.DataFrame) -> go.Figure:
    cm=df.groupby("conversation_id").agg(avg_sentiment=("compound","mean"),turns=("turn_sequence","max")).reset_index()
    fig=px.scatter(cm,x="turns",y="avg_sentiment",color="avg_sentiment",color_continuous_scale="RdYlGn",
                   range_color=[-1,1],hover_name="conversation_id",title="Conversation Performance Map",
                   labels={"turns":"Conversation Length (turns)","avg_sentiment":"Avg Sentiment"})
    fig.update_layout(**CHART_THEME,title_font_size=14); fig.update_coloraxes(colorbar=dict(thickness=10)); return fig

def _chart_tbt_flow(df: pd.DataFrame, conv_id: str) -> go.Figure:
    sub=df[df["conversation_id"]==conv_id].sort_values("turn_sequence")
    fig=go.Figure()
    fig.add_hline(y=0,line_dash="dash",line_color="rgba(255,255,255,0.15)")
    fig.add_trace(go.Scatter(x=sub["turn_sequence"],y=sub["compound"],mode="lines+markers",
                             line=dict(color="#6c63ff",width=2.5),
                             marker=dict(size=9,color=sub["compound"],colorscale="RdYlGn",cmin=-1,cmax=1,
                                         showscale=True,colorbar=dict(thickness=10,title="Score")),
                             text=[f"Turn {r.turn_sequence}<br>{r.speaker}<br>{r.message[:60]}..."
                                   if len(r.message)>60 else f"Turn {r.turn_sequence}<br>{r.speaker}<br>{r.message}"
                                   for _,r in sub.iterrows()],
                             hovertemplate="%{text}<br>Score: %{y:.3f}<extra></extra>"))
    mt=int(sub["turn_sequence"].max()) if not sub.empty else 1
    for pn,(s,e,color) in {"start":(1,3,"rgba(108,99,255,0.08)"),"middle":(4,max(4,mt-3),"rgba(78,205,196,0.06)"),
                            "end":(max(4,mt-2),mt,"rgba(255,107,107,0.08)")}.items():
        if s<=e: fig.add_vrect(x0=s-0.5,x1=e+0.5,fillcolor=color,line_width=0,
                               annotation_text=PHASE_ICONS[pn],annotation_position="top left")
    fig.update_layout(**CHART_THEME,title=f"Turn-by-Turn Flow — {conv_id}",title_font_size=14,
                      xaxis=dict(title="Turn Sequence",  gridcolor="rgba(255,255,255,0.05)"),
                      yaxis=dict(title="Sentiment Score",range=[-1.1,1.1],gridcolor="rgba(255,255,255,0.05)")); return fig

def _chart_sentiment_momentum(df: pd.DataFrame, conv_id: str) -> go.Figure:
    sub=df[df["conversation_id"]==conv_id].sort_values("turn_sequence")
    colors=["#2ecc71" if v>=0 else "#e74c3c" for v in sub["sentiment_momentum"]]
    fig=go.Figure(go.Bar(x=sub["turn_sequence"],y=sub["sentiment_momentum"],marker_color=colors,
                         hovertemplate="Turn %{x}<br>Momentum: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0,line_dash="dash",line_color="rgba(255,255,255,0.2)")
    fig.update_layout(**CHART_THEME,title=f"Sentiment Momentum — {conv_id}",title_font_size=14,showlegend=False,
                      xaxis=dict(title="Turn Sequence",  gridcolor="rgba(255,255,255,0.05)"),
                      yaxis=dict(title="3-Turn Momentum",gridcolor="rgba(255,255,255,0.05)")); return fig

def _chart_speaker_phase_heatmap(df: pd.DataFrame, conv_id: str) -> go.Figure:
    sub=df[df["conversation_id"]==conv_id]
    pivot=(sub.groupby(["speaker","phase"])["compound"].mean().unstack(fill_value=0)
              .reindex(columns=["start","middle","end"],fill_value=0))
    fig=go.Figure(go.Heatmap(z=pivot.values,x=["Start","Middle","End"],y=pivot.index.tolist(),
                             colorscale="RdYlGn",zmin=-1,zmax=1,
                             text=[[f"{v:+.2f}" for v in row] for row in pivot.values],
                             texttemplate="%{text}",showscale=True,colorbar=dict(thickness=10)))
    fig.update_layout(**CHART_THEME,title=f"Sentiment by Speaker × Phase — {conv_id}",title_font_size=14,
                      xaxis=dict(title="Phase"),yaxis=dict(title="Speaker")); return fig


# ===========================================================================
# 11. UI component renderers
# ===========================================================================

def _render_header() -> None:
    st.markdown("""
    <div class="app-header">
        <h1>🎭 Domain Agnostic — TbT Sentiment Analytics</h1>
        <p>Granular Turn-by-Turn Analysis &nbsp;·&nbsp; Start → Middle → End &nbsp;·&nbsp; CSAT / DSAT per Phase</p>
    </div>""", unsafe_allow_html=True)

def _render_kpi_row(insights: Dict[str, Any]) -> None:
    cs=insights["customer_satisfaction"]; ap=insights["agent_performance"]; cp=insights["conversation_patterns"]
    def card(label,value,sub=""):
        return (f'<div class="metric-card"><div class="m-label">{label}</div>'
                f'<div class="m-value">{value}</div><div class="m-sub">{sub}</div></div>')
    overall=insights["overall_sentiment"]["average"]
    esc_c="#e74c3c" if cs["escalation_rate"]>0.15 else "#f39c12" if cs["escalation_rate"]>0.10 else "#2ecc71"
    res_c="#2ecc71" if cs["resolution_rate"]>0.6  else "#f39c12" if cs["resolution_rate"]>0.4  else "#e74c3c"
    html='<div class="metric-grid">'
    html+=card("Conversations",    f"{insights['total_conversations']:,}")
    html+=card("Total Turns",      f"{insights['total_turns']:,}",f"avg {insights['avg_turns_per_conversation']:.1f}/conv")
    html+=card("Overall Sentiment",f'<span style="color:{_score_color(overall)}">{overall:+.3f}</span>')
    html+=card("Customer Avg",     f'<span style="color:{_score_color(cs["average_sentiment"])}">{cs["average_sentiment"]:+.3f}</span>')
    html+=card("Agent Avg",        f'<span style="color:{_score_color(ap["average_sentiment"])}">{ap["average_sentiment"]:+.3f}</span>')
    html+=card("Escalation Rate",  f'<span style="color:{esc_c}">{_fmt_pct(cs["escalation_rate"])}</span>')
    html+=card("Resolution Rate",  f'<span style="color:{res_c}">{_fmt_pct(cs["resolution_rate"])}</span>')
    html+=card("Sentiment Trend",  f'<span style="color:{_score_color(cp["sentiment_improvement"])}">{cp["sentiment_improvement"]:+.3f}</span>',"end − start")
    html+="</div>"
    st.markdown(html, unsafe_allow_html=True)

def _render_phase_table(insights: Dict[str, Any]) -> None:
    pcd=insights.get("phase_csat_dsat",{}); cp=insights.get("conversation_patterns",{})
    rows_html=""
    for pn in ["start","middle","end"]:
        p=pcd.get(pn,{}); csat=p.get("csat_pct",0); dsat=p.get("dsat_pct",0)
        cnt=p.get("count",0); avg=cp.get(f"avg_sentiment_{pn}",0)
        ind="✅" if csat>=0.6 else "⚠️" if csat>=0.4 else "🔴"
        rows_html+=(f"<tr><td>{PHASE_ICONS[pn]} <strong>{pn.capitalize()}</strong></td>"
                    f"<td>{ind}</td><td>{cnt:,}</td>"
                    f"<td><span class='badge-csat'>{_fmt_pct(csat)} CSAT</span></td>"
                    f"<td><span class='badge-dsat'>{_fmt_pct(dsat)} DSAT</span></td>"
                    f"<td>{_score_bar_html(avg)}</td></tr>")
    st.markdown(
        "<table class='phase-table'><thead><tr>"
        "<th>Phase</th><th>Status</th><th>Customer Turns</th>"
        f"<th>CSAT</th><th>DSAT</th><th>Avg Score</th></tr></thead><tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True)

def _render_turn_viewer(df: pd.DataFrame, conv_id: str) -> None:
    sub=df[df["conversation_id"]==conv_id].sort_values("turn_sequence")
    if sub.empty: st.info("No turns found for this conversation."); return
    for _,r in sub.iterrows():
        spk=str(r["speaker"]).upper(); css="turn-customer" if spk=="CUSTOMER" else "turn-agent"
        icon="👤" if spk=="CUSTOMER" else "🎧"
        ts=(f" · {r['timestamp']}" if r.get("timestamp") and str(r["timestamp"]) not in ("nan","None","") else "")
        phase_icon=PHASE_ICONS.get(str(r.get("phase","middle")),"🔄")
        score=float(r["compound"]); lbl=str(r.get("sentiment_label","neutral"))
        st.markdown(
            f'<div class="turn-card {css}">'
            f'<div class="turn-header">{icon} {spk}{ts} &nbsp; {phase_icon} {str(r.get("phase","")).capitalize()} &nbsp; Turn #{int(r["turn_sequence"])}</div>'
            f'<div class="turn-text">{r["message"]}</div>'
            f'<div class="turn-meta">{_badge_html(lbl)} &nbsp; {_score_bar_html(score)} &nbsp; Confidence: {float(r.get("sentiment_confidence",0)):.0%}</div>'
            f'</div>', unsafe_allow_html=True)

def _render_recommendations(insights: Dict[str, Any]) -> None:
    for rec in insights.get("recommendations",[]):
        st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)

def _render_export_section(df_results: pd.DataFrame, insights: dict) -> None:
    """Prominent export section with three clearly labelled download buttons."""
    st.markdown("### ⬇️ Download Results")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="export-card">
            <div class="ex-title">📊 Excel Workbook</div>
            <div class="ex-desc">All Turns · Customer · Agent · Summary (4 sheets)</div>
        </div>""", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Excel (.xlsx)",
            data=_to_excel(df_results, insights),
            file_name=f"tbt_results_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    with col2:
        st.markdown("""
        <div class="export-card">
            <div class="ex-title">📄 CSV File</div>
            <div class="ex-desc">All turns as a flat CSV — ready for Excel / BI tools</div>
        </div>""", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download CSV (.csv)",
            data=_to_csv(df_results),
            file_name=f"tbt_results_{ts}.csv",
            mime="text/csv",
            width="stretch",
        )

    with col3:
        st.markdown("""
        <div class="export-card">
            <div class="ex-title">📦 Full ZIP Bundle</div>
            <div class="ex-desc">CSV + Excel + JSON insights in one archive</div>
        </div>""", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download ZIP (.zip)",
            data=_to_zip(df_results, insights),
            file_name=f"tbt_complete_{ts}.zip",
            mime="application/zip",
            width="stretch",
        )

def _render_landing() -> None:
    col_l, col_r = st.columns([1.2,1])
    with col_l:
        st.markdown("""
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
2. Upload a CSV or Excel file to begin analysis
""")
    with col_r:
        st.markdown("#### 🔍 What you'll see")
        for item in ["📊 KPI dashboard (conversations, turns, sentiment)",
                     "📈 CSAT / DSAT breakdown per phase (Start → Middle → End)",
                     "🔄 Interactive turn-by-turn flow chart + momentum chart",
                     "🗣️ Per-turn detail viewer with sentiment badges",
                     "💡 Automated business recommendations",
                     "⬇️ Download as Excel, CSV, or ZIP"]:
            st.markdown(f"- {item}")


# ===========================================================================
# 12. Sidebar
# ===========================================================================

def _render_sidebar() -> tuple[str, Optional[object]]:
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:.6rem 0 1.2rem">'
            '<span style="font-size:2rem">🎭</span>'
            '<div style="color:#fff;font-weight:700;font-size:1.1rem;margin-top:.3rem">TbT Analytics</div>'
            '<div style="color:#bbb;font-size:.82rem;margin-top:.2rem">Turn-by-Turn Sentiment</div>'
            '</div>',
            unsafe_allow_html=True)

        st.markdown("### ⚙️ Configuration")
        domain_keys   = list(FORMAT_LABELS.keys())
        domain_labels = [FORMAT_LABELS[k] for k in domain_keys]
        sel_idx = st.selectbox(
            "Domain / Format",
            options=range(len(domain_keys)),
            format_func=lambda i: domain_labels[i],
            index=6,
            help="Select the transcript format or leave as Auto-Detect.",
        )
        dataset_type = domain_keys[sel_idx]

        st.markdown("---")
        st.markdown("### 📂 Upload Data")
        uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv","xlsx","xls"],
            help="Upload a file containing conversation transcripts or customer feedback.",
        )

        # Show uploaded file name clearly
        if uploaded is not None:
            st.markdown(
                f'<div style="background:rgba(108,99,255,0.2);border:1px solid #6c63ff;'
                f'border-radius:8px;padding:.5rem .75rem;margin-top:.5rem;">'
                f'<span style="color:#aaa;font-size:.75rem">Loaded:</span><br>'
                f'<span style="color:#fff;font-weight:600;font-size:.88rem">📄 {uploaded.name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<div style="color:#888;font-size:.72rem;text-align:center">'
            'Domain Agnostic · TbT Granular Sentiment v2.2</div>',
            unsafe_allow_html=True,
        )
    return dataset_type, uploaded


# ===========================================================================
# 13. Tab renderers
# ===========================================================================

def _tab_overview(df_results: pd.DataFrame, insights: dict) -> None:
    st.markdown("#### 📊 Phase-Level CSAT / DSAT")
    _render_phase_table(insights)
    st.markdown("#### 📈 Visual Overview")
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(_chart_sentiment_dist(df_results), width='stretch')
    with c2: st.plotly_chart(_chart_speaker_box(df_results),    width='stretch')
    c3,c4=st.columns(2)
    with c3: st.plotly_chart(_chart_phase_comparison(insights),         width='stretch')
    with c4: st.plotly_chart(_chart_escalation_resolution(df_results),  width='stretch')
    st.plotly_chart(_chart_sentiment_progression(df_results), width='stretch')
    st.plotly_chart(_chart_conversation_heatmap(df_results),  width='stretch')

def _tab_tbt_flow(df_results: pd.DataFrame) -> None:
    st.markdown("#### 🔄 Turn-by-Turn Sentiment Flow")
    conv_ids=sorted(df_results["conversation_id"].unique().tolist())
    col1,col2,col3=st.columns([2,1,1])
    with col1: sel_conv    = st.selectbox("Select Conversation",conv_ids,key="flow_conv")
    with col2: speaker_flt = st.selectbox("Speaker",["All","CUSTOMER","AGENT"],key="flow_spk")
    with col3: phase_flt   = st.selectbox("Phase",  ["All","start","middle","end"],key="flow_phase")
    df_view=df_results[df_results["conversation_id"]==sel_conv].copy()
    if speaker_flt!="All": df_view=df_view[df_view["speaker"]==speaker_flt]
    if phase_flt  !="All": df_view=df_view[df_view["phase"]  ==phase_flt]
    st.plotly_chart(_chart_tbt_flow(df_results,sel_conv),             width='stretch')
    st.plotly_chart(_chart_sentiment_momentum(df_results,sel_conv),   width='stretch')
    st.plotly_chart(_chart_speaker_phase_heatmap(df_results,sel_conv),width='stretch')
    cu_sub=df_view[df_view["speaker"]=="CUSTOMER"]; ag_sub=df_view[df_view["speaker"]=="AGENT"]
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Turns (filtered)",len(df_view))
    m2.metric("Customer Avg",f"{cu_sub['compound'].mean():+.3f}" if not cu_sub.empty else "—")
    m3.metric("Agent Avg",   f"{ag_sub['compound'].mean():+.3f}" if not ag_sub.empty else "—")
    m4.metric("Escalation turns",int(df_view["potential_escalation"].sum()) if "potential_escalation" in df_view.columns else 0)

def _tab_conversation_explorer(df_results: pd.DataFrame) -> None:
    st.markdown("#### 🗣️ Conversation Explorer")
    col_l,col_r=st.columns([1.2,1])
    with col_l: sel_conv_exp=st.selectbox("Choose conversation",sorted(df_results["conversation_id"].unique()),key="exp_conv")
    with col_r: phase_exp   =st.selectbox("Filter by phase",["All","start","middle","end"],key="exp_phase")
    df_exp=df_results[df_results["conversation_id"]==sel_conv_exp]
    if phase_exp!="All": df_exp=df_exp[df_exp["phase"]==phase_exp]
    _render_turn_viewer(df_exp,sel_conv_exp)

def _tab_data_table(df_results: pd.DataFrame) -> None:
    st.markdown("#### 📋 Full Results Table")
    col_f1,col_f2,col_f3=st.columns(3)
    with col_f1: f_speaker  =st.selectbox("Speaker",  ["All","CUSTOMER","AGENT"],key="tbl_spk")
    with col_f2: f_sentiment=st.selectbox("Sentiment",["All","positive","neutral","negative"],key="tbl_sent")
    with col_f3: f_phase    =st.selectbox("Phase",    ["All","start","middle","end"],key="tbl_phase")
    df_tbl=df_results.copy()
    if f_speaker  !="All": df_tbl=df_tbl[df_tbl["speaker"]        ==f_speaker]
    if f_sentiment!="All": df_tbl=df_tbl[df_tbl["sentiment_label"]==f_sentiment]
    if f_phase    !="All": df_tbl=df_tbl[df_tbl["phase"]          ==f_phase]
    display_cols=[c for c in ["conversation_id","turn_sequence","phase","speaker","timestamp",
                               "message","sentiment_label","compound","sentiment_confidence",
                               "potential_escalation","potential_resolution"] if c in df_tbl.columns]
    st.markdown(f"**{len(df_tbl):,} rows** after filters")
    st.dataframe(df_tbl[display_cols].reset_index(drop=True), width='stretch', height=450)

def _tab_recommendations_and_export(df_results: pd.DataFrame, insights: dict) -> None:
    st.markdown("#### 💡 Automated Business Recommendations")
    _render_recommendations(insights)
    st.markdown("---")
    _render_export_section(df_results, insights)
    st.markdown("---")
    st.markdown("#### 📊 Raw Insights (JSON)")
    with st.expander("View full insights object"): st.json(insights)


# ===========================================================================
# 14. main()
# ===========================================================================

def main() -> None:
    _render_header()
    dataset_type, uploaded = _render_sidebar()

    # Load data
    df_raw: Optional[pd.DataFrame] = None
    source_label = ""
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            source_label = f"📂 {uploaded.name}"
        except Exception as exc:
            st.error(f"Could not read file: {exc}"); return

    if df_raw is None:
        _render_landing(); return

    # Run pipeline (cached per data + domain)
    cache_key = f"results_{hash(df_raw.values.tobytes())}_{dataset_type}"
    if cache_key not in st.session_state:
        with st.spinner("🔄 Running TbT analysis pipeline…"):
            try:
                st.session_state[cache_key] = run_pipeline(df_raw, dataset_type)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}"); st.exception(exc); return

    df_results, insights, detected = st.session_state[cache_key]

    # Status bar
    col_a, _, col_c = st.columns([3,2,1])
    with col_a:
        st.markdown(
            f'<div style="color:#aaa;font-size:.82rem">{source_label} &nbsp;·&nbsp; '
            f'Format: <span style="color:#6c63ff;font-weight:600">{detected}</span></div>',
            unsafe_allow_html=True)
    with col_c:
        st.download_button(
            "⬇️ Quick Export ZIP",
            data=_to_zip(df_results,insights),
            file_name=f"tbt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            width="stretch",
        )

    st.markdown("---")
    _render_kpi_row(insights)

    tabs = st.tabs(["📊 Overview","🔄 Turn-by-Turn Flow","🗣️ Conversation Explorer",
                    "📋 Data Table","💡 Recommendations & Export"])
    with tabs[0]: _tab_overview(df_results, insights)
    with tabs[1]: _tab_tbt_flow(df_results)
    with tabs[2]: _tab_conversation_explorer(df_results)
    with tabs[3]: _tab_data_table(df_results)
    with tabs[4]: _tab_recommendations_and_export(df_results, insights)


if __name__ == "__main__":
    main()
