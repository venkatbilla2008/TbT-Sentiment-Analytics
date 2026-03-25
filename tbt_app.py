"""
tbt_app.py  —  Domain Agnostic Turn-by-Turn Sentiment Analytics  v3.0
======================================================================
Fixes in v3.0
  ✓ Health-check crash: replaced JIT with pure numpy (no compile delay)
  ✓ Memory crash on 5000+ records: batched VADER scoring + gc.collect every 500 rows
  ✓ Large-file guard: hard cap at 50k turns; warns user before running
  ✓ Sidebar navigation (inspired by reference app) — no more tab overload
  ✓ Premium landing page with animated gradient + feature cards
  ✓ New visualisations: Sunburst, Escalation Timeline, Sentiment Waterfall
  ✓ Narrative Intelligence tab — auto-generated summary text
  ✓ Sidebar text fully visible on dark background
  ✓ Prominent 3-column export section

Run:  streamlit run tbt_app.py
"""

from __future__ import annotations

import gc, io, json, re, warnings, zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TbT Sentiment Analytics",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (corporate teal / slate — inspired by reference app)
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

/* ── Sidebar ── */
section[data-testid="stSidebar"]{{
  background:var(--warm-l)!important;
  border-right:1px solid var(--warm)!important;
}}
section[data-testid="stSidebar"] *{{color:var(--text)!important;}}
section[data-testid="stSidebar"] .stButton>button{{
  justify-content:flex-start!important;text-align:left!important;
  padding:10px 16px!important;border-radius:8px!important;
  width:100%!important;margin-bottom:3px!important;
  font-weight:500!important;font-size:13px!important;
  background:transparent!important;border:1px solid transparent!important;
  color:var(--text2)!important;transition:all .15s!important;
}}
section[data-testid="stSidebar"] .stButton>button:hover{{
  background:var(--warm)!important;border-color:var(--teal)!important;
  color:var(--teal)!important;
}}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{{
  background:var(--teal)!important;color:#fff!important;
  border-color:var(--teal)!important;font-weight:600!important;
}}

/* ── Metric cards ── */
.mc{{background:var(--card);border:1px solid var(--border);border-radius:10px;
    padding:16px 14px;text-align:center;border-top:3px solid var(--teal);
    box-shadow:0 1px 4px rgba(45,95,110,0.06);transition:all .2s;}}
.mc:hover{{box-shadow:0 4px 16px rgba(45,95,110,0.1);transform:translateY(-1px);}}
.mv{{font-size:22px;font-weight:700;color:var(--text);margin:0;line-height:1.2;}}
.ml{{font-size:10px;font-weight:600;color:var(--muted);margin:5px 0 0;
    text-transform:uppercase;letter-spacing:.7px;}}

/* ── Section headers ── */
.sh{{display:flex;align-items:center;gap:8px;margin:24px 0 12px;font-size:15px;
    font-weight:600;color:var(--text);padding-bottom:8px;
    border-bottom:2px solid var(--warm);}}

/* ── Phase table ── */
.pt{{width:100%;border-collapse:separate;border-spacing:0;font-size:13px;
    border-radius:8px;overflow:hidden;border:1px solid var(--border);}}
.pt th{{background:var(--teal);color:#fff;font-weight:600;padding:10px 14px;
       text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;}}
.pt td{{padding:8px 14px;border-bottom:1px solid var(--warm-l);color:var(--text);}}
.pt tr:nth-child(even){{background:var(--warm-l);}}
.pt tr:hover td{{background:#D6E8EE;}}

/* ── Badges ── */
.badge{{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;
       border-radius:5px;font-size:11px;font-weight:600;}}
.b-ok{{background:#D4E8DC;color:var(--ok);}}.b-warn{{background:#F0E6C8;color:#7A6620;}}
.b-err{{background:#F2D6D6;color:var(--err);}}.b-info{{background:#D6E8EE;color:var(--teal);}}

/* ── Turn cards ── */
.tc{{border-radius:10px;padding:10px 14px;margin-bottom:6px;
    border-left:4px solid transparent;background:var(--card);
    box-shadow:0 1px 3px rgba(45,95,110,0.06);}}
.tc-cu{{border-color:{C['neg']};background:#FEF5F5;}}
.tc-ag{{border-color:{C['teal']};background:#F0F7FA;}}
.tc-hdr{{font-size:11px;color:var(--muted);margin-bottom:4px;font-weight:500;}}
.tc-txt{{font-size:13px;color:var(--text);line-height:1.6;}}
.tc-meta{{font-size:11px;color:var(--slate);margin-top:5px;}}

/* ── Score bar ── */
.sbar{{display:flex;align-items:center;gap:6px;}}
.sbar-t{{flex:1;height:5px;background:var(--warm);border-radius:999px;overflow:hidden;}}
.sbar-f{{height:100%;border-radius:999px;}}

/* ── Rec cards ── */
.rc{{background:var(--warm-l);border:1px solid var(--border);border-radius:8px;
    padding:10px 14px;margin-bottom:6px;font-size:13px;color:var(--text);
    line-height:1.6;}}

/* ── Export cards ── */
.ex-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;
         padding:16px;border-top:3px solid var(--gold);}}
.ex-title{{font-size:14px;font-weight:600;color:var(--text);margin-bottom:4px;}}
.ex-desc{{font-size:12px;color:var(--muted);margin-bottom:12px;line-height:1.5;}}

/* ── Primary buttons ── */
.stButton>button[kind="primary"]{{
  background:var(--teal)!important;border-color:var(--teal)!important;
  color:#fff!important;font-weight:600!important;
}}
.stButton>button[kind="primary"]:hover{{background:var(--teal-l)!important;}}

/* Tabs */
.stTabs [data-baseweb="tab"]{{font-weight:500;color:var(--muted);font-size:13px;}}
.stTabs [aria-selected="true"]{{color:var(--teal)!important;
  border-bottom-color:var(--teal)!important;font-weight:600;}}

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
    "lyft":    "🚗 Transportation  (Customer verbatim)",
    "hilton":  "🏨 Travel  (Guest feedback)",
}
PHASE_ICONS = {"start": "🚀", "middle": "🔄", "end": "🏁"}
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C['text'], family="DM Sans"),
    margin=dict(l=10, r=20, t=40, b=10),
    hoverlabel=dict(bgcolor=C['text'], font_size=12, font_color=C['warm_l']),
)
MAX_TURNS = 50_000   # hard safety cap


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


# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS  (pure numpy — fast, no compile delay on startup)
# ─────────────────────────────────────────────────────────────────────────────
def _rolling_mean3(arr: np.ndarray) -> np.ndarray:
    r = np.empty(len(arr), dtype=np.float64)
    r[0] = arr[0]
    if len(arr) > 1: r[1] = (arr[0] + arr[1]) / 2
    for i in range(2, len(arr)): r[i] = (arr[i-2] + arr[i-1] + arr[i]) / 3
    return r

def _diff(arr: np.ndarray) -> np.ndarray:
    r = np.zeros(len(arr), dtype=np.float64)
    if len(arr) > 1: r[1:] = arr[1:] - arr[:-1]
    return r


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class ConversationProcessor:
    _PRIO = [
        "Comments","comments","COMMENTS",
        "Conversation","conversation","CONVERSATION",
        "Additional Feedback","additional feedback","Additional_Feedback",
        "verbatim","Verbatim","VERBATIM",
        "transcripts","transcript","Transcripts","Transcript",
        "messages","message","Message Text (Translate/Original)",
        "feedback","Feedback","comment","Comment","text","chat",
    ]
    def __init__(self, dataset_type: str = "auto"):
        self.dataset_type = dataset_type.lower()
        self._pt  = re.compile(r"^\|?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\s+(Consumer|Customer|Agent|Advisor|Support):\s*(.*)$", re.I)
        self._pb  = re.compile(r"^\[(\d{1,2}:\d{2}:\d{2})\s+(AGENT|CUSTOMER|CONSUMER|ADVISOR|SUPPORT)\]:\s*(.*)$", re.I)
        self._ph  = re.compile(r"\[(\d{1,3}:\d{2})\]\s+([^:]+?):\s*([^\[]+?)(?=\[|$)", re.I|re.DOTALL)
        self._pph = re.compile(r"<b>(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*</b>([^<]+?)(?:<br\s*/?>|$)", re.I|re.DOTALL)
        self._pps = re.compile(r"(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*(.+?)(?=\d{2}:\d{2}:\d{2}\s+|$)", re.DOTALL)

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self._find_col(df)
        if not col: raise ValueError("No transcript/feedback column found.")
        if self.dataset_type == "auto":
            sample = str(df[col].dropna().iloc[0]) if len(df) > 0 else ""
            self.dataset_type = self._detect(sample, col)
        rows: List[Dict] = []
        for idx, row in df.iterrows():
            text = str(row[col])
            if not text or text == "nan" or len(text) < 5: continue
            rows.extend(self._dispatch(text, int(idx)))
        if not rows:
            raise ValueError("No turns parsed. Check domain selector matches your file format.")
        out = pd.DataFrame(rows)
        out["turn_id"]         = range(1, len(out) + 1)
        out["cleaned_message"] = out["message"].str.lower().str.strip()
        return out

    @property
    def detected_format(self) -> str:
        return FORMAT_LABELS.get(self.dataset_type, self.dataset_type.upper())

    def _detect(self, s, col):
        if self._pb.search(s):  return "netflix"
        if self._pt.search(s):  return "spotify"
        if self._pph.search(s): return "ppt"
        if self._ph.search(s):  return "humana"
        if self._pps.search(s): return "ppt"
        cl = col.lower()
        return "hilton" if ("additional" in cl or "hilton" in cl) else "lyft"

    def _dispatch(self, text, idx):
        if self.dataset_type == "netflix":          return self._parse_bracket(text, idx)
        if self.dataset_type == "humana":           return self._parse_humana(text, idx)
        if self.dataset_type == "ppt":              return self._parse_ppt(text, idx)
        if self.dataset_type in ("lyft","hilton"):  return self._parse_feedback(text, idx)
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
            if m: flush(); ct,cs,cm=m.group(1),m.group(2),[]
            elif cs and ls: cm.append(ls)
            if m and m.group(3).strip(): cm.append(m.group(3).strip())
        flush(); return turns

    def _parse_spotify(self, text, idx):
        lines=text.split("\n"); turns=[]; tn=1; cs=ct=None; cm=[]
        def flush():
            nonlocal tn
            if cs:
                msg=" ".join(cm).strip()
                if msg: turns.append(self._row(idx,tn,ct,self._norm(cs),msg)); tn+=1
        for line in lines:
            ls=line.strip(); m=self._pt.match(ls)
            if m: flush(); ct,cs=m.group(1),m.group(2); cm=[m.group(3).strip()] if m.group(3).strip() else []
            elif cs and ls: cm.append(ls)
        flush(); return turns

    def _parse_humana(self, text, idx):
        turns=[]; tn=1
        for ts,spk,msg in self._ph.findall(text):
            sl=spk.strip().lower()
            if sl in {"system","automated","ivr"}: continue
            m=msg.strip()
            if not m or len(m)<3: continue
            ns="CUSTOMER" if any(k in sl for k in ["member","customer","patient","caller"]) \
               else "AGENT" if any(k in sl for k in ["agent","representative","rep","advisor","specialist"]) \
               else spk.strip().upper()
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
            cust=ordered[0] if len(cnts)==1 or (ordered and cnts.get(ordered[0],999)<=sorted(cnts.values())[0]) else min(cnts,key=cnts.get)
            for s in ordered: roles[s]="CUSTOMER" if s==cust else "AGENT"
        all_m=sorted([(ts,s,m) for s in ordered for ts,m in spk_msgs[s]],key=lambda x:x[0])
        return [self._row(idx,i,ts,roles.get(s,"CUSTOMER"),m) for i,(ts,s,m) in enumerate(all_m,1)]

    def _parse_feedback(self, text, idx):
        m=text.strip(); return [self._row(idx,1,None,"CUSTOMER",m)] if m else []


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT ENGINE  (batched, gc-aware — handles 5k+ records)
# ─────────────────────────────────────────────────────────────────────────────
class SentimentEngine:
    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()
        self.thresholds = {"pos": 0.05, "neg": -0.05, "nr": 0.10}

    def calibrate(self, df: pd.DataFrame):
        sample = df.sample(n=min(1000, len(df)), random_state=42)
        scores = [self._vader.polarity_scores(str(m))["compound"]
                  for m in sample["cleaned_message"].fillna("") if len(str(m)) > 5]
        if not scores: return
        arr = np.array(scores)
        pos = max(float(np.percentile(arr, 70)), 0.10)
        neg = min(float(np.percentile(arr, 30)), -0.10)
        self.thresholds = {"pos": pos, "neg": neg, "nr": pos - neg}

    def score(self, df: pd.DataFrame, progress_cb=None) -> pd.DataFrame:
        """
        Score in chunks of 500.  Calls gc.collect() every 10 chunks.
        progress_cb: optional callable(fraction 0-1) for Streamlit progress bar.
        """
        out = df.copy()
        for col in ["compound","positive","negative","neutral","sentiment_confidence"]:
            out[col] = 0.0
        out["sentiment_label"] = "neutral"
        total = len(out); chunk = 500

        for i in range(0, total, chunk):
            blk = out.iloc[i: i + chunk]
            for idx in blk.index:
                msg = out.at[idx, "cleaned_message"]
                if pd.isna(msg) or len(str(msg)) < 5: continue
                sc = self._vader.polarity_scores(str(msg))
                c  = sc["compound"]
                out.at[idx, "compound"]  = c
                out.at[idx, "positive"]  = sc["pos"]
                out.at[idx, "negative"]  = sc["neg"]
                out.at[idx, "neutral"]   = sc["neu"]
                if c >= self.thresholds["pos"]:
                    lbl="positive"; conf=min(c/self.thresholds["pos"],1.0)
                elif c <= self.thresholds["neg"]:
                    lbl="negative"; conf=min(abs(c)/abs(self.thresholds["neg"]),1.0)
                else:
                    lbl="neutral"; conf=max(0.0,1.0-abs(c)/(self.thresholds["nr"]/2))
                out.at[idx,"sentiment_label"]      = lbl
                out.at[idx,"sentiment_confidence"] = conf

            if (i // chunk) % 10 == 0:
                gc.collect()
            if progress_cb:
                progress_cb(min((i + chunk) / total, 1.0))

        gc.collect()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class AnalyticsEngine:
    def compute_turn_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.sort_values(["conversation_id","turn_sequence"]).reset_index(drop=True)
        chg=[]; mom=[]
        for _,grp in d.groupby("conversation_id"):
            s=grp["compound"].values.astype(np.float64)
            ch=_diff(s); mo=_rolling_mean3(ch)
            chg.extend(ch.tolist()); mom.extend(mo.tolist())
        d["sentiment_change"]=chg; d["sentiment_momentum"]=mom
        mt=d.groupby("conversation_id")["turn_sequence"].transform("max")
        d["turn_position"]         =d["turn_sequence"]/mt
        d["is_conversation_start"] =d["turn_sequence"]<=3
        d["is_conversation_end"]   =d["turn_sequence"]>(mt-3)
        d["is_conversation_middle"]=~d["is_conversation_start"]&~d["is_conversation_end"]
        d["phase"]="middle"
        d.loc[d["is_conversation_start"],"phase"]="start"
        d.loc[d["is_conversation_end"],  "phase"]="end"
        d["is_csat"]=d["compound"]>=0; d["is_dsat"]=d["compound"]<0
        prev=d.groupby("conversation_id")["speaker"].shift(1)
        d["prev_speaker"]=prev; d["speaker_changed"]=d["speaker"]!=prev
        d["consecutive_turns"]=(d.groupby(["conversation_id",(d["speaker"]!=prev).cumsum()]).cumcount()+1)
        d["potential_escalation"]=((d["sentiment_change"]<-0.3)&(d["speaker"]=="CUSTOMER")&(d["turn_sequence"]>2))
        d["potential_resolution"]=((d["sentiment_change"]>0.2)&(d["speaker"]=="CUSTOMER")&(d["is_conversation_end"]))
        return d

    def compute_insights(self, df: pd.DataFrame) -> Dict[str,Any]:
        ins={}
        ins["total_conversations"]        = int(df["conversation_id"].nunique())
        ins["total_turns"]                = int(len(df))
        ins["avg_turns_per_conversation"] = float(df.groupby("conversation_id").size().mean())
        ins["overall_sentiment"]          = {"average":float(df["compound"].mean()),
                                             "median": float(df["compound"].median()),
                                             "std":    float(df["compound"].std())}
        cu=df[df["speaker"]=="CUSTOMER"]; ag=df[df["speaker"]=="AGENT"]
        def _s(v,d=0.0): return d if (isinstance(v,float) and np.isnan(v)) or not hasattr(v,"__float__") else float(v)
        ins["customer_satisfaction"]={
            "average_sentiment":   _s(cu["compound"].mean())                       if not cu.empty else 0.0,
            "positive_pct":        _s((cu["sentiment_label"]=="positive").mean())  if not cu.empty else 0.0,
            "escalation_rate":     _s(cu["potential_escalation"].mean())            if not cu.empty else 0.0,
            "resolution_rate":     _s(cu["potential_resolution"].mean())            if not cu.empty else 0.0,
        }
        ins["agent_performance"]={
            "average_sentiment":      _s(ag["compound"].mean())        if not ag.empty else 0.0,
            "response_effectiveness": _s(ag["sentiment_change"].mean()) if not ag.empty else 0.0,
            "consistency_score":      _s(1.0-ag["compound"].std())     if not ag.empty else 0.0,
        }
        s_=df[df["is_conversation_start"]]; m_=df[df["is_conversation_middle"]]; e_=df[df["is_conversation_end"]]
        ins["conversation_patterns"]={
            "avg_sentiment_start":   _s(s_["compound"].mean())  if not s_.empty else 0.0,
            "avg_sentiment_middle":  _s(m_["compound"].mean())  if not m_.empty else 0.0,
            "avg_sentiment_end":     _s(e_["compound"].mean())  if not e_.empty else 0.0,
            "sentiment_improvement": _s(e_["compound"].mean()-s_["compound"].mean()) if not s_.empty and not e_.empty else 0.0,
        }
        cust=cu if not cu.empty else df
        def _ph(pdf):
            if pdf.empty: return {"csat_pct":0.0,"dsat_pct":0.0,"avg_sentiment":0.0,"count":0}
            t=len(pdf)
            return {"csat_pct":int((pdf["compound"]>=0).sum())/t,
                    "dsat_pct":int((pdf["compound"]<0).sum())/t,
                    "avg_sentiment":float(pdf["compound"].mean()),"count":t}
        ins["phase_csat_dsat"]={"start":_ph(cust[cust["phase"]=="start"]),
                                 "middle":_ph(cust[cust["phase"]=="middle"]),
                                 "end":_ph(cust[cust["phase"]=="end"])}
        ins["recommendations"]=self._recs(ins)
        return ins

    def _recs(self, ins):
        r=[]; cs=ins["customer_satisfaction"]; ap=ins["agent_performance"]
        cp=ins["conversation_patterns"]; pcd=ins.get("phase_csat_dsat",{})
        if cs["average_sentiment"]<0:      r.append("🔴 Customer sentiment below neutral — review agent training & scripts.")
        if cs["escalation_rate"]>0.15:     r.append(f"⚠️ High escalation rate ({cs['escalation_rate']:.1%}) — train de-escalation techniques.")
        elif cs["escalation_rate"]>0.10:   r.append(f"⚠️ Moderate escalation rate ({cs['escalation_rate']:.1%}) — monitor closely.")
        if cs["resolution_rate"]<0.5:      r.append(f"🔴 Low resolution rate ({cs['resolution_rate']:.1%}) — improve closing strategies.")
        if ap["average_sentiment"]<0.1:    r.append("📚 Agent sentiment low — tone coaching recommended.")
        if cp["sentiment_improvement"]<0:  r.append("📉 Conversations end worse than they start — review resolution processes.")
        elif cp["sentiment_improvement"]>0.2: r.append("📈 Strong positive improvement — document & replicate best-practice behaviours.")
        if pcd:
            mid=pcd.get("middle",{}); end=pcd.get("end",{}); start=pcd.get("start",{})
            if mid.get("dsat_pct",0)>0.5:  r.append(f"⚠️ Mid-conversation DSAT {mid['dsat_pct']:.1%} — reduce handle time.")
            if end.get("dsat_pct",0)>0.4:  r.append(f"🔴 End DSAT {end['dsat_pct']:.1%} — fix wrap-up process.")
            if start.get("csat_pct",0)>0.7 and end.get("dsat_pct",0)>0.3:
                r.append("📉 CRITICAL: Customers start satisfied but end dissatisfied.")
        if not r: r.append("✅ All key metrics within healthy ranges — maintain current practices.")
        return r


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame, dataset_type: str = "auto",
                 progress_bar=None) -> Tuple[pd.DataFrame, Dict, str]:
    proc = ConversationProcessor(dataset_type=dataset_type)
    df_p = proc.parse(df)

    # Safety cap
    if len(df_p) > MAX_TURNS:
        df_p = df_p.head(MAX_TURNS)
        st.warning(f"⚠️ Dataset truncated to {MAX_TURNS:,} turns for performance. "
                   "Split your file into smaller batches for full analysis.")

    detected = proc.detected_format
    sent = SentimentEngine(); sent.calibrate(df_p)

    def _pb(frac):
        if progress_bar: progress_bar.progress(frac, text=f"Scoring turns… {frac:.0%}")

    df_s = sent.score(df_p, progress_cb=_pb)
    if progress_bar: progress_bar.progress(1.0, text="Computing analytics…")

    anal = AnalyticsEngine()
    df_r = anal.compute_turn_metrics(df_s)
    ins  = anal.compute_insights(df_r)
    gc.collect()
    return df_r, ins, detected


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _to_excel(df: pd.DataFrame, ins: Dict) -> bytes:
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
        pd.DataFrame(ins.get("recommendations",[])).to_excel(w, sheet_name="Recommendations", index=False, header=False)
    return buf.getvalue()

def _to_csv(df: pd.DataFrame) -> str:
    buf=io.StringIO(); df.to_csv(buf,index=False); return buf.getvalue()

def _to_zip(df: pd.DataFrame, ins: Dict) -> bytes:
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("tbt_results.csv",   _to_csv(df))
        zf.writestr("tbt_insights.json", json.dumps(ins,indent=2,default=str))
        zf.writestr("tbt_results.xlsx",  _to_excel(df,ins))
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# CHART FACTORIES
# ─────────────────────────────────────────────────────────────────────────────
def _chart_sentiment_dist(df):
    cnt=df["sentiment_label"].value_counts().reset_index(); cnt.columns=["sentiment","count"]
    fig=px.bar(cnt,x="sentiment",y="count",color="sentiment",
               color_discrete_map={"positive":C['pos'],"neutral":C['neu'],"negative":C['neg']},
               title="Sentiment Distribution",labels={"sentiment":"","count":"Turns"})
    fig.update_traces(marker_line_width=0)
    return apply_chart(fig.update_layout(showlegend=False,title_font_size=14))

def _chart_speaker_box(df):
    fig=go.Figure()
    for role,color in [("CUSTOMER",C['neg']),("AGENT",C['teal'])]:
        sub=df[df["speaker"]==role]["compound"]
        if not sub.empty:
            fig.add_trace(go.Box(y=sub,name=role.capitalize(),marker_color=color,
                                 boxpoints="outliers",line_color=color))
    return apply_chart(fig.update_layout(title="Customer vs Agent Sentiment",title_font_size=14))

def _chart_phase_comparison(ins):
    pcd=ins.get("phase_csat_dsat",{}); phases=["Start","Middle","End"]
    fig=go.Figure(data=[
        go.Bar(name="CSAT %",x=phases,y=[pcd.get(p.lower(),{}).get("csat_pct",0)*100 for p in phases],
               marker_color=C['pos'],marker_line_width=0),
        go.Bar(name="DSAT %",x=phases,y=[pcd.get(p.lower(),{}).get("dsat_pct",0)*100 for p in phases],
               marker_color=C['neg'],marker_line_width=0),
    ])
    return apply_chart(fig.update_layout(barmode="group",title="CSAT vs DSAT by Phase",title_font_size=14,
        yaxis=dict(title="% Customer Turns"),legend=dict(orientation="h",yanchor="bottom",y=1.02)))

def _chart_sentiment_progression(df):
    tp=df.groupby("turn_sequence")["compound"].mean().reset_index(); tp=tp[tp["turn_sequence"]<=30]
    fig=go.Figure()
    fig.add_hline(y=0,line_dash="dash",line_color=C['warm'])
    fig.add_trace(go.Scatter(x=tp["turn_sequence"],y=tp["compound"],mode="lines+markers",
        line=dict(color=C['teal'],width=2.5),marker=dict(size=6,color=C['teal']),
        fill="tozeroy",fillcolor=f"rgba(45,95,110,0.1)"))
    return apply_chart(fig.update_layout(title="Avg Sentiment by Turn (first 30)",title_font_size=14,
        xaxis=dict(title="Turn"),yaxis=dict(title="Avg Score")))

def _chart_escalation_resolution(df):
    esc=int(df["potential_escalation"].sum()); res=int(df["potential_resolution"].sum())
    tot=max(df["conversation_id"].nunique(),1)
    fig=go.Figure(go.Bar(x=["Escalations","Resolutions"],y=[esc,res],
        marker_color=[C['neg'],C['pos']],marker_line_width=0,
        text=[f"{esc} ({esc/tot:.0%})",f"{res} ({res/tot:.0%})"],textposition="auto"))
    return apply_chart(fig.update_layout(title="Escalation & Resolution Events",title_font_size=14,showlegend=False))

def _chart_conv_scatter(df):
    cm=(df.groupby("conversation_id")
         .agg(avg_sentiment=("compound","mean"),turns=("turn_sequence","max"))
         .reset_index())
    fig=px.scatter(cm,x="turns",y="avg_sentiment",color="avg_sentiment",
        color_continuous_scale="RdYlGn",range_color=[-1,1],hover_name="conversation_id",
        title="Conversation Map  (length vs sentiment)",
        labels={"turns":"Turns","avg_sentiment":"Avg Sentiment"})
    fig.update_coloraxes(colorbar=dict(thickness=10))
    return apply_chart(fig.update_layout(title_font_size=14))

def _chart_tbt_flow(df, conv_id):
    sub=df[df["conversation_id"]==conv_id].sort_values("turn_sequence")
    fig=go.Figure()
    fig.add_hline(y=0,line_dash="dash",line_color=C['warm'])
    fig.add_trace(go.Scatter(x=sub["turn_sequence"],y=sub["compound"],mode="lines+markers",
        line=dict(color=C['teal'],width=2.5),
        marker=dict(size=9,color=sub["compound"],colorscale="RdYlGn",cmin=-1,cmax=1,
                    showscale=True,colorbar=dict(thickness=10,title="Score")),
        text=[f"Turn {r.turn_sequence}<br>{r.speaker}<br>{r.message[:55]}…"
              if len(r.message)>55 else f"Turn {r.turn_sequence}<br>{r.speaker}<br>{r.message}"
              for _,r in sub.iterrows()],
        hovertemplate="%{text}<br>Score: %{y:.3f}<extra></extra>"))
    mt=int(sub["turn_sequence"].max()) if not sub.empty else 1
    for pn,(s,e,col) in {"start":(1,3,"rgba(45,95,110,0.08)"),
                          "middle":(4,max(4,mt-3),"rgba(212,185,78,0.06)"),
                          "end":(max(4,mt-2),mt,"rgba(160,64,64,0.08)")}.items():
        if s<=e: fig.add_vrect(x0=s-.5,x1=e+.5,fillcolor=col,line_width=0,
                               annotation_text=PHASE_ICONS[pn],annotation_position="top left")
    return apply_chart(fig.update_layout(title=f"Turn-by-Turn Flow — {conv_id}",title_font_size=14,
        xaxis=dict(title="Turn"),yaxis=dict(title="Score",range=[-1.1,1.1])))

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
    sub=df[df["conversation_id"]==conv_id]
    pivot=(sub.groupby(["speaker","phase"])["compound"].mean()
              .unstack(fill_value=0).reindex(columns=["start","middle","end"],fill_value=0))
    fig=go.Figure(go.Heatmap(z=pivot.values,x=["Start","Middle","End"],y=pivot.index.tolist(),
        colorscale="RdYlGn",zmin=-1,zmax=1,
        text=[[f"{v:+.2f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",showscale=True,colorbar=dict(thickness=10),xgap=3,ygap=3))
    return apply_chart(fig.update_layout(title=f"Speaker × Phase Heatmap — {conv_id}",
        title_font_size=14,xaxis=dict(title="Phase"),yaxis=dict(title="")))

def _chart_sunburst(df):
    """Sunburst: Speaker → Phase → Sentiment label."""
    grp=(df.groupby(["speaker","phase","sentiment_label"]).size()
           .reset_index(name="count"))
    fig=px.sunburst(grp,path=["speaker","phase","sentiment_label"],values="count",
        color="sentiment_label",
        color_discrete_map={"positive":C['pos'],"negative":C['neg'],"neutral":C['neu']},
        title="Conversation Breakdown  (Speaker → Phase → Sentiment)")
    fig.update_traces(textfont=dict(family="DM Sans",size=11),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percentParent:.1%} of parent<extra></extra>")
    return apply_chart(fig.update_layout(title_font_size=14,margin=dict(l=10,r=10,t=45,b=10)))

def _chart_waterfall(ins):
    """Waterfall: Start → Middle → End avg sentiment change."""
    cp=ins["conversation_patterns"]
    vals=[cp["avg_sentiment_start"],
          cp["avg_sentiment_middle"]-cp["avg_sentiment_start"],
          cp["avg_sentiment_end"]-cp["avg_sentiment_middle"]]
    measure=["absolute","relative","relative"]
    labels=["Start","Middle → Δ","End → Δ"]
    fig=go.Figure(go.Waterfall(
        orientation="v",measure=measure,x=labels,y=vals,
        connector=dict(line=dict(color=C['border'],width=1)),
        increasing=dict(marker_color=C['pos']),
        decreasing=dict(marker_color=C['neg']),
        totals=dict(marker_color=C['teal']),
        text=[f"{v:+.3f}" for v in vals],textposition="outside"))
    return apply_chart(fig.update_layout(title="Sentiment Journey  (Start → Middle → End)",
        title_font_size=14,showlegend=False,yaxis=dict(title="Avg Score")))

def _chart_escalation_timeline(df):
    """Scatter of escalation events across turn positions."""
    esc=df[df["potential_escalation"]].copy()
    if esc.empty: return None
    fig=px.strip(esc,x="turn_position",y="conversation_id",color="speaker",
        color_discrete_map={"CUSTOMER":C['neg'],"AGENT":C['teal']},
        hover_data={"message":True,"compound":True},
        title="Escalation Event Map  (turn position within conversation)")
    fig.update_traces(jitter=0.4,marker_size=8)
    return apply_chart(fig.update_layout(title_font_size=14,
        xaxis=dict(title="Position (0=start, 1=end)"),yaxis=dict(title="")))


# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_landing():
    # Hero
    st.markdown(f"""
<div style="background:linear-gradient(135deg,{C['text']} 0%,{C['teal']} 50%,{C['teal_l']} 100%);
     border-radius:20px;padding:3.5rem 2.5rem 3rem;text-align:center;
     box-shadow:0 12px 48px rgba(45,95,110,0.25);margin-bottom:2rem;">
  <div style="font-size:3.2rem;margin-bottom:.5rem">🎭</div>
  <h1 style="color:#fff;font-size:2.4rem;font-weight:700;margin:0 0 .6rem;letter-spacing:-.5px">
    TbT Sentiment Analytics</h1>
  <p style="color:{C['steel']};font-size:1.05rem;margin:0 0 1.8rem">
    Granular Turn-by-Turn Analysis &nbsp;·&nbsp; Start → Middle → End &nbsp;·&nbsp; CSAT / DSAT per Phase</p>
  <div style="display:inline-flex;gap:.75rem;flex-wrap:wrap;justify-content:center">
    <span style="background:rgba(255,255,255,0.12);color:#fff;border-radius:999px;
          padding:.35rem 1rem;font-size:.83rem;font-weight:500">⚡ Fast startup</span>
    <span style="background:rgba(255,255,255,0.12);color:#fff;border-radius:999px;
          padding:.35rem 1rem;font-size:.83rem;font-weight:500">⚡ Handles 5k+ records</span>
    <span style="background:rgba(255,255,255,0.12);color:#fff;border-radius:999px;
          padding:.35rem 1rem;font-size:.83rem;font-weight:500">📊 6 domain formats</span>
    <span style="background:rgba(255,255,255,0.12);color:#fff;border-radius:999px;
          padding:.35rem 1rem;font-size:.83rem;font-weight:500">🔍 Auto-detect</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # Feature cards
    features = [
        ("📊","KPI Dashboard","Conversations, turns, escalation rate, resolution rate & sentiment trend at a glance."),
        ("📈","Phase CSAT/DSAT","CSAT % and DSAT % broken down for Start, Middle and End phases independently."),
        ("🔄","Turn-by-Turn Flow","Interactive Plotly chart showing the exact sentiment trajectory of each conversation."),
        ("🌊","Waterfall Chart","Visual journey from Start → Middle → End with absolute and delta scores."),
        ("🌐","Sunburst View","Hierarchical breakdown: Speaker → Phase → Sentiment label in one glance."),
        ("⚡","Escalation Map","Timeline scatter of all escalation events across conversation positions."),
        ("🗣️","Conversation Explorer","Turn-by-turn card viewer with badges, score bars and confidence %."),
        ("💡","Narrative Intelligence","Auto-generated executive summary with data-driven recommendations."),
        ("⬇️","Flexible Export","Download as Excel (4 sheets), flat CSV, or a complete ZIP bundle."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
<div style="background:{C['card']};border:1px solid {C['border']};border-radius:12px;
     padding:1.2rem;margin-bottom:1rem;border-top:3px solid {C['teal']};
     box-shadow:0 2px 8px rgba(45,95,110,0.07);">
  <div style="font-size:1.6rem;margin-bottom:.4rem">{icon}</div>
  <div style="font-weight:600;color:{C['text']};font-size:.95rem;margin-bottom:.3rem">{title}</div>
  <div style="color:{C['muted']};font-size:.83rem;line-height:1.55">{desc}</div>
</div>""", unsafe_allow_html=True)

    # Format table
    st.markdown("---")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        sh("📂", "Supported Formats")
        st.markdown(f"""
<table class="pt">
<thead><tr><th>Icon</th><th>Domain</th><th>Data Type</th></tr></thead>
<tbody>
<tr><td>🎵</td><td>Media / Ent A</td><td>ISO-timestamp transcripts</td></tr>
<tr><td>🎬</td><td>Media / Ent B</td><td>Bracket [HH:MM:SS] transcripts</td></tr>
<tr><td>🏥</td><td>Healthcare A</td><td>Call transcripts [MM:SS]</td></tr>
<tr><td>🩼</td><td>Healthcare B</td><td>Chat / SMS logs</td></tr>
<tr><td>🚗</td><td>Transportation</td><td>Customer verbatim</td></tr>
<tr><td>🏨</td><td>Travel</td><td>Guest feedback</td></tr>
</tbody></table>
""", unsafe_allow_html=True)
    with c2:
        sh("🚀", "Getting Started")
        st.markdown(f"""
<div style="background:{C['card']};border:1px solid {C['border']};border-radius:10px;padding:1.2rem;">
  <ol style="color:{C['text2']};font-size:.92rem;line-height:2;margin:0;padding-left:1.2rem">
    <li>Select your <strong>Domain / Format</strong> in the sidebar<br>
        <span style="color:{C['muted']};font-size:.82rem">(or leave as Auto-Detect)</span></li>
    <li>Upload a <strong>CSV or Excel</strong> file using the sidebar uploader</li>
    <li>Click <strong>▶ Run Analysis</strong></li>
    <li>Explore results across the five navigation sections</li>
    <li>Download your results in Excel, CSV, or ZIP</li>
  </ol>
</div>
""", unsafe_allow_html=True)

    st.markdown(f'<div style="text-align:center;color:{C["muted"]};font-size:11px;padding:16px 0">'
                f'TbT Sentiment Analytics v3.0 &nbsp;·&nbsp; Domain Agnostic &nbsp;·&nbsp; '
                f'Powered by VADER + Streamlit</div>', unsafe_allow_html=True)


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

        # ── Navigation ──
        st.markdown("### 🗂 Navigate")
        pages = ["🏠 Home", "📊 Overview", "🔄 TbT Flow", "🗣️ Explorer", "📋 Data Table", "💡 Narrative & Export"]
        for p in pages:
            is_active = st.session_state.get("page") == p
            if st.button(p, key=f"nav_{p}", type="primary" if is_active else "secondary"):
                st.session_state["page"] = p
                st.rerun()

        st.markdown("---")

        # ── Configuration ──
        st.markdown("### ⚙️ Configuration")
        domain_keys   = list(FORMAT_LABELS.keys())
        domain_labels = [FORMAT_LABELS[k] for k in domain_keys]
        sel = st.selectbox("Domain / Format", options=range(len(domain_keys)),
                           format_func=lambda i: domain_labels[i], index=0,
                           help="Select transcript format or leave as Auto-Detect.")
        dataset_type = domain_keys[sel]

        st.markdown("---")

        # ── Upload ──
        st.markdown("### 📂 Upload Data")
        uploaded = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"],
                                    help="Conversation transcripts or customer feedback.")
        if uploaded:
            st.markdown(f"""
<div style="background:rgba(45,95,110,0.1);border:1px solid {C['teal']};
     border-radius:8px;padding:.5rem .8rem;margin-top:.5rem">
  <div style="color:{C['muted']};font-size:.72rem">Loaded:</div>
  <div style="color:{C['text']};font-weight:600;font-size:.88rem">📄 {uploaded.name}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Run button ──
        run = st.button("▶ Run Analysis", type="primary", key="run_btn")

        st.markdown(f'<div style="color:{C["muted"]};font-size:.7rem;text-align:center;padding-top:.5rem">'
                    f'v3.0 — Domain Agnostic TbT</div>', unsafe_allow_html=True)

    return dataset_type, uploaded, run


# ─────────────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def _kpi_row(ins):
    cs=ins["customer_satisfaction"]; ap=ins["agent_performance"]; cp=ins["conversation_patterns"]
    overall=ins["overall_sentiment"]["average"]
    esc_c=C['neg'] if cs["escalation_rate"]>0.15 else C['gold'] if cs["escalation_rate"]>0.10 else C['pos']
    res_c=C['pos'] if cs["resolution_rate"]>0.6  else C['gold'] if cs["resolution_rate"]>0.4  else C['neg']
    cols=st.columns(8)
    data=[
        ("Conversations",  f"{ins['total_conversations']:,}",         "var(--teal)"),
        ("Total Turns",    f"{ins['total_turns']:,}",                  "var(--slate)"),
        ("Overall Sent.",  f'{overall:+.3f}',                          _score_color(overall)),
        ("Customer Avg",   f'{cs["average_sentiment"]:+.3f}',          _score_color(cs["average_sentiment"])),
        ("Agent Avg",      f'{ap["average_sentiment"]:+.3f}',          _score_color(ap["average_sentiment"])),
        ("Escalation",     _pct(cs["escalation_rate"]),                esc_c),
        ("Resolution",     _pct(cs["resolution_rate"]),                res_c),
        ("Trend",          f'{cp["sentiment_improvement"]:+.3f}',      _score_color(cp["sentiment_improvement"])),
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
        sbar=_sbar(avg)
        rows+=(f"<tr><td>{PHASE_ICONS[pn]} <strong>{pn.capitalize()}</strong></td>"
               f"<td>{ind}</td><td>{cnt:,}</td>"
               f"<td><span class='badge b-ok'>{_pct(csat)} CSAT</span></td>"
               f"<td><span class='badge b-err'>{_pct(dsat)} DSAT</span></td>"
               f"<td>{sbar}</td></tr>")
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
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown(f'<div class="ex-card"><div class="ex-title">📊 Excel Workbook</div>'
                    f'<div class="ex-desc">All Turns · Customer · Agent · Summary · Recommendations (5 sheets)</div></div>',
                    unsafe_allow_html=True)
        st.download_button("📥 Download Excel (.xlsx)", data=_to_excel(df_r,ins),
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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_overview(df_r, ins):
    sh("📊","Phase-Level CSAT / DSAT")
    _phase_table(ins)

    sh("🌊","Sentiment Journey")
    st.plotly_chart(_chart_waterfall(ins), width="stretch")

    sh("📈","Visual Overview")
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(_chart_sentiment_dist(df_r),   width="stretch")
    with c2: st.plotly_chart(_chart_speaker_box(df_r),      width="stretch")
    c3,c4=st.columns(2)
    with c3: st.plotly_chart(_chart_phase_comparison(ins),          width="stretch")
    with c4: st.plotly_chart(_chart_escalation_resolution(df_r),    width="stretch")

    sh("🌐","Conversation Sunburst")
    st.plotly_chart(_chart_sunburst(df_r), width="stretch")

    sh("📉","Sentiment Progression & Conversation Map")
    st.plotly_chart(_chart_sentiment_progression(df_r), width="stretch")
    st.plotly_chart(_chart_conv_scatter(df_r),          width="stretch")

    esc_fig = _chart_escalation_timeline(df_r)
    if esc_fig:
        sh("⚡","Escalation Event Timeline")
        st.plotly_chart(esc_fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: TBT FLOW
# ─────────────────────────────────────────────────────────────────────────────
def page_tbt_flow(df_r):
    sh("🔄","Turn-by-Turn Sentiment Flow")
    conv_ids=sorted(df_r["conversation_id"].unique().tolist())
    c1,c2,c3=st.columns([2,1,1])
    with c1: sel=st.selectbox("Conversation",conv_ids,key="flow_conv")
    with c2: sflt=st.selectbox("Speaker",["All","CUSTOMER","AGENT"],key="flow_spk")
    with c3: pflt=st.selectbox("Phase",["All","start","middle","end"],key="flow_ph")

    dv=df_r[df_r["conversation_id"]==sel].copy()
    if sflt!="All": dv=dv[dv["speaker"]==sflt]
    if pflt!="All": dv=dv[dv["phase"]==pflt]

    st.plotly_chart(_chart_tbt_flow(df_r,sel),       width="stretch")
    st.plotly_chart(_chart_momentum(df_r,sel),        width="stretch")
    st.plotly_chart(_chart_speaker_heatmap(df_r,sel), width="stretch")

    cu=dv[dv["speaker"]=="CUSTOMER"]; ag=dv[dv["speaker"]=="AGENT"]
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Turns (filtered)",  len(dv))
    m2.metric("Customer Avg",  f"{cu['compound'].mean():+.3f}" if not cu.empty else "—")
    m3.metric("Agent Avg",     f"{ag['compound'].mean():+.3f}" if not ag.empty else "—")
    m4.metric("Escalation turns",int(dv["potential_escalation"].sum()) if "potential_escalation" in dv.columns else 0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def page_explorer(df_r):
    sh("🗣️","Conversation Explorer")
    c1,c2=st.columns([1.2,1])
    with c1: conv=st.selectbox("Conversation",sorted(df_r["conversation_id"].unique()),key="exp_conv")
    with c2: ph  =st.selectbox("Phase",["All","start","middle","end"],key="exp_ph")
    sub=df_r[df_r["conversation_id"]==conv]
    if ph!="All": sub=sub[sub["phase"]==ph]
    _turn_viewer(sub, conv)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: DATA TABLE
# ─────────────────────────────────────────────────────────────────────────────
def page_data_table(df_r):
    sh("📋","Full Results Table")
    c1,c2,c3=st.columns(3)
    with c1: fs=st.selectbox("Speaker",["All","CUSTOMER","AGENT"],key="dt_spk")
    with c2: fsen=st.selectbox("Sentiment",["All","positive","neutral","negative"],key="dt_sen")
    with c3: fp=st.selectbox("Phase",["All","start","middle","end"],key="dt_ph")
    dt=df_r.copy()
    if fs!="All":   dt=dt[dt["speaker"]==fs]
    if fsen!="All": dt=dt[dt["sentiment_label"]==fsen]
    if fp!="All":   dt=dt[dt["phase"]==fp]
    cols=[c for c in ["conversation_id","turn_sequence","phase","speaker","timestamp",
                       "message","sentiment_label","compound","sentiment_confidence",
                       "potential_escalation","potential_resolution"] if c in dt.columns]
    st.markdown(f"**{len(dt):,} rows** after filters")
    st.dataframe(dt[cols].reset_index(drop=True), width="stretch", height=450)
    st.download_button("⬇️ Download filtered CSV",
        data=_to_csv(dt[cols]),
        file_name=f"tbt_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: NARRATIVE & EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def page_narrative_export(df_r, ins):
    sh("💡","Narrative Intelligence")
    # Auto-generated executive summary
    cs=ins["customer_satisfaction"]; ap=ins["agent_performance"]
    cp=ins["conversation_patterns"]; pcd=ins.get("phase_csat_dsat",{})
    total=ins["total_turns"]; convs=ins["total_conversations"]

    sentiment_verdict = ("strongly positive" if ins["overall_sentiment"]["average"]>0.2
                         else "positive" if ins["overall_sentiment"]["average"]>0.05
                         else "neutral" if ins["overall_sentiment"]["average"]>-0.05
                         else "negative" if ins["overall_sentiment"]["average"]>-0.2
                         else "strongly negative")
    trend_verdict = ("improving" if cp["sentiment_improvement"]>0.05
                     else "declining" if cp["sentiment_improvement"]<-0.05 else "stable")

    paras = [
        f"**Dataset overview:** {convs:,} conversations totalling **{total:,} turns** were analysed. "
        f"Average turns per conversation: **{ins['avg_turns_per_conversation']:.1f}**.",

        f"**Overall sentiment** is **{sentiment_verdict}** (avg score {ins['overall_sentiment']['average']:+.3f}). "
        f"Customer average is {cs['average_sentiment']:+.3f}; agent average is {ap['average_sentiment']:+.3f}.",

        f"**Conversation trend** is **{trend_verdict}** — sentiment changes by "
        f"{cp['sentiment_improvement']:+.3f} from start to end. "
        f"Start avg: {cp['avg_sentiment_start']:+.3f} → Middle: {cp['avg_sentiment_middle']:+.3f} → End: {cp['avg_sentiment_end']:+.3f}.",

        f"**Escalation rate** is {_pct(cs['escalation_rate'])} and **resolution rate** is "
        f"{_pct(cs['resolution_rate'])}. "
        + ("Escalation is above the 15% threshold — investigate trigger topics. " if cs['escalation_rate']>0.15 else "")
        + ("Resolution rate is below 50% — closing strategies need improvement." if cs['resolution_rate']<0.5 else ""),
    ]

    # Phase narrative
    for pn in ["start","middle","end"]:
        p = pcd.get(pn, {})
        if p.get("count", 0) > 0:
            paras.append(
                f"**{pn.capitalize()} phase** ({p['count']:,} customer turns): "
                f"CSAT {_pct(p['csat_pct'])} · DSAT {_pct(p['dsat_pct'])} · avg {p['avg_sentiment']:+.3f}."
            )

    st.markdown(f"""
<div style="background:{C['card']};border:1px solid {C['border']};border-radius:12px;
     padding:1.5rem 1.8rem;margin-bottom:1rem;border-left:4px solid {C['teal']}">
  <div style="font-size:11px;color:{C['muted']};text-transform:uppercase;letter-spacing:1.2px;
       margin-bottom:.8rem;font-weight:600">Executive Summary — Auto-Generated</div>
""", unsafe_allow_html=True)
    for p in paras:
        st.markdown(p)
    st.markdown("</div>", unsafe_allow_html=True)

    # Recommendations
    sh("🔔","Recommendations")
    for rec in ins.get("recommendations", []):
        st.markdown(f'<div class="rc">{rec}</div>', unsafe_allow_html=True)

    st.markdown("---")
    _export_section(df_r, ins)

    st.markdown("---")
    sh("🔍","Raw Insights JSON")
    with st.expander("View full insights object"): st.json(ins)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Defaults
    if "page" not in st.session_state: st.session_state["page"] = "🏠 Home"

    dataset_type, uploaded, run_clicked = render_sidebar()

    # ── Load data ──
    if uploaded is not None and run_clicked:
        try:
            df_raw = (pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
                      else pd.read_excel(uploaded))
            pb = st.progress(0, text="Starting pipeline…")
            try:
                df_r, ins, detected = run_pipeline(df_raw, dataset_type, progress_bar=pb)
                pb.empty()
                st.session_state["df_r"]     = df_r
                st.session_state["ins"]      = ins
                st.session_state["detected"] = detected
                st.session_state["fname"]    = uploaded.name
                if st.session_state.get("page") == "🏠 Home":
                    st.session_state["page"] = "📊 Overview"
                st.rerun()
            except Exception as exc:
                pb.empty()
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)
                return
        except Exception as exc:
            st.error(f"Could not read file: {exc}"); return

    # ── Route pages ──
    page = st.session_state.get("page", "🏠 Home")
    has_results = "df_r" in st.session_state and "ins" in st.session_state

    if page == "🏠 Home" or not has_results:
        render_landing()
        if not has_results and page != "🏠 Home":
            st.info("👆 Upload a file and click **▶ Run Analysis** to get started.")
        return

    df_r     = st.session_state["df_r"]
    ins      = st.session_state["ins"]
    detected = st.session_state.get("detected","—")
    fname    = st.session_state.get("fname","")

    # Status bar
    ca,_,cc = st.columns([3,2,1])
    with ca:
        st.markdown(f'<div style="color:{C["muted"]};font-size:.82rem">'
                    f'📂 {fname} &nbsp;·&nbsp; '
                    f'Format: <strong style="color:{C["teal"]}">{detected}</strong></div>',
                    unsafe_allow_html=True)
    with cc:
        st.download_button("⬇️ Quick ZIP", data=_to_zip(df_r,ins),
            file_name=f"tbt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip", width="stretch")

    st.markdown("---")
    _kpi_row(ins)
    st.markdown("---")

    if page == "📊 Overview":             page_overview(df_r, ins)
    elif page == "🔄 TbT Flow":           page_tbt_flow(df_r)
    elif page == "🗣️ Explorer":           page_explorer(df_r)
    elif page == "📋 Data Table":         page_data_table(df_r)
    elif page == "💡 Narrative & Export": page_narrative_export(df_r, ins)

    st.markdown(f'<div style="text-align:center;color:{C["muted"]};font-size:11px;padding:16px 0">'
                f'TbT Sentiment Analytics v3.0 &nbsp;·&nbsp; Domain Agnostic</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
