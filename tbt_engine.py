"""
tbt_engine.py
=============
Clean, modular backend for Turn-by-Turn Sentiment Analytics.
Extracted & refactored from the Domain Agnostic TbT Granular Sentiment Notebook.

No Colab / ipywidgets / tqdm / print dependencies.
Consumed exclusively by tbt_app.py (Streamlit entrypoint).

Architecture
------------
ConversationProcessor  — domain-agnostic transcript / feedback parser
SentimentEngine        — VADER scorer with adaptive per-dataset thresholds
AnalyticsEngine        — turn-level metrics + aggregated business insights
run_pipeline()         — single entry-point that chains all three stages
"""

from __future__ import annotations

import gc
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Public constants (imported by tbt_app.py)
# ---------------------------------------------------------------------------

#: Maps internal dataset-type keys → human-readable format labels (sidebar)
FORMAT_LABELS: Dict[str, str] = {
    "spotify": "Media / Entertainment A  (Timestamp transcript)",
    "netflix": "Media / Entertainment B  (Bracket [HH:MM:SS])",
    "humana":  "Healthcare A  (Call transcript [MM:SS])",
    "ppt":     "Healthcare B  (Chat / SMS)",
    "lyft":    "Transportation  (Customer verbatim)",
    "hilton":  "Travel  (Guest feedback)",
    "auto":    "🔍 Auto-Detect",
}

#: Maps internal keys → short display names for UI headings
DOMAIN_DISPLAY: Dict[str, str] = {
    "spotify": "Media / Entertainment A",
    "netflix": "Media / Entertainment B",
    "humana":  "Healthcare A",
    "ppt":     "Healthcare B",
    "lyft":    "Transportation",
    "hilton":  "Travel",
    "auto":    "",
}


# ---------------------------------------------------------------------------
# Numba-accelerated math helpers
# ---------------------------------------------------------------------------

@njit
def _fast_rolling_mean_3(arr: np.ndarray) -> np.ndarray:
    """Compute a causal 3-point rolling mean (no look-ahead)."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            result[i] = arr[i]
        elif i == 1:
            result[i] = (arr[i - 1] + arr[i]) / 2.0
        else:
            result[i] = (arr[i - 2] + arr[i - 1] + arr[i]) / 3.0
    return result


@njit
def _fast_sentiment_change(arr: np.ndarray) -> np.ndarray:
    """Return first-difference of sentiment scores; index 0 is always 0."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[0] = 0.0
    for i in range(1, n):
        result[i] = arr[i] - arr[i - 1]
    return result


# ---------------------------------------------------------------------------
# ConversationProcessor
# ---------------------------------------------------------------------------

class ConversationProcessor:
    """
    Domain-agnostic parser for six conversation / feedback transcript formats.

    Parameters
    ----------
    dataset_type : str
        One of ``'spotify' | 'netflix' | 'humana' | 'ppt' | 'lyft' | 'hilton' | 'auto'``.
        Pass ``'auto'`` to let the engine sniff the format from the data.

    Usage
    -----
    >>> proc = ConversationProcessor(dataset_type="auto")
    >>> df_turns = proc.parse(df_raw)
    >>> print(proc.detected_format)
    """

    # Column names that are likely to hold transcript / feedback text,
    # checked in priority order before falling back to heuristics.
    _PRIORITY_COLS: List[str] = [
        "Comments", "comments", "COMMENTS",
        "Conversation", "conversation", "CONVERSATION",
        "Additional Feedback", "additional feedback", "Additional_Feedback",
        "verbatim", "Verbatim", "VERBATIM",
        "transcripts", "transcript", "Transcripts", "Transcript",
        "messages", "message", "Message Text (Translate/Original)",
        "feedback", "Feedback", "comment", "Comment", "text", "chat",
    ]

    def __init__(self, dataset_type: str = "auto") -> None:
        self.dataset_type = dataset_type.lower()

        # Compiled regex patterns — compiled once, reused across all rows.
        self._pat_timestamp = re.compile(
            r"^\|?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4})"
            r"\s+(Consumer|Customer|Agent|Advisor|Support):\s*(.*)$",
            re.IGNORECASE,
        )
        self._pat_bracket = re.compile(
            r"^\[(\d{1,2}:\d{2}:\d{2})\s+(AGENT|CUSTOMER|CONSUMER|ADVISOR|SUPPORT)\]:\s*(.*)$",
            re.IGNORECASE,
        )
        self._pat_humana = re.compile(
            r"\[(\d{1,3}:\d{2})\]\s+([^:]+?):\s*([^\[]+?)(?=\[|$)",
            re.IGNORECASE | re.DOTALL,
        )
        self._pat_ppt_html = re.compile(
            r"<b>(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*</b>([^<]+?)(?:<br\s*/?>|$)",
            re.IGNORECASE | re.DOTALL,
        )
        self._pat_ppt_sms = re.compile(
            r"(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*(.+?)(?=\d{2}:\d{2}:\d{2}\s+|$)",
            re.DOTALL,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse a raw DataFrame and return a tidy *turns* DataFrame.

        Each output row represents one conversational turn with columns:
        ``conversation_id``, ``turn_sequence``, ``timestamp``, ``speaker``,
        ``message``, ``turn_id``, ``cleaned_message``.

        Raises
        ------
        ValueError
            If no transcript column can be located, or if parsing yields 0 rows.
        """
        col = self._find_col(df)
        if not col:
            raise ValueError(
                "Could not find a transcript / feedback column in the uploaded file. "
                "Expected one of: Comments, Conversation, Transcripts, verbatim, etc."
            )

        # Auto-detect format from the first non-null row
        if self.dataset_type == "auto":
            sample = str(df[col].dropna().iloc[0]) if len(df) > 0 else ""
            self.dataset_type = self._detect(sample, col)

        rows: List[Dict] = []
        for idx, row in df.iterrows():
            text = str(row[col])
            if not text or text == "nan" or len(text) < 5:
                continue
            rows.extend(self._dispatch(text, int(idx)))  # type: ignore[arg-type]

        # Fallback: if selected type parsed nothing, try auto-detect
        if not rows and self.dataset_type != "auto":
            sample = str(df[col].dropna().iloc[0]) if len(df) > 0 else ""
            detected = self._detect(sample, col)
            if detected != self.dataset_type:
                self.dataset_type = detected
                for idx, row in df.iterrows():
                    text = str(row[col])
                    if not text or text == "nan" or len(text) < 5:
                        continue
                    rows.extend(self._dispatch(text, int(idx)))  # type: ignore[arg-type]

        if not rows:
            raise ValueError(
                "No turns could be parsed. "
                "Please verify the file format matches the selected domain."
            )

        out = pd.DataFrame(rows)
        out["turn_id"] = range(1, len(out) + 1)
        out["cleaned_message"] = out["message"].str.lower().str.strip()
        return out

    @property
    def detected_format(self) -> str:
        """Human-readable label for the detected / selected format."""
        return FORMAT_LABELS.get(self.dataset_type, self.dataset_type.upper())

    # ------------------------------------------------------------------
    # Private — format detection
    # ------------------------------------------------------------------

    def _detect(self, sample: str, col: str) -> str:
        """Sniff the dataset type from a representative text sample."""
        if self._pat_bracket.search(sample):
            return "netflix"
        if self._pat_timestamp.search(sample):
            return "spotify"
        if self._pat_ppt_html.search(sample):
            return "ppt"
        if self._pat_humana.search(sample):
            return "humana"
        if self._pat_ppt_sms.search(sample):
            return "ppt"
        cl = col.lower()
        return "hilton" if ("additional" in cl or "hilton" in cl) else "lyft"

    def _dispatch(self, text: str, idx: int) -> List[Dict]:
        """Route a raw text cell to the correct parser."""
        if self.dataset_type == "netflix":
            return self._parse_netflix(text, idx)
        if self.dataset_type == "humana":
            return self._parse_humana(text, idx)
        if self.dataset_type == "ppt":
            return self._parse_ppt(text, idx)
        if self.dataset_type in ("lyft", "hilton"):
            return self._parse_feedback(text, idx)
        return self._parse_spotify(text, idx)

    def _find_col(self, df: pd.DataFrame) -> Optional[str]:
        """Locate the transcript / feedback column via priority list then heuristics."""
        for name in self._PRIORITY_COLS:
            if name in df.columns:
                return name
        # Pattern-match heuristic
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                s = str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else ""
                if (self._pat_bracket.search(s) or self._pat_timestamp.search(s)
                        or self._pat_humana.search(s)
                        or self._pat_ppt_html.search(s) or self._pat_ppt_sms.search(s)):
                    return col
        # Last resort: longest string column
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                if df[col].dropna().astype(str).str.len().mean() > 20:
                    return col
        return None

    # ------------------------------------------------------------------
    # Private — normalisation helpers
    # ------------------------------------------------------------------

    def _norm(self, spk: str) -> str:
        """Normalise speaker labels to CUSTOMER | AGENT."""
        s = spk.upper().strip()
        if s in {"AGENT", "ADVISOR", "SUPPORT", "REP", "REPRESENTATIVE", "SPECIALIST"}:
            return "AGENT"
        if s in {"CUSTOMER", "CONSUMER", "CLIENT", "USER", "MEMBER", "PATIENT", "CALLER"}:
            return "CUSTOMER"
        return s

    def _row(self, idx: int, seq: int, ts: Optional[str], spk: str, msg: str) -> Dict:
        """Build a standardised turn record."""
        return {
            "conversation_id": f"CONV_{idx + 1:04d}",
            "turn_sequence":   seq,
            "timestamp":       ts,
            "speaker":         spk,
            "message":         msg,
        }

    # ------------------------------------------------------------------
    # Private — format-specific parsers
    # ------------------------------------------------------------------

    def _parse_netflix(self, text: str, idx: int) -> List[Dict]:
        """Parse bracket-style [HH:MM:SS SPEAKER]: … transcripts."""
        lines = text.split("\n")
        turns: List[Dict] = []
        tn = 1
        cs = ct = None
        cm: List[str] = []

        def flush() -> None:
            nonlocal tn
            if cs:
                msg = " ".join(cm).strip()
                if msg:
                    turns.append(self._row(idx, tn, ct, self._norm(cs), msg))
                    tn += 1

        for line in lines:
            ls = line.strip()
            m = self._pat_bracket.match(ls)
            if m:
                flush()
                ct, cs, cm = m.group(1), m.group(2), []
                r = m.group(3).strip()
                if r:
                    cm.append(r)
            elif cs and ls:
                cm.append(ls)
        flush()
        return turns

    def _parse_spotify(self, text: str, idx: int) -> List[Dict]:
        """Parse ISO-timestamp-based transcripts (YYYY-MM-DD HH:MM:SS ±ZZZZ Speaker: …)."""
        lines = text.split("\n")
        turns: List[Dict] = []
        tn = 1
        cs = ct = None
        cm: List[str] = []

        def flush() -> None:
            nonlocal tn
            if cs:
                msg = " ".join(cm).strip()
                if msg:
                    turns.append(self._row(idx, tn, ct, self._norm(cs), msg))
                    tn += 1

        for line in lines:
            ls = line.strip()
            m = self._pat_timestamp.match(ls)
            if m:
                flush()
                ct, cs = m.group(1), m.group(2)
                r = m.group(3).strip()
                cm = [r] if r else []
            elif cs and ls:
                cm.append(ls)
        flush()
        return turns

    def _parse_humana(self, text: str, idx: int) -> List[Dict]:
        """Parse call-centre transcripts with [MM:SS] Speaker: message format."""
        matches = self._pat_humana.findall(text)
        turns: List[Dict] = []
        tn = 1
        for ts, spk, msg in matches:
            sl = spk.strip().lower()
            # Skip system / IVR messages
            if sl in {"system", "automated", "ivr", "automated system"}:
                continue
            m = msg.strip()
            if not m or len(m) < 3:
                continue
            if any(k in sl for k in ["member", "customer", "patient", "caller"]):
                ns = "CUSTOMER"
            elif any(k in sl for k in ["agent", "representative", "rep", "advisor", "specialist"]):
                ns = "AGENT"
            else:
                ns = spk.strip().upper()
            turns.append(self._row(idx, tn, ts, ns, m))
            tn += 1
        return turns

    def _parse_ppt(self, text: str, idx: int) -> List[Dict]:
        """Parse HTML or SMS-style chat transcripts (Healthcare B format)."""
        html_m = self._pat_ppt_html.findall(text)
        if html_m:
            return self._ppt_turns(html_m, idx, is_sms=False)
        sms_m = self._pat_ppt_sms.findall(text)
        if sms_m:
            return self._ppt_turns(sms_m, idx, is_sms=True)
        return []

    def _ppt_turns(self, matches: list, idx: int, is_sms: bool) -> List[Dict]:
        """Shared builder for HTML and SMS variants of the PPT format."""
        spk_msgs: Dict[str, List] = {}
        ordered: List[str] = []

        for ts, spk, msg in matches:
            sl = spk.strip().lower()
            if sl == "system":
                continue
            if is_sms:
                # Strip PII placeholders from SMS logs
                msg = re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.]+Z\w*$", "", msg)
                msg = re.sub(r"Looks up Phone Number.*?digits-\d+", "", msg)
                msg = re.sub(r"Looks up SSN number.*?digits-\d+", "", msg)
                msg = re.sub(r"Phone Numbers rule for Chat|SSN rule for Chat", "", msg)
            m = msg.strip()
            if not m:
                continue
            if sl not in spk_msgs:
                spk_msgs[sl] = []
                ordered.append(sl)
            spk_msgs[sl].append((ts, m))

        if not spk_msgs:
            return []

        # Assign CUSTOMER / AGENT roles
        roles: Dict[str, str] = {}
        if is_sms:
            for s in ordered:
                roles[s] = "CUSTOMER" if re.match(r"^\d+$", s) else "AGENT"
        else:
            cnts = {s: len(msgs) for s, msgs in spk_msgs.items()}
            if len(cnts) == 1:
                cust = list(cnts.keys())[0]
            else:
                srt = sorted(cnts.items(), key=lambda x: x[1])
                f = ordered[0] if ordered else None
                cust = f if f and cnts.get(f, 999) <= srt[0][1] else srt[0][0]
            for s in ordered:
                roles[s] = "CUSTOMER" if s == cust else "AGENT"

        # Merge and sort all turns chronologically
        all_m = [(ts, s, m) for s in ordered for ts, m in spk_msgs[s]]
        all_m.sort(key=lambda x: x[0])
        return [self._row(idx, i, ts, roles.get(s, "CUSTOMER"), m)
                for i, (ts, s, m) in enumerate(all_m, 1)]

    def _parse_feedback(self, text: str, idx: int) -> List[Dict]:
        """Parse plain customer feedback / verbatim as a single-turn conversation."""
        m = text.strip()
        if not m:
            return []
        return [self._row(idx, 1, None, "CUSTOMER", m)]


# ---------------------------------------------------------------------------
# SentimentEngine
# ---------------------------------------------------------------------------

class SentimentEngine:
    """
    VADER-based sentiment scorer with adaptive per-dataset thresholds.

    Calibration samples up to 1,000 turns to compute dataset-specific
    percentile thresholds, avoiding the one-size-fits-all ±0.05 defaults.

    Usage
    -----
    >>> engine = SentimentEngine()
    >>> engine.calibrate(df_turns)
    >>> df_scored = engine.score(df_turns)
    """

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()
        # Default thresholds — overwritten after calibrate()
        self.thresholds: Dict[str, float] = {
            "positive":     0.05,
            "negative":    -0.05,
            "neutral_range": 0.10,
        }

    def calibrate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute dataset-specific positive / negative thresholds.

        Uses the 30th and 70th percentiles of a random sample so minority
        datasets (e.g. heavily negative call centres) aren't biased by global
        VADER defaults.
        """
        sample = df.sample(n=min(1_000, len(df)), random_state=42)
        scores = [
            self._vader.polarity_scores(str(m))["compound"]
            for m in sample["cleaned_message"].fillna("")
            if len(str(m)) > 5
        ]
        if not scores:
            return self.thresholds

        arr = np.array(scores)
        pos = max(float(np.percentile(arr, 70)), 0.10)
        neg = min(float(np.percentile(arr, 30)), -0.10)
        self.thresholds = {
            "positive":      pos,
            "negative":      neg,
            "neutral_range": pos - neg,
        }
        return self.thresholds

    def score(self, df: pd.DataFrame, chunk_size: int = 500) -> pd.DataFrame:
        """
        Score every turn and append sentiment columns.

        Adds: ``compound``, ``positive``, ``negative``, ``neutral``,
        ``sentiment_label``, ``sentiment_confidence``.
        Processes in chunks and calls gc.collect() periodically to manage
        memory on large datasets.
        """
        out = df.copy()
        for col in ["compound", "positive", "negative", "neutral", "sentiment_confidence"]:
            out[col] = 0.0
        out["sentiment_label"] = "neutral"

        for i in range(0, len(out), chunk_size):
            chunk = out.iloc[i: i + chunk_size]
            for idx in chunk.index:
                msg = out.at[idx, "cleaned_message"]
                if pd.isna(msg) or len(str(msg)) < 5:
                    continue
                sc = self._vader.polarity_scores(str(msg))
                out.at[idx, "compound"]  = sc["compound"]
                out.at[idx, "positive"]  = sc["pos"]
                out.at[idx, "negative"]  = sc["neg"]
                out.at[idx, "neutral"]   = sc["neu"]
                c = sc["compound"]
                if c >= self.thresholds["positive"]:
                    lbl  = "positive"
                    conf = min(c / self.thresholds["positive"], 1.0)
                elif c <= self.thresholds["negative"]:
                    lbl  = "negative"
                    conf = min(abs(c) / abs(self.thresholds["negative"]), 1.0)
                else:
                    lbl  = "neutral"
                    conf = max(0.0, 1.0 - abs(c) / (self.thresholds["neutral_range"] / 2))
                out.at[idx, "sentiment_label"]      = lbl
                out.at[idx, "sentiment_confidence"] = conf
            if i % (chunk_size * 10) == 0:
                gc.collect()

        return out


# ---------------------------------------------------------------------------
# AnalyticsEngine
# ---------------------------------------------------------------------------

class AnalyticsEngine:
    """
    Computes turn-level metrics and aggregated business insights.

    Turn metrics
    ------------
    - sentiment_change    : first-difference of compound score
    - sentiment_momentum  : 3-turn rolling mean of sentiment_change
    - turn_position       : normalised position within conversation (0–1)
    - phase               : start | middle | end  (first 3, last 3, rest)
    - potential_escalation: sharp negative drop by CUSTOMER (not in first 2 turns)
    - potential_resolution: positive upturn by CUSTOMER near end

    Business insights (dict)
    ------------------------
    Returned by ``compute_insights()`` — contains nested keys for
    overall_sentiment, customer_satisfaction, agent_performance,
    conversation_patterns, phase_csat_dsat, and recommendations.
    """

    def compute_turn_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add turn-level derived columns; sorts by conversation then sequence."""
        d = df.sort_values(["conversation_id", "turn_sequence"]).reset_index(drop=True)

        changes:  List[float] = []
        momentum: List[float] = []
        for _, grp in d.groupby("conversation_id"):
            s  = grp["compound"].values.astype(np.float64)
            ch = _fast_sentiment_change(s)
            mo = _fast_rolling_mean_3(ch)
            changes.extend(ch.tolist())
            momentum.extend(mo.tolist())

        d["sentiment_change"]   = changes
        d["sentiment_momentum"] = momentum

        max_turn = d.groupby("conversation_id")["turn_sequence"].transform("max")
        d["turn_position"]         = d["turn_sequence"] / max_turn
        d["is_conversation_start"] = d["turn_sequence"] <= 3
        d["is_conversation_end"]   = d["turn_sequence"] > (max_turn - 3)
        d["is_conversation_middle"] = ~d["is_conversation_start"] & ~d["is_conversation_end"]

        d["phase"] = "middle"
        d.loc[d["is_conversation_start"], "phase"] = "start"
        d.loc[d["is_conversation_end"],   "phase"] = "end"

        d["is_csat"] = d["compound"] >= 0
        d["is_dsat"] = d["compound"] <  0

        prev = d.groupby("conversation_id")["speaker"].shift(1)
        d["prev_speaker"]      = prev
        d["speaker_changed"]   = d["speaker"] != prev
        d["consecutive_turns"] = (
            d.groupby(["conversation_id", (d["speaker"] != prev).cumsum()]).cumcount() + 1
        )
        d["potential_escalation"] = (
            (d["sentiment_change"] < -0.3)
            & (d["speaker"] == "CUSTOMER")
            & (d["turn_sequence"] > 2)
        )
        d["potential_resolution"] = (
            (d["sentiment_change"] > 0.2)
            & (d["speaker"] == "CUSTOMER")
            & (d["is_conversation_end"])
        )
        return d

    def compute_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Aggregate df_results into a nested insights dict.

        All float values are safe to JSON-serialise (NaN → 0.0).
        """
        ins: Dict[str, Any] = {}

        ins["total_conversations"]      = int(df["conversation_id"].nunique())
        ins["total_turns"]              = int(len(df))
        ins["avg_turns_per_conversation"] = float(df.groupby("conversation_id").size().mean())

        ins["overall_sentiment"] = {
            "average": float(df["compound"].mean()),
            "median":  float(df["compound"].median()),
            "std":     float(df["compound"].std()),
        }

        cu = df[df["speaker"] == "CUSTOMER"]
        ag = df[df["speaker"] == "AGENT"]

        def _safe(series, default: float = 0.0) -> float:
            v = float(series) if hasattr(series, "__float__") else default
            return default if np.isnan(v) else v

        ins["customer_satisfaction"] = {
            "average_sentiment":       _safe(cu["compound"].mean())                                  if not cu.empty else 0.0,
            "positive_interactions":   _safe((cu["sentiment_label"] == "positive").mean())           if not cu.empty else 0.0,
            "escalation_rate":         _safe(cu["potential_escalation"].mean())                      if not cu.empty else 0.0,
            "resolution_rate":         _safe(cu["potential_resolution"].mean())                      if not cu.empty else 0.0,
        }
        ins["agent_performance"] = {
            "average_sentiment":        _safe(ag["compound"].mean())        if not ag.empty else 0.0,
            "response_effectiveness":   _safe(ag["sentiment_change"].mean()) if not ag.empty else 0.0,
            "consistency_score":        _safe(1.0 - ag["compound"].std())   if not ag.empty else 0.0,
        }

        st_  = df[df["is_conversation_start"]]
        mid_ = df[df["is_conversation_middle"]]
        en_  = df[df["is_conversation_end"]]
        ins["conversation_patterns"] = {
            "avg_sentiment_start":   _safe(st_["compound"].mean())  if not st_.empty  else 0.0,
            "avg_sentiment_middle":  _safe(mid_["compound"].mean()) if not mid_.empty else 0.0,
            "avg_sentiment_end":     _safe(en_["compound"].mean())  if not en_.empty  else 0.0,
            "sentiment_improvement": (
                _safe(en_["compound"].mean() - st_["compound"].mean())
                if not st_.empty and not en_.empty else 0.0
            ),
        }

        # Phase CSAT / DSAT — computed on customer turns only
        cust = cu if not cu.empty else df

        def _phase_stats(phase_df: pd.DataFrame) -> Dict:
            if phase_df.empty:
                return {"csat_pct": 0.0, "dsat_pct": 0.0, "avg_sentiment": 0.0, "count": 0}
            t   = len(phase_df)
            cs_ = int((phase_df["compound"] >= 0).sum())
            ds_ = int((phase_df["compound"] <  0).sum())
            return {
                "csat_pct":      cs_ / t,
                "dsat_pct":      ds_ / t,
                "avg_sentiment": float(phase_df["compound"].mean()),
                "count":         t,
            }

        ins["phase_csat_dsat"] = {
            "start":  _phase_stats(cust[cust["phase"] == "start"]),
            "middle": _phase_stats(cust[cust["phase"] == "middle"]),
            "end":    _phase_stats(cust[cust["phase"] == "end"]),
        }

        ins["recommendations"] = self._recommendations(ins)
        return ins

    def _recommendations(self, ins: Dict) -> List[str]:
        """Generate rule-based, actionable business recommendations."""
        r: List[str] = []
        cs  = ins["customer_satisfaction"]
        ap  = ins["agent_performance"]
        cp  = ins["conversation_patterns"]
        pcd = ins.get("phase_csat_dsat", {})

        if cs["average_sentiment"] < 0:
            r.append("🔴 Customer sentiment is below neutral — review agent training and script quality.")
        if cs["escalation_rate"] > 0.15:
            r.append(f"⚠️ High escalation rate ({cs['escalation_rate']:.1%}) — analyse trigger topics and train de-escalation.")
        elif cs["escalation_rate"] > 0.10:
            r.append(f"⚠️ Moderate escalation rate ({cs['escalation_rate']:.1%}) — monitor closely and introduce early-intervention protocols.")
        if cs["resolution_rate"] < 0.5:
            r.append(f"🔴 Low resolution rate ({cs['resolution_rate']:.1%}) — strengthen closing techniques and first-contact resolution targets.")
        if ap["average_sentiment"] < 0.1:
            r.append("📚 Agent sentiment is low — consider tone coaching and positive-language training.")
        if cp["sentiment_improvement"] < 0:
            r.append("📉 Conversations end worse than they start — review resolution processes and end-of-call scripts.")
        elif cp["sentiment_improvement"] > 0.2:
            r.append("📈 Strong positive sentiment improvement across conversations — document and replicate best-practice agent behaviours.")

        if pcd:
            mid = pcd.get("middle", {})
            if mid.get("dsat_pct", 0) > 0.5:
                r.append(f"⚠️ Mid-conversation DSAT at {mid['dsat_pct']:.1%} — customers losing patience during problem resolution; reduce handle time.")
            end = pcd.get("end", {})
            if end.get("dsat_pct", 0) > 0.4:
                r.append(f"🔴 End-conversation DSAT at {end['dsat_pct']:.1%} — customers leaving dissatisfied; improve wrap-up and follow-through.")
            start = pcd.get("start", {})
            if start.get("csat_pct", 0) > 0.7 and end.get("dsat_pct", 0) > 0.3:
                r.append("📉 CRITICAL: Customers start satisfied but end dissatisfied — the resolution process itself may be the failure point.")

        if not r:
            r.append("✅ All key metrics are within healthy ranges — maintain current practices and track trends over time.")
        return r


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    dataset_type: str = "auto",
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Execute the full Turn-by-Turn analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame (loaded from CSV / Excel by the Streamlit app).
    dataset_type : str
        Domain key from FORMAT_LABELS, or ``'auto'`` for auto-detection.

    Returns
    -------
    df_results : pd.DataFrame
        Enriched turns DataFrame with all sentiment and metric columns.
    insights : dict
        Nested dictionary of aggregated business insights.
    detected : str
        Human-readable label of the detected / selected data format.
    """
    processor   = ConversationProcessor(dataset_type=dataset_type)
    df_parsed   = processor.parse(df)
    detected    = processor.detected_format

    sentiment   = SentimentEngine()
    sentiment.calibrate(df_parsed)
    df_scored   = sentiment.score(df_parsed)

    analytics   = AnalyticsEngine()
    df_results  = analytics.compute_turn_metrics(df_scored)
    insights    = analytics.compute_insights(df_results)

    return df_results, insights, detected
