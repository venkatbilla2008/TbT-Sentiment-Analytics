# TbT Sentiment Analytics v5.1

Granular **Turn-by-Turn** sentiment analysis for call-centre conversations and customer transcripts. Upload a CSV or Excel file, select your format, and get interactive dashboards across sentiment, escalation, resolution, and speaker behaviour — no configuration required.

Built with Streamlit · VADER · Polars · Plotly.

---

## Project Structure

```
tbt_app.py       ← Single-file app — all logic self-contained here
requirements.txt ← Python dependencies
CLAUDE.md        ← Codebase guidance for Claude Code
.gitignore       ← Keeps venv/ and cache out of git
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run tbt_app.py
```

App opens at **http://localhost:8501**

---

## Supported Transcript Formats

| Key | Domain Label | Format Style |
|-----|-------------|--------------|
| `auto` | Auto-Detect | Sniffs format from first rows |
| `netflix` | Media / Entertainment | `[HH:MM:SS SPEAKER]: message` |
| `spotify` | Media / Entertainment | ISO-timestamp `Speaker: message` |
| `humana` | Healthcare A | `[MM:SS] Speaker: message` call transcripts |
| `ppt` | Healthcare B | HTML `<b>HH:MM:SS name:</b>` or SMS chat logs |

Upload a **CSV** or **Excel** file (.xlsx / .xls). For Excel files with multiple sheets, a sheet selector appears automatically.

---

## Pipeline

```
Upload CSV / Excel
       │
       ▼
ConversationProcessor.parse()
  Detects format, extracts turns into tidy rows
  Columns: conversation_id · turn_sequence · timestamp · speaker · message
       │
       ▼
SentimentEngine.calibrate() + .score()
  Adaptive VADER thresholds (30th / 70th percentile of dataset sample)
  Parallel scoring — ThreadPoolExecutor (4 workers)
  Adds: compound · pos · neg · neu · sentiment_label · confidence
       │
       ▼
AnalyticsEngine.compute_turn_metrics() + .compute_insights()
  Adds: sentiment_change · sentiment_momentum · phase · potential_escalation
  Adds: resolution_status (Truly Resolved / Partially Resolved / Unresolved / Escalated+Unrecovered)
  Produces: aggregated KPIs · CSAT/DSAT by phase · recommendations dict
       │
       ▼
_precompute_aggs()
  Polars groupbys for all charts — cached once per file
```

Each stage is independently cached with `@st.cache_data`. Changing domain or PII settings busts only the affected stage.

---

## Pages

| Page | What you get |
|------|-------------|
| **Overview** | KPI cards · sentiment distribution · phase CSAT/DSAT · escalation/resolution rates · top phrase clusters |
| **Sankey Flows** | Phase sentiment flow · Turn-by-Turn transitions · Outcome journey (Start → Resolution → End) |
| **Conversation Explorer** | Per-conversation sentiment timeline · turn-level detail table · animated playback |
| **Escalation Analysis** | Escalation triggers · agent effectiveness · category deterioration · resolution signal audit |
| **Export** | Download Excel workbook (multi-sheet) · CSV · JSON insights · ZIP bundle |

---

## Key Features

- **4 transcript formats** — auto-detected or manually selected
- **Hybrid resolution classification** — 4-way scoring (sentiment 30% + language 50% + outcome 20%)
- **Phase-level CSAT / DSAT** — Start → Middle → End sentiment breakdown per conversation
- **Escalation intelligence** — trigger phrase clusters, agent effectiveness buckets, category deterioration
- **Turn-by-Turn Sankey** — consecutive sentiment transition flows, filterable by speaker
- **Outcome Flow Sankey** — start sentiment → resolution status → end sentiment journey
- **PII redaction** — mask or tokenise 8 pattern types (email, phone, card, SSN, MRN, DOB, IP, address)
- **Multi-sheet Excel upload** — sheet selector shown automatically for workbooks with multiple sheets
- **Adaptive thresholds** — VADER calibrated to each dataset's distribution
- **5-stage pipeline cache** — no recomputation on UI interactions
- **Streaming scoring** — chunked VADER batches cap peak RAM regardless of dataset size
- **One-click export** — Excel (All Turns · Customer · Agent · Summary · Recommendations) + ZIP

---

## Performance Limits (auto-detected by RAM)

| Environment | Max Turns | Chunk Size | VADER Workers |
|-------------|-----------|------------|---------------|
| Streamlit Cloud (≤1 GB) | 100,000 | 5,000 | 2 |
| Local ≥16 GB RAM | 500,000 | 25,000 | up to 6 |
| Local ≥32 GB RAM | 1,000,000 | 50,000 | up to 8 |

Override via environment variable: `MAX_TURNS=500000 streamlit run tbt_app.py`

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
polars>=0.20.0
numpy>=1.26.0
plotly>=5.18.0
vaderSentiment>=3.3.2
openpyxl>=3.1.0
```

Install all: `pip install -r requirements.txt`

---

## PII Redaction

Enable in the sidebar before running. Two modes:

- **Mask** — replaces values with `[TYPE:REDACTED]` (e.g. `[EMAIL:REDACTED]`)
- **Tokenise** — replaces with stable tokens (e.g. `[EMAIL_001]`) so the same value maps to the same token across turns

Redaction is applied immediately after parsing, before scoring. The sidebar shows a count of redacted rows after analysis.

---

## Environment Variable Reference

| Variable | Effect |
|----------|--------|
| `MAX_TURNS` | Hard cap on turns processed (default auto-detected) |
| `STREAMLIT_SHARING_MODE` | Forces Cloud profile (low RAM limits) |
| `IS_STREAMLIT_CLOUD` | Same as above |
