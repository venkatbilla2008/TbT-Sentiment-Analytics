# 🎭 Domain Agnostic — TbT Sentiment Analytics

Granular **Turn-by-Turn** sentiment analysis for call-centre conversations and
customer feedback. Built with Streamlit, VADER, Numba, and Plotly.

---

## Project Structure

```
tbt_app.py       ← Single-file app — all logic self-contained here
requirements.txt ← Python dependencies
.gitignore       ← Keeps venv/ and cache out of git
```

Everything — the parser, sentiment engine, analytics, charts, UI, and exports — lives inside `tbt_app.py`. No other Python files needed.

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

## Supported Domains / Formats

| Key | Label | Format Description |
|-----|-------|-------------------|
| `spotify` | Media / Entertainment A | ISO-timestamp transcripts |
| `netflix` | Media / Entertainment B | `[HH:MM:SS SPEAKER]:` bracket style |
| `humana`  | Healthcare A | `[MM:SS] Speaker: message` call transcripts |
| `ppt`     | Healthcare B | HTML `<b>HH:MM:SS name:</b>` or SMS chat |
| `lyft`    | Transportation | Plain customer verbatim / feedback |
| `hilton`  | Travel | Plain guest feedback |
| `auto`    | Auto-Detect | System sniffs format from first row |

---

## App Sections (inside tbt_app.py)

| Section | What it does |
|---------|-------------|
| `ConversationProcessor` | Parses raw transcripts into tidy turn rows |
| `SentimentEngine` | VADER scoring with adaptive per-dataset thresholds |
| `AnalyticsEngine` | Turn-level metrics + aggregated business insights |
| `run_pipeline()` | Chains all three stages into one call |
| Export helpers | Builds Excel (multi-sheet) and ZIP downloads |
| Chart factories | All Plotly figures |
| UI renderers | HTML/CSS components and Streamlit calls |
| Sidebar | Domain selector + file uploader |
| Tab renderers | One function per tab |
| `main()` | Entry point — wires everything together |

---

## Pipeline Flow

```
Upload CSV / Excel
       │
       ▼
ConversationProcessor.parse()
  → Detects format, extracts turns into tidy DataFrame
  → Columns: conversation_id, turn_sequence, timestamp, speaker, message
       │
       ▼
SentimentEngine.calibrate() + .score()
  → Adaptive VADER thresholds (30th / 70th percentile per dataset)
  → Adds: compound, positive, negative, neutral, sentiment_label, confidence
       │
       ▼
AnalyticsEngine.compute_turn_metrics()
  → Adds: sentiment_change, sentiment_momentum (Numba-accelerated)
  → Adds: phase (start / middle / end), potential_escalation, potential_resolution
       │
       ▼
AnalyticsEngine.compute_insights()
  → Aggregated dict: KPIs, CSAT/DSAT by phase, recommendations
```

---

## Key Features

- **6 domain formats** — auto-detected or manually selected
- **Phase-level CSAT / DSAT** — Start → Middle → End breakdown
- **Turn-by-turn flow chart** — interactive Plotly chart per conversation
- **Sentiment momentum** — 3-turn rolling change (Numba-accelerated)
- **Speaker × Phase heatmap** — spot who drives negativity and when
- **Escalation / Resolution detection** — rule-based signal flags
- **Adaptive thresholds** — VADER calibrated per dataset
- **Pipeline caching** — same file + domain = no re-run on rerenders
- **One-click export** — Excel (multi-sheet) + CSV + JSON in a ZIP

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.18.0
vaderSentiment>=3.3.2
numba>=0.59.0
openpyxl>=3.1.0
xlrd>=2.0.1
pyarrow>=14.0.0
```
