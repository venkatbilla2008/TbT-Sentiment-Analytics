# 🎭 Domain Agnostic — TbT Sentiment Analytics

Granular **Turn-by-Turn** sentiment analysis for call-centre conversations and
customer feedback.  Built with Streamlit, VADER, Numba, and Plotly.

---

## Project Structure

```
tbt_app.py       ← Streamlit entrypoint (UI routing & session state)
tbt_engine.py    ← Parsing, VADER scoring, analytics pipeline
tbt_charts.py    ← Plotly chart factories
tbt_ui.py        ← Reusable HTML/CSS component renderers
tbt_demo.py      ← Sample dataset generators for each domain
tbt_export.py    ← Excel / ZIP export helpers
requirements.txt
README.md
```

### Why this split?

| File | Responsibility | Who imports it |
|------|---------------|----------------|
| `tbt_engine.py` | All data logic (no Streamlit) | `tbt_app.py` |
| `tbt_charts.py` | All Plotly charts (no Streamlit) | `tbt_app.py` |
| `tbt_ui.py` | HTML/CSS helpers (uses Streamlit) | `tbt_app.py` |
| `tbt_demo.py` | Sample data only (no Streamlit) | `tbt_app.py` |
| `tbt_export.py` | Export helpers (no Streamlit) | `tbt_app.py` |

This means the *engine*, *charts*, *demo data*, and *exports* can all be
unit-tested without a live Streamlit session.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run tbt_app.py
```

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

## Pipeline Overview

```
Raw CSV / Excel
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
  → Adds: phase (start/middle/end), potential_escalation, potential_resolution
      │
      ▼
AnalyticsEngine.compute_insights()
  → Aggregated dict: KPIs, CSAT/DSAT by phase, recommendations
```

---

## Key Features

- **6 domain formats** — auto-detected or manually selected
- **Phase-level CSAT / DSAT** — Start → Middle → End breakdown
- **Turn-by-turn flow** — interactive Plotly chart per conversation
- **Sentiment momentum** — 3-turn rolling change (Numba-accelerated)
- **Speaker × Phase heatmap** — quickly spot who drives negativity and when
- **Escalation / Resolution detection** — rule-based signal flags
- **Adaptive thresholds** — VADER calibrated per dataset (not global defaults)
- **Pipeline caching** — same file + domain = no re-run on Streamlit rerenders
- **One-click export** — Excel (multi-sheet) + CSV + JSON in a ZIP

---

## Adding a New Domain

1. **Parser** — add a regex + `_parse_<name>()` method in `ConversationProcessor`.
2. **Registry** — add the key to `FORMAT_LABELS` and `DOMAIN_DISPLAY` in `tbt_engine.py`.
3. **Demo** — add a `@_register("Label", "key")` generator in `tbt_demo.py`.
4. That's it — the sidebar, auto-detection, and exports update automatically.

---

## Performance Notes

- Pipeline results are cached in `st.session_state` keyed by a hash of the raw
  data bytes + dataset_type.  Re-uploading the same file is instant.
- VADER scoring processes in chunks of 500 with periodic `gc.collect()` to keep
  memory flat on large datasets (~50k turns tested).
- Numba JIT functions (`_fast_rolling_mean_3`, `_fast_sentiment_change`) compile
  on first call; subsequent calls are near-native speed.
