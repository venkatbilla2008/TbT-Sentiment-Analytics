# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run tbt_app.py
```

App opens at **http://localhost:8501**. There are no tests or build steps.

## Architecture

**Single-file app** — all logic lives in `tbt_app.py` (~4200 lines). No modules, no subpackages.

### Pipeline (data flows top to bottom)

```
Upload → ConversationProcessor.parse()
       → SentimentEngine.calibrate() + .score()
       → AnalyticsEngine.compute_turn_metrics() + .compute_insights()
       → _precompute_aggs()  ← cached aggregations used by all pages
```

Each stage is wrapped in a `@st.cache_data` function (`_cached_parse`, `_cached_score`, `_cached_analytics`). Cache key is a file checksum (`_file_checksum`). Do not bypass caching unless debugging a specific stage.

`run_pipeline()` is the outer orchestrator — it calls each cached stage in order and returns `(df_r, ins, detected, pii_meta)` (4-tuple).

### Key classes

| Class | Responsibility |
|-------|---------------|
| `ConversationProcessor` | Parses 4 transcript formats (netflix/spotify/humana/ppt) + auto-detect into tidy turn rows |
| `SentimentEngine` | Parallel VADER scoring (4-worker ThreadPoolExecutor), adaptive 30th/70th-percentile thresholds |
| `AnalyticsEngine` | Turn-level metrics (momentum, phase, escalation/resolution flags) + aggregated insight dict |
| `PIIRedactor` | Regex-based PII scrubbing applied during parse |

### Pages (rendered via `main()`)

`page_overview` · `page_sankey` · `page_explorer` · `page_escalation` · `page_narrative_export`

Sidebar is rendered by `render_sidebar()` — returns `(dataset_type, uploaded, run_clicked, pii_enabled, pii_mode, excel_sheet)`. Landing screen by `render_landing()`.

### Performance conventions

- **Polars** for all groupby/aggregation inside `_precompute_aggs` — do not replace with pandas groupby.
- `_to_pd()` converts Polars → pandas at the chart boundary.
- Scatter/sunburst charts subsample to ≤ 2 000 points.
- Data table is paginated at 200 rows.
- `_detect_env_limits()` caps workers/chunk-size based on available RAM.
- `_safe_collect(lf)` is the version-safe Polars collect helper — use it instead of `.collect()` directly inside `_precompute_aggs`.

### Supported transcript formats

| Key | Style |
|-----|-------|
| `netflix` | `[HH:MM:SS SPEAKER]: message` |
| `spotify` | ISO-timestamp `Speaker: message` |
| `humana` | `[MM:SS] Speaker: message` |
| `ppt` | HTML `<b>HH:MM:SS name:</b>` or SMS chat |
| `auto` | Format sniffed from first rows |

lyft/hilton (single-turn verbatim) were removed in v5.1.

### Excel upload

`_cached_parse` accepts an `excel_sheet` parameter (name or index, default `0`). When a multi-sheet Excel file is uploaded, `render_sidebar()` reads the sheet names via openpyxl and renders a selectbox — the chosen sheet name is forwarded through `run_pipeline()` → `_cached_parse()` → `pd.read_excel()`.

### `_precompute_aggs` return keys

| Key | Content |
|-----|---------|
| `sent_dist` | Sentiment label distribution DataFrame |
| `conv_map` | Per-conversation summary |
| `phase_speaker` | Phase × speaker sentiment aggregates |
| `turn_prog` | Turn-progression data |
| `esc_res` | Escalation/resolution event counts |
| `sample` | Subsampled DataFrame for scatter/sunburst (≤ CHART_SAMPLE rows) |
| `sm_flow` | Start→Middle phase flow (Sankey) |
| `me_flow` | Middle→End phase flow (Sankey) |
| `se_flow` | Start→End direct arc (Sankey) |
| `phase_pivot` | Conversation-level phase pivot |
| `spk_phase_sent` | Speaker × phase × sentiment counts |
| `turn_flow` | Consecutive turn sentiment transitions (source/target/speaker/count) |
| `outcome_flow_df` | Start sentiment → resolution status → end sentiment (Outcome Sankey) |

### UI conventions

- `mc(label, value)` — metric card HTML (`.mc` CSS class)
- `sh(icon, text)` — section header HTML (`.sh` CSS class)
- `apply_chart(fig)` — applies consistent Plotly theme/layout
- Global colour palette is the `C` dict at the top of the file

### Import conventions

All stdlib imports are at the top of the file: `gc, hashlib, io, json, os, re, warnings, zipfile`. Do not re-import these inside function bodies. `psutil` is imported locally inside `render_sidebar` because it is optional.

### Warning filters

`warnings.filterwarnings` at module level suppresses `DeprecationWarning`, `FutureWarning`, and Streamlit `UserWarning` only. Runtime/overflow warnings from numpy are not suppressed.
