# Map — one page, so the whole thing fits in one look

Read this instead of the code. The code is what fills the tables below; the
tables and the rules are the system.

Three documents, three kinds of knowledge:

| file | holds |
|---|---|
| `MAP.md` | the SHAPE — where things live and what each part owns |
| `METHOD.md` | the RULES and the benchmark definitions |
| `ENTSOE.md` | facts about the SOURCE, its gaps, and the correspondence |

## The flow, once through

```
        ENTSO-E API                        File Library (history bundles)
             │                                        │
     ingest.fetch_raw                        parse_filelibrary
     data/raw/YYYY/MM/                              │
             │                                build_history
     parse_da / parse_im                  history_{da,im,load}.feather
             │                                        │
             └────────► ZONE STORE ◄──────────────────┘
          zones/<AREA>/da.parquet, im.parquet   (archive fill: by hand, optional)
                            │
                        as_of(current_ts)          ← the only causality rule
                            │
                       benchmarks_core
              build_matrix → add_pnl → signal_for
                            │
              forecasts.parquet ──► scores.parquet
```

Three systemd timers on the server (`scripts/make_systemd.py`), all through
`scripts/run.sh`:
`daily` at 09:40 UTC (update -> forecast -> score -> publish — the one decision,
and it must finish before the 12:00 market-local auction gate), `catchup` at
14:00 UTC (update -> score -> publish, NEVER forecast: the gate has closed), and
`monthly` on the 20th (re-request the previous month for the as-final pass).

## Files — what each one owns

**The core (pure, no I/O, no network)**

| file | owns | the rule it enforces |
|---|---|---|
| `benchmarks_core.py` | the benchmark maths and `as_of` | causality: you only see what was published |
| `period.py` | market day ↔ UTC instants | a market day is local midnight to local midnight (23/24/25 h) |

**Reading the outside world (raw in, DataFrame out)**

| file | owns | the rule it enforces |
|---|---|---|
| `ingest.py` | the HTTP request, saving raw bytes + sidecar | keep exactly what the source said; never log the key |
| `parse_da.py` / `parse_im.py` | live A44 / A85 documents | read the currency, never assume it; a price published per hour fills each of its four quarter-hours |
| `parse_filelibrary.py` | the monthly extracts (all zones) | Positive = A04/long, Negative = A05/short |
| `parse_fx.py`, `parse_fx_nbu.py` | ECB and NBU rates → one daily table | no rate is invented outside a currency's own span |

**The workflow (orchestration, no maths)**

| file | owns | the rule it enforces |
|---|---|---|
| `workflow.py` | the per-zone store, logging, seed, migrate, `for_each_area` | versions are kept on write, collapsed on read |
| `rebuild.py` | the store, rebuilt from `data/raw` alone | the store is a fold over raw: same parsing, same diff as `update` |
| `update.py` | polling, diffing, the poll log | only CHANGES are stored |
| `workflow_compute.py` | forecast / score | a day is scored exactly twice, and never forecast on a day that is not whole |
| `publish.py` | rendering the tables as published XML | a published file is frozen; it computes nothing — including the version, which is read from the row |
| `status.py` | `outputs/status.xml` | the ONE mutable file, and facts only — no self-assessment |
| `summary.py` | `outputs/v1/` — scores and forecasts, one row per zone per day; monthly sums of the same rows | derived values only; nothing before `publication.public_from`; no wall clock |
| `site_push.py` | the public data, pushed to `powertradinglab-site/data/` | only v1, status and the archive files v1 cites; a mirror, rebuilt every run |
| `backup.py` | the off-machine copy | dated snapshots, written atomically |
| `config.py` | `settings.json` | every tunable number lives in json, not in code |

**Every path is anchored to the repository**, never to the current directory:
each module computes `ROOT = Path(__file__).resolve().parents[1]` and builds its
paths from it. A relative path does not fail when a script is run from somewhere
else — it quietly builds a parallel tree. That happened: running an ad-hoc script
with the working directory inside `modelling/` created a second `data/raw` and
`logs/` there, and hid two fetched files well enough that we went looking for a
network fault which did not exist.

No module is named after a zone. What is particular to a zone — EIC, time
zone, expected currency, `shadow` or `live` — lives in `settings.json -> zones`,
and `areas` says which zones run. The store can always be rebuilt from
`data/raw` alone (`src/rebuild.py`); it then starts where raw starts.

**Around it**

| file | owns |
|---|---|
| `report/benchmarks.py` | the yearly report over all zones → `yearly.csv` |
| `validate_as_of.py` (notes repository) | proof that live == backtest (NL 2025, exact) |
| `scripts/make_systemd.py` | the server schedule (three timers, UTC) |
| `scripts/make_launchd.py` | the same schedule for a Mac; kept as the way back |
| `tests/` | 108 cases, each docstring says WHY — faster to read than the code |

## The five tables that matter

All timestamps UTC, tz-aware. `delivery_ts` = the 15-min PTU. `knownby_ts` =
when the row became known.

| table | one row per | key | filled by |
|---|---|---|---|
| `zones/<AREA>/da.parquet` | PTU **per version** | `delivery_ts, knownby_ts` | seed + update |
| `zones/<AREA>/im.parquet` | PTU **per version** | `delivery_ts, knownby_ts` | seed + update |
| `zones/<AREA>/polls.parquet` | one API request | `fetched_ts, doc` | update |
| `results/forecasts.parquet` | PTU of the target day | `area, knownby_ts, delivery_ts` | forecast() |
| `results/scores.parquet` | settled PTU, per pass | `area, knownby_ts, delivery_ts` | score() |

`area` is in the key of the two result tables, and it is what lets them BE two
tables: without it an FR row and an NL row for the same PTU computed at the same
instant would share a key and the append would drop one without saying so.

The layout question is settled (2026-09-20). The store is one folder per zone,
named by the area code; the results are one table per kind, across all zones.
The split is by what the thing is. The store is machinery — 235k rows per zone
per product in a file parquet rewrites whole on every append — so twenty-five
zones in one file would rewrite six million rows daily to add ninety-six. The
results are the product: small, keyed by area, and meant to be compared across
zones, which nobody should do by opening twenty-five files.

Columns worth knowing:

- store: `currency`, `price_da` / `price_im_pos`, `price_im_neg`
- forecasts: `action` (+1 buy, −1 sell, `<NA>` no action), `sig_buy`, `sig_sell`,
  `n_obs`, `data_through` (the freshest fact the signal actually saw),
  `skip_reason` (empty = the signal did not form; set = the forecast was REFUSED
  because its input day was not whole, and this delivery day will never have an
  action — see METHOD, rule 9)
- scores: `action_pnl` next to `ideal`, `devil`, `always_buy`, `always_sell`
- both result tables: `schema_version`, `methodology_version` — the two published
  numbers as they stood when the row was COMPUTED, so a re-render never
  relabels history with today's methodology

`data/archive/` holds the File Library downloads. DA and IM come from ONE
full-history bundle each (`EnergyPrices_12.1.D_r3.1.zip`,
`ImbalancePrices_17.1.G_r3.1.zip`); LOAD has no bundle and stays per month. The
per-month DA/IM zips are the July snapshot, kept as evidence and NOT read by the
build — the globs for those two carry no leading `*_`, which is the only thing
telling the two kinds of name apart. Which months are taken is one setting,
`history` in `settings.json`: from 2020_01 to the last CLOSED month.
- polls: `n_new`, `n_changed`, `http_ok` — this is the instrument that answers
  "when does imbalance actually appear"

`data/raw/YYYY/MM/` holds the frozen source: every response, successful or
refused, plus a `.meta.json` sidecar. Nothing scans it; the filename carries the
whole request.

`outputs/` holds what the world sees. Everything under a dated folder is frozen —
written once, never rewritten — and `outputs/status.xml` at the root is the only
file that changes. The boundary is visible from the path alone, which is the
point of putting it there.

## Where to look when something is odd

| question | look at |
|---|---|
| is it alive at all, and whose fault if not? | `outputs/status.xml` — answers it without our logs |
| did it run on time? | `logs/workflow.log` — every line carries its `slot` |
| why is there no forecast for a day? | `logs/workflow.log` for `NO FORECAST` — it names the missing input day and the backfill command |
| what did the source say? | `data/raw/YYYY/MM/` + the sidecar |
| did anything change today? | `polls.parquet`, columns `n_new` / `n_changed` |
| what did we decide, and on what? | `forecasts.parquet`, column `data_through` |
| do the report numbers still hold? | `validate_as_of.py`, kept in the notes repository |
| is the whole thing still sane? | `uv run pytest tests/ -q` |
| the store is lost or suspect | `uv run python -m src.rebuild --force` — starts where `data/raw` starts |
