# PowerTradingLab — 2020–2026 Power Trading Opportunities Retrospective

This folder holds everything behind the report *A 2020–2026 Power Trading
Opportunities Retrospective*: the LaTeX source, the monthly result table every
number in the report is taken from, and the scripts that turn that table into the
report's figures and tables. The point of publishing it is that a reader can
recompute any figure quoted in the report instead of taking it on trust.

- Report PDF: `powertradinglab_2020_2026_report.pdf`
- Report source: `powertradinglab_2020_2026_report.tex`
- Data cut-off: 2 September 2026, the date the ENTSO-E archives were downloaded
- Horizon: January 2020 – August 2026
- Web: https://powertradinglab.org

---

## What is in the folder

| File | What it is |
|---|---|
| `powertradinglab_2020_2026_report.tex` | the report source |
| `powertradinglab_2020_2026_report.pdf` | the compiled report |
| `yearly_update.csv` | **the result table.** One row per bidding zone and month, 2020–2026. Every number in the report comes from here |
| `benchmarks.py` | the script that produced `yearly_update.csv` from the raw ENTSO-E archives (see *Scope of this folder* below) |
| `tables.py` | `yearly_update.csv` → the six CSVs behind the report's tables |
| `plot_01_ev_decay.py` | `yearly_update.csv` → `figures/group_A_decay.png`, `figures/group_B_decay.png`, the report's two figures |
| `plot_02_zones.py` | `yearly_update.csv` → one two-panel PNG per bidding zone in `figures/`. Not used in the report; this is the eyeball tool the zone grouping was made with |
| `plot_03_renewable_share.py` | wind and solar share per group. Needs `vre_share_monthly.csv`, which is not part of this folder; its output is not used in the report |
| `plot.html` | a local gallery for the per-zone PNGs. Open it in a browser after running `plot_02_zones.py` |
| `figures/` | the generated PNGs, including the two the report embeds |
| `ptl-full-light.pdf`, `ptl-compact-light.pdf` | logos used by the title page and the running header |
| `LICENSE` | MIT for the scripts, CC BY 4.0 for the report, the table and the figures |

## Scope of this folder

`yearly_update.csv` is the published starting point, and it is the file to check
the report against.

`benchmarks.py` is included so that the computation is readable, but it will not
run here: it reads the raw 15-minute price archives from `data/processed/` and
imports the shared pipeline from `src/benchmarks_core.py`, both of which live in
the private PowerTradingLab core repository. Reproducing `yearly_update.csv` from
raw ENTSO-E data therefore means re-collecting the archives yourself; everything
downstream of it reproduces from this folder alone.

The raw inputs are day-ahead and imbalance prices from the ENTSO-E Transparency
Platform, converted to EUR at European Central Bank reference rates.

---

## The result table

`yearly_update.csv`, one row per `(area, year, month)`:

| Column | Meaning |
|---|---|
| `year`, `month` | the delivery month, UTC |
| `area` | bidding zone, ENTSO-E spelling |
| `ideal` | **Extractable Value** — the better of the two sides in every interval, chosen with perfect hindsight |
| `devil` | **Adverse Value** — the worse side in every interval |
| `always_buy` | **Passive DA Buy Value** — buy at day-ahead in every interval, settle in imbalance |
| `always_sell` | **Passive DA Sell Value** — the mirror: sell at day-ahead, settle in imbalance |
| `stat_action_pnl` | **Trailing Bias Value** — the side picked from the trailing bias of the three previous days |
| `price_da` | mean day-ahead price of the month, EUR/MWh |
| `gap_mean` | mean of \|DA − IM_mid\| over the month, EUR/MWh, where IM_mid is the average of the positive and the negative imbalance price |
| `gap_vs_price_da` | `gap_mean / price_da` — the volatility measure the report uses |
| `da_volatility` | standard deviation of the day-ahead price over its mean. The textbook measure, kept for comparison; the report does not use it |

The five benchmark columns are **euro per 1 MW of traded quantity, summed over
the month**. The four price columns are monthly averages.

### Conventions

These hold throughout the table and the report, and they are the things to get
right before comparing a number against your own:

- **Per 1 MW.** Every benchmark is the result of trading exactly 1 MW in every
  15-minute interval of the month.
- **×0.25.** Prices are quoted in EUR/MWh and a 15-minute delivery of 1 MW is
  0.25 MWh, so each per-interval result is the price difference times 0.25.
- **UTC** throughout, so a month is a UTC month.
- **No fees.** Exchange and settlement fees are not included.
- **Mean month, never a yearly sum.** Where the report shows a year it shows the
  average month of that year. 2026 stops in August, and only a mean month puts a
  part-year in the same column as a whole one.
- **90% coverage.** A month enters the table only if at least 90% of its
  15-minute intervals are present in both the day-ahead and the imbalance series.
  Months below that are dropped, not interpolated, which is why the per-zone
  charts show gaps as breaks in the line.
- **2021 for DK, FI, SE and NO rests on October–December only.** ENTSO-E
  publishes these zones from October 2021.

### Zones

The table carries 46 areas; the report describes 25 of them. The rest are in the
file because dropping them would have meant hiding the evidence for excluding
them — ES, PT, CH and SI are left out of the report because of how their
imbalance price is built, and the Italian zones, SK, LT, ME, AL, MK and GR
because of gaps in the ENTSO-E data.

`DK`, `NO`, `SE` and `IT` are extra rows, not replacements: each is the **mean**
of its member zones, and the member zones stay in the table. The mean, never a
sum — the mean of per-1-MW numbers is still per 1 MW. A month enters such an
average only if every member zone has that month, so the average never steps
because a zone entered or left the coverage filter.

### Single and dual imbalance pricing

The report's `S/D` column is read from the data, not from a list. The imbalance
band — the distance between the two imbalance prices — comes back out of the
table as

```
band = -4 * (ideal + devil) / N        # EUR/MWh, N = 15-minute intervals in the month
```

It is zero in a single-price zone and positive in a dual-price zone. Over
2020–2026 five of the 25 reported zones used dual pricing at some point: CZ until
June 2024, HU until April 2021 and again from August 2026, RO until June 2026, and
FR and NL throughout. In every other reported zone the band is zero in all 80
months, and `devil = -ideal` exactly.

---

## Reproducing the report

Python 3.10 or newer, with `pandas` and `matplotlib`:

```bash
pip install pandas matplotlib
```

Run everything from this folder.

**1. The report's tables.**

```bash
python tables.py
```

Writes six CSVs next to the script:

| File | Table in the report |
|---|---|
| `table_by_countries.csv` | Extractable Value by zone and year, both groups |
| `adverse_value.csv` | Adverse Value, the zones that were ever dual-priced |
| `passive_da_s.csv` | Passive DA Sell Value, single-price zones |
| `passive_da_d.csv` | Passive DA Buy and Sell Values, dual-price zones |
| `trailing_bias_values.csv` | Trailing Bias Value |
| `tb_efficiency.csv` | Trailing Bias Value as a share of Extractable Value |

`tb_efficiency.csv` carries two summaries of 2022–2026 and they are not equal.
The report quotes `sums_2022-2026`: the yearly Trailing Bias Values of the span
added up, divided by the yearly Extractable Values of the same span added up.
That is the one a reader can redo from the two printed tables with a single
division. `mean_2022-2026`, the mean of the yearly shares, is kept as a control —
where the two disagree, one year is carrying the summary.

**2. The report's figures.**

```bash
python plot_01_ev_decay.py yearly_update.csv
```

Writes `figures/group_A_decay.png` and `figures/group_B_decay.png`. Both groups
are drawn the same way: every zone's monthly series is divided by its own maximum
over the horizon, smoothed with a 3-month centred rolling median, and then
summarised across the zones of the group — the Extractable Value twice, as the
mean (solid) and the median (long dashes), the day-ahead price and the volatility
as the median only. `--smooth 1` turns the smoothing off.

The distance between the solid and the dashed line is itself a result: it opens
where a minority of zones runs hotter than the rest.

**3. The per-zone charts** (optional, not in the report):

```bash
python plot_02_zones.py yearly_update.csv
open plot.html
```

**4. The report.**

```bash
pdflatex powertradinglab_2020_2026_report.tex
pdflatex powertradinglab_2020_2026_report.tex   # twice, for the table of contents
```

The table values are typed into the `.tex` by hand from the CSVs of step 1; the
figures are read from `figures/`. A LaTeX installation with `booktabs`,
`graphicx`, `titlesec`, `fancyhdr`, `enumitem`, `caption`, `hyperref` and
`eurosym` is enough. Without `eurosym` the preamble falls back to `textcomp`.

---

## Checking a single number

The quickest way to confirm that this folder and the report agree:

```python
import pandas as pd
y = pd.read_csv("yearly_update.csv")

# Extractable Value, mean month of 2026, RO — the report's Table 3
ro = y[(y.area == "RO") & (y.year == 2026)]
print(round(ro.ideal.mean()))        # 100082

# the largest Extractable Value of the whole study
z = y[y.area.isin(["BE","CZ","DE_LU","DK","EE","FI","FR","HR","HU","LV","PL",
                   "RO","RS","SE","AT","NL","NO"])]
print(z.loc[z.ideal.idxmax(), ["area", "year", "month", "ideal"]].tolist())
# ['RO', 2024, 6, 536049.3...]
```

---

## Attribution

Day-ahead and imbalance prices: ENTSO-E Transparency Platform. FX reference
rates: European Central Bank. The figures published here are derived benchmark
results, not raw market price series.

The scripts are under the MIT licence; the report, the result table, the figures
and this document are under CC BY 4.0. The PowerTradingLab name and logos are
excluded from both. See `LICENSE` for the terms and for the suggested citation.

If you use this material, please cite the report by its DOI (see the title page).

This document does not constitute trading advice.
