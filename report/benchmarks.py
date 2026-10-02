"""
Report generator: yearly benchmark table per zone -> yearly.csv.

Thin wrapper over src/benchmarks_core.py (the shared pipeline), so the report and
the live NL workflow use the SAME computation. Numbers are unchanged vs the old
inline version. n_days comes from settings.json via config (default 3 today; can
become region-dependent later with no code change).

Run (from repo root):
    uv run python modelling/benchmarks.py
"""

import os
import sys

import pandas as pd
from datetime import datetime, UTC

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src.benchmarks_core as C   # noqa: E402
import src.config as cfg          # noqa: E402

df_da = pd.read_feather("data/processed/history_da.feather")
df_im = pd.read_feather("data/processed/history_im.feather")
df_fx = pd.read_feather("data/processed/fx_daily.feather")
df_fx_uah = pd.read_feather("data/processed/fx_uah.feather")

dates = pd.date_range(datetime(2020, 1, 1, tzinfo=UTC),
                      # datetime(2025, 12, 31, hour=23, minute=45, tzinfo=UTC),
                      datetime(2026, 8, 31, hour=23, minute=45, tzinfo=UTC),
                      freq='15min')

df_fx = pd.concat([df_fx, df_fx_uah]).reset_index(drop=True)

# Germany relabel (2023): map IM codes to the DA bidding-zone codes.
df_im = C.apply_im_alias(df_im)

ON = ["delivery_ts", "area", "area_eic"]
yearly_list, last_list = [], []

# Monthly coverage should be higher than 90%
COVERAGE_EDGE = 0.9
for area in df_da["area"].unique():
# for area in ["BE"]:
    if area in ['GB', 'CY', 'BA', 'BG', 'IE_SEM', 'IT-SACOAC', 'IT-Rossano', 'IT-SACODC',
                'NO2NSL', 'UA_BEI', 'XK', 'UA_IPS', '']:
        continue

    da_a = df_da[df_da["area"] == area]
    im_a = df_im[df_im["area"] == area]
    if da_a.empty or im_a.empty:
        print(f"For {area}: len DA {len(da_a)}, len IM {len(im_a)}")
        continue

    df = pd.DataFrame(dates, columns=["delivery_ts"])

    m = C.build_matrix(da_a, im_a, df_fx, on=ON)
    m = C.add_pnl(m)
    actions = C.forecast_actions(m, cfg.n_days(area))
    m = C.apply_actions(m, actions)

    m = m.assign(year=m["delivery_ts"].dt.year)
    m = m.assign(month=m["delivery_ts"].dt.month)
    m = m.assign(mid=m[['price_im_pos_eur', 'price_im_neg_eur']].mean(axis=1))
    m = m.assign(gap=abs(m["price_da"] - m["mid"]))

    df = df.merge(m, on=['delivery_ts'], how='left')
    first_nans_in_area = df['area'].isna().cumprod().astype(bool)
    df = df[~first_nans_in_area]

    y_sum = df.groupby(["year", "month"], as_index=False).sum(numeric_only=True)
    y_count_total = df[['year', 'month', 'always_buy']].groupby(["year", "month"], as_index=False).count()
    y_count_total = y_count_total.rename(columns={"always_buy": "count_total"})
    df.dropna(subset=["area"], inplace=True)
    y_count = df[['year', 'month', 'always_buy']].groupby(["year", "month"], as_index=False).count()
    y_count = y_count.rename(columns={"always_buy": "count_real"})
    y_share = y_count_total.merge(y_count, on=['year', 'month'], how='left')
    y_share = y_share.assign(monthly_share=y_share['count_real'] / y_share['count_total'])


    m = m.merge(y_share, on=['year', 'month'], how='left')
    m = m[m.monthly_share >= COVERAGE_EDGE]
    y = m[['year', 'month', 'ideal', 'devil', 'always_buy', 'always_sell', 'stat_action_pnl']].groupby(by=["year", "month"], as_index=False).sum(numeric_only=True)
    y_prices = m.groupby(["year", "month"], as_index=False).agg(
        price_da=("price_da", "mean"), price_da_std=("price_da", "std"), gap_mean=("gap", "mean"))
    y_monthly = y.merge(y_prices, on=['year', 'month'], how='left')
    y_monthly = y_monthly.assign(da_volatility=y_monthly['price_da_std'] / y_monthly['price_da'])
    y_monthly = y_monthly.assign(gap_vs_price_da=y_monthly['gap_mean']/y_monthly['price_da'])
    y_monthly["area"] = area

    yearly_list.append(y_monthly)
    print(f"Area {area} is calculated")

yearly = pd.concat(yearly_list).reset_index(drop=True)
# last_years = pd.concat(last_list).reset_index(drop=True)
# last_years.to_csv('report/last.csv', index=False)

# ---------------------------------------------------------------------------
# Country averages for the three multi-zone families, added to cut the number
# of charts. MEAN, never sum: the mean of per-1-MW numbers is still per 1 MW,
# a sum would quietly turn it into per N MW and break comparability with every
# other row. The member zones stay in the table — nothing is replaced.
#
# A month enters the average only if EVERY member zone has it. Otherwise the
# average would step whenever a zone appears or drops out of the coverage
# filter, and that step would read as a market move.
#
# The four ratios are recomputed from the averaged inputs rather than averaged,
# so they mean the same thing they mean for a single zone.
# ---------------------------------------------------------------------------
GROUPS = {"DK": ["DK1", "DK2"],
          "NO": ["NO1", "NO2", "NO3", "NO4", "NO5"],
          "SE": ["SE1", "SE2", "SE3", "SE4"],
          "IT": ["IT-CNORTH", "IT-CSOUTH", "IT-Calabria", "IT-NORTH", "IT-SOUTH", "IT-Sardinia", "IT-Sicily"]}

group_list = []
for name, members in GROUPS.items():
    g = yearly[yearly["area"].isin(members)]
    keep = g.groupby(["year", "month"])["area"].count()
    keep = keep[keep == len(members)].index
    if keep.empty:
        print(f"Group {name}: no month is covered by all of {members}, skipped")
        continue
    g = g.set_index(["year", "month"]).loc[keep].reset_index()
    a = g.groupby(["year", "month"], as_index=False).mean(numeric_only=True)
    a = a.assign(gap_vs_price_da=a["gap_mean"] / a["price_da"])
    a["area"] = name
    group_list.append(a)
    print(f"Group {name} averaged over {members}: {len(a)} months")

yearly = pd.concat([yearly] + group_list).reset_index(drop=True)
yearly.drop(columns=["price_da_std"], inplace=True)

yearly.to_csv("report/yearly_update.csv", index=False)
yearly.to_feather("report/yearly_update.feather")

