"""
File Library (bulk CSV) parsers for BACKFILL -> clean pandas DataFrames.

ENTSO-E's File Library ships long history as monthly ZIPs; each inner CSV is one
tab-separated extract holding ALL areas (so adding a zone later needs no
re-download - just re-filter). This module turns those extracts into the SAME
schema our live API parsers produce, so backfill and live data stack cleanly.

Two products, two layouts:

  Day-ahead (parse_filelibrary_da):
      cols: InstanceCode DateTime(UTC) ResolutionCode AreaCode ... ContractType
            Sequence Price[Currency/MWh] Currency UpdateTime(UTC)
      -> delivery_ts, knownby_ts, price_da

  Imbalance (parse_filelibrary_im):
      cols: DateTime(UTC) ResolutionCode AreaCode ...
            PositiveImbalancePrice[..] (+Scarcity/Incentive/FinancialNeutrality)
            NegativeImbalancePrice[..] (+ same components) Currency Status
            UpdateTime(UTC)
      -> delivery_ts, knownby_ts, price_im_pos, price_im_neg
      We keep the source's own Positive/Negative labels verbatim (no long/short
      re-interpretation): price_im_pos = PositiveImbalancePrice, price_im_neg =
      NegativeImbalancePrice. Store both, never average. The Scarcity / Incentive
      / FinancialNeutrality columns are price components we do not need here.

      Pos/Neg is the CANONICAL naming for this project (BE and File Library use it).
      Its correspondence to the API A85 long/short (A04/A05) is now SETTLED, and it
      is the intuitive one:
          PositiveImbalancePrice == A04 (long / surplus)
          NegativeImbalancePrice == A05 (short / deficit)
      Cross-checked row for row on NL 2026-07-25 (saved API zip vs the monthly
      extract): 40 of the 96 PTU were dual-priced, and every one matched exactly.
      An earlier note here guessed the opposite from the ordering of the two
      prices — that guess was wrong; the ordering varies by PTU and says nothing
      about the labels. Do not "fix" this back: because the benchmark takes
      max/min of the pair, swapping the columns would corrupt the data without
      moving a single published number.

Shared rules (both):
  - DateTime(UTC) is already UTC, so no CET conversion. delivery_ts = it, in UTC.
  - knownby_ts = UpdateTime(UTC). Treat it as PROVENANCE, not as publication time:
    the monthly extracts are rewritten wholesale after the fact, so the stamp
    belongs to the file rather than to the row. Observed on NL 2026-07-25, whose
    UpdateTime is 2026-07-01 22:50 — three weeks BEFORE the delivery it describes.
    So it cannot serve as an as-known/as-final marker; those come from our own
    polling (see update), where knownby_ts is the instant of the request.
  - Resolution changed to 15-min on 2025-10-01. Older rows are PT60M and are
    UPSAMPLED to the 15-min grid (PT30M too; PT15M passes through) to match A44/A85.
  - De-dup: if a delivery_ts appears more than once for the zone, keep the row
    with the LATEST UpdateTime (the most-final value we hold).
  - Filter by AreaCode (EIC) so a multi-area file yields one zone.
"""

import csv
import re
import zipfile
from dataclasses import dataclass, field

import pandas as pd

_RESOLUTION_MINUTES = {"PT15M": 15, "PT30M": 30, "PT60M": 60}
_GRID_MINUTES = 15  # target MTU grid

_DA_PRICE_COL = "Price[Currency/MWh]"
_IM_POS_COL = "PositiveImbalancePrice[Currency/MWh]"
_IM_NEG_COL = "NegativeImbalancePrice[Currency/MWh]"
_LOAD_COL = "TotalLoad[MW]"
# 14.1.D publishes three forecast vintages per MTU. We take the DAY-AHEAD one,
# for two reasons that happen to agree. It is the most complete (98.7% of rows
# in 2026-08, against 96.0% intraday and 90.2% current), and it is the only one
# that was knowable BEFORE the day-ahead auction closed — so "the share of
# renewables the market was pricing against" is a driver of the price, while a
# later revision is partly a consequence of it.
_VRE_COL = "DayAheadGenerationForecast[MW]"


@dataclass
class ParseResult:
    df: pd.DataFrame
    meta: dict = field(default_factory=dict)


def _frame_from_text(text: str) -> pd.DataFrame:
    """
    Turn the raw tab-separated extract text into a DataFrame of strings,
    tolerating the whole-line double-quote wrapping some exports carry.
    """
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        if len(ln) >= 2 and ln.startswith('"') and ln.endswith('"'):
            ln = ln[1:-1]
        if ln.strip():
            cleaned.append(ln)
    rows = list(csv.reader(cleaned, delimiter="\t"))
    header, data = rows[0], rows[1:]
    return pd.DataFrame(data, columns=[h.strip() for h in header])


def _read_rows(path: str) -> pd.DataFrame:
    """Read a plain CSV/TSV extract file into a string DataFrame."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return _frame_from_text(f.read())


# A File Library member is always named for the month it holds:
# "2020_01_EnergyPrices_12.1.D_r3.1.csv".
_MEMBER_MONTH = re.compile(r"(\d{4}_\d{2})_")


def member_month(name: str) -> str | None:
    """The "YYYY_MM" a member belongs to, from its own filename."""
    m = _MEMBER_MONTH.match(name.rsplit("/", 1)[-1])
    return m.group(1) if m else None


def iter_zip_frames(zip_path: str, month_from: str | None = None,
                    month_to: str | None = None):
    """
    Yield (member name, string DataFrame) for every monthly CSV in a File
    Library ZIP, oldest month first. Unzipped in memory; nothing hits disk.

    ONE shape covers both kinds of download, which is why there is no branch
    here. A per-month zip holds a single CSV; the platform's "download all"
    bundle holds the whole history as a FLAT list of exactly those files — same
    names, same header, same bytes (checked on 2020_01 for both products). So a
    monthly zip is simply a bundle of one, and the only thing that ever differs
    is how many members there are.

    `month_from` / `month_to` are inclusive "YYYY_MM" bounds matched against the
    member's own name. Filtering by NAME rather than after parsing means a month
    outside the window is never decompressed: of the 143 members in the DA
    bundle, 62 are then never touched at all. A member whose name carries no
    month is kept — it cannot be judged, and dropping it silently would be worse
    than parsing it.
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"No CSV inside {zip_path} (members: {zf.namelist()})")
        for name in sorted(members):
            month = member_month(name)
            if month is not None:
                if month_from and month < month_from:
                    continue
                if month_to and month > month_to:
                    continue
            yield name, _frame_from_text(zf.read(name).decode("utf-8-sig"))


def read_zip_frame(zip_path: str) -> pd.DataFrame:
    """
    The first (or only) CSV inside a File Library ZIP, as a string DataFrame.

    Kept for the single-month case; anything walking an archive wants
    `iter_zip_frames`, which is the same read without the assumption that a zip
    holds exactly one month.
    """
    for _, frame in iter_zip_frames(zip_path):
        return frame
    raise ValueError(f"No CSV inside {zip_path}")


def _to_grid(df: pd.DataFrame, value_cols: list[str],
             group_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Common finishing step: type the timestamps/prices, de-dup keeping the latest
    UpdateTime, and upsample every native interval onto the 15-min grid, carrying
    value_cols (and any group_cols, e.g. 'area') through.

    Expects df with: DateTime(UTC), UpdateTime(UTC), ResolutionCode, value_cols,
    and any group_cols. De-dup is per (group_cols + native_ts) so multiple zones
    in one file do not collapse into each other.
    """
    group_cols = group_cols or []
    df = df.copy()
    df["native_ts"] = pd.to_datetime(df["DateTime(UTC)"], utc=True)
    df["knownby_ts"] = pd.to_datetime(df["UpdateTime(UTC)"], utc=True)
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad = sorted(set(df["ResolutionCode"]) - set(_RESOLUTION_MINUTES))
    if bad:
        raise ValueError(f"Unexpected ResolutionCode(s): {bad}")

    # One row per (group..., native_ts): keep the latest UpdateTime.
    df = df.sort_values("knownby_ts").drop_duplicates(
        [*group_cols, "native_ts"], keep="last")

    # Upsample each native interval to the 15-min grid, vectorised so column
    # names with brackets (e.g. 'Price[Currency/MWh]') are not a problem.
    df = df.reset_index(drop=True)
    n_sub = (df["ResolutionCode"].map(_RESOLUTION_MINUTES) // _GRID_MINUTES).astype(int)
    rep = df.loc[df.index.repeat(n_sub)].copy()
    rep["_k"] = rep.groupby(level=0).cumcount()
    rep["delivery_ts"] = rep["native_ts"] + pd.to_timedelta(
        rep["_k"] * _GRID_MINUTES, unit="m")

    out = rep[["delivery_ts", "knownby_ts", *group_cols, *value_cols]].copy()
    return out.sort_values([*group_cols, "delivery_ts"]).reset_index(drop=True)


def _filter_zone(raw: pd.DataFrame, zone: str) -> pd.DataFrame:
    return raw[raw["AreaCode"] == zone].copy()


def parse_da_frame(raw: pd.DataFrame, zones: list[str] | None = None,
                   contract: str = "Day-ahead") -> ParseResult:
    """
    Parse an already-read DA extract (all areas) ->
        delivery_ts, knownby_ts, area, currency, price_da.

    zones=None keeps ALL areas; otherwise filters AreaCode to the given list.
    Used by build_history.py to fold every monthly zip into one table.

    DA is NOT always EUR: most zones publish in EUR (SDAC), but some non-euro
    zones publish DA in local currency (e.g. UA_IPS in UAH). We keep the native
    `Currency` column (never overwrite with EUR); the EUR conversion is a derived
    step in the benchmark, symmetric with IM.
    """
    n_total = len(raw)
    df = raw[raw["ContractType"] == contract].copy()
    if zones is not None:
        df = df[df["AreaCode"].isin(zones)]
    # area = human-readable short code (NL, BE, DE_LU, SE3, IT-NORTH...);
    # area_eic = the canonical EIC (join key / traceability).
    df = df.rename(columns={"AreaCode": "area_eic", "AreaMapCode": "area"})
    n_kept = len(df)

    out_cols = ["delivery_ts", "knownby_ts", "area", "area_eic", "currency", "price_da"]
    if df.empty:
        return ParseResult(df=pd.DataFrame(columns=out_cols),
                           meta={"product": "DA", "n_rows_file": n_total,
                                 "n_rows_kept": 0, "rows_out": 0})

    # Carry Currency through the grid via group_cols (it is constant per area/day).
    grid = _to_grid(df, [_DA_PRICE_COL], group_cols=["area_eic", "area", "Currency"])
    grid = grid.rename(columns={_DA_PRICE_COL: "price_da",
                                "Currency": "currency"})[out_cols]

    meta = {
        "product": "DA", "n_rows_file": n_total, "n_rows_kept": n_kept,
        "n_areas": grid["area_eic"].nunique(),
        "resolutions_seen": sorted(set(df["ResolutionCode"])),
        "rows_out": len(grid),
    }
    return ParseResult(df=grid, meta=meta)


def parse_im_frame(raw: pd.DataFrame, zones: list[str] | None = None) -> ParseResult:
    """
    Parse an already-read IM extract (all areas) ->
        delivery_ts, knownby_ts, area, currency, price_im_pos, price_im_neg.

    zones=None keeps ALL areas; otherwise filters AreaCode to the given list.
    Positive/Negative labels are carried verbatim (canonical for the project);
    currency is kept native (some non-euro zones publish IM in local currency),
    EUR conversion is a derived step done later against the ECB FX table.
    Used by build_history.py to fold every monthly zip into one table.
    """
    n_total = len(raw)
    df = raw.copy()
    if zones is not None:
        df = df[df["AreaCode"].isin(zones)]
    # area = human-readable short code; area_eic = canonical EIC.
    df = df.rename(columns={"AreaCode": "area_eic", "AreaMapCode": "area",
                            "Currency": "currency"})
    n_kept = len(df)

    out_cols = ["delivery_ts", "knownby_ts", "area", "area_eic", "currency",
                "price_im_pos", "price_im_neg"]
    if df.empty:
        return ParseResult(df=pd.DataFrame(columns=out_cols),
                           meta={"product": "IM", "n_rows_file": n_total,
                                 "n_rows_kept": 0, "rows_out": 0})

    grid = _to_grid(df, [_IM_POS_COL, _IM_NEG_COL],
                    group_cols=["area_eic", "area", "currency"])
    grid = grid.rename(columns={_IM_POS_COL: "price_im_pos",
                                _IM_NEG_COL: "price_im_neg"})[out_cols]

    meta = {
        "product": "IM", "n_rows_file": n_total, "n_rows_kept": n_kept,
        "n_areas": grid["area_eic"].nunique(),
        "currencies_seen": sorted(set(grid["currency"].dropna())),
        "resolutions_seen": sorted(set(df["ResolutionCode"])),
        "rows_out": len(grid),
    }
    return ParseResult(df=grid, meta=meta)


def parse_load_frame(raw: pd.DataFrame, zones: list[str] | None = None) -> ParseResult:
    """
    Parse an already-read Actual Total Load (6.1.A) extract (all areas) ->
        delivery_ts, knownby_ts, area, area_eic, area_type, load_mw.

    Load is a VOLUME (MW), no currency. Unlike prices, ENTSO-E publishes load at
    several aggregation levels (AreaTypeCode BZN / CTA / CTY and combinations), so
    the same country can appear more than once at different levels. We keep
    area_type so those levels stay distinguishable; area_eic is the identity.
    Used by build_history.py for the zone-selection cache.
    """
    n_total = len(raw)
    df = raw.copy()
    if zones is not None:
        df = df[df["AreaCode"].isin(zones)]
    df = df.rename(columns={"AreaCode": "area_eic", "AreaMapCode": "area",
                            "AreaTypeCode": "area_type"})
    n_kept = len(df)

    out_cols = ["delivery_ts", "knownby_ts", "area", "area_eic", "area_type", "load_mw"]
    if df.empty:
        return ParseResult(df=pd.DataFrame(columns=out_cols),
                           meta={"product": "LOAD", "n_rows_file": n_total,
                                 "n_rows_kept": 0, "rows_out": 0})

    grid = _to_grid(df, [_LOAD_COL], group_cols=["area_eic", "area", "area_type"])
    grid = grid.rename(columns={_LOAD_COL: "load_mw"})[out_cols]

    meta = {
        "product": "LOAD", "n_rows_file": n_total, "n_rows_kept": n_kept,
        "n_areas": grid["area_eic"].nunique(),
        "area_types_seen": sorted(set(grid["area_type"].dropna())),
        "resolutions_seen": sorted(set(df["ResolutionCode"])),
        "rows_out": len(grid),
    }
    return ParseResult(df=grid, meta=meta)


def parse_vre_frame(raw: pd.DataFrame, zones: list[str] | None = None) -> ParseResult:
    """
    Parse a Wind & Solar Generation Forecast (14.1.D) extract (all areas) ->
        delivery_ts, knownby_ts, area, area_eic, area_type, production_type, vre_mw

    Long format, one row per production type — Solar, Wind Onshore, Wind
    Offshore. They are NOT summed here: a zone that loses one series for a
    while would then look like it simply had less renewables, and the drop
    would be indistinguishable from a real one. Summing is a decision for the
    caller, which can also see what it is summing.

    `production_type` MUST be in the grouping key, and that is not cosmetic:
    the resolution is declared per (area, type), so one zone can publish solar
    at PT15M and wind at PT60M for the same instant (Austria did, in 2014).
    Grouped by area alone, those two collide in the de-dup and one is dropped.

    Aggregation levels behave exactly as in 6.1.A: AreaTypeCode carries BZN,
    CTA, CTY and their combinations, and we keep the level so a share against
    load is computed at a matching level rather than across two of them.
    """
    n_total = len(raw)
    df = raw.copy()
    if zones is not None:
        df = df[df["AreaCode"].isin(zones)]
    df = df.rename(columns={"AreaCode": "area_eic", "AreaMapCode": "area",
                            "AreaTypeCode": "area_type",
                            "ProductionType": "production_type"})
    n_kept = len(df)

    group_cols = ["area_eic", "area", "area_type", "production_type"]
    out_cols = ["delivery_ts", "knownby_ts", *group_cols, "vre_mw"]
    if df.empty:
        return ParseResult(df=pd.DataFrame(columns=out_cols),
                           meta={"product": "VRE", "n_rows_file": n_total,
                                 "n_rows_kept": 0, "rows_out": 0})

    grid = _to_grid(df, [_VRE_COL], group_cols=group_cols)
    grid = grid.rename(columns={_VRE_COL: "vre_mw"})[out_cols]

    meta = {
        "product": "VRE", "n_rows_file": n_total, "n_rows_kept": n_kept,
        "n_areas": grid["area_eic"].nunique(),
        "types_seen": sorted(set(grid["production_type"].dropna())),
        "area_types_seen": sorted(set(grid["area_type"].dropna())),
        "resolutions_seen": sorted(set(df["ResolutionCode"])),
        "rows_out": len(grid),
    }
    return ParseResult(df=grid, meta=meta)


def parse_filelibrary_da(path: str, zone: str, contract: str = "Day-ahead") -> ParseResult:
    """Parse a File Library DA extract for ONE zone -> delivery_ts/knownby_ts/price_da."""
    raw = _read_rows(path)
    n_total = len(raw)
    df = _filter_zone(raw, zone)
    df = df[df["ContractType"] == contract]
    n_zone = len(df)

    if df.empty:
        return ParseResult(
            df=pd.DataFrame(columns=["delivery_ts", "knownby_ts", "price_da"]),
            meta={"product": "DA", "zone": zone, "n_rows_file": n_total,
                  "n_rows_zone": 0, "rows_out": 0},
        )

    grid = _to_grid(df, [_DA_PRICE_COL])
    grid = grid.rename(columns={_DA_PRICE_COL: "price_da"})[
        ["delivery_ts", "knownby_ts", "price_da"]]

    meta = {
        "product": "DA", "zone": zone, "contract": contract,
        "n_rows_file": n_total, "n_rows_zone": n_zone,
        "resolutions_seen": sorted(set(df["ResolutionCode"])),
        "native_span": [df["DateTime(UTC)"].min(), df["DateTime(UTC)"].max()],
        "rows_out": len(grid),
    }
    return ParseResult(df=grid, meta=meta)


def parse_filelibrary_im(path: str, zone: str) -> ParseResult:
    """
    Parse a File Library IM extract for ONE zone ->
        delivery_ts, knownby_ts, price_im_pos, price_im_neg.

    Positive/Negative labels are carried through verbatim (see module docstring);
    their correspondence to the API long/short is confirmed by cross-check later.
    """
    raw = _read_rows(path)
    n_total = len(raw)
    df = _filter_zone(raw, zone)
    n_zone = len(df)

    # currency is kept verbatim: some non-euro zones publish IM in local currency,
    # and we store native + currency, converting to EUR only as a derived step.
    out_cols = ["delivery_ts", "knownby_ts", "currency",
                "price_im_pos", "price_im_neg"]
    if df.empty:
        return ParseResult(
            df=pd.DataFrame(columns=out_cols),
            meta={"product": "IM", "zone": zone, "n_rows_file": n_total,
                  "n_rows_zone": 0, "rows_out": 0},
        )

    df = df.rename(columns={"Currency": "currency"})
    grid = _to_grid(df, [_IM_POS_COL, _IM_NEG_COL], group_cols=["currency"])
    grid = grid.rename(columns={_IM_POS_COL: "price_im_pos",
                                _IM_NEG_COL: "price_im_neg"})[out_cols]

    meta = {
        "product": "IM", "zone": zone,
        "n_rows_file": n_total, "n_rows_zone": n_zone,
        "currencies_seen": sorted(set(df["currency"].dropna())),
        "resolutions_seen": sorted(set(df["ResolutionCode"])),
        "statuses_seen": sorted(set(df.get("Status", pd.Series(dtype=str)).fillna(""))),
        "native_span": [df["DateTime(UTC)"].min(), df["DateTime(UTC)"].max()],
        "rows_out": len(grid),
    }
    return ParseResult(df=grid, meta=meta)


# --- manual test run --------------------------------------------------------
# Usage: applied in build_history.py, run the latter

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4 or sys.argv[1] not in ("da", "im"):
        print("Usage: uv run python parse_filelibrary.py <da|im> <extract.csv> <zone_eic>")
        raise SystemExit(1)

    kind, path, zone = sys.argv[1], sys.argv[2], sys.argv[3]
    result = (parse_filelibrary_da if kind == "da" else parse_filelibrary_im)(path, zone)

    print("META:")
    for k, v in result.meta.items():
        print(f"  {k}: {v}")
    print("-" * 74)
    print(result.df.head(12).to_string(index=False))
    print(f"\nTotal rows: {len(result.df)}")
