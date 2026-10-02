"""
Period helper: turn a delivery DATE + market timezone into the
periodStart / periodEnd strings ENTSO-E expects (YYYYMMDDHHMM).

Why this exists:
  A market day is anchored to LOCAL midnight (a NL day runs local 00:00->00:00),
  but ENTSO-E interprets the periodStart/periodEnd string as UTC. So this helper
  is the single point where the local timezone (CET/CEST) enters: it takes the
  local-midnight bounds of a delivery day and emits their UTC instants as the
  YYYYMMDDHHMM strings the API actually wants. Everything downstream stays UTC.

Two subtleties handled here, both via zoneinfo (stdlib, no install):
  1. DST offset: a NL summer day is 22:00Z->22:00Z, a winter day 23:00Z->23:00Z.
     We never hardcode the offset - zoneinfo computes it from the actual date.
  2. DST transition days have 23 or 25 hours. Because we build both bounds as
     local-midnight and convert to UTC, the span is automatically 23/24/25h.

Why UTC and not local wall-clock in the string (learned the hard way):
  The period string is interpreted by the platform as UTC. Proof: requesting
  '202607260000' for NL returns an IM interval starting 2026-07-26T00:00Z, not
  the NL market midnight (22:00Z). If we emit local wall-clock '...0000', we are
  off by the DST offset: IM comes back shifted, and DA - which snaps to whole
  CET delivery days overlapping the UTC window - bleeds into an extra day.
  Emitting the UTC instant makes the window bound exactly one delivery day for
  BOTH A44 and A85.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

try:
    import src.config as cfg
except ImportError:          # run as a plain script from src/
    import config as cfg

_FMT = "%Y%m%d%H%M"  # ENTSO-E period format, interpreted by the platform as UTC


def market_tz(area: str) -> ZoneInfo:
    """
    The market's IANA timezone, by AREA code ("NL"), not by EIC.

    Zones used to be identified here by their EIC — unreadable in logs, and a
    second place to remember when adding a zone. The registry now lives in
    settings.json; this is only the lookup.
    """
    return ZoneInfo(cfg.zone_tz_name(area))


def auction_gate(delivery_day: date) -> datetime:
    """
    When the day-ahead auction for `delivery_day` closes, as a UTC instant.

    ONE auction, ONE gate: 12:00 Europe/Brussels on the day before delivery
    (settings -> day_ahead), which is 10:00 UTC in summer and 11:00 in winter.
    The two numbers are one rule seen twice, so neither is written down — but
    the CLOCK the rule is read on belongs to the auction, not to the zone.

    That distinction is what this function got wrong until 2026-09-20. It took
    noon in the ZONE's own timezone, which is right for NL by coincidence — NL
    is CET — and wrong for everyone else. For RO it produced 12:00 EEST = 09:00
    UTC, while the real gate is 12:00 CET = 13:00 EEST = 10:00 UTC (OPCOM
    publishes the Romanian DAM gate as 12:00 CET / 13:00 EEST). The decision run
    at 09:40 UTC would then have sat forty minutes past an imaginary gate, and
    RO would have published `<no_forecast reason="after_gate"/>` every single
    day, with the source and the data perfectly healthy. The gate MOVES in local
    time from zone to zone; the auction does not move at all.

    It matters because a decision issued after this instant could not have been
    placed. The numbers behind it stay causal — they only use data through the
    previous day — but the benchmark claims to measure what a mechanical rule
    would have EARNED on the market, and a bid nobody could submit earns
    nothing.
    """
    h, m, tzname = cfg.day_ahead_gate()
    local = datetime.combine(delivery_day - timedelta(days=1), time(h, m),
                             tzinfo=ZoneInfo(tzname))
    return local.astimezone(ZoneInfo("UTC"))


def market_day_bounds(delivery_day: date, area: str) -> tuple[datetime, datetime]:
    """
    The [start, end) UTC INSTANTS of one market day, as tz-aware datetimes.

    Same maths as market_day_period, but returns objects instead of API strings,
    so callers that need a delivery grid (the workflow) do not go through text.
    The span is automatically 23/24/25 h on DST days.
    """
    tz = market_tz(area)

    # Local midnight of the delivery day, and local midnight of the next day.
    start_local = datetime(delivery_day.year, delivery_day.month,
                           delivery_day.day, 0, 0, tzinfo=tz)
    next_day = delivery_day + timedelta(days=1)
    end_local = datetime(next_day.year, next_day.month,
                         next_day.day, 0, 0, tzinfo=tz)

    utc = ZoneInfo("UTC")
    return start_local.astimezone(utc), end_local.astimezone(utc)


def market_day_period(delivery_day: date, area: str) -> tuple[str, str]:
    """
    Return (periodStart, periodEnd) covering one full market day.

    delivery_day : the calendar date of delivery in the MARKET's local timezone
    area         : our zone code, e.g. 'NL' (timezone from settings)

    The bounds are local midnight -> next local midnight, converted to their
    UTC instants and formatted as the string ENTSO-E interprets as UTC. On DST
    days the real span is 23 or 25 hours, handled automatically by the tz math.

    The platform reads the string as UTC, so we emit the UTC instant of each
    local-midnight bound (summer 22:00Z, winter 23:00Z). This makes the window
    bound exactly one delivery day for both A44 and A85.
    """
    start_utc, end_utc = market_day_bounds(delivery_day, area)
    return start_utc.strftime(_FMT), end_utc.strftime(_FMT)


def market_range_period(start_day: date, end_day: date, area: str) -> tuple[str, str]:
    """
    Return (periodStart, periodEnd) covering [start_day 00:00 local,
    end_day+1 00:00 local) - i.e. inclusive of end_day. Useful for backfill.
    """
    ps, _ = market_day_period(start_day, area)
    _, pe = market_day_period(end_day, area)
    return ps, pe


# --- quick self-check -------------------------------------------------------
# Demonstrates the DST shift: a summer day vs a winter day for NL.
# Run: uv run python period.py
if __name__ == "__main__":
    area = "NL"

    summer = date(2026, 7, 25)   # CEST, UTC+2
    winter = date(2026, 1, 15)   # CET,  UTC+1
    mar_dst = date(2026, 3, 29)  # spring-forward day (23h)
    oct_dst = date(2026, 10, 25) # fall-back day (25h)

    for d in (summer, winter, mar_dst, oct_dst):
        ps, pe = market_day_period(d, area)
        # ps/pe are already UTC instants now, so parse them straight as UTC.
        utc = ZoneInfo("UTC")
        s_utc = datetime.strptime(ps, _FMT).replace(tzinfo=utc)
        e_utc = datetime.strptime(pe, _FMT).replace(tzinfo=utc)
        span_h = (e_utc - s_utc).total_seconds() / 3600
        print(f"{d}: periodStart={ps} periodEnd={pe} | "
              f"UTC {s_utc:%H:%MZ}->{e_utc:%H:%MZ} span={span_h:.0f}h")