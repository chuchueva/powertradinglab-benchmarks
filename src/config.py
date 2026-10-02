"""
Single source of truth for settings. Values live in settings.json (edit there,
no code changes); this module loads them and exposes typed accessors.

Why json + this loader: settings.json is pure data — safe to tweak anytime
(run time, area list, n_days) without risking the code; config.py is the one
place that reads it. Region-dependent params (e.g. stat_action n_days) are a
default + per-area overrides, so `3 -> 4 -> 5` is a one-line json edit and a
per-area value is just an entry in `n_days_by_area`.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]      # repo root (parent of src/)
SETTINGS_PATH = _ROOT / "settings.json"

with open(SETTINGS_PATH, encoding="utf-8") as _f:
    SETTINGS: dict = json.load(_f)

RUN_TIME_UTC: str = SETTINGS["run_time_utc"]      # "HH:MM", pre-auction (09:30)
AREAS: list[str] = SETTINGS["areas"]
PATHS: dict = SETTINGS["paths"]

_STAT = SETTINGS["stat_action"]
_N_DAYS_DEFAULT: int = _STAT["n_days_trailing_bias_default"]
_N_DAYS_BY_AREA: dict = _STAT["n_days_by_area"]


_ZONES: dict = SETTINGS.get("zones", {})


def zone(area: str) -> dict:
    """
    Everything that identifies one bidding zone: its EIC and its market timezone.

    ONE registry, because these three facts always travel together and used to be
    scattered: the EIC lived in two modules as a constant, the timezone in a third
    keyed BY the EIC. Adding a zone meant remembering all three.

    The AREA CODE is the project's identifier — short, readable, what appears in
    logs, filenames and published files. The EIC is an API detail and belongs only
    where a request is built.
    """
    if area not in _ZONES:
        raise KeyError(f"zone {area!r} is not in settings.json -> zones "
                       f"(known: {', '.join(sorted(_ZONES))})")
    return _ZONES[area]


ZONE_MODES = ("shadow", "live")


def zone_mode(area: str) -> str:
    """
    How much of a zone is real: "shadow" or "live".

    A zone joins by POLLING first. In shadow it runs the whole chain — poll,
    store, forecast, score — and stores nothing and publishes nothing: the
    results are computed and logged, exactly as a hand-run with an explicit
    current_ts already behaves. Several days of that answer the questions no
    amount of reasoning can (when does this TSO publish imbalance, does the
    input gate open, does the rate arrive in time), and any code still shaped
    like the first zone fails in the log instead of in a published file.
    Promotion is one word in settings; the first run after it writes.

    There is NO DEFAULT. Adding a zone means saying what it is for — and
    defaulting either way is wrong in a different direction: "live" publishes a
    zone nobody has watched yet, "shadow" would quietly stop publishing a zone
    that was already live if the field went missing.
    """
    m = zone(area).get("mode")
    if m is None:
        raise KeyError(
            f"zone {area} has no \"mode\" in settings.json -> zones. "
            f"Say \"shadow\" (runs and is logged, stores and publishes nothing) "
            f"or \"live\" (part of the published benchmark).")
    if m not in ZONE_MODES:
        raise ValueError(f"zone {area}: mode {m!r} is not one of {ZONE_MODES}")
    return m


def expected_currency(area: str, leg: str) -> str | None:
    """
    What a zone is EXPECTED to trade in on one leg ("DA" or "IM"), or None.

    An expectation, not a rule. The currency of a price is read from the
    document and carried in the row (rule 7); this only lets us notice when the
    document says something else. A currency is permanent until it is not — HRK
    ended on 2022-12-31 and no setting would have warned — so a mismatch is a
    WARNING and a flag in status.xml, never a stop. The day a zone joins the
    euro, the data keeps flowing and the system tells us what happened.
    """
    return zone(area).get("expect_currency", {}).get(leg)


def eic(area: str) -> str:
    """The zone's EIC code, for building an API request and nothing else."""
    return str(zone(area)["eic"])


def zone_tz_name(area: str) -> str:
    """IANA timezone of the market. The only place local time enters, via period."""
    return str(zone(area)["tz"])


_UPDATE = SETTINGS.get("update", {})
_SCORING = SETTINGS.get("scoring", {})
_PUBLICATION = SETTINGS.get("publication", {})


def versions() -> dict:
    """
    The two version numbers stamped into every published file.

    `schema` describes the SHAPE of the file: bump it when a parser written for
    the previous version could break — an element renamed, removed, or moved.
    Adding a sixth benchmark to the family counts: the numbers already published
    do not change, but the set of elements does.

    `methodology` describes the NUMBERS: bump it when the same input could now
    produce a different output — a changed benchmark definition, causality rule,
    or scoring window.

    Kept apart because they have different readers. Whoever wrote a parser cares
    about the first; whoever compares March against September cares about the
    second. One combined number would force a recalculation notice over a purely
    cosmetic change to the markup.

    Both are raised BY HAND, deliberately. That is also why every published file
    carries the parameters that actually drive the numbers (n_days,
    final_day_of_month, run_time_utc) beside the version: if the bump is ever
    forgotten — and eventually it will be — the difference still shows up in the
    parameters. The version is for quick comparison, the parameters are the proof.
    """
    return {"schema": int(_PUBLICATION.get("schema_version", 1)),
            "methodology": int(_PUBLICATION.get("methodology_version", 1))}


_BACKUP = SETTINGS.get("backup", {})


def backup_mode() -> str:
    """"git" (push a snapshot), "dir" (dated folders), or empty to disable."""
    return str(_BACKUP.get("mode", "")).strip()


def backup_repo() -> str:
    """The PRIVATE repository the snapshot is pushed to. Private is not a
    preference: data/raw carries ENTSO-E data we may not redistribute."""
    return str(_BACKUP.get("repo", "")).strip()


def backup_work_dir() -> str:
    """The checkout the snapshot is assembled in — OUTSIDE this repository, so
    one git tree never ends up inside another."""
    return str(_BACKUP.get("work_dir", "~/ptl-backup")).strip()


def backup_key() -> str:
    """The deploy key with write access to the backup repository only."""
    return str(_BACKUP.get("key", "")).strip()


def backup_paths() -> list:
    """What goes in. The reasoning for each path is in src/backup.py."""
    return list(_BACKUP.get("paths", []))


def backup_dir() -> str:
    """Where the off-machine copy goes — normally a sync folder (a cloud drive,
    iCloud). Empty disables the step; a missing backup must never be a reason to
    lose a day of data collection."""
    return str(_BACKUP.get("dir", "")).strip()


def backup_keep_days() -> int:
    """How many dated snapshots to keep. Dated rather than overwritten, so that a
    corrupted store cannot quietly replace the last good copy."""
    return int(_BACKUP.get("keep_days", 14))

_SITE = SETTINGS.get("site", {})


def site_repo() -> str:
    """The PUBLIC site repository the data is pushed to. Empty disables the step."""
    return str(_SITE.get("repo", "")).strip()


def site_work_dir() -> str:
    """The server's checkout of the site repository — outside this one, beside
    the backup checkout, so one git tree never ends up inside another."""
    return str(_SITE.get("work_dir", "~/ptl-site")).strip()


def site_key() -> str:
    """The deploy key with write access to the site repository only."""
    return str(_SITE.get("key", "")).strip()


def site_data_dir() -> str:
    """The one folder of the site repository this service owns. It is deleted
    and rebuilt on every push; nothing else in that repository is touched."""
    return str(_SITE.get("data_dir", "data")).strip()



def publisher() -> dict:
    """
    Who publishes the file. Kept OUT of the structure on purpose.

    The root element name and the element names are the format; they are frozen
    the moment anything is published, and renaming one breaks every parser that
    reads it. A project name and a domain are not that kind of thing — they get
    rebranded, they move. So identity lives in an element whose CONTENT can
    change freely, and the structure never has to.
    """
    d = {"name": str(_PUBLICATION.get("publisher_name", "TradingLab"))}
    url = str(_PUBLICATION.get("publisher_url", "")).strip()
    if url:
        d["url"] = url
    return d


def public_from() -> date | None:
    """
    The first delivery day the PUBLIC interface shows.

    Not the same thing as `scoring.score_from`, though today they are the same
    date. The scoring floor decides what the store puts forward for grading,
    and days before it are still graded when an action was issued for them —
    36 NL days wait for their as-final pass on that rule. This floor decides
    what a reader is shown. Everything before it is a working draft: it stays
    in outputs/ and in the private backup, and never reaches outputs/v1 or the
    site. Moving one date must not move the other, hence two keys.

    It changes no published number, so it is not in publication_parameters()
    and it is not a version.
    """
    v = str(_PUBLICATION.get("public_from", "")).strip()
    return date.fromisoformat(v) if v else None


def publication_parameters() -> dict:
    """Everything in settings that can move a published number, in one dict —
    for the header of every published file. Extend this when a new setting
    starts influencing results."""
    return {"n_days_trailing_bias_default": _N_DAYS_DEFAULT,
            "final_day_of_month": final_day_of_month(),
            "run_time_utc": RUN_TIME_UTC,
            "causality": "as_of"}


def day_ahead_gate() -> tuple[int, int, str]:
    """
    When the day-ahead auction closes: (hour, minute, timezone name).

    A property of the AUCTION, not of a zone. Day-ahead in the coupled European
    market is ONE auction with one gate — 12:00 Europe/Brussels on the day
    before delivery — and every zone in it sits behind that same instant,
    whatever its own clock reads at the time. A zone that ever trades outside
    this auction gets a registry of auctions; it does not get a gate of its own
    derived from its timezone, which is the mistake this replaces.
    """
    g = SETTINGS.get("day_ahead", {})
    hhmm = str(g.get("gate_local", "12:00"))
    h, m = (int(x) for x in hhmm.split(":"))
    return h, m, str(g.get("gate_tz", "Europe/Brussels"))


def final_day_of_month() -> int:
    """Day of the month FOLLOWING delivery after which imbalance prices are
    taken as final. Market practice puts the last revision at the 15th; we wait
    until the 20th, because being a few days late costs nothing and catching a
    still-moving value costs a wrong published number."""
    return int(_SCORING.get("final_day_of_month", 20))


def score_from() -> date | None:
    """
    The first delivery day the STORE may put forward for scoring on its own.

    Four of the five benchmarks — ideal, devil, always_buy, always_sell — are
    properties of the day's prices and need no decision of ours, so a settled
    day can be graded on them whether or not a forecast was issued for it. What
    that cannot mean is "grade everything in the store": the NL slice is seeded
    back to 2020 and the first run would write six years of history in one pass,
    dated today. Hence a floor, set once when the live series began.

    Days BEFORE the floor are not lost — any day an action was issued for is
    scored regardless, which is what keeps the existing series and its pending
    as-final passes intact.
    """
    v = str(_SCORING.get("score_from", "")).strip()
    return date.fromisoformat(v) if v else None


def n_days(area: str) -> int:
    """stat_action trailing window for a zone (per-area override, else default)."""
    return int(_N_DAYS_BY_AREA.get(area, _N_DAYS_DEFAULT))


def history_months() -> tuple[str, str]:
    """
    Inclusive "YYYY_MM" bounds of the File Library archive build.

    `month_from` is where the methodology starts, and it is a decision, not a
    property of the download: the platform's full-history bundles reach back to
    2013/2014, and everything before 2020 is deliberately cut.

    `month_to` empty means the last CLOSED month, computed rather than written
    down so it cannot go stale. History is made of finished months. A three-day
    tail of the month in progress adds nothing to an annual table and actively
    harms the thing dated snapshots are FOR: the next download would carry a
    different partial month, and a diff meant to reveal backfills would show
    that instead.
    """
    h = SETTINGS.get("history", {})
    m_from = str(h.get("month_from", "2020_01"))
    m_to = str(h.get("month_to") or "")
    if not m_to:
        # UTC, like every other date in this project: on the 1st of a
        # month a laptop in UTC+4 and the VM would otherwise disagree about
        # which month was the last closed one.
        today = datetime.now(timezone.utc).date()
        m_to = f"{today.replace(day=1) - timedelta(days=1):%Y_%m}"
    return m_from, m_to


def catchup_time_utc() -> str:
    """
    "HH:MM" UTC of the second run of the day, which deliberately does NOT
    forecast.

    It exists because data often arrives after the decision run — late
    imbalance, a source outage that clears by lunchtime — and everything except
    the decision benefits from a second look: the store gets the day, settled
    days get scored, the files get published, status stops being stale.

    The decision is excluded on purpose. This slot is hours past the auction
    gate, and a forecast issued then could not have been placed. Recovering a
    forecast has exactly one window, from the decision run to gate closure, and
    nothing later will do.
    """
    return str(SETTINGS.get("catchup_time_utc", "14:00"))


def update_window_days() -> int:
    """How many delivery days back each poll re-requests (catches late arrivals
    and revisions without a separate 'backfill' code path)."""
    return int(_UPDATE.get("window_days", 3))


def max_query_days() -> int:
    """
    The longest range ONE request to the source may cover, in delivery days.
    0 means no limit.

    ENTSO-E imposed a temporary 30-day cap on 2026-09-03, during the platform's
    stabilisation after the migration. The three-day daily poll never notices
    it; the monthly re-fetch of a 31-day month does.

    Kept as a number here rather than as a branch on "is this the monthly pass",
    because the cap is a property of the SOURCE, not of one caller. The next
    outage leaves a forty-day hole, and the backfill that repairs it meets the
    same wall at a far worse moment. When the limit is lifted, set this to 0 and
    nothing else has to change.
    """
    return int(_UPDATE.get("max_query_days", 30))


def refetch_time_utc() -> str:
    """When the monthly re-request runs, on `final_day_of_month`. Deliberately
    NOT a separate day setting: the re-fetch exists to feed the as-final scoring
    pass, so its date is that rule's date and cannot drift away from it. Keep it
    earlier than `run_time_utc` — the ordering is guaranteed by nothing but the
    clock."""
    return str(_UPDATE.get("refetch_time_utc", "08:00"))


def poll_times_utc() -> list[str]:
    """
    "HH:MM" UTC times to poll at.

    Started at five a morning to MEASURE when imbalance appears. That question is
    answered — five days running it was complete by 05:30, long before the
    decision — so the schedule is down to one, at the decision itself. What the
    extra polls also provided by accident was redundancy against a bad moment at
    the source, and that is now provided on purpose by `retry_attempts`.
    """
    return list(_UPDATE.get("poll_times_utc", [RUN_TIME_UTC]))


def retry_attempts() -> int:
    """How many times a poll may try before the run goes on without fresh data.
    Four attempts five minutes apart covers a 15-minute wobble and still lands
    ahead of the 10:00 UTC summer gate."""
    return int(_UPDATE.get("retry_attempts", 4))


def retry_delay_s() -> int:
    return int(_UPDATE.get("retry_delay_s", 300))


def scheduled_times_utc() -> list[str]:
    """
    EVERY time something is scheduled — polls, the decision, the monthly
    re-fetch. Used for labelling a run with the slot it belongs to.

    Not the same list as poll_times_utc, and the difference matters: with one
    poll a day, a monthly re-fetch at 08:00 would otherwise be matched against
    yesterday's 09:30 and reported as twenty-two hours late.
    """
    return sorted({*poll_times_utc(), RUN_TIME_UTC, refetch_time_utc(),
                   catchup_time_utc()})


def path(key: str) -> str:
    """A configured folder path by key (e.g. 'zone_store')."""
    return PATHS[key]
