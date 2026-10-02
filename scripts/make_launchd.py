"""
Generate the launchd jobs that run the NL workflow.

Why a generator instead of two hand-written plists: launchd's
StartCalendarInterval is in the machine's LOCAL time, while every schedule in
this project is UTC. Converting by hand is exactly the kind of silent one-hour
error the UTC-only rule exists to prevent, so the conversion is done here, once,
and printed so it can be checked.

Jobs, because the workflow has several cadences:
    ...nl.poll    - update only, at the non-decision poll times. This is the
                    measuring instrument for when imbalance appears.
    ...nl.daily   - update -> forecast -> score -> publish, at run_time_utc.
                    The ONE decision of the day, and it must finish before the
                    auction gate.
    ...nl.catchup - update -> score -> publish, at catchup_time_utc, with no
                    forecast. Late data still lands, settled days still get
                    scored; the decision is excluded because by then the auction
                    has closed.
They are separate so that the decision run is a single ordered chain — polling
and computing at the same minute as independent jobs would race, and the compute
could read the store before the poll had written to it.

DST warning: if the machine's timezone observes DST, the local times below shift
twice a year and the jobs drift by an hour against UTC. Yerevan (UTC+4) does not.
Re-run this and reload the jobs after any timezone change. A drift is visible in
the data rather than hidden — polls record their real fetch time, and a forecast
records its own current_ts — but the decision run must stay ahead of the 12:00
CET auction gate, so do check it.

Run (from repo root):
    uv run python scripts/make_launchd.py            # print the plists
    uv run python scripts/make_launchd.py --install  # write them to LaunchAgents
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, time, timezone
from pathlib import Path

LABEL = "com.tradinglab"
# The labels the jobs carried while there was one zone. They are gone from the
# configuration, but a plist left in LaunchAgents keeps firing on its own
# schedule and nothing says so — the failure this file already guards against
# for the poll job. Swept on every --install until they are all gone.
LEGACY_LABELS = [f"com.tradinglab.nl.{m}"
                 for m in ("poll", "daily", "catchup", "monthly")]
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))       # run as a script, not as a module

import src.config as cfg  # noqa: E402
AGENTS = Path.home() / "Library" / "LaunchAgents"


def _loaded() -> set[str]:
    """Our labels that launchd currently has loaded, straight from the horse."""
    r = subprocess.run(["launchctl", "list"], capture_output=True, text=True,
                       check=False)
    return {line.split()[-1] for line in r.stdout.splitlines()
            if line.split() and line.split()[-1].startswith(LABEL)}


def _verify(expected: set[str], removed: set[str]) -> None:
    """
    Check the END STATE rather than trusting the commands that produced it.

    `launchctl unload` returns non-zero for perfectly ordinary reasons ("not
    loaded"), so its exit code cannot be read strictly — which is how the removal
    of a job could fail silently and leave it firing on an old schedule with no
    plist on disk to show for it. Nothing would report that; it would surface
    days later as extra runs in the log, if anyone noticed. So: do the work, then
    ask launchd what is actually loaded.
    """
    loaded = _loaded()
    missing = sorted(expected - loaded)
    lingering = sorted(removed & loaded)
    for name in missing:
        print(f"  NOT LOADED: {name}")
    for name in lingering:
        print(f"  STILL LOADED although no longer configured: {name}")
        print(f"    remove it by label: launchctl bootout gui/$(id -u)/{name}")
    if missing or lingering:
        raise SystemExit(1)
    print(f"\nverified against launchctl: exactly the {len(loaded)} configured "
          "jobs are loaded")


def _to_local(hhmm: str) -> tuple[int, int, str]:
    """A "HH:MM" UTC schedule time -> (local hour, local minute, tz name)."""
    h, m = (int(x) for x in hhmm.split(":"))
    utc_today = datetime.combine(datetime.now(timezone.utc).date(),
                                 time(h, m), tzinfo=timezone.utc)
    local = utc_today.astimezone()
    return local.hour, local.minute, local.tzname() or "local"


def _plist(name: str, mode: str, times_utc: list[str],
           day_of_month: int | None = None) -> str:
    day = f"""
            <key>Day</key><integer>{day_of_month}</integer>""" if day_of_month else ""
    entries = "".join(
        f"""
        <dict>{day}
            <key>Hour</key><integer>{h}</integer>
            <key>Minute</key><integer>{m}</integer>
        </dict>""" for h, m, _ in map(_to_local, times_utc))
    utc_note = ", ".join(times_utc) + (f" on day {day_of_month}" if day_of_month else "")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- Times below are LOCAL. They correspond to {utc_note} UTC. -->
<plist version="1.0">
<dict>
    <key>Label</key><string>{name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>{REPO}/scripts/run.sh</string>
        <string>{mode}</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>{entries}
    </array>
    <key>StandardOutPath</key><string>{REPO}/logs/launchd_{mode}.log</string>
    <key>StandardErrorPath</key><string>{REPO}/logs/launchd_{mode}.log</string>
    <key>WorkingDirectory</key><string>{REPO}</string>
</dict>
</plist>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--install", action="store_true",
                   help="write to ~/Library/LaunchAgents and load the jobs")
    a = p.parse_args()

    run_utc = cfg.RUN_TIME_UTC
    catchup_utc = cfg.catchup_time_utc()
    poll_utc = [t for t in cfg.poll_times_utc() if t != run_utc]
    refetch_utc = cfg.refetch_time_utc()
    refetch_day = cfg.final_day_of_month()

    print(f"repo: {REPO}")
    print(f"machine timezone: {datetime.now().astimezone().tzname()}\n")
    print(f"{'UTC':>8}  {'local':>8}   job")
    names = {run_utc: "daily (update + forecast + score + publish)",
             catchup_utc: "catchup (update + score + publish, NO forecast)"}
    for t in sorted(poll_utc + [run_utc, catchup_utc]):
        h, m, tzname = _to_local(t)
        print(f"{t:>8}  {h:02d}:{m:02d} {tzname}   "
              f"{names.get(t, 'poll (update only)')}")
    h, m, tzname = _to_local(refetch_utc)
    print(f"{refetch_utc:>8}  {h:02d}:{m:02d} {tzname}   monthly, day {refetch_day} "
          f"(re-request last month, for the as-final pass)")
    if refetch_utc >= run_utc:
        print(f"\n  WARNING: the monthly re-fetch ({refetch_utc}) is not earlier than "
              f"the daily run ({run_utc}).\n  The as-final scoring would then read "
              f"data that has not been refreshed yet.")

    jobs = {f"{LABEL}.daily": ("daily", [run_utc], None),
            f"{LABEL}.catchup": ("catchup", [catchup_utc], None),
            f"{LABEL}.monthly": ("monthly", [refetch_utc], refetch_day)}
    if poll_utc:
        jobs[f"{LABEL}.poll"] = ("poll", poll_utc, None)
    else:
        # With a single poll time, and that time being the decision itself,
        # there is nothing left for a separate poll job to do. Any plist
        # installed by an earlier configuration must be actively removed —
        # otherwise it keeps firing on the old schedule, invisibly.
        print("\n  no separate poll job: the only poll time is the decision run")
    obsolete = [f"{LABEL}.{m}" for m in ("poll", "daily", "catchup", "monthly")
                if f"{LABEL}.{m}" not in jobs] + LEGACY_LABELS

    if not a.install:
        for name, (mode, times, day) in jobs.items():
            print(f"\n----- {name} -----\n{_plist(name, mode, times, day)}")
        print("Re-run with --install to write and load these.")
        return

    os.chmod(REPO / "scripts" / "run.sh", 0o755)
    AGENTS.mkdir(parents=True, exist_ok=True)

    for name in obsolete:
        path = AGENTS / f"{name}.plist"
        if path.exists():
            r = subprocess.run(["launchctl", "unload", "-w", str(path)],
                               capture_output=True, text=True, check=False)
            if r.returncode != 0:
                print(f"  note: unload of {name} returned {r.returncode}: "
                      f"{r.stderr.strip() or r.stdout.strip()}")
            path.unlink()
            print(f"removed job no longer configured: {path}")

    for name, (mode, times, day) in jobs.items():
        path = AGENTS / f"{name}.plist"
        path.write_text(_plist(name, mode, times, day), encoding="utf-8")
        subprocess.run(["launchctl", "unload", str(path)],
                       capture_output=True, check=False)   # ignore "not loaded"
        # -w, and it is not decoration. `unload -w` writes the label into
        # launchd's disabled database, which is what makes a pause survive a
        # reboot; a plain `load` then REFUSES that label with "Load failed: 5:
        # Input/output error". Installing means "I want these running", and
        # that includes lifting a pause. (2026-09-22: the first --install after
        # a real pause was refused on all three jobs.)
        subprocess.run(["launchctl", "load", "-w", str(path)], check=True)
        # Not "loaded": `launchctl load` exits 0 even when it refuses, so this
        # line can only honestly claim the request. _verify below is what knows.
        print(f"\ninstalled, load requested: {path}")

    _verify(set(jobs), set(obsolete))

    print("\nCheck with:  launchctl list | grep tradinglab")
    print(f"Watch with:  tail -f {REPO}/logs/launchd_daily.log")
    print("Run once now: launchctl start " + f"{LABEL}.daily")


if __name__ == "__main__":
    main()
