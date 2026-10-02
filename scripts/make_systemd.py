"""
Generate the systemd units that run the workflow on the server.

The launchd twin, and simpler for one reason: the server's clock is UTC, so the
schedule in settings.json goes into the units unchanged. Every local-time
conversion in `make_launchd.py` exists because macOS schedules in local time,
and every one of them is a place an hour can go missing. Here there are none —
the times are written with an explicit UTC suffix so that a machine which is
later moved to another zone does not silently take the schedule with it.

Three units, the same three cadences the Mac has:
    powertradinglab-daily    update -> forecast -> score -> publish, at
                             run_time_utc. The one decision of the day, and it
                             must finish before the auction gate.
    powertradinglab-catchup  update -> score -> publish at catchup_time_utc,
                             with NO forecast: by then the auction has closed.
    powertradinglab-monthly  the whole previous month re-requested on
                             final_day_of_month, so the as-final pass has
                             something to look at.

Persistent=true on all three: a slot missed because the machine was down runs
once when it comes back, which is launchd's behaviour on wake and the one we
have been relying on. It cannot produce a late DECISION — the gate check
refuses a forecast issued after the auction closed and records the refusal —
so the worst case is a poll and a score arriving late, which is what we want.

System units with User=ptl rather than user units: no lingering to remember,
nothing tied to a login session, and `systemctl` says the truth about them from
any shell.

Run (from the repo root, on the server):
    uv run python scripts/make_systemd.py            # print the units
    uv run python scripts/make_systemd.py --install  # write, enable, verify
"""

import argparse
import getpass
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.config as cfg  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
UNIT_DIR = Path("/etc/systemd/system")
PREFIX = "powertradinglab"
# uv is installed per user and systemd starts with a bare PATH, so the unit says
# where things are. The same class of mistake as launchd's bare PATH.
BIN_PATH = "/home/{user}/.local/bin:/usr/local/bin:/usr/bin:/bin"


def _service(mode: str, user: str) -> str:
    return f"""[Unit]
Description=PowerTradingLab {mode} run
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User={user}
Group={user}
WorkingDirectory={REPO}
Environment=PATH={BIN_PATH.format(user=user)}
ExecStart=/bin/bash {REPO}/scripts/run.sh {mode}
StandardOutput=append:{REPO}/logs/run_{mode}.log
StandardError=append:{REPO}/logs/run_{mode}.log
"""


def _timer(mode: str, calendar: str) -> str:
    return f"""[Unit]
Description=PowerTradingLab {mode} schedule ({calendar})

[Timer]
OnCalendar={calendar}
Persistent=true
AccuracySec=1s
Unit={PREFIX}-{mode}.service

[Install]
WantedBy=timers.target
"""


def _calendars() -> dict[str, str]:
    """mode -> OnCalendar, straight from settings. No conversion anywhere."""
    day = cfg.final_day_of_month()
    return {
        "daily": f"*-*-* {cfg.RUN_TIME_UTC}:00 UTC",
        "catchup": f"*-*-* {cfg.catchup_time_utc()}:00 UTC",
        "monthly": f"*-*-{day:02d} {cfg.refetch_time_utc()}:00 UTC",
    }


def _verify(modes: list[str]) -> None:
    """Ask systemd what it has, rather than trust the commands just issued.

    The same rule as the launchd generator, learned the same way: a scheduler
    that quietly did not take a job is the failure nobody notices until a
    morning goes missing.
    """
    bad = []
    for mode in modes:
        timer = f"{PREFIX}-{mode}.timer"
        r = subprocess.run(["systemctl", "is-enabled", timer],
                           capture_output=True, text=True, check=False)
        state = r.stdout.strip() or r.stderr.strip()
        active = subprocess.run(["systemctl", "is-active", timer],
                                capture_output=True, text=True, check=False).stdout.strip()
        if state != "enabled" or active != "active":
            bad.append(f"{timer}: is-enabled={state}, is-active={active}")
    print("\nnext runs:")
    subprocess.run(["systemctl", "list-timers", f"{PREFIX}-*", "--all",
                    "--no-pager"], check=False)
    if bad:
        print("\nNOT INSTALLED CORRECTLY:")
        for line in bad:
            print(f"  {line}")
        raise SystemExit(1)
    print(f"\nverified against systemd: {len(modes)} timers enabled and active")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--install", action="store_true",
                   help="write the units to /etc/systemd/system, enable, verify")
    p.add_argument("--user", default=getpass.getuser(),
                   help="the account the runs belong to (default: the current one)")
    a = p.parse_args()

    cal = _calendars()
    print(f"repo: {REPO}\nuser: {a.user}\n")
    print(f"{'unit':<32} {'when (UTC)':<26} what")
    what = {"daily": "update + forecast + score + publish",
            "catchup": "update + score + publish, NO forecast",
            "monthly": "re-request last month, for the as-final pass"}
    for mode, when in cal.items():
        print(f"{f'{PREFIX}-{mode}.timer':<32} {when:<26} {what[mode]}")

    if cfg.refetch_time_utc() >= cfg.RUN_TIME_UTC:
        print(f"\n  WARNING: the monthly re-fetch ({cfg.refetch_time_utc()}) is not "
              f"earlier than the daily run ({cfg.RUN_TIME_UTC}).\n  The as-final "
              f"scoring would then read data that has not been refreshed yet.")

    if not a.install:
        for mode, when in cal.items():
            print(f"\n----- {PREFIX}-{mode}.service -----\n{_service(mode, a.user)}")
            print(f"----- {PREFIX}-{mode}.timer -----\n{_timer(mode, when)}")
        print("Re-run with --install to write and enable these.")
        return

    (REPO / "logs").mkdir(exist_ok=True)
    for mode, when in cal.items():
        for name, text in ((f"{PREFIX}-{mode}.service", _service(mode, a.user)),
                           (f"{PREFIX}-{mode}.timer", _timer(mode, when))):
            subprocess.run(["sudo", "tee", str(UNIT_DIR / name)],
                           input=text, text=True, capture_output=True, check=True)
            print(f"wrote {UNIT_DIR / name}")

    subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
    for mode in cal:
        subprocess.run(["sudo", "systemctl", "enable", "--now",
                        f"{PREFIX}-{mode}.timer"], check=True)
    _verify(list(cal))


if __name__ == "__main__":
    main()
