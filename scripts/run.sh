#!/bin/bash
# NL workflow entry point for launchd. One argument:
#
#   poll   - just refresh the store (measures when IM appears)
#   daily  - refresh, then decide tomorrow's action and grade what has settled
#
# Steps run in sequence and each reports its own exit code. A failed poll does
# NOT abort the rest: with the as-of rule a forecast on a slightly older window
# is still correct (it simply says so in data_through), whereas skipping the
# forecast leaves a hole in a daily series. Failures are loud in the log.
#
# launchd starts jobs with a bare PATH, so uv has to be findable explicitly.

# poll    - refresh the store only
# daily   - refresh, decide, score, publish
# catchup - refresh, score, publish, but NEVER decide. It runs hours after the
#           day-ahead auction has closed, and a decision issued then could not
#           have been placed. Everything else gains from a second look: data
#           that arrived late lands in the store, days that settled get scored,
#           and status stops being stale. Recovering a FORECAST has exactly one
#           window — from the decision run to gate closure — and nothing later
#           will do.
# monthly - re-request the whole previous month, so that the as-final scoring
#           pass has something to look at. Runs BEFORE the daily slot on the
#           same day; the clock is what guarantees the order.
MODE="${1:-poll}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
# Marks the run as belonging to a scheduled slot. Only this script sets it, so a
# run typed by hand logs "manual" and never claims to be late.
export TRADINGLAB_SCHEDULED=1
cd "$REPO" || exit 1

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
run() {
    echo "$(stamp) running: $*"
    "$@"
    local code=$?          # capture BEFORE anything else touches $?
    if [ "$code" -ne 0 ]; then
        echo "$(stamp) FAILED (exit $code): $*"
        return 1
    fi
}

# launchd owns logs/launchd_$MODE.log (StandardOutPath), so the Python side's
# rotation cannot reach it, and it receives a copy of everything the console
# handler prints. Keep it bounded here: past 2 MB, move it aside, keep one
# generation. The durable record is logs/workflow.log, which rotates daily.
LOGFILE="$REPO/logs/launchd_$MODE.log"
# Size via wc, not stat: the flag for "size in bytes" is -f%z on macOS and
# -c%s on Linux, and this script now runs on both. wc -c is the same word
# on either, and a wrong stat flag fails SILENTLY into "no rotation".
if [ -f "$LOGFILE" ] && [ "$(wc -c < "$LOGFILE" 2>/dev/null || echo 0)" -gt 2000000 ]; then
    mv "$LOGFILE" "$LOGFILE.old"
fi

echo "$(stamp) === run.sh $MODE in $REPO ==="
rc=0

# Only the first command of the chain may raise the "started late" alarm; the
# rest are the same run and would repeat it verbatim.
export TRADINGLAB_FIRST_STEP=1
if [ "$MODE" = "monthly" ]; then
    run uv run python -m src.update --last-month || rc=1
else
    run uv run python -m src.update || rc=1
fi
unset TRADINGLAB_FIRST_STEP

# Exchange rates, before anything is valued. Euro zones do not need them — EUR
# is a unit, never looked up — so a failure here is counted and the chain goes
# on; a non-euro zone that lacks a rate is refused downstream with leg="FX".
if [ "$MODE" = "daily" ] || [ "$MODE" = "catchup" ]; then
    run uv run python -m src.fx || rc=1
fi

if [ "$MODE" = "daily" ]; then
    run uv run python -m src.workflow_compute forecast || rc=1
fi

if [ "$MODE" = "daily" ] || [ "$MODE" = "catchup" ]; then
    run uv run python -m src.workflow_compute score || rc=1
    # Publishing renders whatever the tables hold and skips what is already
    # there, so it catches up by itself if a previous day failed — no argument
    # needed, and no harm in running it after a step that produced nothing.
    run uv run python -m src.publish || rc=1
fi

# The reader's shape, rebuilt from the results table after the archive is
# written — never before, because every row cites the frozen file it came from
# and a citation must point at something that exists. It runs on every kind of
# run: a poll that adds nothing leaves the bytes untouched and costs a second.
run uv run python -m src.summary || rc=1

# The status file is rewritten by EVERY run, whatever the mode: its own
# timestamp is the heartbeat readers rely on, and a heartbeat that only beats
# once a day cannot say much.
run uv run python -m src.status || rc=1

# Last, and after every kind of run — the store changes on polls too, so a copy
# taken only on the daily run would leave the afternoon's measurements exposed.
run uv run python -m src.backup || rc=1

# The public data, to the site repository: v1, status.xml and the archive files
# v1 cites. After the backup, so a failure here never delays the private copy.
run uv run python -m src.site_push || rc=1

echo "$(stamp) === done, exit $rc ==="
exit $rc
