"""
The data the site is built from, pushed to the site repository.

WHERE THINGS ARE. Three repositories, two machines. This step runs on the
SERVER, at the end of every scheduled run, after the backup. It keeps its own
checkout of `powertradinglab-site` in `site.work_dir` (`~/ptl-site`), outside
this repository and beside `~/ptl-backup`, and writes into ONE folder of it,
`site.data_dir` (`data/`). Everything else in that repository — the generator,
the workflow, the pages — belongs to the site session and is never touched here.

WHAT GOES IN, and nothing else:

    outputs/v1/*            -> data/v1/*
    outputs/status.xml      -> data/status.xml
    every `source_file` named in outputs/v1/<ZONE>_scores.csv
    and outputs/v1/<ZONE>_forecasts.csv
                            -> data/<source_file>

The folder mirrors the publication root, so a `source_file` value resolves
against it unchanged. Archive files that no v1 row cites are NOT pushed: the
drafts before `publication.public_from` stay in outputs/ and in the private
backup and never reach a public repository. The cut is made once, in
`summary`; this step only follows it.

A MIRROR, NOT AN APPEND. Unlike the backup, the data folder is deleted and
rebuilt on every run, so a file that leaves the published set leaves the
repository too. That is also why the folder name is checked before anything is
deleted: an empty or relative-climbing name would turn "rebuild data/" into
"delete the repository".

TWO WRITERS. The site session pushes to the same repository, so every run
starts from the remote's newest state (fetch + hard reset), and a rejected push
is retried once from a fresh fetch. A run that changes nothing commits nothing.

Failure is logged and reported in the exit code; nothing else in the run waits
for it, and the next run pushes the same files again.

Run (from repo root, after summary and status):
    uv run python -m src.site_push
"""

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import src.config as cfg
from src.backup import _git
from src.publish import OUT_DIR
from src.workflow import setup_logging

V1 = "v1"
STATUS = "status.xml"


def published_set(out_dir: Path, log) -> list[str]:
    """Paths relative to the publication root: v1, the heartbeat, and the
    archive files v1 cites. Sorted, so the copy order never varies."""
    out_dir = Path(out_dir)
    v1 = out_dir / V1
    if not v1.is_dir():
        return []
    names = {f"{V1}/{f.name}" for f in v1.iterdir() if f.is_file()}
    if (out_dir / STATUS).is_file():
        names.add(STATUS)
    # Only the full-history files: the 90-day and latest ones cite a subset.
    tables = sorted(v1.glob("*_scores.csv")) + sorted(v1.glob("*_forecasts.csv"))
    for table in tables:
        with table.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                src = (row.get("source_file") or "").strip()
                if not src:
                    continue
                if (out_dir / src).is_file():
                    names.add(src)
                else:
                    log.warning("site_push: %s cites %s, which is not on disk; "
                                "not pushed", table.name, src)
    return sorted(names)


def _safe_data_dir(name: str) -> str:
    """One plain folder name, or refuse. This folder is deleted on every run."""
    p = Path(name)
    if (not name or p.is_absolute() or len(p.parts) != 1
            or name in (".", "..") or name.startswith(".")):
        raise ValueError(f"site.data_dir must be one plain folder name, got {name!r}")
    return name


def _sync(out_dir: Path, work: Path, data_dir: str, log) -> int:
    """Delete the data folder in the checkout and rebuild it from the set."""
    target = work / data_dir
    if target.exists():
        shutil.rmtree(target)
    names = published_set(out_dir, log)
    for rel in names:
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(out_dir) / rel, dst)
    return len(names)


def _generated_from(work: Path, data_dir: str) -> str:
    """The newest fact in the data, from index.json — not the wall clock."""
    try:
        idx = json.loads((work / data_dir / V1 / "index.json").read_text("utf-8"))
        return str(idx.get("generated_from", "")) or "unknown"
    except (OSError, ValueError):
        return "unknown"


def _start_from_remote(work: Path, repo: str, key: str, log) -> None:
    """Make the checkout exactly the remote's main, cloning it the first time."""
    if not (work / ".git").exists():
        work.mkdir(parents=True, exist_ok=True)
        _git(work, "init", "-q", "-b", "main")
        _git(work, "remote", "add", "origin", repo)
        log.info("site_push: initialised %s -> %s", work, repo)
    # An empty repository has no main yet; then we start from nothing.
    if _git(work, "fetch", "-q", "origin", "main", key=key, check=False).returncode == 0:
        _git(work, "reset", "-q", "--hard", "origin/main")


def push_site(repo: str, work_dir: str, data_dir: str, key: str, log,
              out_dir: Path = OUT_DIR) -> str | None:
    """Rebuild data/, commit if anything moved, push. Returns the commit or None."""
    data_dir = _safe_data_dir(data_dir)
    work = Path(work_dir).expanduser()

    for attempt in (1, 2):
        _start_from_remote(work, repo, key, log)
        n = _sync(out_dir, work, data_dir, log)
        _git(work, "add", "-A", "--", data_dir)
        if _git(work, "diff", "--cached", "--quiet", check=False).returncode == 0:
            log.info("site_push: nothing changed (%d files in %s/)", n, data_dir)
            return None
        _git(work, "-c", "user.name=PowerTradingLab server",
             "-c", "user.email=ptl@powertradinglab.org",
             "commit", "-q", "-m",
             "data through %s — %d files" % (_generated_from(work, data_dir), n))
        r = _git(work, "push", "-q", "origin", "main", key=key, check=False)
        if r.returncode == 0:
            sha = _git(work, "rev-parse", "--short", "HEAD").stdout.strip()
            log.info("site_push: pushed %s — %d files in %s/", sha, n, data_dir)
            return sha
        if attempt == 1:
            last = ((r.stderr or r.stdout).strip().splitlines() or [""])[-1]
            log.info("site_push: push rejected (%s); refetching once", last)
    raise RuntimeError("push rejected twice: %s" % (r.stderr or r.stdout).strip())


def site_push(log=None) -> str | None:
    log = log or setup_logging()
    repo = cfg.site_repo()
    if not repo:
        log.info("site_push: no repository configured (settings.json -> site.repo), skipped")
        return None
    # Same rule as the backup: the key is the server's; elsewhere, fall back to
    # the machine's own git identity.
    key = cfg.site_key()
    if key and not os.path.exists(os.path.expanduser(key)):
        log.info("site_push: %s is not on this machine, using the default git identity", key)
        key = ""
    return push_site(repo, cfg.site_work_dir(), cfg.site_data_dir(), key, log)


def main() -> None:
    log = setup_logging()
    try:
        site_push(log)
    except Exception as e:                 # noqa: BLE001 - reported, never fatal
        log.error("site_push: failed — %s. The data is on the machine; only the "
                  "site is behind.", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
