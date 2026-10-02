"""
Backup step: push the irreplaceable files into a private git repository.

WHAT IS COPIED, AND WHY ONLY THIS

    data/raw                 the source of truth: every response from ENTSO-E and
                             the ECB exactly as it arrived, with its fetch time
    data/processed/results   what we DECIDED and what we GRADED
    outputs                  the frozen published files — the product
    settings.json            the parameters those files were produced under
    logs                     what the machine did, run by run

Git is good at what is APPENDED and bad at what is REWRITTEN WHOLE. Every path
above grows; a parquet store does not, it is rewritten in full every morning, and
a binary rewrite costs its entire size in every commit — about 5 MB per zone per
day, which is 3.6 GB a year at two zones and absurd at twenty-five.

So the store (data/processed/zones) is NOT here, and that is a claim, not an
omission: the store is a fold over data/raw by (delivery_ts, knownby_ts), and raw
keeps every response with the instant it was fetched. Whatever the store knows
about when a value appeared, raw knows first. Keeping raw and dropping the store
backs up the cause instead of the effect.

The honest hole in that: no code rebuilds the store from raw today — the
2020-2026 seed came from the bulk archive, not from raw. Until that exists, the
store's protection is the machine's own snapshots, and this repository is the
guarantee that nothing which cannot be recomputed is ever lost with the disk.

The history caches (data/processed/history_*.feather) and data/archive are large,
static and re-downloadable, and by our own rule they are a dev convenience rather
than a source. They stay out.

WHY GIT RATHER THAN A DATED FOLDER

The old step copied into dated folders in a sync directory, so that a bug which
corrupted the store could not quietly overwrite the last good copy. A commit is
that argument taken to its conclusion: every snapshot is addressable for ever, a
corrupted file is one `git diff` away from being visible, and nothing needs
pruning. It also removes the failure the old step kept reporting — the sync
folder lived on a laptop, and the service now runs on a server that never had it.

PRIVATE, AND IT HAS TO STAY PRIVATE. data/raw carries ENTSO-E market data we are
not licensed to redistribute. The repository is separate from the code repository
for exactly this reason: the code will be public, and a visibility switch should
not be able to publish data by accident. The deploy key the server holds can
write to this repository and nothing else.

Configure in settings.json:

    "backup": { "mode": "git",
                "repo": "git@github.com:chuchueva/powertradinglab-backup.git",
                "work_dir": "~/ptl-backup",
                "key": "~/.ssh/ptl_backup",
                "paths": ["data/raw", "data/processed/results", "outputs",
                          "settings.json", "logs"] }

`mode: "dir"` keeps the old dated-folder behaviour (`dir`, `keep_days`), which is
the way back. An empty mode disables the step. A failed backup is reported in the
exit code and never fails the run: losing a day of collection to protect a copy
would be the wrong trade.

Run (from repo root):
    uv run python -m src.backup
"""

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import src.config as cfg
from src.workflow import STORE_ROOT, setup_logging
from src.workflow_compute import RESULTS_DIR

DATED = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _copy_atomic(src: Path, dst: Path) -> int:
    """Copy, then rename into place. A sync client never sees a partial file."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copy2(src, tmp)
    if tmp.stat().st_size != src.stat().st_size:
        tmp.unlink(missing_ok=True)
        raise OSError(f"short copy of {src}: {tmp.stat().st_size} of {src.stat().st_size} bytes")
    os.replace(tmp, dst)          # atomic within one filesystem
    return dst.stat().st_size


def _prune(root: Path, keep_days: int, log) -> int:
    """Drop dated folders older than the window. Touches NOTHING else — only
    directories whose name is exactly a date, so an unrelated folder sharing the
    destination can never be deleted by us."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    removed = 0
    for d in sorted(root.iterdir()):
        if d.is_dir() and DATED.match(d.name) and d.name < cutoff:
            shutil.rmtree(d)
            removed += 1
            log.info("backup: removed old snapshot %s", d.name)
    return removed


README = """PowerTradingLab — backup. PRIVATE, and it must stay private.

Contains ENTSO-E derived market data (data/raw) which we are not licensed to
redistribute. Do not make this repository public and do not mirror it.

Written by the server at the end of every scheduled run,
through a write deploy key. Nothing here is
edited by hand: the next snapshot would overwrite it without noticing.

What is here, and what is deliberately not, is explained in src/backup.py of
the powertradinglab-benchmarks repository.
"""


def _git(work, *args, key: str = "", check: bool = True):
    """One git command in the backup checkout, with the deploy key if there is one."""
    cmd = ["git", "-C", str(work)]
    if key:
        cmd += ["-c", "core.sshCommand=ssh -i %s -o IdentitiesOnly=yes "
                      "-o StrictHostKeyChecking=accept-new" % os.path.expanduser(key)]
    cmd += [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError("git %s: %s" % (args[0], (r.stderr or r.stdout).strip()))
    return r


def _sync(paths, work: Path, log) -> tuple[int, int]:
    """Copy the backed-up paths into the checkout, keeping their shape.

    The shape is the point: restoring is then a copy, not a reconstruction. A
    path that does not exist on this machine is said once and skipped — the Mac
    and the server do not hold the same things, and neither is wrong.
    """
    for rel in paths:
        src = Path(rel)
        if not src.exists():
            log.info("backup: %s is not on this machine, skipped", rel)
            continue
        dst = work / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            _copy_atomic(src, dst)
    n = b = 0
    for f in work.rglob("*"):
        if f.is_file() and ".git/" not in str(f.relative_to(work)):
            n += 1
            b += f.stat().st_size
    return n, b


def push_snapshot(repo: str, work_dir: str, paths: list, key: str, log,
                  when=None) -> str | None:
    """Copy, commit if anything moved, push. Returns the commit, or None."""
    work = Path(work_dir).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    if not (work / ".git").exists():
        _git(work, "init", "-b", "main")
        _git(work, "remote", "add", "origin", repo)
        (work / "README.md").write_text(README, encoding="utf-8")
        log.info("backup: initialised %s -> %s", work, repo)
        # An empty repository has no main to fetch; that is not an error.
        if _git(work, "fetch", "origin", "main", key=key, check=False).returncode == 0:
            _git(work, "reset", "--hard", "origin/main")

    n_files, n_bytes = _sync(paths, work, log)
    _git(work, "add", "-A")
    if _git(work, "diff", "--cached", "--quiet", check=False).returncode == 0:
        log.info("backup: nothing changed since the last snapshot")
        return None

    stamp = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%MZ")
    _git(work, "-c", "user.name=PowerTradingLab server",
         "-c", "user.email=ptl@powertradinglab.org",
         "commit", "-q", "-m",
         "snapshot %s — %d files, %.1f MB" % (stamp, n_files, n_bytes / 1048576))
    _git(work, "push", "-q", "-u", "origin", "main", key=key)
    sha = _git(work, "rev-parse", "--short", "HEAD").stdout.strip()
    log.info("backup: pushed %s — %d files, %.1f MB in the repository",
             sha, n_files, n_bytes / 1048576)
    return sha


def backup_git(log) -> str | None:
    repo = cfg.backup_repo()
    if not repo:
        log.info("backup: no repository configured (settings.json -> backup.repo), skipped")
        return None
    # The key is the SERVER's. On a machine that does not have it, fall back to
    # that machine's own git identity rather than failing with IdentitiesOnly.
    key = cfg.backup_key()
    if key and not os.path.exists(os.path.expanduser(key)):
        log.info("backup: %s is not on this machine, using the default git identity", key)
        key = ""
    return push_snapshot(repo, cfg.backup_work_dir(), cfg.backup_paths(), key, log)


def backup_dir_snapshot(log) -> Path | None:
    """The old behaviour: the store into today's dated folder. Kept as the way back."""
    conf_dir = cfg.backup_dir()
    if not conf_dir:
        log.info("backup: not configured (settings.json -> backup.dir), skipped")
        return None

    root = Path(conf_dir).expanduser()
    if not root.parent.exists():
        # The parent missing means the sync folder itself is gone — a wrong path,
        # or Drive not mounted. Say so loudly, but do not fail the run.
        log.warning("backup: %s is not reachable, skipped (is the sync folder mounted?)",
                    root)
        return None

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target = root / day
    target.mkdir(parents=True, exist_ok=True)

    # Every zone under its own folder name, and the shared results beside them
    # — the backup keeps the shape of what it copies, so restoring it is a copy
    # and not a reconstruction. The results are the PRODUCT: leaving them out
    # because they moved out of the zone folders would have backed up everything
    # except the thing worth having.
    total = 0
    files = (sorted(Path(STORE_ROOT).glob("*/*.parquet"))
             + sorted(Path(RESULTS_DIR).glob("*.parquet")))
    for f in files:
        dest = target / f.parent.name / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        total += _copy_atomic(f, dest)

    removed = _prune(root, cfg.backup_keep_days(), log)
    log.info("backup: %d files, %.1f MB -> %s (%d old snapshots removed)",
             len(files), total / 1048576, target, removed)
    return target


def backup(log=None):
    """Dispatch on the configured mode. Two mechanisms, one contract."""
    log = log or setup_logging()
    mode = cfg.backup_mode()
    if mode == "git":
        return backup_git(log)
    if mode == "dir":
        return backup_dir_snapshot(log)
    log.info("backup: not configured (settings.json -> backup.mode), skipped")
    return None


def main() -> None:
    log = setup_logging()
    try:
        backup(log)
    except Exception as e:                 # noqa: BLE001 - reported, never fatal
        log.error("backup: failed — %s. The data is on the machine; only the "
                  "off-machine copy is behind.", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
