"""
Tests for the backup step — specifically the part that DELETES.

Everything else here only copies; pruning is the one operation that can destroy
something, and it runs unattended inside a folder the user also keeps other
things in. So that is what is tested.

Run:
    uv run pytest tests/test_backup.py -q
    uv run python tests/test_backup.py
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.backup as B  # noqa: E402


class _Log:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


def _day(offset: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset)).strftime("%Y-%m-%d")


def test_prune_only_touches_dated_folders():
    """
    The destination is the user's own sync folder and will hold other things.

    Only directories whose name is exactly a date may be removed — never a file,
    never a folder with any other name, however old it looks.
    """
    root = Path(tempfile.mkdtemp())
    (root / _day(-100)).mkdir()                 # old snapshot: goes
    (root / _day(-1)).mkdir()                   # recent snapshot: stays
    (root / "notes").mkdir()                    # not ours: stays
    (root / "2026-08-01-old-copy").mkdir()      # date-LIKE, not a date: stays
    (root / "readme.txt").write_text("x")       # a file, not a folder: stays

    removed = B._prune(root, keep_days=14, log=_Log())

    left = sorted(p.name for p in root.iterdir())
    assert removed == 1, left
    assert _day(-100) not in left
    assert _day(-1) in left
    assert "notes" in left
    assert "2026-08-01-old-copy" in left
    assert "readme.txt" in left


def test_prune_keeps_everything_inside_the_window():
    root = Path(tempfile.mkdtemp())
    for offset in (0, -1, -13):
        (root / _day(offset)).mkdir()
    assert B._prune(root, keep_days=14, log=_Log()) == 0
    assert len(list(root.iterdir())) == 3


def test_copy_leaves_no_partial_file_behind():
    """
    The temp file must not survive the copy.

    A `.part` left in a sync folder would be uploaded as if it were data, and a
    truncated parquet still looks like a file to anything that finds it later.
    """
    src_dir, dst_dir = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    src = src_dir / "nl_da.parquet"
    src.write_bytes(b"x" * 5000)

    size = B._copy_atomic(src, dst_dir / src.name)

    assert size == 5000
    assert (dst_dir / "nl_da.parquet").read_bytes() == b"x" * 5000
    assert list(dst_dir.glob("*.part")) == []


def test_the_backup_set_is_the_inputs_and_the_outputs_not_the_middle():
    """
    Git stores a rewritten binary in full, every commit. The store is rewritten
    every morning and costs about 5 MB per zone per day; data/raw only grows, and
    the store is a fold over it. So raw is in and the store is out — and the
    results are in, because those are decisions, which nothing recomputes.
    """
    import src.config as cfg

    paths = cfg.backup_paths()
    for wanted in ("data/raw", "data/processed/results", "outputs"):
        assert wanted in paths, f"{wanted} cannot be recovered from anywhere else"
    joined = " ".join(paths)
    assert "zones" not in joined, "the store is rewritten whole; back up data/raw instead"
    assert "history_" not in joined and "archive" not in joined, (
        "large, static and re-downloadable — a dev cache is not a source")


def test_a_snapshot_commits_once_and_says_nothing_when_nothing_moved():
    """
    A real repository, a real push, no network: the remote is a bare repo in a
    temp folder. What matters is that an unchanged tree produces NO commit — a
    daily job that commits regardless would bury the days something happened.
    """
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bare = tmp / "remote.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
        src = tmp / "src"
        (src / "outputs").mkdir(parents=True)
        (src / "outputs" / "a.xml").write_text("<a/>", encoding="utf-8")
        (src / "settings.json").write_text("{}", encoding="utf-8")

        paths = ["outputs", "settings.json", "data/raw"]      # the last one is absent
        log, cwd = _Log(), os.getcwd()
        os.chdir(src)
        try:
            first = B.push_snapshot(str(bare), str(tmp / "work"), paths, "", log)
            assert first, "the first run has everything to say"
            assert B.push_snapshot(str(bare), str(tmp / "work"), paths, "", log) is None
            (src / "outputs" / "b.xml").write_text("<b/>", encoding="utf-8")
            third = B.push_snapshot(str(bare), str(tmp / "work"), paths, "", log)
            assert third and third != first
        finally:
            os.chdir(cwd)

        def git(*a):
            return subprocess.run(["git", "--git-dir", str(bare), *a],
                                  capture_output=True, text=True).stdout

        # The bare repo's HEAD still points at its own default branch, which we
        # never created — ask for main by name.
        assert git("log", "--oneline", "main").count("snapshot") == 2
        files = git("ls-tree", "-r", "--name-only", "main")
        assert "outputs/a.xml" in files and "outputs/b.xml" in files
        assert "README.md" in files, "the private-only warning travels with the data"

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
