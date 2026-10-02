"""
Tests for the push of public data to the site repository.

A local bare repository stands in for GitHub. What is tested is what can go
wrong in public: a draft reaching the site, a file of the site session being
deleted, an empty commit every run, and a folder name that would delete the
repository.

Run:
    uv run pytest tests/test_site_push.py -q
    uv run python tests/test_site_push.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.site_push as SP  # noqa: E402


class _Log:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


def _sh(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   cwd=cwd, check=True, capture_output=True)


def _remote(tmp: Path) -> Path:
    """A bare 'GitHub' holding what the site session would have pushed, plus a
    stale file in data/ that the next push must remove."""
    bare = tmp / "site.git"
    _sh(tmp, "init", "-q", "--bare", "-b", "main", str(bare))
    seed = tmp / "seed"
    _sh(tmp, "clone", "-q", str(bare), str(seed))
    (seed / "README.md").write_text("site\n")
    (seed / "data").mkdir()
    (seed / "data" / "stale.csv").write_text("old\n")
    _sh(seed, "add", "-A")
    _sh(seed, "commit", "-q", "-m", "seed")
    _sh(seed, "push", "-q", "origin", "main")
    return bare


def _outputs(tmp: Path) -> Path:
    """One cited live file, one uncited draft, v1 and the heartbeat."""
    out = tmp / "outputs"
    live = "nl/2026/09/score_NL_2026-09-25_20260926T094016Z.xml"
    draft = "nl/2026/07/score_NL_2026-07-21_20260722T094000Z.xml"
    for rel in (live, draft):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text("<x/>\n")
    (out / "v1").mkdir()
    (out / "v1" / "NL_scores.csv").write_text(
        "zone,delivery_day,source_file\nNL,2026-09-25,%s\n" % live)
    fc = "nl/2026/09/forecast_NL_2026-09-26_20260925T094016Z.xml"
    fc_draft = "nl/2026/07/forecast_NL_2026-07-22_20260721T094000Z.xml"
    for rel in (fc, fc_draft):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text("<x/>\n")
    (out / "v1" / "NL_forecasts.csv").write_text(
        "zone,delivery_day,source_file\nNL,2026-09-26,%s\n" % fc)
    (out / "v1" / "index.json").write_text('{"generated_from": "2026-09-26T09:40:16Z"}\n')
    (out / "status.xml").write_text("<status/>\n")
    return out


def _tree(bare: Path, tmp: Path) -> set:
    look = tmp / "look"
    if look.exists():
        _sh(look, "pull", "-q")
    else:
        _sh(tmp, "clone", "-q", str(bare), str(look))
    return {str(p.relative_to(look)) for p in look.rglob("*")
            if p.is_file() and ".git" not in p.parts}


def test_only_the_cited_set_reaches_the_site_and_nothing_else_is_touched():
    """
    v1, the heartbeat and the archive files v1 cites, scores and forecasts —
    and nothing else. The drafts are on disk and must not travel; the stale file in data/ must go; the
    site session's README must stay.
    """
    tmp = Path(tempfile.mkdtemp())
    bare, out = _remote(tmp), _outputs(tmp)
    sha = SP.push_site(str(bare), str(tmp / "work"), "data", "", _Log(), out_dir=out)
    assert sha
    assert _tree(bare, tmp) == {
        "README.md",
        "data/status.xml",
        "data/v1/NL_scores.csv",
        "data/v1/index.json",
        "data/v1/NL_forecasts.csv",
        "data/nl/2026/09/forecast_NL_2026-09-26_20260925T094016Z.xml",
        "data/nl/2026/09/score_NL_2026-09-25_20260926T094016Z.xml",
    }


def test_a_run_that_changes_nothing_commits_nothing():
    """Two runs a day with no new data must not leave two commits a day."""
    tmp = Path(tempfile.mkdtemp())
    bare, out = _remote(tmp), _outputs(tmp)
    work = str(tmp / "work")
    assert SP.push_site(str(bare), work, "data", "", _Log(), out_dir=out)
    assert SP.push_site(str(bare), work, "data", "", _Log(), out_dir=out) is None


def test_the_site_sessions_commit_in_between_is_kept():
    """The site session pushes to the same repository. Our next run starts from
    its commit, not from our stale checkout, and does not undo it."""
    tmp = Path(tempfile.mkdtemp())
    bare, out = _remote(tmp), _outputs(tmp)
    work = str(tmp / "work")
    SP.push_site(str(bare), work, "data", "", _Log(), out_dir=out)

    other = tmp / "other"
    _sh(tmp, "clone", "-q", str(bare), str(other))
    (other / "build.py").write_text("# generator\n")
    _sh(other, "add", "-A")
    _sh(other, "commit", "-q", "-m", "generator")
    _sh(other, "push", "-q", "origin", "main")

    (out / "status.xml").write_text("<status run='2'/>\n")
    assert SP.push_site(str(bare), work, "data", "", _Log(), out_dir=out)
    assert "build.py" in _tree(bare, tmp)


def test_a_folder_name_that_could_delete_the_repository_is_refused():
    """data/ is deleted on every run. Anything but one plain name is refused
    before a single file is touched."""
    for bad in ("", ".", "..", "/data", "../data", "a/b", ".git"):
        try:
            SP._safe_data_dir(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted")
    assert SP._safe_data_dir("data") == "data"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
