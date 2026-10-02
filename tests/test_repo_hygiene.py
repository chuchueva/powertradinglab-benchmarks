"""
Tests for what this PUBLIC repository must never contain.

The internal notes live in `private/`, a clone of a separate private repository
sitting inside this folder. The only thing keeping them out of the public
history is one line of .gitignore, so that line is tested.

Run:
    uv run pytest tests/test_repo_hygiene.py -q
"""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ignore_lines() -> set:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    return {ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.startswith("#")}


def test_private_notes_and_secrets_are_ignored():
    """private/ holds the hosting notes and the working log; .env holds the
    ENTSO-E token; data/ holds raw prices we may not redistribute."""
    lines = _ignore_lines()
    for needed in ("/private/", ".env", "/data", "/logs/", "/outputs/"):
        assert needed in lines, f"{needed} missing from .gitignore"


def test_git_really_ignores_private():
    """The line could be present and still not match. Ask git itself."""
    if not (ROOT / ".git").exists():
        return
    r = subprocess.run(["git", "check-ignore", "-q", "private/HANDOFF.md"],
                       cwd=ROOT, capture_output=True)
    assert r.returncode == 0, "git does not ignore private/"


if __name__ == "__main__":
    test_private_notes_and_secrets_are_ignored()
    test_git_really_ignores_private()
    print("ok")
