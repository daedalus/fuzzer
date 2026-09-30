"""Throwaway git worktree with ``git bisect`` driver.

Bisecting inside a private worktree leaves the caller's checkout, index and
bisect state untouched.

    repo/.git  <-- shared object store
    /tmp/crash_bisect_xxx/   (detached worktree; bisect state is per-worktree)
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path

_FIRST_BAD_RE = re.compile(r"^([0-9a-f]{40}) is the first bad commit", re.MULTILINE)
_SKIPPED_ONLY = "only 'skip'ped commits left"
_SHA_LINE_RE = re.compile(r"^([0-9a-f]{40})$", re.MULTILINE)


class GitError(RuntimeError):
    """A git command failed."""


class BisectMark(Enum):
    """Verdict word understood by ``git bisect``."""

    GOOD = "good"
    BAD = "bad"
    SKIP = "skip"


class BisectStep(Enum):
    """What ``git bisect`` reported after a start/mark."""

    CONTINUE = "continue"
    FOUND = "found"
    SKIPPED_OUT = "skipped_out"


class GitWorktree:
    """Detached worktree of *repo*, removed on exit."""

    def __init__(self, repo: str) -> None:
        self._repo = repo
        self._dir: Path | None = None
        self.culprit: str | None = None
        self.candidates: list[str] = []

    @property
    def path(self) -> Path:
        if self._dir is None:
            raise GitError("worktree not open")
        return self._dir

    def _run(
        self, *args: str, cwd: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        r = subprocess.run(
            ["git", *args],
            cwd=cwd or self._repo,
            capture_output=True,
            text=True,
        )
        if check and r.returncode != 0:
            raise GitError((r.stderr or r.stdout).strip() or f"git {args[0]} failed")
        return r

    def resolve(self, rev: str) -> str:
        """Full sha of *rev*; GitError if it is not a commit."""
        return self._run("rev-parse", "--verify", f"{rev}^{{commit}}").stdout.strip()

    def subject(self, sha: str) -> str:
        return self._run("log", "-1", "--format=%s", sha).stdout.strip()

    def __enter__(self) -> GitWorktree:
        self._run("rev-parse", "--git-dir")
        self._dir = Path(tempfile.mkdtemp(prefix="crash_bisect_"))
        self._run("worktree", "add", "--detach", "-f", str(self._dir), "HEAD")
        return self

    def __exit__(self, *_exc) -> None:
        if self._dir is None:
            return
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(self._dir)],
            cwd=self._repo,
            capture_output=True,
        )
        shutil.rmtree(self._dir, ignore_errors=True)
        subprocess.run(["git", "worktree", "prune"], cwd=self._repo, capture_output=True)
        self._dir = None

    def checkout(self, sha: str) -> None:
        self._run("checkout", "-q", "--detach", "-f", sha, cwd=str(self.path))
        self._run("clean", "-fdxq", cwd=str(self.path))

    def head(self) -> str:
        return self._run("rev-parse", "HEAD", cwd=str(self.path)).stdout.strip()

    def bisect_start(self, bad: str, good: str) -> BisectStep:
        return self._bisect("start", bad, good)

    def bisect_mark(self, mark: BisectMark) -> BisectStep:
        return self._bisect(mark.value)

    def _bisect(self, *args: str) -> BisectStep:
        # git exits 2 when only skipped commits remain: that is a result, not a failure.
        r = self._run("bisect", *args, cwd=str(self.path), check=False)
        if _SKIPPED_ONLY in r.stdout + r.stderr:
            self.candidates = _SHA_LINE_RE.findall(r.stdout + r.stderr)
            return BisectStep.SKIPPED_OUT
        if r.returncode != 0:
            raise GitError((r.stderr or r.stdout).strip() or "git bisect failed")
        return self._step(r.stdout)

    def _step(self, out: str) -> BisectStep:
        m = _FIRST_BAD_RE.search(out)
        if m:
            self.culprit = m.group(1)
            return BisectStep.FOUND
        return BisectStep.CONTINUE
