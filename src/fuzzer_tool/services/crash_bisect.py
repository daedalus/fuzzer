"""Crash bisector: first commit that introduces (or fixes) a crash.

    good ---o---o---o---o--- bad        build + run crash file at each midpoint
              (git bisect in a private worktree; caller's checkout untouched)

Endpoints are verified first and the crash signature is pinned from the
crashing one, so an unrelated crash on the way is not blamed. Commits that
do not build (or lack the target) are skipped, never blamed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from fuzzer_tool.adapters.git_worktree import (
    BisectMark,
    BisectStep,
    GitError,
    GitWorktree,
)
from fuzzer_tool.adapters.process import SIGNAL_CRASH_CODES, run_target_file, run_target_stdin
from fuzzer_tool.core.sanitizer import SanitizerReport

_TIMEOUT_CODES = (-2, -1)
_DEFAULT_BUILD_TIMEOUT = 1800.0


class CrashBisectError(RuntimeError):
    """Bisection could not produce a verdict."""


class BisectMode(Enum):
    INTRODUCED = "introduced"
    FIXED = "fixed"


class SigMatch(Enum):
    """How strictly a crash must match the pinned one."""

    CLASS = "class"  # sanitizer + error type (survives line shifts)
    EXACT = "exact"  # full signature with top frames
    ANY = "any"  # any crash


class BisectVerdict(Enum):
    CRASH = "crash"
    CLEAN = "clean"
    SKIP = "skip"


@dataclass
class BisectResult:
    culprit: str
    subject: str
    mode: BisectMode
    signature: str | None
    probes: int
    skipped: int

    def report(self) -> str:
        verb = "introduced" if self.mode is BisectMode.INTRODUCED else "fixed"
        sig = self.signature or "any crash"
        return (
            f"Crash {verb} by {self.culprit}\n"
            f"  subject:   {self.subject}\n"
            f"  signature: {sig}\n"
            f"  probes:    {self.probes} ({self.skipped} skipped)\n"
        )


def _crash_key(rc: int, stderr: str, match: SigMatch) -> str | None:
    """Crash identity for one run, or None when the run did not crash."""
    if rc in _TIMEOUT_CODES:
        return None
    report = SanitizerReport.parse(stderr)
    if report and report.is_valid():
        if match is SigMatch.EXACT:
            return report.signature
        return f"{report.sanitizer}:{report.error_type}"
    if rc in SIGNAL_CRASH_CODES or rc < 0:
        return f"signal:{abs(rc)}"
    return None


class CrashBisector:
    """Bisect a repo history against one crashing input."""

    def __init__(
        self,
        repo: str,
        target: str,
        crash_file: str,
        build_cmd: str,
        timeout: float = 1.0,
        file_mode: bool = False,
        target_args: list[str] | None = None,
        mode: BisectMode = BisectMode.INTRODUCED,
        match: SigMatch = SigMatch.CLASS,
        build_timeout: float = _DEFAULT_BUILD_TIMEOUT,
    ) -> None:
        self._repo = repo
        self._target = target
        self._crash_file = crash_file
        self._build_cmd = build_cmd
        self._timeout = timeout
        self._file_mode = file_mode
        self._target_args = target_args or []
        self._mode = mode
        self._match = match
        self._build_timeout = build_timeout
        self._pinned: str | None = None
        self._probes = 0
        self._skipped = 0
        self._data = b""

    # -- one commit -> one verdict ------------------------------------

    def _build(self, wt: GitWorktree) -> bool:
        try:
            r = subprocess.run(
                self._build_cmd,
                shell=True,
                cwd=str(wt.path),
                capture_output=True,
                timeout=self._build_timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0

    def _run_target(self, exe: str) -> tuple[int, str]:
        if not self._file_mode:
            rc, err, _pid = run_target_stdin(exe, self._data, self._timeout)
            return rc, err
        tmp = tempfile.mkdtemp(prefix="crash_bisect_in_")
        try:
            rc, err, _pid = run_target_file(exe, self._data, self._timeout, tmp, self._target_args)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return rc, err

    def _probe(self, wt: GitWorktree) -> BisectVerdict:
        """Build and run the current worktree HEAD."""
        self._probes += 1
        exe = wt.path / self._target
        if not self._build(wt) or not exe.is_file():
            self._skipped += 1
            return BisectVerdict.SKIP
        rc, err = self._run_target(str(exe))
        key = _crash_key(rc, err, self._match)
        if key is None:
            return BisectVerdict.CLEAN
        if self._match is SigMatch.ANY:
            return BisectVerdict.CRASH
        if self._pinned is None:
            self._pinned = key
        return BisectVerdict.CRASH if key == self._pinned else BisectVerdict.CLEAN

    # -- endpoints ----------------------------------------------------

    def _endpoint(self, wt: GitWorktree, sha: str, label: str, want: BisectVerdict) -> None:
        wt.checkout(sha)
        got = self._probe(wt)
        if got is BisectVerdict.SKIP:
            raise CrashBisectError(f"{label} endpoint {sha[:12]} does not build")
        if got is not want:
            state = "does not crash" if want is BisectVerdict.CRASH else "crashes"
            raise CrashBisectError(f"{label} endpoint {sha[:12]} {state}")

    def _check_endpoints(self, wt: GitWorktree, good: str, bad: str) -> None:
        # Crashing endpoint first: it pins the signature for the other.
        if self._mode is BisectMode.INTRODUCED:
            self._endpoint(wt, bad, "bad", BisectVerdict.CRASH)
            self._endpoint(wt, good, "good", BisectVerdict.CLEAN)
            return
        self._endpoint(wt, good, "good", BisectVerdict.CRASH)
        self._endpoint(wt, bad, "bad", BisectVerdict.CLEAN)

    # -- search -------------------------------------------------------

    def _mark_for(self, verdict: BisectVerdict) -> BisectMark:
        if verdict is BisectVerdict.SKIP:
            return BisectMark.SKIP
        crash_is_bad = self._mode is BisectMode.INTRODUCED
        is_bad = (verdict is BisectVerdict.CRASH) == crash_is_bad
        return BisectMark.BAD if is_bad else BisectMark.GOOD

    def _search(self, wt: GitWorktree, good: str, bad: str) -> str:
        step = wt.bisect_start(bad, good)
        while step is BisectStep.CONTINUE:
            step = wt.bisect_mark(self._mark_for(self._probe(wt)))
        if step is BisectStep.SKIPPED_OUT or wt.culprit is None:
            cands = ", ".join(c[:12] for c in wt.candidates)
            raise CrashBisectError(
                f"only skipped (unbuildable) commits left; culprit is one of: {cands}"
            )
        return wt.culprit

    def run(self, good: str, bad: str = "HEAD") -> BisectResult:
        """Bisect ``good..bad``; INTRODUCED: good is clean, FIXED: good crashes."""
        crash = Path(self._crash_file)
        if not crash.is_file():
            raise CrashBisectError(f"crash file not found: {self._crash_file}")
        self._data = crash.read_bytes()
        self._pinned, self._probes, self._skipped = None, 0, 0

        try:
            with GitWorktree(self._repo) as wt:
                try:
                    good_sha, bad_sha = wt.resolve(good), wt.resolve(bad)
                except GitError as e:
                    raise CrashBisectError(f"bad revision: {e}") from e
                self._check_endpoints(wt, good_sha, bad_sha)
                culprit = self._search(wt, good_sha, bad_sha)
                return BisectResult(
                    culprit=culprit,
                    subject=wt.subject(culprit),
                    mode=self._mode,
                    signature=self._pinned,
                    probes=self._probes,
                    skipped=self._skipped,
                )
        except GitError as e:
            raise CrashBisectError(f"git: {e}") from e


def crash_bisect(**kw) -> BisectResult | None:
    """CLI wrapper: print report, return None on failure."""
    run_kw = {"good": kw.pop("good"), "bad": kw.pop("bad")}
    try:
        res = CrashBisector(**kw).run(**run_kw)
    except CrashBisectError as e:
        print(f"[-] {e}", file=sys.stderr)
        return None
    print(res.report())
    return res
