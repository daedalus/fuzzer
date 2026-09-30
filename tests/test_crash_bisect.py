"""Crash bisector: first-bad-commit search on throwaway git repos.

Targets are committed shell scripts. ``kill -SEGV $$`` yields returncode -11,
so no compiler is needed and every verdict is deterministic.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.services.crash_bisect import (
    BisectMode,
    BisectVerdict,
    CrashBisectError,
    CrashBisector,
    SigMatch,
)

CRASH_INPUT = b"BOOM"
SAFE = "#!/bin/sh\ncat >/dev/null\nexit 0\n"
SEGV = "#!/bin/sh\ncat >/dev/null\nkill -SEGV $$\n"
ABRT = "#!/bin/sh\ncat >/dev/null\nkill -ABRT $$\n"


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip()


def _commit(repo: Path, body: str, msg: str) -> str:
    (repo / f"{msg}.txt").write_text(msg)  # unique file: identical bodies still commit
    tgt = repo / "target.sh"
    tgt.write_text(body)
    tgt.chmod(0o755)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _history(repo: Path, bodies: list[str]) -> list[str]:
    _git(repo, "init", "-q")
    return [_commit(repo, b, f"c{i}") for i, b in enumerate(bodies)]


def _bisector(repo: Path, crash: Path, **kw) -> CrashBisector:
    return CrashBisector(
        repo=str(repo),
        target="target.sh",
        crash_file=str(crash),
        build_cmd="true",
        timeout=5.0,
        **kw,
    )


@pytest.fixture
def crash_file(tmp_path: Path) -> Path:
    p = tmp_path / "crash.bin"
    p.write_bytes(CRASH_INPUT)
    return p


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    return r


def test_finds_introducing_commit(repo, crash_file):
    h = _history(repo, [SAFE, SAFE, SAFE, SEGV, SEGV, SEGV])
    res = _bisector(repo, crash_file).run(good=h[0], bad=h[-1])
    assert res.culprit == h[3]
    assert res.mode is BisectMode.INTRODUCED


def test_finds_fixing_commit(repo, crash_file):
    h = _history(repo, [SEGV, SEGV, SEGV, SAFE, SAFE])
    res = _bisector(repo, crash_file, mode=BisectMode.FIXED).run(good=h[0], bad=h[-1])
    assert res.culprit == h[3]


def test_adjacent_range(repo, crash_file):
    h = _history(repo, [SAFE, SEGV])
    assert _bisector(repo, crash_file).run(good=h[0], bad=h[1]).culprit == h[1]


def test_probe_count_is_logarithmic(repo, crash_file):
    h = _history(repo, [SAFE] * 16 + [SEGV] * 16)
    res = _bisector(repo, crash_file).run(good=h[0], bad=h[-1])
    assert res.culprit == h[16]
    assert res.probes <= 2 + 6  # 2 endpoint checks + ceil(log2(31)) + slack


def test_user_repo_untouched(repo, crash_file):
    h = _history(repo, [SAFE, SAFE, SEGV, SEGV])
    before = _git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain")
    _bisector(repo, crash_file).run(good=h[0], bad=h[-1])
    after = _git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain")
    assert before == after
    assert "bisect" not in _git(repo, "worktree", "list").replace(str(repo), "")


def test_worktree_removed_after_run(repo, crash_file):
    h = _history(repo, [SAFE, SEGV])
    _bisector(repo, crash_file).run(good=h[0], bad=h[1])
    assert len(_git(repo, "worktree", "list").splitlines()) == 1


# --- falsification: oracle must be able to say no -------------------------


def test_good_endpoint_that_crashes_is_rejected(repo, crash_file):
    h = _history(repo, [SEGV, SEGV, SEGV])
    with pytest.raises(CrashBisectError, match="good"):
        _bisector(repo, crash_file).run(good=h[0], bad=h[-1])


def test_bad_endpoint_that_does_not_crash_is_rejected(repo, crash_file):
    h = _history(repo, [SAFE, SAFE, SAFE])
    with pytest.raises(CrashBisectError, match="bad"):
        _bisector(repo, crash_file).run(good=h[0], bad=h[-1])


def test_control_bisecting_a_no_crash_history_never_blames(repo, crash_file):
    """Control against itself: same call shape, no crash anywhere -> no culprit."""
    h = _history(repo, [SAFE, SAFE, SAFE, SAFE])
    with pytest.raises(CrashBisectError):
        _bisector(repo, crash_file).run(good=h[0], bad=h[-1])


# --- adversarial ----------------------------------------------------------


def test_signature_pinning_ignores_a_different_crash(repo, crash_file):
    """c2..c3 abort (other bug); segv starts at c4. Pinned to SEGV -> c4."""
    h = _history(repo, [SAFE, SAFE, ABRT, ABRT, SEGV, SEGV])
    res = _bisector(repo, crash_file).run(good=h[0], bad=h[-1])
    assert res.culprit == h[4]


def test_any_crash_mode_blames_first_of_any_crash(repo, crash_file):
    h = _history(repo, [SAFE, SAFE, ABRT, ABRT, SEGV, SEGV])
    res = _bisector(repo, crash_file, match=SigMatch.ANY).run(good=h[0], bad=h[-1])
    assert res.culprit == h[2]


def test_unbuildable_commits_are_skipped_not_blamed(repo, crash_file, tmp_path):
    """3rd build (= first midpoint, after 2 endpoint builds) fails; culprit is far from it."""
    h = _history(repo, [SAFE, SEGV, SEGV, SEGV, SEGV, SEGV])
    counter = tmp_path / "builds"
    bis = CrashBisector(
        repo=str(repo),
        target="target.sh",
        crash_file=str(crash_file),
        build_cmd=f'echo x >> {counter}; test "$(wc -l < {counter})" -ne 3',
        timeout=5.0,
    )
    res = bis.run(good=h[0], bad=h[-1])
    assert res.culprit == h[1]
    assert res.skipped == 1


def test_all_candidates_unbuildable_raises_not_guesses(repo, crash_file):
    h = _history(repo, [SAFE, SAFE, SEGV])
    bis = CrashBisector(
        repo=str(repo),
        target="target.sh",
        crash_file=str(crash_file),
        build_cmd='test "$(git log -1 --format=%s)" = c0 -o "$(git log -1 --format=%s)" = c2',
        timeout=5.0,
    )
    with pytest.raises(CrashBisectError, match="skip") as exc:
        bis.run(good=h[0], bad=h[-1])
    assert h[1][:12] in str(exc.value) and h[2][:12] in str(exc.value)


def test_missing_target_is_skip_verdict(repo, crash_file):
    h = _history(repo, [SAFE, SEGV])
    bis = CrashBisector(
        repo=str(repo),
        target="nope.sh",
        crash_file=str(crash_file),
        build_cmd="true",
        timeout=5.0,
    )
    with pytest.raises(CrashBisectError):
        bis.run(good=h[0], bad=h[1])


def test_bad_revision_names_fail_cleanly(repo, crash_file):
    _history(repo, [SAFE, SEGV])
    with pytest.raises(CrashBisectError, match="revision"):
        _bisector(repo, crash_file).run(good="deadbeef", bad="HEAD")


def test_not_a_git_repo(tmp_path, crash_file):
    with pytest.raises(CrashBisectError, match="git"):
        _bisector(tmp_path, crash_file).run(good="a", bad="b")


def test_missing_crash_file(repo):
    h = _history(repo, [SAFE, SEGV])
    bis = _bisector(repo, repo / "absent.bin")
    with pytest.raises(CrashBisectError, match="crash"):
        bis.run(good=h[0], bad=h[1])


def test_verdict_enum_values_are_distinct():
    assert len({v.value for v in BisectVerdict}) == len(list(BisectVerdict))


def test_merge_history_finds_culprit_on_side_branch(repo, crash_file):
    """Non-linear DAG: culprit lives on a merged branch; git-bisect semantics required."""
    _git(repo, "init", "-q")
    base = _commit(repo, SAFE, "base")
    _git(repo, "checkout", "-q", "-b", "side")
    _commit(repo, SAFE, "s1")
    s2 = _commit(repo, SEGV, "s2")
    _git(repo, "checkout", "-q", "-")
    _commit(repo, SAFE, "m1")
    _git(repo, "merge", "-q", "--no-edit", "-X", "theirs", "side")
    head = _git(repo, "rev-parse", "HEAD")
    assert _bisector(repo, crash_file).run(good=base, bad=head).culprit == s2


def test_cli_end_to_end_and_exit_codes(repo, crash_file, tmp_path):
    import os
    import sys

    src = str(Path(__file__).resolve().parent.parent / "src")
    env = {**os.environ, "PYTHONPATH": src}
    h = _history(repo, [SAFE, SAFE, SEGV, SEGV])
    out = tmp_path / "rep.txt"
    base = [sys.executable, "-m", "fuzzer_tool", "crash-bisect", str(repo), str(crash_file)]
    common = ["--target", "target.sh", "--build-cmd", "true", "-t", "5"]
    ok = subprocess.run(
        [*base, "--good", h[0], "--bad", h[-1], *common, "-O", str(out)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert ok.returncode == 0, ok.stderr
    assert h[2] in out.read_text()
    # falsification: swapped endpoints must fail, not blame someone
    bad = subprocess.run(
        [*base, "--good", h[-1], "--bad", h[0], *common],
        capture_output=True,
        text=True,
        env=env,
    )
    assert bad.returncode == 1
