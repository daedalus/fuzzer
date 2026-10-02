"""Regression: masked crashes that surface as timeouts.

Chalmers "Escaping the Fuzz" (2016) §5.3.1 hid every crash from AFL by
running ``main`` in a forked child and exiting 0 when it was signalled; some
of those crashes showed up as *hangs*. Reproduced here against this tool:

  * the loader (``fuzz_loader.c``) SIGKILLed only its direct child on a
    timeout, so the wrapper died, the crashing grandchild lived on, held the
    stderr pipe open past the deadline and leaked as an orphan;
  * ``is_crash`` returned False for rc -1 before parsing stderr, so a
    sanitizer report that arrived with a timeout was filed as a hang;
  * the spawn paths replaced a timed-out run's stderr with ``"timeout"``,
    dropping whatever report the target had already written;
  * a timed-out input was never re-run, so any crash slower than the
    deadline was lost in every mode.

Targets are ``/bin/sh`` scripts: the code under test is process handling.
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.adapters.forkserver import ForkserverRunner
from fuzzer_tool.adapters.process import (
    _STDERR_CAP,
    run_target_fast,
    run_target_file,
    run_target_stdin,
)
from fuzzer_tool.services import runner as runner_mod
from fuzzer_tool.services.runner import TargetRunner

REPORT = "==1==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x1\n"

# Short enough to keep the suite fast, long enough to absorb sh start-up.
DEADLINE = 0.3


@pytest.fixture
def script(tmp_path):
    """Write an executable /bin/sh target and hand back its path."""
    made: list[str] = []

    def _make(name: str, body: str) -> str:
        p = tmp_path / f"{name}.sh"
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(0o755)
        made.append(str(p))
        return str(p)

    yield _make
    for path in made:
        if os.path.exists(path):
            os.unlink(path)


def _alive(pid: int) -> bool:
    """True when *pid* runs and is not a zombie (PID 1 may never reap it)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ")[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _kill_all(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            continue


def _fake_fuzzer(**kw) -> MagicMock:
    """A MagicMock fuzzer routed to the spawn fast path."""
    f = MagicMock()
    f._target_shm_covs = {}
    f.target = kw.get("target", "/bin/true")
    f.multi_targets = None
    f.shm_cov = None
    f._cmplog = None
    f._inprocess_runner = None
    f._persistent_runner = None
    f._network_runner = None
    f.ptrace_cov = None
    f._forkserver = kw.get("forkserver")
    f.use_coverage = False
    f.file_mode = False
    f.timeout = DEADLINE
    f.exec_count = 0
    f._perf_counters = None
    f._pt_session = None
    f._lbr_session = None
    f.pt_cov = None
    f.branch_cov = None
    f.extra_crash_codes = ()
    return f


# ── (a) loader kills the whole process group ─────────────────────────────


def test_regression_loader_kills_grandchild_on_timeout(script, tmp_path):
    """A grandchild must die with its wrapper, inside the deadline."""
    pidfile = tmp_path / "pids"
    target = script(
        "wrap_hang",
        f"sh -c 'echo $$ >> {pidfile}; exec sleep 30'\nexit 0\n",
    )
    fs = ForkserverRunner(target, timeout=DEADLINE)
    try:
        assert fs.start(), "fuzz_loader did not start"
        t0 = time.monotonic()
        rc, _ = fs.run_one(b"x")
        elapsed = time.monotonic() - t0
    finally:
        fs.stop()

    pids = [int(p) for p in pidfile.read_text().split()]
    survivors = [p for p in pids if _alive(p)]
    _kill_all(survivors)

    assert rc == -1
    # Under the loader's grace second: no Python-side restart was needed.
    assert elapsed < DEADLINE + 0.5, f"deadline not enforced on descendants ({elapsed:.2f}s)"
    assert pids, "target never started its grandchild"
    assert survivors == [], f"orphaned grandchildren: {survivors}"


def test_loader_clean_run_unaffected(script):
    """Falsification: a wrapper whose child exits cleanly reports its status."""
    target = script("wrap_ok", "sh -c 'exit 3'\nexit $?\n")
    fs = ForkserverRunner(target, timeout=2.0)
    try:
        assert fs.start()
        rc, _ = fs.run_one(b"x")
    finally:
        fs.stop()
    assert rc == 3


# ── (b) a sanitizer report that came with a timeout is a crash ───────────


def test_regression_timeout_with_report_is_crash():
    assert TargetRunner(_fake_fuzzer()).is_crash(-1, REPORT)


@pytest.mark.parametrize(
    "rc, stderr",
    [
        (-1, "timeout"),  # plain hang
        (-1, ""),  # forkserver hang sentinel
        (-1, "AddressSanitizer"),  # keyword without an error type
        (-2, REPORT),  # infrastructure failure never vouches for a crash
    ],
)
def test_timeout_without_valid_report_is_not_crash(rc, stderr):
    """Adversarial: only a parseable report upgrades a timeout."""
    assert not TargetRunner(_fake_fuzzer()).is_crash(rc, stderr)


# ── (c) partial stderr survives a timeout ────────────────────────────────


def _spawn(kind: str, target: str, tmp_path) -> tuple[int, str]:
    if kind == "fast":
        rc, err, _ = run_target_fast(target, b"x", timeout=DEADLINE)
    elif kind == "stdin":
        rc, err, _ = run_target_stdin(target, b"x", timeout=DEADLINE)
    else:
        rc, err, _ = run_target_file(target, b"x", DEADLINE, str(tmp_path), [])
    return rc, err


@pytest.mark.parametrize("kind", ["fast", "stdin", "file"])
def test_regression_timeout_keeps_partial_stderr(kind, script, tmp_path):
    target = script("report_then_hang", f"printf '{REPORT.strip()}\\n' >&2\nexec sleep 30\n")
    rc, err = _spawn(kind, target, tmp_path)
    assert rc == -1
    assert "AddressSanitizer: stack-buffer-overflow" in err


@pytest.mark.parametrize("kind", ["fast", "stdin", "file"])
def test_silent_timeout_keeps_sentinel(kind, script, tmp_path):
    """Falsification: no output still reads as the "timeout" sentinel."""
    rc, err = _spawn(kind, script("silent_hang", "exec sleep 30\n"), tmp_path)
    assert (rc, err) == (-1, "timeout")


@pytest.mark.parametrize("kind", ["fast", "stdin", "file"])
def test_timeout_stderr_bounded(kind, script, tmp_path):
    """Adversarial: a flood before the hang stays within the stderr cap."""
    target = script("flood_hang", "head -c 300000 /dev/zero | tr '\\0' 'A' >&2\nexec sleep 30\n")
    rc, err = _spawn(kind, target, tmp_path)
    assert rc == -1
    assert 0 < len(err) <= _STDERR_CAP


# ── (d) hang confirmation ────────────────────────────────────────────────


def test_confirm_hang_reruns_with_longer_deadline(monkeypatch):
    f = _fake_fuzzer()
    seen: list[float] = []

    def _spy(*_a, **kw):
        seen.append(kw["timeout"])
        return (1, REPORT, 4242)

    monkeypatch.setattr(runner_mod, "run_target_fast", _spy)
    result = TargetRunner(f).confirm_hang(b"x")

    assert result == (1, REPORT)
    assert seen == [DEADLINE * runner_mod.HANG_CONFIRM_FACTOR]
    assert f.timeout == DEADLINE, "deadline not restored"


def test_confirm_hang_restores_deadline_on_error(monkeypatch):
    """Adversarial: a backend that raises must not leave the long deadline."""
    f = _fake_fuzzer()

    def _boom(*_a, **_kw):
        raise RuntimeError("backend died")

    monkeypatch.setattr(runner_mod, "run_target_fast", _boom)
    with pytest.raises(RuntimeError):
        TargetRunner(f).confirm_hang(b"x")
    assert f.timeout == DEADLINE


def test_confirm_hang_budget(monkeypatch):
    """Adversarial: a target that always hangs cannot multiply the cost."""
    f = _fake_fuzzer()
    calls = []
    monkeypatch.setattr(
        runner_mod, "run_target_fast", lambda *a, **kw: calls.append(1) or (-1, "timeout", 1)
    )
    tr = TargetRunner(f)

    for _ in range(runner_mod.HANG_CONFIRM_FREE):
        assert tr.confirm_hang(b"x") is not None
    assert tr.confirm_hang(b"x") is None, "budget not enforced"
    assert len(calls) == runner_mod.HANG_CONFIRM_FREE

    f.exec_count = runner_mod.HANG_CONFIRM_EVERY
    assert tr.confirm_hang(b"x") is not None, "budget does not grow with executions"


@pytest.mark.parametrize("backend", ["_inprocess_runner", "_persistent_runner", "_network_runner"])
def test_confirm_hang_skips_fixed_deadline_backends(backend, monkeypatch):
    """Adversarial: backends whose deadline cannot be stretched are not re-run."""
    f = _fake_fuzzer()
    setattr(f, backend, MagicMock())
    monkeypatch.setattr(runner_mod, "run_target_fast", MagicMock(side_effect=AssertionError))
    assert TargetRunner(f).confirm_hang(b"x") is None
    getattr(f, backend).run_one.assert_not_called()


def test_regression_slow_masked_crash_found_via_loader(script):
    """End to end: Listing 5 wrapper, child reports after the deadline."""
    target = script(
        "wrap_slow_crash",
        'sh -c \'sleep 0.6; printf "%s\\n" "' + REPORT.strip() + "\" >&2; kill -ABRT $$'\nexit 0\n",
    )
    fs = ForkserverRunner(target, timeout=DEADLINE)
    try:
        assert fs.start()
        f = _fake_fuzzer(target=target, forkserver=fs)
        tr = TargetRunner(f)
        first = tr.run_target(b"x")
        confirmed = tr.confirm_hang(b"x")
        restored = fs.timeout
    finally:
        fs.stop()

    assert first[0] == -1
    assert confirmed is not None
    assert tr.is_crash(*confirmed), f"masked crash not recovered: {confirmed!r}"
    assert restored == pytest.approx(DEADLINE)


# ── FuzzRound wiring ─────────────────────────────────────────────────────


@pytest.fixture
def fuzzer(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        yield Fuzzer(
            target="/bin/true",
            corpus_dir=str(tmp_path / "corpus"),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=256,
            timeout=1,
            mutations_per_input=2,
        )


def _round(f, run_result, confirm_result):
    with (
        patch.object(f, "_dedup_mutate", return_value=b"MUTANT01"),
        patch.object(f, "_run_target", return_value=run_result),
        patch.object(f, "_confirm_hang", return_value=confirm_result) as confirm,
    ):
        f.fuzz_one(f.corpus[0])
    return confirm


def test_regression_round_counts_reported_timeout_as_crash(fuzzer):
    crashes, timeouts = fuzzer.crash_count, fuzzer.timeout_count
    _round(fuzzer, (-1, REPORT), None)
    assert fuzzer.crash_count == crashes + 1
    assert fuzzer.timeout_count == timeouts


def test_round_confirms_hang_into_crash(fuzzer):
    crashes, timeouts = fuzzer.crash_count, fuzzer.timeout_count
    confirm = _round(fuzzer, (-1, "timeout"), (1, REPORT))
    confirm.assert_called_once_with(b"MUTANT01")
    assert fuzzer.crash_count == crashes + 1
    assert fuzzer.timeout_count == timeouts


def test_round_confirmed_hang_stays_timeout(fuzzer):
    """Falsification: a real hang is still a hang after confirmation."""
    crashes, timeouts = fuzzer.crash_count, fuzzer.timeout_count
    _round(fuzzer, (-1, "timeout"), (-1, "timeout"))
    assert fuzzer.crash_count == crashes
    assert fuzzer.timeout_count == timeouts + 1


def test_round_skips_confirmation_when_not_timed_out(fuzzer):
    """Adversarial: clean runs never pay for a re-run."""
    confirm = _round(fuzzer, (0, ""), None)
    confirm.assert_not_called()
