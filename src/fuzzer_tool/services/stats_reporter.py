"""Statistics collection and crash replay for the fuzzer."""

import time
from array import array
from enum import IntEnum

from fuzzer_tool.adapters.process import SIGNAL_CRASH_CODES, stderr_crash_marker
from fuzzer_tool.core.sanitizer import SanitizerReport

# Adapter sentinels: -1 timeout, -2 infrastructure failure.
_SENTINEL_CODES = (-1, -2)


class ReplayOutcome(IntEnum):
    """Verdict of one crash replay, stored in ``_crash_replays``."""

    CLEAN = 0
    CRASHED = 1
    ERROR = -2
    MISSING = -3


def _replay_crashed(rc: int, stderr: str) -> bool:
    """Did a replay crash? Raw ``rc`` is not enough: signals come back negative."""
    if rc in SIGNAL_CRASH_CODES:
        return True

    if rc < 0 and rc not in _SENTINEL_CODES:
        return True

    report = SanitizerReport.parse(stderr)
    if report and report.is_valid():
        return True

    return stderr_crash_marker(rc, stderr) is not None


def repro_rate(replays) -> float:
    """Fraction of replays that crashed again; 0.0 when none ran."""
    if not replays:
        return 0.0
    return sum(1 for r in replays if r == ReplayOutcome.CRASHED) / len(replays)


def format_elapsed(start_time: float, now: float | None = None) -> str:
    # `now` from the run's decision clock, so --clock virtual reads consistently.
    elapsed = (time.time() if now is None else now) - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def record_discovery_snapshot(
    exec_count: int,
    shm_cov,
    ptrace_cov,
    discovery_history: tuple[array, array, array],
) -> None:
    """Record (exec_count, cumulative_edges, timestamp) for discovery rate calculation."""
    edges = 0
    if shm_cov:
        edges = shm_cov.cumulative_edges
    elif ptrace_cov:
        edges = ptrace_cov.cumulative_edges
    discovery_execs, discovery_edges, discovery_timestamps = discovery_history
    discovery_execs.append(exec_count)
    discovery_edges.append(edges)
    discovery_timestamps.append(time.time())
    if len(discovery_execs) > 500:
        del discovery_execs[:250]
        del discovery_edges[:250]
        del discovery_timestamps[:250]


def discovery_rate(discovery_history: tuple[array, array, array]) -> float:
    """Edges discovered per 1000 execs, over a sliding window of last 5 snapshots."""
    discovery_execs, discovery_edges, _ = discovery_history
    if len(discovery_execs) < 2:
        return 0.0
    recent = list(zip(discovery_execs[-5:], discovery_edges[-5:], strict=True))
    first_exec, first_edges = recent[0]
    last_exec, last_edges = recent[-1]
    exec_delta = last_exec - first_exec
    edge_delta = last_edges - first_edges
    if exec_delta <= 0:
        return 0.0
    return edge_delta / exec_delta * 1000


def _scan_crash_file(crashes_dir, sig: str, seed_key_fn):
    """Fallback: first crash file whose content hash equals ``sig``."""
    for f in crashes_dir.iterdir():
        if f.is_file() and not f.name.endswith((".json", ".txt")):
            try:
                if seed_key_fn(f.read_bytes()) == sig:
                    return f
            except Exception:
                continue
    return None


def _find_crash_file(crashes_dir, sig: str, crash_files, seed_key_fn):
    """Recorded ``save_crash()`` file for ``sig``, else content-hash scan."""
    base_name = (crash_files or {}).get(sig)
    if base_name:
        candidate = crashes_dir / f"{base_name}.bin"
        if candidate.is_file():
            return candidate
    return _scan_crash_file(crashes_dir, sig, seed_key_fn)


def run_crash_replays(
    crashes_dir,
    target: str,
    timeout: float,
    crash_replays: dict[str, list[int]],
    replay_n: int,
    seed_key_fn,
    budget_ms: float = 200,
    crash_files: dict[str, str] | None = None,
) -> None:
    """Replay pending crashes for reproducibility scoring (non-blocking).

    ``crash_files`` maps a crash signature to the base name ``save_crash()``
    wrote it under, which is the only exact way back to the input that
    produced that signature.

    Without it the lookup was a guess, and a wrong one: the fallback matched
    ``f.stem.startswith(sig[:12])`` against names shaped
    ``crash_<unix_ts>_<cluster>_<san>_<err>``, so twelve characters cover
    ``crash_`` plus six digits of a ten-digit timestamp -- every crash within
    the same 10^4-second (~2.7h) window compares equal, and the FIRST such
    file in directory order won. Reproducibility scores were therefore
    computed by replaying some other crash's input. The sibling
    ``seed_key_fn(crash_data) == sig`` test compared a content hash against a
    crash signature and could not match at all. Finding #22.

    The scan survives only as a fallback for signatures with no recorded
    file -- state restored from an older run -- and now uses the content-hash
    identity alone, never a prefix.
    """
    if replay_n <= 0 or not crash_replays:
        return
    from fuzzer_tool.adapters.process import run_target_stdin

    t0 = time.monotonic()
    pending = [(sig, replays) for sig, replays in crash_replays.items() if len(replays) < replay_n]
    for sig, replays in pending:
        if (time.monotonic() - t0) * 1000 > budget_ms:
            break
        crash_file = _find_crash_file(crashes_dir, sig, crash_files, seed_key_fn)
        if crash_file is None:
            replays.append(ReplayOutcome.MISSING)
            continue
        try:
            data = crash_file.read_bytes()
            rc, stderr, _pid = run_target_stdin(target, data, timeout)
            crashed = _replay_crashed(rc, stderr)
            replays.append(ReplayOutcome.CRASHED if crashed else ReplayOutcome.CLEAN)
        except Exception:
            replays.append(ReplayOutcome.ERROR)
