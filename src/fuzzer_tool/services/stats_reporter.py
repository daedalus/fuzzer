"""Statistics collection and crash replay for the fuzzer."""

import time


def format_elapsed(start_time: float) -> str:
    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def record_discovery_snapshot(
    exec_count: int,
    shm_cov,
    ptrace_cov,
    discovery_history: list[tuple[int, int]],
) -> None:
    """Record (exec_count, cumulative_edges) for discovery rate calculation."""
    edges = 0
    if shm_cov:
        edges = shm_cov.cumulative_edges
    elif ptrace_cov:
        edges = ptrace_cov.cumulative_edges
    discovery_history.append((exec_count, edges))
    if len(discovery_history) > 500:
        del discovery_history[:250]


def discovery_rate(discovery_history: list[tuple[int, int]]) -> float:
    """Edges discovered per 1000 execs, over a sliding window of last 5 snapshots."""
    if len(discovery_history) < 2:
        return 0.0
    window = discovery_history[-5:]
    first_exec, first_edges = window[0]
    last_exec, last_edges = window[-1]
    exec_delta = last_exec - first_exec
    edge_delta = last_edges - first_edges
    if exec_delta <= 0:
        return 0.0
    return edge_delta / exec_delta * 1000


def run_crash_replays(
    crashes_dir,
    target: str,
    timeout: float,
    crash_replays: dict[str, list[int]],
    replay_n: int,
    seed_key_fn,
    budget_ms: float = 200,
) -> None:
    """Replay pending crashes for reproducibility scoring (non-blocking)."""
    if replay_n <= 0 or not crash_replays:
        return
    from fuzzer_tool.adapters.process import run_target_stdin

    t0 = time.monotonic()
    pending = [(sig, replays) for sig, replays in crash_replays.items() if len(replays) < replay_n]
    for sig, replays in pending:
        if (time.monotonic() - t0) * 1000 > budget_ms:
            break
        crash_file = None
        for f in crashes_dir.iterdir():
            if f.is_file() and not f.name.endswith((".json", ".txt")):
                try:
                    crash_data = f.read_bytes()
                    if seed_key_fn(crash_data) == sig or f.stem.startswith(sig[:12]):
                        crash_file = f
                        break
                except Exception:
                    continue
        if crash_file is None:
            for f in crashes_dir.iterdir():
                if f.is_file() and sig[:12] in f.name:
                    crash_file = f
                    break
        if crash_file is None:
            replays.append(-3)
            continue
        try:
            data = crash_file.read_bytes()
            rc, _ = run_target_stdin(target, data, timeout)
            replays.append(rc)
        except Exception:
            replays.append(-2)
