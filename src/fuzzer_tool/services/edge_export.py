"""Edge export: replay every corpus seed and pickle its edge ids. No pruning.

    corpus/seeds/** --replay--> ShmCoverage --> {hash_data(seed): uint32[edge ids]} --> out.pkl

Replay backends (spawn, loader, direct_lite, direct) are minimize's.
"""

import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.adapters.process import disable_aslr
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.services.minimize import (
    _TIMEOUT_RC,
    _TRIM_EVERY,
    DEFAULT_INPROCESS_FUNC,
    ExecMode,
    _discover_corpus_files,
    _maxrss_kb,
    _open_runner,
    _print_progress,
    _runner_replay,
    _spawn_replay,
    _split_oversized,
    _trim_heap,
)

# Seeds between progress lines, as in minimize.
_PROGRESS_EVERY = 10


def export_edges(
    target: str,
    corpus_dir: str,
    out_path: str,
    timeout: float = 1.0,
    file_mode: bool = False,
    target_args: list[str] | None = None,
    exec_mode: ExecMode = ExecMode.SPAWN,
    inprocess_func: str = DEFAULT_INPROCESS_FUNC,
    max_seed_bytes: int = 0,
) -> int:
    """Pickle ``{seed_id: sorted uint32 edge ids}`` for every live seed.

    seed_id is ``hash_data(seed)``, the corpus filename key. The corpus is
    read only. A total blackout (no edges anywhere) is a failed measurement:
    nothing is written.

    Returns:
        Seeds exported; 0 when nothing was written.
    """
    files = _discover_corpus_files(Path(corpus_dir))
    if not files:
        print(f"[-] Corpus is empty: {corpus_dir}", file=sys.stderr)
        return 0

    to_replay, oversized = _split_oversized(files, max_seed_bytes)
    if oversized:
        print(f"[*] Skipped: {len(oversized)} seeds over {max_seed_bytes} bytes")

    edges = _replay_all(
        to_replay, target, timeout, file_mode, target_args, exec_mode, inprocess_func
    )
    if edges is None:
        return 0

    # Blackout guard, same reading as cmin: no edges means no measurement.
    if not any(a.size for a in edges.values()):
        print("[-] No edges recorded for any seed -- nothing written.", file=sys.stderr)
        return 0

    _write_atomic(Path(out_path), edges)
    print(f"[+] Exported {len(edges)} seeds -> {out_path}")
    return len(edges)


def _replay_all(files, target, timeout, file_mode, target_args, exec_mode, func):
    """``{seed_id: edge ids}`` over *files*, or None when the runner cannot start."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="edump_"))

    # Caller-context edge ids hash a return address; ASLR makes them noise.
    disable_aslr()

    shm = ShmCoverage()
    env = os.environ.copy()
    env["__AFL_SHM_ID"] = shm.env_id
    env["AFL_MAP_SIZE"] = str(shm.num_entries)

    edges: dict[str, np.ndarray] = {}
    timeouts = 0
    runner = None
    try:
        # Backend chosen once; the loop does not branch on mode.
        if exec_mode is ExecMode.SPAWN:
            replay = _spawn_replay(target, timeout, file_mode, tmp_dir, target_args, env)
        else:
            runner = _open_runner(target, exec_mode, func, timeout, shm)
            if runner is None:
                return None
            replay = _runner_replay(runner)

        n = len(files)
        start_time = time.time()
        unique: set[int] = set()
        peak_rss_kb = _maxrss_kb()
        for i, fpath in enumerate(files):
            if not fpath.exists():
                print(f"[!] Skipping missing seed: {fpath}")
                continue

            data = fpath.read_bytes()
            shm.reset_edge_map()
            timeouts += replay(data)[0] == _TIMEOUT_RC
            ids = shm.edge_id_array()
            edges[hash_data(data)] = ids
            unique.update(ids.tolist())
            peak_rss_kb = max(peak_rss_kb, _maxrss_kb())

            if (i + 1) % _TRIM_EVERY == 0:
                _trim_heap()

            if (i + 1) % _PROGRESS_EVERY == 0 or i + 1 == n:
                _print_progress(i + 1, n, start_time, len(unique), peak_rss_kb)
    finally:
        if runner is not None:
            runner.stop()
        shm.cleanup()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print()
    if timeouts:
        print(f"[*] Timed out: {timeouts}/{len(files)} seeds (partial edges kept)")
    return edges


def _write_atomic(out: Path, edges: dict[str, np.ndarray]) -> None:
    """Pickle *edges* to *out* via tmp + rename: no half-written file on crash."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(edges, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)
