"""Igor root-cause crash clustering (E4, CCS '21): ``fuzzer-tool crash-cluster``.

Stack hashes split one bug into many buckets. Igor instead shrinks each
crash's *path*, then compares paths::

    crash_*.bin --TraceProbe--> Trace(sig, graph)
        |
        CoverageReducer: mutate, keep a mutant iff same sig and fewer edges
        |
        wl_features -> kernel_matrix -> cluster_kernel | cluster_spectral   (core/crash_graph.py)
        |
    root-cause clusters

Byte minimization (tmin) shortens the input; coverage reduction drops the
code the bug does not need, which is what makes two paths comparable.
"""

import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from fuzzer_tool.adapters.process import disable_aslr, run_target_file, run_target_stdin
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.crash_graph import (
    WL_ITERS,
    CrashGraph,
    Linkage,
    cluster_kernel,
    cluster_spectral,
    graph_from_edges,
    graph_from_fires,
    kernel_matrix,
    wl_features,
)
from fuzzer_tool.core.elf import estimate_map_size
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.run_mode import InputChannel
from fuzzer_tool.services.tmin import crash_sig

CRASH_INPUT_GLOB = "crash_*.bin"
# Coverage-reduction execs per crash.
REDUCE_EXECS = 200
# WL cosine at or above which two crashes share a root cause.
CLUSTER_THRESHOLD = 0.7
_BYTE_MAX = 255


@dataclass(frozen=True, slots=True)
class Trace:
    """One run: crash signature (None = no crash) and its edge graph."""

    sig: str | None
    graph: CrashGraph


class TraceProbe:
    """Driver: run one input on a spawned target, return its Trace.

    Edges come from the SHM map; arcs from ``$__AFL_FIRES_OUT`` when the
    target was built with ``-D__AFL_TRACE_FIRES=1`` (otherwise the log stays
    empty and the graph is the edge set).
    """

    def __init__(
        self,
        target: str,
        timeout: float,
        channel: InputChannel = InputChannel.STDIN,
        target_args: list[str] | None = None,
    ):
        # Caller-context edge ids hash return addresses; ASLR makes them noise.
        disable_aslr()

        self._tmp = Path(tempfile.mkdtemp(prefix="crash_cluster_"))
        self._fires = self._tmp / "fires.log"
        self._shm = ShmCoverage(estimate_map_size(target))

        env = os.environ.copy()
        env["__AFL_SHM_ID"] = self._shm.env_id
        env["AFL_MAP_SIZE"] = str(self._shm.num_entries)
        env["__AFL_FIRES_OUT"] = str(self._fires)

        # Delivery decided once; __call__ does not branch on it.
        if channel is InputChannel.FILE:
            args = target_args or []
            self._run = lambda d: run_target_file(target, d, timeout, str(self._tmp), args, env=env)
        else:
            self._run = lambda d: run_target_stdin(target, d, timeout, env=env)

    def __call__(self, data: bytes) -> Trace:
        self._fires.unlink(missing_ok=True)
        self._shm.reset_edge_map()

        rc, stderr, _pid = self._run(data)

        return Trace(crash_sig(rc, stderr), self._graph())

    def _graph(self) -> CrashGraph:
        """Fire-log graph when the build logs fires, else the SHM edge set."""
        if self._fires.is_file() and self._fires.stat().st_size:
            with self._fires.open() as fh:
                return graph_from_fires(int(line) for line in fh)
        return graph_from_edges(self._shm.edge_id_array().tolist())

    def close(self) -> None:
        self._shm.cleanup()
        shutil.rmtree(self._tmp, ignore_errors=True)


class CoverageReducer:
    """Coverage-reduction fuzzing: shrink the path, keep the bug.

    A mutant replaces the current input iff it crashes with the same
    signature and touches fewer edges (equal edges: shorter input).
    """

    def __init__(self, probe: Callable[[bytes], Trace], rng, max_execs: int = REDUCE_EXECS):
        self._probe = probe
        self._rng = rng
        self._max_execs = max_execs
        self._ops = (self._cut, self._byte, self._trunc)

    def reduce(self, data: bytes, trace: Trace) -> tuple[bytes, Trace]:
        """Reduce *data* (whose run is *trace*) within the exec budget."""
        last = len(self._ops) - 1

        for _ in range(self._max_execs):
            if not data:
                break

            cand = self._ops[self._rng.randint(0, last)](data)
            got = self._probe(cand)
            if got.sig != trace.sig:
                continue

            # Fewer edges wins; at equal edges only a shorter input does.
            key = (got.graph.size, len(cand))
            if key < (trace.graph.size, len(data)):
                data, trace = cand, got

        return data, trace

    def _cut(self, data: bytes) -> bytes:
        """Delete a random chunk."""
        start = self._rng.randint(0, len(data) - 1)
        ln = self._rng.randint(1, len(data) - start)
        return data[:start] + data[start + ln :]

    def _byte(self, data: bytes) -> bytes:
        """Overwrite one byte: may steer around code the bug does not need."""
        i = self._rng.randint(0, len(data) - 1)
        return data[:i] + bytes((self._rng.randint(0, _BYTE_MAX),)) + data[i + 1 :]

    def _trunc(self, data: bytes) -> bytes:
        """Drop a random tail."""
        return data[: self._rng.randint(0, len(data) - 1)]


def cluster_inputs(
    paths: list[Path],
    probe: Callable[[bytes], Trace],
    rng,
    max_execs: int = REDUCE_EXECS,
    iters: int = WL_ITERS,
    threshold: float = CLUSTER_THRESHOLD,
    linkage: Linkage = Linkage.THRESHOLD,
) -> list[list[int]]:
    """Cluster crash inputs by reduced-path similarity.

    Returns clusters of indices into *paths*; inputs that do not reproduce
    a crash are left out. *threshold* applies to ``Linkage.THRESHOLD`` only.
    """
    # Grouping chosen once, not per call.
    group = {
        Linkage.THRESHOLD: lambda sim: cluster_kernel(sim, threshold),
        Linkage.SPECTRAL: cluster_spectral,
    }[linkage]
    reducer = CoverageReducer(probe, rng, max_execs)
    kept: list[int] = []
    feats: list[dict[int, int]] = []

    for i, path in enumerate(paths):
        data = path.read_bytes()
        trace = probe(data)
        if trace.sig is None:
            continue

        _data, trace = reducer.reduce(data, trace)
        kept.append(i)
        feats.append(wl_features(trace.graph, iters))

    groups = group(kernel_matrix(feats))
    return [[kept[j] for j in g] for g in groups]


def _crash_inputs(crashes_dir: str) -> Iterator[Path]:
    return iter(sorted(Path(crashes_dir).glob(CRASH_INPUT_GLOB)))


def crash_cluster(
    target: str,
    crashes_dir: str,
    timeout: float = 1.0,
    channel: InputChannel = InputChannel.STDIN,
    target_args: list[str] | None = None,
    max_execs: int = REDUCE_EXECS,
    iters: int = WL_ITERS,
    threshold: float = CLUSTER_THRESHOLD,
    seed: int | None = None,
    linkage: Linkage = Linkage.THRESHOLD,
) -> list[list[Path]]:
    """Cluster ``crash_*.bin`` in *crashes_dir* by root cause; print the groups."""
    paths = list(_crash_inputs(crashes_dir))
    probe = TraceProbe(target, timeout, channel, target_args)
    try:
        groups = cluster_inputs(paths, probe, RandPool(seed), max_execs, iters, threshold, linkage)
    finally:
        probe.close()

    clusters = [[paths[i] for i in g] for g in groups]
    kept = sum(len(c) for c in clusters)
    print(
        f"[*] {kept}/{len(paths)} crash(es) reproduced -> {len(clusters)} "
        f"root-cause cluster(s) ({linkage.value})"
    )
    for n, members in enumerate(clusters):
        print(f"  cluster {n} ({len(members)}): " + ", ".join(p.name for p in members))
    return clusters
