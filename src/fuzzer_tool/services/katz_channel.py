"""Runtime K-Scheduler channel: node bitmap → horizon graph → Katz energy.

Owns the lifecycle the W2/W3/W4 modules assume:

1. **build/upload** — whole-program ICFG (W1), the probe-key→node table,
   a ``DistanceTableShm`` carrying ``node_idx``, and a ``NodeBitmapShm``
   segment exported via ``__AFL_NODE_BITMAP_ID``. Only viable on trace-pc
   targets; returns None otherwise so non-instrumented campaigns are
   untouched.
2. **sample/record** — Python reads-and-clears the bitmap after EVERY
   execution (eager C-side writes need no destructor). Global per-node
   hit counts feed β; per-seed OR-accumulated masks feed V and seed
   attachment. Masks are keyed by the same content hash EdgeTracker uses.
3. **scores** — lazily recomputed horizon+Katz when new coverage arrived
   and a minimum interval elapsed (the paper's recompute trigger).
"""

import logging
import os

import numpy as np

from fuzzer_tool.core.horizon import HorizonGraph, build_horizon_graph
from fuzzer_tool.core.icfg import (
    InterproceduralCFG,
    build_interprocedural_cfg,
    probe_key_node_table,
)
from fuzzer_tool.core.schedulers.katz import katz_scores

log = logging.getLogger(__name__)

# Minimum executions between Katz recomputes (the paper's timer trigger;
# new-coverage dirtiness alone would recompute every exec early on).
_RECOMPUTE_MIN_INTERVAL = 50


class KatzChannel:
    """Per-campaign K-Scheduler state and SHM plumbing."""

    def __init__(self, icfg: InterproceduralCFG, node_of: dict[int, int]):
        self.icfg = icfg
        self.node_of = node_of
        self.n_nodes = icfg.n_nodes
        self.hit_counts = np.zeros(self.n_nodes, dtype=np.float64)
        self._masks: dict[str, bytes] = {}
        self.table_shm = None
        self.bmp = None
        self._horizon: HorizonGraph | None = None
        self._scores = None
        self._dirty = True
        self._last_recompute_exec = -_RECOMPUTE_MIN_INTERVAL
        self.exec_count = 0  # caller updates so the interval gate works

    # ── setup ────────────────────────────────────────────────────────

    @classmethod
    def build(cls, target: str, use_cfg_cache: bool = True) -> "KatzChannel | None":
        """Detect viability and build the ICFG; None when not applicable."""
        td_load_ok = _target_has_trace_pc(target)
        if not td_load_ok:
            return None
        from fuzzer_tool.core.distance import TargetDistance

        td = TargetDistance(target, use_cfg_cache=use_cfg_cache)
        if not td.load():
            return None
        icfg = build_interprocedural_cfg(td)
        if icfg is None or icfg.n_nodes == 0:
            return None
        node_of = probe_key_node_table(td, icfg)
        if not node_of:
            return None
        ch = cls(icfg, node_of)
        ch._td = td
        return ch

    def upload(self) -> bool:
        """Upload distance/node tables + bitmap; export env vars."""
        from fuzzer_tool.adapters.shm import DistanceTableShm, NodeBitmapShm

        dist = self._td.pc_distance_table()
        table = {key: float(dist.get(key, 0.0)) for key in self.node_of}
        try:
            self.table_shm = DistanceTableShm(table, node_of=self.node_of)
            self.bmp = NodeBitmapShm(num_nodes=self.n_nodes)
        except OSError as e:
            log.warning("Katz channel SHM upload failed: %s", e)
            self.table_shm = self.bmp = None
            return False
        if self.table_shm.shm_id >= 0:
            os.environ["__AFL_DIST_SHM_ID"] = self.table_shm.env_id
        if self.bmp.shm_id >= 0:
            os.environ["__AFL_NODE_BITMAP_ID"] = self.bmp.env_id
        return True

    def cleanup(self):
        if self.bmp is not None:
            self.bmp.cleanup()
        if self.table_shm is not None:
            self.table_shm.cleanup()

    # ── sampling ─────────────────────────────────────────────────────

    def sample(self) -> np.ndarray | None:
        """Read-and-clear one execution's bitmap; None without a channel."""
        if self.bmp is None:
            return None
        buf = self.bmp.read_and_clear()
        bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")[
            : self.n_nodes
        ].astype(bool)
        return bits

    def record(self, bits: np.ndarray, seed_key: str | None = None):
        """Accumulate one execution: global hits always, per-seed mask only
        for inputs that earned corpus membership (has_new_coverage)."""
        self.hit_counts += bits.astype(np.float64)
        if seed_key is not None:
            packed = np.packbits(bits, bitorder="little").tobytes()
            old = self._masks.get(seed_key)
            if old is None:
                self._masks[seed_key] = packed
            else:
                cur = np.frombuffer(old, dtype=np.uint8)
                self._masks[seed_key] = np.bitwise_or(
                    cur, np.frombuffer(packed, dtype=np.uint8)
                ).tobytes()
            # New OR merged bits change V / seed attachment either way.
            self._dirty = True
        self.exec_count += 1

    # ── scores ───────────────────────────────────────────────────────

    def ensure_scores(self, force: bool = False):
        """Recompute horizon+Katz when dirty and the interval allows."""
        due = self.exec_count - self._last_recompute_exec >= _RECOMPUTE_MIN_INTERVAL
        if self._scores is None or (self._dirty and due) or force:
            masks = {k: v for k, v in self._masks.items() if len(v) * 8 >= self.n_nodes}
            self._horizon = build_horizon_graph(self.icfg, masks)
            self._scores = katz_scores(self._horizon, hit_counts=self.hit_counts.copy())
            self._dirty = False
            self._last_recompute_exec = self.exec_count
        return self._scores

    def seed_energy(self, seed_key: str) -> float:
        """Normalized [0,1] centrality of a corpus seed's graph node."""
        res = self.ensure_scores()
        n_u = self._horizon.n_u
        idx = (
            n_u + self._horizon.seed_names.index(seed_key)
            if seed_key in self._horizon.seed_names
            else None
        )
        if idx is None:
            return 0.0
        peak = float(res.scores.max()) if res.scores.size else 0.0
        if peak <= 0:
            return 0.0
        return min(float(res.scores[idx]) / peak, 1.0)

    # ── persistence ──────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "hit_counts": self.hit_counts.tolist(),
            "masks": {k: v.hex() for k, v in self._masks.items()},
            "exec_count": self.exec_count,
        }

    def load_state_dict(self, state: dict):
        hc = state.get("hit_counts")
        if isinstance(hc, list) and len(hc) == self.n_nodes:
            self.hit_counts = np.asarray(hc, dtype=np.float64)
        for key, hexmask in (state.get("masks") or {}).items():
            try:
                packed = bytes.fromhex(hexmask)
            except ValueError:
                continue
            if len(packed) * 8 >= self.n_nodes:
                self._masks[key] = packed
        self.exec_count = int(state.get("exec_count", 0))
        self._dirty = True
        # Resume must refresh promptly rather than wait out the interval.
        self._last_recompute_exec = -_RECOMPUTE_MIN_INTERVAL


def _target_has_trace_pc(target: str) -> bool:
    """Same detection the distance channel uses: shim flush symbol or bare
    trace-pc reference in the symbol table."""
    try:
        from fuzzer_tool.core.elf import _symbol_names

        for name in _symbol_names(target):
            if name == "__afl_dist_flush" or name == "__sanitizer_cov_trace_pc":
                return True
    except Exception:  # noqa: BLE001
        return False
    return False
