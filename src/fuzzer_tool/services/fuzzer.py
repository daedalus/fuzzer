"""Fuzzer orchestration: coordinates mutations, execution, and coverage."""

import atexit
import contextlib
import ctypes
import hashlib
import json
import logging
import math
import os
import random
import shutil
import resource
import signal
import struct
import tempfile
import threading
import time
from pathlib import Path

from fuzzer_tool.adapters.filesystem import load_corpus, save_crash, save_to_corpus
from fuzzer_tool.adapters.process import (
    SIGNAL_CRASH_CODES,
    _child_pids,
    run_target_file,
    run_target_stdin,
)
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.markov import MarkovChain, MarkovEnsemble
from fuzzer_tool.core.montecarlo import (
    MOptScheduler,
    MonteCarloScheduler,
    ReplicatorScheduler,
    ShapleyAttribution,
)
from fuzzer_tool.core.secretary import DEFAULT_EXPLORATION_FRAC, SecretaryStopping
from fuzzer_tool.core.mi import MutualInformationTracker
from fuzzer_tool.core.mutations import (
    DICT_MUTATIONS,
    FORMAT_MUTATIONS,
    INTERESTING_8,
    INTERESTING_16,
    INTERESTING_32,
    MUTATIONS,
    splice,
)
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.services.ptrace_coverage import (
    HAS_CAPSTONE,
    INT3,
    PTRACE_CONT,
    PTRACE_GETREGS,
    PTRACE_PEEKDATA,
    PTRACE_POKEDATA,
    PTRACE_SETOPTIONS,
    PTRACE_SETREGS,
    PTRACE_SINGLESTEP,
    PTRACE_TRACEME,
    PtraceCoverage,
)
from fuzzer_tool.services.te_position import (
    get_te_weighted_position,
    update_te_causal_map,
)
from fuzzer_tool.services.stats_reporter import (
    discovery_rate as _discovery_rate,
    format_elapsed as _format_elapsed_fn,
    record_discovery_snapshot as _record_discovery_snapshot_fn,
    run_crash_replays as _run_crash_replays_fn,
)

log = logging.getLogger(__name__)

_shutdown = False
_active_dmesg_parser = None  # module-level ref for atexit cleanup


def _kill_children(sig=None, frame=None):
    global _shutdown
    _shutdown = True
    for pid in list(_child_pids):
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    _child_pids.clear()
    # Stop dmesg streaming to avoid orphan -w subprocess
    if _active_dmesg_parser is not None:
        _active_dmesg_parser.stop_stream()


atexit.register(_kill_children)
signal.signal(signal.SIGTERM, _kill_children)
signal.signal(signal.SIGINT, _kill_children)


def _handle_sigsegv(signum, frame):
    """Handle SIGSEGV in the fuzzer process itself."""
    import traceback

    print("\n[FATAL] Segmentation fault in fuzzer process!")
    print(f"Signal: {signum}")
    if frame:
        print(f"Frame: {frame}")
    traceback.print_stack(frame)
    sys.exit(1)


signal.signal(signal.SIGSEGV, _handle_sigsegv)


def _write_and_close(fd: int, data: bytes) -> None:
    """Write *data* to *fd* then close it — designed to run in a thread."""
    try:
        os.write(fd, data)
    finally:
        try:
            os.close(fd)
        except OSError:
            log.debug("Failed to close fd %d (already closed?)", fd)


def _cleanup_tmp_dir(path: Path) -> None:
    """Remove temp directory on exit."""
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        log.debug("Failed to clean up %s", path, exc_info=True)


class Fuzzer:
    def __init__(
        self,
        target,
        corpus_dir,
        crashes_dir,
        max_len=4096,
        timeout=5,
        mutations_per_input=8,
        use_coverage=False,
        deep_coverage=False,
        max_bps=50000,
        dictionary=None,
        file_mode=False,
        target_args=None,
        markov_order=1,
        markov_generate=False,
        markov_blend=False,
        mc_bandit=False,
        mc_cem=False,
        mopt=False,
        targets=None,
        anneal_budget=0,
        mc_elite_frac=0.1,
        mc_refit_interval=1000,
        mc_decay_interval=100,
        pairwise_blend=0.0,
        stats_file=None,
        stats_interval=1000,
        coverage_report=None,
        coverage_log=None,
        grammar=None,
        persistent=False,
        inprocess=False,
        inprocess_direct=False,
        inprocess_func="LLVMFuzzerTestOneInput",
        cmplog=False,
        max_corpus=0,
        minimize_every_execs=0,
        no_shm=False,
        resume=False,
        trace_crashes=False,
        seed=42,
        extra_crash_codes=None,
        replay_n=0,
        schedule_ablation=None,
        replicator=False,
        shapley=False,
        mi_guided=False,
        renyi_weight=False,
        transfer_entropy=False,
        secretary=False,
        secretary_window=500,
        secretary_exploration=None,
        elo=False,
        sensitivity=False,
        ga=False,
        ga_pop_size=200,
        ga_gen_size=500,
        ga_elite_frac=0.1,
        ga_crossover_rate=0.7,
        ga_mutation_rate=0.3,
        ga_tournament_size=3,
        ga_speciation_threshold=0.3,
        calibrate=0,
        stall_threshold=1000,
        map_size=0,
        max_collision_risk=30,
        continue_until_crash=False,
    ):
        self.target = target
        # Record boot time at init — before any child processes are spawned.
        # Use -2s tolerance so crashes logged just before this read are included.
        try:
            with open("/proc/uptime") as f:
                self._run_boot_start = float(f.read().split()[0]) - 2.0
        except OSError:
            self._run_boot_start = 0.0
        self.corpus_dir = Path(corpus_dir)
        self.crashes_dir = Path(crashes_dir)
        self.resume = resume
        self.continue_until_crash = continue_until_crash
        self._calibrate = calibrate
        self._stall_threshold = stall_threshold
        self._max_collision_risk = max_collision_risk
        self._last_new_edge_exec = 0
        self._stall_recovery_active = False
        self._stall_recovery_count = 0  # times recovery was activated
        self._stall_recovery_execs = 0  # execs spent in recovery mode
        self.extra_crash_codes = set(extra_crash_codes) if extra_crash_codes else set()
        self.max_len = max_len
        self.timeout = timeout
        self.mutations_per_input = mutations_per_input
        self.use_coverage = use_coverage
        self.dictionary = dictionary or []
        self.file_mode = file_mode
        self.target_args = target_args or []
        self.max_corpus = max_corpus
        self.minimize_every_execs = minimize_every_execs
        self.coverage_report = Path(coverage_report) if coverage_report else None
        self.coverage_log = Path(coverage_log) if coverage_log else None
        if self.coverage_log:
            self.coverage_log.parent.mkdir(parents=True, exist_ok=True)
        self.grammar = grammar
        self.persistent = persistent
        self.seed = seed
        random.seed(seed)

        # GA lifecycle parameters
        self._ga_enabled = ga
        self._ga_pop_size = ga_pop_size
        self._ga_gen_size = ga_gen_size
        self._ga_elite_frac = ga_elite_frac
        self._ga_crossover_rate = ga_crossover_rate
        self._ga_mutation_rate = ga_mutation_rate
        self._ga_tournament_size = ga_tournament_size
        self._ga_speciation_threshold = ga_speciation_threshold
        self.ga = None  # Initialized in run() when --ga is set

        # Edge bitmap size: use provided value or auto-size from branch density
        if map_size > 0:
            self.map_size = map_size
        else:
            from fuzzer_tool.core.elf import estimate_map_size

            self.map_size = estimate_map_size(target)

        # Static analysis: profile target for string extraction, function
        # boundaries, input format hints, and call graph structure.
        from fuzzer_tool.core.target_profiler import TargetProfiler

        self._profile = TargetProfiler(target).profile()

        # Auto-populate dictionary from extracted strings and magic bytes
        if self._profile.interesting_strings:
            for s in self._profile.interesting_strings[:200]:
                token = s.encode("utf-8", errors="replace")
                if token not in self.dictionary:
                    self.dictionary.append(token)
        if self._profile.magic_bytes:
            for mb in self._profile.magic_bytes:
                if mb not in self.dictionary:
                    self.dictionary.append(mb)

        # Cmplog: comparison tracing via LD_PRELOAD
        self._cmplog = None
        if cmplog:
            from fuzzer_tool.core.cmplog import CmplogCollector

            self._cmplog = CmplogCollector()
            if self._cmplog.start():
                print("[*] Cmplog: comparison tracing enabled (memcmp/strcmp/strncmp/memchr)")
            else:
                print("[!] Cmplog: failed to compile shim, disabling")
                self._cmplog = None

        if self.file_mode:
            self._tmp_dir = Path(tempfile.mkdtemp(prefix="fuzzer_"))
            atexit.register(_cleanup_tmp_dir, self._tmp_dir)
        else:
            self._tmp_dir = Path("/tmp") / f"fuzzer_{os.getpid()}"

        self.ptrace_cov: PtraceCoverage | None = None
        self.shm_cov: ShmCoverage | None = None
        self._forkserver = None
        if self.use_coverage:
            if no_shm:
                self._setup_ptrace(target, deep_coverage, max_bps)
            else:
                try:
                    self.shm_cov = ShmCoverage(size=self.map_size)
                    print(f"[*] Coverage: AFL SHM bitmap, id={self.shm_cov.env_id}")
                except OSError:
                    self._setup_ptrace(target, deep_coverage, max_bps, fallback_hint=True)

        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir.mkdir(parents=True, exist_ok=True)

        self.corpus: list[bytes] = []
        self.seen_hashes: set[str] = set()
        self.bloom = BloomFilter(capacity=100_000)
        self.bloom.init_fuzzy(max_recent=200)
        self.crash_hashes: set[str] = set()
        self.crash_sigs: dict[str, int] = {}
        self.crash_frames: dict[str, list[str]] = {}  # sig -> frames for clustering
        self.exec_count = 0
        self.crash_count = 0
        self.timeout_count = 0
        self.start_time = time.time()
        self.last_report: SanitizerReport | None = None
        self.op_counts: dict[str, int] = {}
        self.op_success: dict[str, int] = {}
        self._peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._discovery_history: list[tuple[int, int]] = []  # (exec_count, edges)
        self._crash_rate_history: list[tuple[int, int]] = []  # (exec_count, crash_count)
        self._duplicate_reject_count = 0
        self._total_corpus_attempts = 0
        self._pruned_count = 0
        self._exec_baseline = 0
        self._peak_eps = 0.0
        self._total_exec_time = 0.0
        self._replay_budget_ms: float = 0.2  # max 200ms per batch for crash replay
        self._crash_replays: dict[str, list[int]] = {}  # sig -> list of replay return codes
        self.replay_n: int = replay_n  # --replay-N: replay each crash N times

        # Execution time tracking for adaptive timeout calibration
        from fuzzer_tool.core.execution_time import ExecutionTimeTracker

        self._exec_time_tracker = ExecutionTimeTracker()

        # Kernel-level crash verification via dmesg
        from fuzzer_tool.core.dmesg import DmesgParser

        self._dmesg = DmesgParser()
        self._kernel_crashes: list = []
        self._last_child_pid: int | None = None
        self._dmesg.start_stream()
        # Register for atexit cleanup to avoid orphan dmesg -w subprocess
        global _active_dmesg_parser
        _active_dmesg_parser = self._dmesg
        self.stats_file = Path(stats_file) if stats_file else None
        self.stats_interval = stats_interval

        # Schedule ablation: per-iteration CSV log of signal contributions
        self._ablation_path = Path(schedule_ablation) if schedule_ablation else None
        self._ablation_file = None
        if self._ablation_path:
            self._ablation_path.parent.mkdir(parents=True, exist_ok=True)
            self._ablation_file = open(self._ablation_path, "w")  # noqa: SIM115
            self._ablation_file.write(
                "iter,seed_idx,seed_hash,fuzz_count,coverage_edges,age_s,"
                "base_w,burst,penalty,subsumption,diversity,spatial,mdl,"
                "final_w,new_coverage,new_crash\n"
            )
            self._ablation_file.flush()

        # Support multiple markov orders via comma-separated list or single int
        if isinstance(markov_order, str):
            orders = [int(o.strip()) for o in markov_order.split(",")]
        elif isinstance(markov_order, list):
            orders = markov_order
        else:
            orders = [markov_order]
        if len(orders) > 1:
            self.markov = MarkovEnsemble(orders=orders, blend=markov_blend)
        else:
            self.markov = MarkovChain(order=orders[0])
        self.markov_generate = markov_generate
        self.markov_trained = False
        self._markov_path = self.corpus_dir / "markov.json"
        self._mi_path = self.corpus_dir / "mi.json"

        self._load_corpus()
        self._init_seed_metadata()
        # Load persisted Markov state; skip retrain if loaded (avoids
        # double-counting the same corpus transitions across restarts)
        loaded = False
        if self._markov_path.exists():
            loaded = self.markov.load(str(self._markov_path))
        if self.corpus and not loaded:
            self.markov.train_corpus(self.corpus)
        self.markov_trained = self.markov.is_trained()

        self.mc_bandit = mc_bandit
        self.mc_cem = mc_cem
        self._use_mopt = mopt
        self.mc = (
            MonteCarloScheduler(
                elite_frac=mc_elite_frac,
                refit_interval=mc_refit_interval,
                pairwise_blend=pairwise_blend,
                decay_interval=mc_decay_interval,
            )
            if (mc_bandit or mc_cem or mopt)
            else None
        )
        self._mopt = None
        if mopt:
            self._mopt = MOptScheduler(n_particles=5, window_size=200)
            log.info("MOpt PSO scheduling enabled (5 particles, window=200)")
        self._use_replicator = replicator
        self._seed_strategy = None
        self._op_dispatch = self._build_dispatch()
        self._replicator = None
        if replicator:
            self._replicator = ReplicatorScheduler(window_size=200, learning_rate=0.1)
            log.info("Replicator dynamics scheduling enabled (window=200, eta=0.1)")
        self._use_shapley = shapley
        self._shapley = ShapleyAttribution(n_samples=100, window_size=500) if shapley else None
        self._use_mi = mi_guided
        self._mi = (
            MutualInformationTracker(max_positions=max_len, min_observations=50)
            if mi_guided
            else None
        )
        # Load persisted MI state
        if self._use_mi and self._mi and self._mi_path.exists():
            self._mi.load(str(self._mi_path))
            log.info("MI tracker loaded from %s", self._mi_path)

        # Crash MI tracker: I(byte_position; crash_outcome)
        from fuzzer_tool.core.crash_eta import CrashMITracker

        self._crash_mi = CrashMITracker(max_positions=max_len, min_observations=20)
        self._crash_mi_path = self.corpus_dir / "crash_mi.json"
        if self._crash_mi_path.exists():
            try:
                self._crash_mi.load(json.loads(self._crash_mi_path.read_text()))
                log.info("Crash MI tracker loaded: %d execs, %d crashes",
                         self._crash_mi.total_execs, self._crash_mi.total_crashes)
            except (OSError, json.JSONDecodeError):
                pass

        self._use_renyi_weight = renyi_weight
        self._use_transfer_entropy = transfer_entropy
        self._te = None
        self._te_byte_edges: dict[int, dict[int, int]] = {}  # pos → {edge: count}
        if transfer_entropy:
            from fuzzer_tool.core.transfer_entropy import TransferEntropy

            self._te = TransferEntropy(history_length=1)
            self._te_input_history: list[bytes] = []
            self._te_edge_history: list[bytes] = []
            self._te_history_max = 500
            log.info("Transfer entropy tracking enabled")
        self._last_ops_used: list[str] = []
        self._last_hamming_distance: int = -1

        # Per-byte sensitivity tracker (Lyapunov exponent)
        self._use_sensitivity = sensitivity
        from fuzzer_tool.core.sensitivity import ByteSensitivityTracker

        self._sensitivity = ByteSensitivityTracker(
            max_seeds=50, max_bytes=max_len, sample_rate=0.02
        )

        # Critical slowing down detector
        from fuzzer_tool.core.critical_slowing import CriticalSlowingDown

        self._csd = CriticalSlowingDown(window_size=50, rise_threshold=1.5, min_observations=20)

        # Elo rating system for operator scheduling
        self._use_elo = elo
        self._elo = None
        if elo:
            from fuzzer_tool.core.elo import EloTracker

            self._elo = EloTracker(k_factor=16, decay=0.99, crash_track=True, min_matches=10)
            self._elo_path = self.corpus_dir / "elo.json"
            if self._elo_path.exists():
                self._elo.load(str(self._elo_path))
                log.info("Elo tracker loaded from %s", self._elo_path)
            log.info("Elo rating system enabled (k=32, decay=0.99)")
            self._elo_decay_interval = 100  # apply decay every N iterations
            self._elo_match_window: list[tuple[str, str, float, bool]] = []

        # Elo arbitrates between all available strategies when enabled
        self._meta_strategy: str | None = None
        if self._use_elo:
            log.info("Meta-scheduler enabled: Elo arbitrating bandit vs MOpt")
            self._meta_strategy_choices: list[str] = []

        # Secretary-problem optimal stopping
        self._secretary = secretary
        self._secretary_window = secretary_window
        self._secretary_exploration = (
            secretary_exploration if secretary_exploration is not None else DEFAULT_EXPLORATION_FRAC
        )
        self._seed_secretary: dict[str, SecretaryStopping] = {}
        self._op_secretary: dict[str, SecretaryStopping] = {}
        self._corpus_secretary = (
            SecretaryStopping(
                window_size=secretary_window,
                exploration_frac=self._secretary_exploration,
                min_observations=30,
            )
            if secretary
            else None
        )

        # Directed distance for targeted fuzzing
        self._distance = None
        self._distance_targets = targets
        self._anneal_progress = 0.0  # 0.0 = pure coverage, 1.0 = pure distance
        if targets:
            from fuzzer_tool.core.distance import TargetDistance

            self._distance = TargetDistance(target, targets)
            if self._distance.load():
                print(
                    f"[*] Directed mode: {len(self._distance.target_addrs)} target(s), "
                    f"{len(self._distance.functions)} functions mapped"
                )
            else:
                print(
                    "[!] Directed mode: failed to load target distances, falling back to coverage"
                )
                self._distance = None

        # Simulated annealing temperature schedule
        self._anneal_budget = anneal_budget  # 0 = no annealing (temperature always 1.0)
        self._temperature = 1.0

        # Crash tracing: GDB backtrace + strace on crash inputs
        self._tracer = None
        if trace_crashes:
            from fuzzer_tool.core.trace import CrashTracer

            self._tracer = CrashTracer(target)

        def _register_arms(scheduler):
            """Register all mutation arms on a scheduler (mc, mopt, replicator, elo)."""
            for op in MUTATIONS:
                scheduler.init_arm(op)
            for op in DICT_MUTATIONS:
                scheduler.init_arm(op)
            scheduler.init_arm("markov_bytes")
            scheduler.init_arm("cem_bytes")
            if self.grammar:
                scheduler.init_arm("grammar_mutate")
                scheduler.init_arm("grammar_tree_mutate")
            for op in FORMAT_MUTATIONS:
                scheduler.init_arm(op)

        if self.mc and self.mc_bandit:
            _register_arms(self.mc)
        if self._mopt:
            _register_arms(self._mopt)
        if self._replicator:
            _register_arms(self._replicator)
        if self._elo:
            _register_arms(self._elo)

        self._persistent_runner = None
        if self.persistent:
            from fuzzer_tool.adapters.persistent import PersistentRunner

            self._persistent_runner = PersistentRunner(target=self.target, timeout=self.timeout)
            if self._persistent_runner.start():
                print("[*] Persistent mode: target started")
            else:
                print("[!] Persistent mode: failed to start target, falling back to fork")
                self._persistent_runner = None

        self._inprocess_runner = None
        if inprocess:
            from fuzzer_tool.adapters.inprocess import InProcessRunner

            cov_env_id = self.shm_cov.env_id if self.shm_cov else None
            self._inprocess_runner = InProcessRunner(
                target=self.target,
                function_name=inprocess_func,
                timeout=self.timeout,
                shm_size=self.map_size,
                direct=inprocess_direct,
                coverage_env_id=cov_env_id,
                cov=bool(cov_env_id),
            )
            mode = "direct ctypes" if inprocess_direct else "subprocess loader"
            cov_note = f", SHM cov id={cov_env_id}" if cov_env_id else ""
            print(f"[*] In-process mode ({mode}{cov_note}): {self.target}::{inprocess_func}")
            if self._inprocess_runner._persistent:
                print("[*] Persistent loader: enabled (1 process, many calls)")

        # Forkserver: use C fuzz_loader for default execution path when available.
        # Currently disabled: fuzz_loader reads bitmap from file while target
        # writes to SHM — these are disconnected. Enable when fuzz_loader.c
        # is updated to read from SHM via __AFL_SHM_ID.
        # if not self._inprocess_runner and not self._persistent_runner and not self.ptrace_cov:
        #     from fuzzer_tool.adapters.forkserver import ForkserverRunner
        #     self._forkserver = ForkserverRunner(target, timeout=self.timeout)
        #     if self._forkserver.start():
        #         log.info("Forkserver started for default execution path")

    def _setup_ptrace(self, target, deep_coverage, max_bps, fallback_hint=False):
        cov = PtraceCoverage(target, deep_coverage=deep_coverage, max_bps=max_bps)
        if cov.bb_addrs:
            self.ptrace_cov = cov
            mode = "deep (capstone)" if cov.deep_coverage else "function-entry"
            print(f"[*] Coverage: {len(cov.bb_addrs)} breakpoints ({mode}), map={cov.map_size}")
        else:
            print(
                "[!] Coverage: no symbols found in ELF, "
                "coverage disabled (use -g to compile with symbols)"
            )
            if fallback_hint:
                print(
                    "[!] For closed-source binaries, use AFL++ QEMU mode: afl-qemu-trace ./target"
                )

    def _load_corpus(self):
        self.corpus, self.seen_hashes = load_corpus(self.corpus_dir, self.bloom)

    def _init_seed_metadata(self):
        self._state_path = self.corpus_dir / "state.json"
        self._edge_tracker_path = self.corpus_dir / "edge_tracker.json"
        now = time.time()
        self.seed_meta: dict[bytes, dict] = {}
        for seed in self.corpus:
            self.seed_meta[seed] = {
                "fuzz_count": 0,
                "coverage_edges": 0,
                "momentum": 0.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "added_at": now,
            }
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        self._edge_tracker = EdgeTracker(map_size=self.map_size)
        self._corpus_size_history: list[int] = []

        # Load persisted state if resuming
        if self.resume:
            self._load_state()

    def _seed_key(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    def _save_state(self):
        """Persist fuzzer state for resume."""
        state = {
            "exec_count": self.exec_count,
            "crash_count": self.crash_count,
            "timeout_count": self.timeout_count,
            "crash_sigs": self.crash_sigs,
            "op_counts": self.op_counts,
            "op_success": self.op_success,
            "corpus_size_history": self._corpus_size_history[-500:],
            "seed_meta": {},
            "crash_frames": self.crash_frames,
        }
        for seed, meta in self.seed_meta.items():
            key = seed.hex()
            # Serialize redqueen_matches as hex strings for JSON compat
            rm = meta.get("redqueen_matches", [])
            rm_ser = [[m[0], m[1].hex(), m[2].hex()] for m in rm]
            state["seed_meta"][key] = {
                "fuzz_count": meta["fuzz_count"],
                "coverage_edges": meta["coverage_edges"],
                "momentum": meta.get("momentum", 0.0),
                "redqueen_offsets": meta["redqueen_offsets"],
                "redqueen_matches": rm_ser,
                "added_at": meta["added_at"],
                "lineage_depth": meta.get("lineage_depth", 0),
                "hamming_distance": meta.get("hamming_distance", -1),
            }
        try:
            self._state_path.write_text(json.dumps(state, separators=(",", ":")))
        except OSError as e:
            log.debug("Failed to save state: %s", e)
        self._edge_tracker.save(str(self._edge_tracker_path))
        if self._use_elo and self._elo:
            self._elo.save(str(self._elo_path))
        # Save sensitivity tracker
        sens_path = self.corpus_dir / "sensitivity.json"
        try:
            sens_path.write_text(json.dumps(self._sensitivity.save(), separators=(",", ":")))
        except OSError:
            pass
        # Save crash MI tracker
        try:
            self._crash_mi_path.write_text(json.dumps(self._crash_mi.save(), separators=(",", ":")))
        except OSError:
            pass

    def _load_state(self):
        """Load persisted fuzzer state for resume."""
        if not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            log.debug("Failed to load state: %s", e)
            return
        self.exec_count = state.get("exec_count", 0)
        self.crash_count = state.get("crash_count", 0)
        self.timeout_count = state.get("timeout_count", 0)
        self.crash_sigs = state.get("crash_sigs", {})
        self.crash_frames = state.get("crash_frames", {})
        self.op_counts = state.get("op_counts", {})
        self.op_success = state.get("op_success", {})
        self._corpus_size_history = state.get("corpus_size_history", [])
        # Merge seed metadata for seeds still in corpus
        saved_meta = state.get("seed_meta", {})
        for seed in self.corpus:
            key = seed.hex()
            if key in saved_meta:
                sm = saved_meta[key]
                self.seed_meta[seed].update(
                    {
                        "fuzz_count": sm.get("fuzz_count", 0),
                        "coverage_edges": sm.get("coverage_edges", 0),
                        "momentum": sm.get("momentum", 0.0),
                        "redqueen_offsets": sm.get("redqueen_offsets", []),
                        "added_at": sm.get("added_at", self.seed_meta[seed]["added_at"]),
                        "lineage_depth": sm.get("lineage_depth", 0),
                        "hamming_distance": sm.get("hamming_distance", -1),
                    }
                )
                # Deserialize redqueen_matches from hex strings
                rm_ser = sm.get("redqueen_matches", [])
                if rm_ser:
                    self.seed_meta[seed]["redqueen_matches"] = [
                        (m[0], bytes.fromhex(m[1]), bytes.fromhex(m[2])) for m in rm_ser
                    ]
        self._edge_tracker.load(str(self._edge_tracker_path))
        # Load sensitivity tracker
        sens_path = self.corpus_dir / "sensitivity.json"
        if sens_path.exists():
            try:
                self._sensitivity.load(json.loads(sens_path.read_text()))
            except (OSError, json.JSONDecodeError):
                pass
        if self.resume:
            print(
                f"[*] Resumed: {self.exec_count} execs, "
                f"{self.crash_count} crashes, {len(self.corpus)} seeds"
            )
        log.info(
            "Fuzzer state loaded: execs=%d, crashes=%d, corpus=%d",
            self.exec_count,
            self.crash_count,
            len(self.corpus),
        )

    def _run_target(self, data: bytes) -> tuple[int, str]:
        if self._inprocess_runner:
            if self.shm_cov:
                self.shm_cov.reset_edge_map()
            rc, err = self._inprocess_runner.run_one(data)
            # Read coverage bitmap from runner and copy into SHM
            if self.shm_cov:
                bitmap = self._inprocess_runner.read_bitmap()
                if bitmap and len(bitmap) <= self.shm_cov.size:
                    ctypes.memmove(self.shm_cov._ptr, bitmap, len(bitmap))
            return rc, err

        if self._persistent_runner:
            return self._persistent_runner.run_one(data)

        if self.ptrace_cov:
            return self._run_target_ptrace(data)

        # Forkserver: use C fuzz_loader (avoids Python subprocess overhead)
        if self._forkserver and self._forkserver._ready:
            rc, bitmap = self._forkserver.run_one(data)
            if bitmap and self.shm_cov and len(bitmap) <= self.shm_cov.size:
                ctypes.memmove(self.shm_cov._ptr, bitmap, len(bitmap))
            return rc, ""

        if self.shm_cov:
            self.shm_cov.reset_edge_map()

        env = os.environ.copy()
        if self.use_coverage:
            env["AFL_MAP_SIZE"] = str(self.map_size)
        if self.shm_cov:
            env["__AFL_SHM_ID"] = self.shm_cov.env_id
        if self._cmplog:
            env = self._cmplog.setup_env(env)

        if self.file_mode:
            rc, stderr, pid = run_target_file(
                self.target,
                data,
                self.timeout,
                str(self._tmp_dir),
                self.target_args,
                env=env,
            )
            self._last_child_pid = pid
            return rc, stderr
        rc, stderr, pid = run_target_stdin(self.target, data, self.timeout, env=env)
        self._last_child_pid = pid
        return rc, stderr

    def _ptrace_handle_breakpoint(self, pid: int, libc, cov: PtraceCoverage, regs_buf) -> bool:
        """Handle a SIGTRAP: restore bp, record edge, re-exec if RSP is valid.

        Before the stack is initialized (RSP=0), breakpoints fire during
        dynamic linker and libc startup. We skip edge recording and
        re-execution for those — just restore the byte and continue.
        Once we observe RSP > 0x1000, the stack is set up and all
        subsequent breakpoints are safe to instrument.

        Returns True if execution should continue, False to break the loop.
        """
        if not cov._is_x86_64:
            log.warning("ptrace coverage requires x86_64")
            return False
        libc.ptrace(PTRACE_GETREGS, pid, None, regs_buf)
        rip = struct.unpack_from("<Q", bytes(regs_buf), 128)[0]
        bp_addr = rip - 1

        if bp_addr not in cov.original_bytes:
            libc.ptrace(PTRACE_CONT, pid, None, None)
            return True

        orig = cov.original_bytes[bp_addr]
        val = cov._read_memory(pid, bp_addr)
        cov._write_memory(pid, bp_addr, (val & ~0xFF) | orig)
        del cov.original_bytes[bp_addr]

        rsp = struct.unpack_from("<Q", bytes(regs_buf), 128 + 48)[0]
        if rsp > 0x1000:
            cov._stack_initialized = True
            cov.record_edge(bp_addr)
            cov.discover_new_bbs(pid, bp_addr)
            regs_buf2 = (ctypes.c_char * (27 * 8))()
            libc.ptrace(PTRACE_GETREGS, pid, None, regs_buf2)
            regs = bytearray(regs_buf2)
            struct.pack_into("<Q", regs, 128, bp_addr)
            libc.ptrace(PTRACE_SETREGS, pid, None, bytes(regs))
        # Continue — at RSP=0 just skip the breakpoint past the
        # early-init instruction.
        libc.ptrace(PTRACE_CONT, pid, None, None)
        return True

    def _run_target_ptrace(self, data: bytes) -> tuple[int, str]:
        cov = self.ptrace_cov
        cov.reset_edge_map()
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.ptrace.argtypes = [
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        libc.ptrace.restype = ctypes.c_long

        stdin_r, stdin_w = os.pipe()
        writer = None
        pid = os.fork()
        self._last_child_pid = pid
        if pid == 0:
            os.setsid()
            os.dup2(stdin_r, 0)
            os.close(stdin_r)
            os.close(stdin_w)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
            # Strip LD_PRELOAD to avoid conflicts with ASAN
            # (e.g. libksm_preload.so loaded before ASAN causes abort)
            ld_preload = os.environ.get("LD_PRELOAD", "")
            if ld_preload:
                cleaned = [p for p in ld_preload.split(":") if "ksm_preload" not in p]
                if cleaned:
                    os.environ["LD_PRELOAD"] = ":".join(cleaned)
                else:
                    os.environ.pop("LD_PRELOAD", None)
            libc.ptrace(PTRACE_TRACEME, 0, None, None)
            signal.signal(signal.SIGTRAP, signal.SIG_IGN)
            os.execv(self.target, [self.target])
            os._exit(127)

        os.close(stdin_r)
        # Write data in a thread to avoid deadlock when data > PIPE_BUF (~64KB).
        # The child may be stopped at exec's SIGTRAP before reading stdin, so a
        # blocking write would stall the parent before it can call waitpid.
        writer = threading.Thread(target=_write_and_close, args=(stdin_w, data))
        writer.start()

        try:
            _, status = os.waitpid(pid, 0)
            if os.WIFSTOPPED(status) and os.WSTOPSIG(status) == signal.SIGTRAP:
                pass  # normal: child stopped at exec, install breakpoints
            elif os.WIFSTOPPED(status):
                sig = os.WSTOPSIG(status)
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return -sig, ""  # child crashed before we could instrument it
            elif os.WIFSIGNALED(status):
                return -os.WTERMSIG(status), ""
            elif os.WIFEXITED(status):
                return os.WEXITSTATUS(status), ""
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return -2, "exec failed"

            cov.install_breakpoints(pid)
            libc.ptrace(PTRACE_CONT, pid, None, None)

            deadline = time.time() + self.timeout

            last_action = None
            last_sig = 0
            returncode = 0
            child_reaped = False
            while time.time() < deadline:
                _, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                if status == 0:
                    time.sleep(0.0005)
                    continue

                if os.WIFEXITED(status):
                    returncode = os.WEXITSTATUS(status)
                    child_reaped = True
                    break
                if os.WIFSIGNALED(status):
                    returncode = -os.WTERMSIG(status)
                    child_reaped = True
                    break

                if os.WIFSTOPPED(status):
                    sig = os.WSTOPSIG(status)
                    last_sig = sig
                    if sig == signal.SIGTRAP:
                        regs_buf = (ctypes.c_char * (27 * 8))()
                        if self._ptrace_handle_breakpoint(pid, libc, cov, regs_buf):
                            last_action = "cont"
                        else:
                            break
                    else:
                        break

            if child_reaped:
                pass  # loop already captured the definitive returncode
            elif last_action == "cont" and last_sig == signal.SIGTRAP:
                # Child stopped at breakpoint but loop exited (deadline?)
                # Resume and wait for final outcome.
                _, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                if status != 0 and os.WIFSTOPPED(status):
                    libc.ptrace(PTRACE_CONT, pid, None, None)
                    _, status = os.waitpid(pid, 0)
                elif status != 0:
                    if os.WIFSIGNALED(status):
                        returncode = -os.WTERMSIG(status)
                    elif os.WIFEXITED(status):
                        returncode = os.WEXITSTATUS(status)
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)

            if returncode == 0 and not child_reaped:
                if os.WIFSIGNALED(status):
                    returncode = -os.WTERMSIG(status)
                elif os.WIFEXITED(status):
                    returncode = os.WEXITSTATUS(status)
                elif os.WIFSTOPPED(status):
                    returncode = -os.WSTOPSIG(status)
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
            return returncode, ""

        except ChildProcessError:
            # Child already reaped (race with watchdog). Return -2
            # (unknown) instead of 0 (success) to avoid masking crashes.
            return -2, ""
        except Exception as e:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except Exception:
                log.debug("Failed to kill orphan pid %d", pid, exc_info=True)
            return -2, str(e)
        finally:
            if writer is not None:
                writer.join(timeout=self.timeout)

    def _verify_kernel_crash(self, child_pid: int | None) -> bool:
        """Try to verify a crash via dmesg. Returns True if dmesg confirmed it.

        Checks for:
        1. Target process crashes (matched by child_pid)
        2. Python subprocess crashes (any python3/python process) - these indicate
           fuzzer infrastructure issues or malformed inputs causing interpreter crashes
        """
        if not child_pid:
            return False

        # Drain the async stream
        kernel_hits = self._dmesg.drain_stream(pid=child_pid)
        if not kernel_hits:
            # Let the stream reader consume /dev/kmsg
            import time as _time

            _time.sleep(0.05)
            kernel_hits = self._dmesg.drain_stream(pid=child_pid)
        if not kernel_hits:
            # Synchronous fallback: read dmesg from last known timestamp
            text_crashes = self._dmesg._poll_text(since=self._dmesg._last_ts)
            if text_crashes:
                kernel_hits = [kc for kc in text_crashes if kc.pid == child_pid]

        if kernel_hits:
            for kc in kernel_hits:
                self._kernel_crashes.append(kc)
                log.info(
                    "Kernel crash verified: %s at ip=%s (ts=%.3f)",
                    kc.crash_type,
                    kc.ip or "?",
                    kc.timestamp,
                )
            return True

        # Also check for Python process crashes (fuzzer infrastructure issues)
        self._check_python_crashes()

        return False

    def _check_python_crashes(self):
        """Check for Python process crashes in dmesg that may indicate fuzzer issues."""
        # Get recent crashes from dmesg (not filtered by PID)
        all_crashes = self._dmesg._poll_text(since=self._dmesg._last_ts)
        for kc in all_crashes:
            # Check for Python process crashes
            if kc.process_name and "python" in kc.process_name.lower():
                if kc.crash_type == "segfault":
                    print(
                        f"\n[*] Python process crash detected: pid={kc.pid}, ip={kc.ip or '?'}, "
                        f"type={kc.crash_type} (may indicate fuzzer infrastructure issue)"
                    )
                    # Record as a special type of crash for diagnostics
                    kc.crash_type = "python_segfault"
                    self._kernel_crashes.append(kc)

    def _is_interesting(self, returncode: int, stderr: str) -> bool:
        if returncode in SIGNAL_CRASH_CODES or returncode in self.extra_crash_codes:
            return True
        if returncode < 0 and returncode != -1:
            return True
        if returncode in (-1, 0) and "ASAN" in stderr:
            return True
        if "Segmentation fault" in stderr:
            return True
        return "Aborted" in stderr

    def _is_crash(self, returncode: int, stderr: str) -> bool:
        self.last_report = None
        if returncode in (-2, -1):
            return False

        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            self.last_report = report
            return True

        if returncode in SIGNAL_CRASH_CODES or returncode in self.extra_crash_codes:
            return True
        if returncode < 0:
            return True
        return any(
            sig in stderr
            for sig in [
                "SIGSEGV",
                "SIGABRT",
                "SIGFPE",
                "SIGBUS",
                "Segmentation fault",
                "Aborted",
            ]
        )

    def mutate(self, data: bytes) -> bytes:
        from fuzzer_tool.core.similarity import hamming_distance

        buf = bytearray(data)
        if not buf:
            buf = bytearray(b"\x00" * random.randint(1, 32))

        ops = self._build_ops(data)
        self._last_ops_used = []
        self._last_mopt_particles = []
        if not hasattr(self, "_prev_bandit_op"):
            self._prev_bandit_op = None
        self._meta_strategy = None

        # Stall recovery: use more mutations
        n_mutations = self.mutations_per_input
        if self._stall_recovery_active:
            n_mutations = max(n_mutations, 16)

        for _ in range(n_mutations):
            op = self._select_op(ops)
            self._last_ops_used.append(op)

            byte_idx = self._select_position(buf, data)

            result = self._op_dispatch[op](buf, byte_idx, data)
            if result is not None:
                # Handler returned new bytes (havoc, format mutators, dict overwrite, etc.)
                if op == "havoc":
                    self._last_hamming_distance = (
                        hamming_distance(data, result) if len(data) == len(result) else -1
                    )
                    return result
                buf = bytearray(result[: self.max_len])

        result = bytes(buf)
        self._last_hamming_distance = (
            hamming_distance(data, result) if len(data) == len(result) else -1
        )
        return result

    def _build_ops(self, data: bytes) -> list[str]:
        """Build the list of available mutation operators from ground truth."""
        ops = list(MUTATIONS)
        if self.dictionary:
            ops.extend(DICT_MUTATIONS)
        if self.markov_trained:
            ops.append("markov_bytes")
        if self.mc and self.mc_cem and self.mc.cem_fitted:
            ops.append("cem_bytes")
        if self.grammar:
            ops.append("grammar_mutate")
            ops.append("grammar_tree_mutate")
        ops.extend(FORMAT_MUTATIONS)
        parent_meta = self.seed_meta.get(data)
        if parent_meta and (
            parent_meta.get("redqueen_matches") or parent_meta.get("redqueen_offsets")
        ):
            ops.append("redqueen")
        return ops

    def _select_op(self, ops: list[str]) -> str:
        """Select a mutation operator using the active scheduling strategy.

        Hierarchy: Elo arbitrates all strategies when enabled.
        Falls through to individual strategies if Elo is not active.
        Stall recovery: pick randomly to maximize diversity.
        """
        # Stall recovery: bypass all strategies, pick randomly
        if self._stall_recovery_active:
            self._meta_strategy = "random_stall"
            return random.choice(ops)

        # Build list of available strategies
        available = []
        if self._use_replicator and self._replicator:
            available.append("replicator")
        if self.mc and self.mc_bandit:
            available.append("bandit")
        if self._use_mopt and self._mopt:
            available.append("mopt")

        # Elo sits on top: pick which strategy to use
        if self._use_elo and self._elo and len(available) >= 2:
            strategy = self._elo.select_strategy(available)
            self._meta_strategy = strategy
        elif self._use_elo and self._elo and available:
            strategy = available[0]
            self._meta_strategy = strategy
        else:
            strategy = None

        # Execute selected strategy
        if strategy == "replicator" and self._replicator:
            op = self._replicator.select_op(ops)
            self._last_mopt_particles.append(None)
        elif strategy == "mopt" and self._mopt:
            op, pid = self._mopt.select_op(ops)
            self._last_mopt_particles.append(pid)
        elif strategy == "bandit" and self.mc and self.mc_bandit:
            op = self.mc.select_op(ops, prev_op=self._prev_bandit_op)
            self._prev_bandit_op = op
            self._last_mopt_particles.append(None)
        elif self._use_replicator and self._replicator:
            op = self._replicator.select_op(ops)
            self._last_mopt_particles.append(None)
        elif self._use_mopt and self._mopt:
            op, pid = self._mopt.select_op(ops)
            self._last_mopt_particles.append(pid)
        elif self.mc and self.mc_bandit:
            op = self.mc.select_op(ops, prev_op=self._prev_bandit_op)
            self._prev_bandit_op = op
            self._last_mopt_particles.append(None)
        else:
            op = random.choice(ops)
            self._last_mopt_particles.append(None)
        return op

    def _select_position(self, buf: bytearray, data: bytes) -> int:
        """Select a byte position for mutation using MI/TE/sensitivity/random."""
        if not buf:
            return 0
        te_pos = (
            self._get_te_weighted_position(len(buf))
            if self._use_transfer_entropy and self._te
            else None
        )
        mi_pos = self._mi.weighted_position(len(buf)) if self._use_mi and self._mi else None
        sens_pos = self._sensitivity.get_weighted_position(data, len(buf))
        candidates = [p for p in [sens_pos, te_pos, mi_pos] if p is not None]
        if candidates:
            return random.choice(candidates)
        return random.randint(0, len(buf) - 1)

    # ── Operator handlers ──────────────────────────────────────────────
    # Each handler: (buf, byte_idx, data) -> None (in-place) or bytes (replace buf)

    def _op_bit_flip(self, buf, byte_idx, _data):
        if buf:
            buf[byte_idx] ^= 1 << random.randint(0, 7)

    def _op_byte_flip(self, buf, byte_idx, _data):
        if buf:
            buf[byte_idx] ^= 0xFF

    def _op_interesting_8(self, buf, byte_idx, _data):
        if buf:
            buf[byte_idx] = random.choice(INTERESTING_8) & 0xFF

    def _op_interesting_16(self, buf, _byte_idx, _data):
        if len(buf) >= 2:
            idx = random.randint(0, len(buf) - 2)
            struct.pack_into("<h", buf, idx, random.choice(INTERESTING_16))

    def _op_interesting_32(self, buf, _byte_idx, _data):
        if len(buf) >= 4:
            idx = random.randint(0, len(buf) - 4)
            struct.pack_into("<i", buf, idx, random.choice(INTERESTING_32))

    def _op_arithmetic(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import ARITHMETIC_DELTAS

        width = random.choice([1, 2, 4, 8])
        if len(buf) >= width:
            max_start = len(buf) - width
            idx = (random.randint(0, max_start) // width) * width
            delta = random.choice(ARITHMETIC_DELTAS)
            if random.random() < 0.5:
                delta = -delta
            endian = random.choice(["<", ">"])
            if width == 1:
                buf[idx] = (buf[idx] + delta) & 0xFF
            elif width == 2:
                val = (struct.unpack_from(f"{endian}H", buf, idx)[0] + delta) & 0xFFFF
                struct.pack_into(f"{endian}H", buf, idx, val)
            elif width == 4:
                val = (struct.unpack_from(f"{endian}I", buf, idx)[0] + delta) & 0xFFFFFFFF
                struct.pack_into(f"{endian}I", buf, idx, val)
            elif width == 8:
                val = (struct.unpack_from(f"{endian}Q", buf, idx)[0] + delta) & 0xFFFFFFFFFFFFFFFF
                struct.pack_into(f"{endian}Q", buf, idx, val)

    def _op_random_bytes(self, buf, _byte_idx, _data):
        if buf:
            buf[random.randint(0, len(buf) - 1)] = random.randint(0, 255)

    def _op_block_insert(self, buf, _byte_idx, _data):
        if len(buf) < self.max_len:
            idx = random.randint(0, len(buf))
            size = random.randint(1, min(32, self.max_len - len(buf)))
            buf[idx:idx] = bytes(random.randint(0, 255) for _ in range(size))

    def _op_block_delete(self, buf, _byte_idx, _data):
        if len(buf) > 1:
            idx = random.randint(0, len(buf) - 1)
            max_size = min(32, len(buf) - idx, len(buf) - 1)
            if max_size >= 1:
                del buf[idx : idx + random.randint(1, max_size)]

    def _op_block_duplicate(self, buf, _byte_idx, _data):
        if len(buf) < self.max_len:
            idx = random.randint(0, len(buf) - 1)
            size = random.randint(1, min(16, len(buf) - idx))
            block = buf[idx : idx + size]
            ins = random.randint(0, len(buf))
            buf[ins:ins] = block

    def _op_dict_insert(self, buf, _byte_idx, _data):
        if self.dictionary:
            token = random.choice(self.dictionary)
            if len(buf) + len(token) <= self.max_len:
                buf[random.randint(0, len(buf)) : 0] = token  # insert at random pos

    def _op_dict_replace(self, buf, _byte_idx, _data):
        if self.dictionary and buf:
            token = random.choice(self.dictionary)
            idx = random.randint(0, len(buf) - 1)
            end = min(idx + len(token), len(buf))
            buf[idx:end] = token[: end - idx]

    def _op_dict_overwrite(self, buf, _byte_idx, _data):
        if self.dictionary:
            return bytearray(random.choice(self.dictionary)[: self.max_len])

    def _op_dict_prepend(self, buf, _byte_idx, _data):
        if self.dictionary:
            token = random.choice(self.dictionary)
            if len(buf) + len(token) <= self.max_len:
                return bytearray(token) + buf

    def _op_dict_append(self, buf, _byte_idx, _data):
        if self.dictionary:
            token = random.choice(self.dictionary)
            if len(buf) + len(token) <= self.max_len:
                buf.extend(token)

    def _op_checksum_repair(self, buf, _byte_idx, _data):
        import zlib

        if buf and len(buf) >= 4:
            pos = random.randint(0, max(0, len(buf) - 4))
            buf[pos : pos + 4] = zlib.crc32(bytes(buf[:pos])).to_bytes(4, "big")

    def _op_token_dup(self, buf, _byte_idx, _data):
        if self.dictionary and buf:
            token = random.choice(self.dictionary)
            if len(buf) + len(token) <= self.max_len:
                buf[random.randint(0, len(buf)) : 0] = token

    def _op_markov_bytes(self, buf, _byte_idx, _data):
        if buf:
            idx = random.randint(0, len(buf) - 1)
            ctx = bytes(buf[max(0, idx - self.markov.order) : idx]) if self.markov.order else b""
            buf[idx] = self.markov.sample_byte(ctx)

    def _op_cem_bytes(self, buf, _byte_idx, _data):
        if self.mc and self.mc.cem_fitted:
            if buf:
                buf[random.randint(0, len(buf) - 1)] = self.mc.cem_byte(
                    random.randint(0, len(buf) - 1)
                )
            else:
                return bytearray(self.mc.cem_sample(random.randint(1, min(32, self.max_len))))

    def _op_splice(self, buf, _byte_idx, data):
        if len(self.corpus) >= 2:
            a = random.choice(self.corpus)
            b = random.choice(self.corpus)
            if a is not data and b is not data:
                return bytearray(splice(a, b)[: self.max_len])
            others = [c for c in self.corpus if c is not data]
            if others:
                return bytearray(splice(bytes(buf), random.choice(others))[: self.max_len])

    def _op_crossover(self, buf, _byte_idx, data):
        from fuzzer_tool.core.mutations import crossover

        if len(self.corpus) >= 2 and buf:
            a = random.choice(self.corpus)
            b = random.choice(self.corpus)
            if a is not data and b is not data:
                return bytearray(crossover(a, b)[: self.max_len])
            others = [c for c in self.corpus if c is not data]
            if others:
                return bytearray(crossover(bytes(buf), random.choice(others))[: self.max_len])

    def _op_type_replace(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import type_replace

        if buf:
            return bytearray(type_replace(bytes(buf))[: self.max_len])

    def _op_ascii_num(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import ascii_num_replace

        if buf:
            return bytearray(ascii_num_replace(bytes(buf))[: self.max_len])

    def _op_byte_shuffle(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_shuffle

        if buf and len(buf) > 1:
            return bytearray(byte_shuffle(bytes(buf))[: self.max_len])

    def _op_byte_delete(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_delete

        if buf and len(buf) > 1:
            return bytearray(byte_delete(bytes(buf))[: self.max_len])

    def _op_byte_insert(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import byte_insert

        if buf and len(buf) < self.max_len:
            return bytearray(byte_insert(bytes(buf), self.max_len)[: self.max_len])

    def _op_insert_ascii_num(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import insert_ascii_num

        if buf and len(buf) < self.max_len:
            return bytearray(insert_ascii_num(bytes(buf), self.max_len)[: self.max_len])

    def _op_transpose_16(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 2:
            return bytearray(transpose_bytes(bytes(buf), 2)[: self.max_len])

    def _op_transpose_32(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 4:
            return bytearray(transpose_bytes(bytes(buf), 4)[: self.max_len])

    def _op_transpose_64(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import transpose_bytes

        if len(buf) >= 8:
            return bytearray(transpose_bytes(bytes(buf), 8)[: self.max_len])

    def _op_bit_transpose_8(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if buf:
            return bytearray(bit_transpose(bytes(buf), 1)[: self.max_len])

    def _op_bit_transpose_16(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 2:
            return bytearray(bit_transpose(bytes(buf), 2)[: self.max_len])

    def _op_bit_transpose_32(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 4:
            return bytearray(bit_transpose(bytes(buf), 4)[: self.max_len])

    def _op_bit_transpose_64(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.mutations import bit_transpose

        if len(buf) >= 8:
            return bytearray(bit_transpose(bytes(buf), 8)[: self.max_len])

    def _op_length_grow(self, buf, _byte_idx, _data):
        if buf and len(buf) < self.max_len:
            size = random.randint(1, min(64, self.max_len - len(buf)))
            if size > 0:
                buf.extend(random.randint(0, 255) for _ in range(size))

    def _op_length_shrink(self, buf, _byte_idx, _data):
        if len(buf) > 2:
            del buf[random.randint(1, len(buf) - 1) :]

    def _op_repeat_clone(self, buf, _byte_idx, _data):
        if buf and len(buf) < self.max_len:
            idx = random.randint(0, len(buf) - 1)
            size = random.randint(1, min(16, len(buf) - idx))
            block = buf[idx : idx + size]
            ins = idx + size
            if ins <= len(buf) and len(buf) + len(block) <= self.max_len:
                buf[ins:ins] = block

    def _op_truncate(self, buf, _byte_idx, _data):
        if len(buf) > 2:
            del buf[random.randint(2, len(buf)) :]

    def _op_swap_regions(self, buf, _byte_idx, _data):
        if len(buf) >= 4:
            i = random.randint(0, len(buf) - 3)
            j = random.randint(i + 2, len(buf) - 1)
            size = random.randint(1, min(j - i, 16))
            a, b = buf[i : i + size], buf[j : j + size]
            buf[i : i + size] = b
            buf[j : j + size] = a

    def _op_swap_bytes(self, buf, _byte_idx, _data):
        if len(buf) >= 2:
            i, j = random.sample(range(len(buf)), 2)
            buf[i], buf[j] = buf[j], buf[i]

    def _op_endianness_swap(self, buf, _byte_idx, _data):
        if buf:
            width = random.choice([2, 4, 8])
            if len(buf) >= width:
                idx = random.randint(0, len(buf) - width)
                val = int.from_bytes(buf[idx : idx + width], "little")
                buf[idx : idx + width] = val.to_bytes(width, "big")

    def _op_grammar_mutate(self, buf, _byte_idx, _data):
        if self.grammar:
            return bytearray(self.grammar.mutate(bytes(buf), max_len=self.max_len)[: self.max_len])

    def _op_grammar_tree_mutate(self, buf, _byte_idx, _data):
        if self.grammar:
            from fuzzer_tool.core.grammar import TreeMutator

            if not hasattr(self, "_tree_mutator"):
                self._tree_mutator = TreeMutator(self.grammar)
            tree = self._tree_mutator.parse(bytes(buf))
            return bytearray(
                self._tree_mutator.mutate_tree(tree, max_len=self.max_len)[: self.max_len]
            )

    def _op_png_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.png_mutations import PngChunkMutator, parse_png_chunks

        if not hasattr(self, "_png_mutator"):
            self._png_mutator = PngChunkMutator()
        if parse_png_chunks(bytes(buf)):
            mutated = self._png_mutator.mutate(bytes(buf), max_len=self.max_len)
        else:
            mutated = self._png_mutator._generate_random_png(self.max_len)
        return bytearray(mutated[: self.max_len])

    def _op_jpeg_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.jpeg_mutations import JpegMutator, parse_jpeg_markers

        if not hasattr(self, "_jpeg_mutator"):
            self._jpeg_mutator = JpegMutator()
        if parse_jpeg_markers(bytes(buf)):
            mutated = self._jpeg_mutator.mutate(bytes(buf), max_len=self.max_len)
        else:
            mutated = self._jpeg_mutator._generate_random_jpeg(max_len=self.max_len)
        return bytearray(mutated[: self.max_len])

    def _op_jpeg_crc_fix(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.jpeg_mutations import (
            parse_jpeg_markers,
            serialize_jpeg_markers,
            STANDALONE_MARKERS,
        )

        if buf:
            markers = parse_jpeg_markers(bytes(buf))
            if markers and len(markers) > 2:
                candidates = [
                    i
                    for i, m in enumerate(markers)
                    if m.marker not in STANDALONE_MARKERS and len(m.data) > 0
                ]
                if candidates:
                    idx = random.choice(candidates)
                    marker = markers[idx]
                    data = bytearray(marker.data)
                    for _ in range(random.randint(1, min(4, len(data)))):
                        data[random.randint(0, len(data) - 1)] ^= 1 << random.randint(0, 7)
                    marker.data = bytes(data)
                    return bytearray(serialize_jpeg_markers(markers)[: self.max_len])

    def _op_gzip_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.gzip_mutations import GzipMutator, parse_gzip

        if not hasattr(self, "_gzip_mutator"):
            self._gzip_mutator = GzipMutator()
        if parse_gzip(bytes(buf)):
            mutated = self._gzip_mutator.mutate(bytes(buf), max_len=self.max_len)
        else:
            mutated = self._gzip_mutator._generate_random_gzip(max_len=self.max_len)
        return bytearray(mutated[: self.max_len])

    def _op_bmp_chunk_mutate(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.bmp_mutations import BmpMutator, parse_bmp

        if not hasattr(self, "_bmp_mutator"):
            self._bmp_mutator = BmpMutator()
        if parse_bmp(bytes(buf)):
            mutated = self._bmp_mutator.mutate(bytes(buf), max_len=self.max_len)
        else:
            mutated = self._bmp_mutator._generate_random_bmp(max_len=self.max_len)
        return bytearray(mutated[: self.max_len])

    def _op_png_crc_fix(self, buf, _byte_idx, _data):
        from fuzzer_tool.core.png_mutations import parse_png_chunks, serialize_png_chunks

        if buf:
            chunks = parse_png_chunks(bytes(buf))
            if chunks and len(chunks) > 1:
                candidates = [i for i, c in enumerate(chunks) if c.chunk_type != b"IEND"]
                if candidates:
                    idx = random.choice(candidates)
                    chunk = chunks[idx]
                    if chunk.data:
                        data = bytearray(chunk.data)
                        for _ in range(random.randint(1, min(4, len(data)))):
                            data[random.randint(0, len(data) - 1)] ^= 1 << random.randint(0, 7)
                        chunk.data = bytes(data)
                    else:
                        chunk.data = bytes(
                            random.randint(0, 255) for _ in range(random.randint(1, 32))
                        )
                    return bytearray(serialize_png_chunks(chunks)[: self.max_len])

    def _op_redqueen(self, buf, _byte_idx, data):
        parent_meta = self.seed_meta.get(data)
        if not (buf and parent_meta):
            return
        matches = parent_meta.get("redqueen_matches", [])
        offsets = parent_meta.get("redqueen_offsets", [])
        if matches:
            for _ in range(random.randint(1, min(4, len(matches)))):
                off, op_a, op_b = random.choice(matches)
                end = off + len(op_a)
                if end <= len(buf) and bytes(buf[off:end]) == op_a:
                    for j, b_val in enumerate(op_b):
                        if off + j < len(buf):
                            buf[off + j] = b_val
        elif offsets and self._cmplog and self._cmplog.tokens:
            for _ in range(random.randint(1, min(4, len(offsets)))):
                off = random.choice(offsets)
                if off < len(buf):
                    token = random.choice(self._cmplog.tokens)
                    for j, b_val in enumerate(token):
                        if off + j < len(buf):
                            buf[off + j] = b_val
        elif offsets:
            for _ in range(random.randint(1, min(4, len(offsets)))):
                off = random.choice(offsets)
                if off < len(buf):
                    buf[off] ^= 0xFF

    def _op_havoc(self, buf, _byte_idx, data):
        return bytes(self._havoc_mutate(buf))

    # ── Dispatch table: op name → handler method ───────────────────────
    def _build_dispatch(self):
        return {
            "bit_flip": self._op_bit_flip,
            "byte_flip": self._op_byte_flip,
            "interesting_8": self._op_interesting_8,
            "interesting_16": self._op_interesting_16,
            "interesting_32": self._op_interesting_32,
            "arithmetic": self._op_arithmetic,
            "random_bytes": self._op_random_bytes,
            "block_insert": self._op_block_insert,
            "block_delete": self._op_block_delete,
            "block_duplicate": self._op_block_duplicate,
            "dict_insert": self._op_dict_insert,
            "dict_replace": self._op_dict_replace,
            "dict_overwrite": self._op_dict_overwrite,
            "dict_prepend": self._op_dict_prepend,
            "dict_append": self._op_dict_append,
            "checksum_repair": self._op_checksum_repair,
            "token_dup": self._op_token_dup,
            "markov_bytes": self._op_markov_bytes,
            "cem_bytes": self._op_cem_bytes,
            "splice": self._op_splice,
            "crossover": self._op_crossover,
            "type_replace": self._op_type_replace,
            "ascii_num": self._op_ascii_num,
            "byte_shuffle": self._op_byte_shuffle,
            "byte_delete": self._op_byte_delete,
            "byte_insert": self._op_byte_insert,
            "insert_ascii_num": self._op_insert_ascii_num,
            "transpose_16": self._op_transpose_16,
            "transpose_32": self._op_transpose_32,
            "transpose_64": self._op_transpose_64,
            "bit_transpose_8": self._op_bit_transpose_8,
            "bit_transpose_16": self._op_bit_transpose_16,
            "bit_transpose_32": self._op_bit_transpose_32,
            "bit_transpose_64": self._op_bit_transpose_64,
            "length_grow": self._op_length_grow,
            "length_shrink": self._op_length_shrink,
            "repeat_clone": self._op_repeat_clone,
            "truncate": self._op_truncate,
            "swap_regions": self._op_swap_regions,
            "swap_bytes": self._op_swap_bytes,
            "endianness_swap": self._op_endianness_swap,
            "grammar_mutate": self._op_grammar_mutate,
            "grammar_tree_mutate": self._op_grammar_tree_mutate,
            "png_chunk_mutate": self._op_png_chunk_mutate,
            "jpeg_chunk_mutate": self._op_jpeg_chunk_mutate,
            "jpeg_crc_fix": self._op_jpeg_crc_fix,
            "gzip_chunk_mutate": self._op_gzip_chunk_mutate,
            "bmp_chunk_mutate": self._op_bmp_chunk_mutate,
            "png_crc_fix": self._op_png_crc_fix,
            "redqueen": self._op_redqueen,
            "havoc": self._op_havoc,
        }

    def _havoc_mutate(self, buf: bytearray) -> bytearray:
        for _ in range(random.randint(2, 8)):
            self._apply_single_mutation(buf)
        return buf

    def _apply_single_mutation(self, buf: bytearray):
        if not buf:
            buf.extend(random.randint(0, 255) for _ in range(random.randint(1, 16)))
            return
        op = random.randint(0, 10)
        if op == 0:
            idx = random.randint(0, len(buf) - 1)
            buf[idx] ^= 1 << random.randint(0, 7)
        elif op == 1:
            idx = random.randint(0, len(buf) - 1)
            buf[idx] = random.randint(0, 255)
        elif op == 2 and len(buf) > 1:
            i, j = random.sample(range(len(buf)), 2)
            buf[i], buf[j] = buf[j], buf[i]
        elif op == 3 and len(buf) < self.max_len:
            idx = random.randint(0, len(buf))
            buf.insert(idx, random.randint(0, 255))
        elif op == 4 and len(buf) > 1:
            idx = random.randint(0, len(buf) - 1)
            size = random.randint(1, min(len(buf) - 1, len(buf) - idx))
            del buf[idx : idx + size]
        elif op == 5 and len(buf) >= 4:
            import zlib

            pos = random.randint(0, max(0, len(buf) - 4))
            buf[pos : pos + 4] = zlib.crc32(bytes(buf[:pos])).to_bytes(4, "big")
        elif op == 6 and len(buf) >= 2:
            i = random.randint(0, len(buf) - 2)
            j = random.randint(i + 1, len(buf) - 1)
            size = random.randint(1, min(j - i, 8))
            a = buf[i : i + size]
            b = buf[j : j + size]
            buf[i : i + size] = b
            buf[j : j + size] = a
        elif op == 7 and buf:
            width = random.choice([2, 4])
            if len(buf) >= width:
                idx = random.randint(0, len(buf) - width)
                val = int.from_bytes(buf[idx : idx + width], "little")
                buf[idx : idx + width] = val.to_bytes(width, "big")
        elif op == 8 and len(buf) > 2:
            del buf[random.randint(2, len(buf) - 1) :]
        elif op == 9 and buf and len(buf) < self.max_len:
            size = random.randint(1, min(16, self.max_len - len(buf)))
            if size > 0:
                buf.extend(random.randint(0, 255) for _ in range(size))
        elif op == 10 and buf and len(buf) < self.max_len:
            idx = random.randint(0, len(buf) - 1)
            size = random.randint(1, min(16, len(buf) - idx))
            block = buf[idx : idx + size]
            ins = idx + size
            if ins <= len(buf) and len(buf) + len(block) <= self.max_len:
                buf[ins:ins] = block

    def save_crash(self, data: bytes, returncode: int, stderr: str):
        from fuzzer_tool.adapters.filesystem import hash_data
        from fuzzer_tool.core.crash_metadata import CrashMetadata, find_nearest_corpus

        meta = CrashMetadata()
        meta.exec_count = self.exec_count
        meta.corpus_size = len(self.corpus)
        meta.target = self.target
        meta.mutation_ops = list(self._last_ops_used)
        meta.elapsed = self._format_elapsed()

        # Parent seed hash (the seed that was mutated)
        if self.corpus:
            parent = self._last_parent_seed if hasattr(self, "_last_parent_seed") else None
            if parent:
                meta.parent_seed_hash = hash_data(parent)

        # Target SHA256 (computed once, cached)
        if not hasattr(self, "_target_sha256"):
            try:
                self._target_sha256 = hashlib.sha256(Path(self.target).read_bytes()).hexdigest()[
                    :16
                ]
            except Exception:
                self._target_sha256 = "unknown"
        meta.target_sha256 = self._target_sha256

        # Nearest corpus entry
        if self.corpus:
            label, sim, diffs = find_nearest_corpus(data, self.corpus)
            meta.nearest_corpus_file = label
            meta.nearest_similarity = sim
            meta.diff_bytes = diffs

        # Register state from ptrace (if active)
        if self.ptrace_cov and hasattr(self, "_last_regs"):
            meta.rip = self._last_regs.get("rip", 0)
            meta.rsp = self._last_regs.get("rsp", 0)
            meta.rbp = self._last_regs.get("rbp", 0)

        # Extract frames for crash clustering
        from fuzzer_tool.core.sanitizer import SanitizerReport

        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            sig = report.signature
            if sig not in self.crash_frames:
                self.crash_frames[sig] = report.frames

        return save_crash(
            data,
            returncode,
            stderr,
            self.crashes_dir,
            self.crash_hashes,
            self.crash_sigs,
            metadata=meta,
        )

    def save_to_corpus(self, data: bytes, parent: bytes | None = None):
        # Compute lineage depth: child depth = parent depth + 1
        parent_depth = 0
        if parent is not None:
            parent_meta = self.seed_meta.get(parent)
            if parent_meta is not None:
                parent_depth = parent_meta.get("lineage_depth", 0)

        self._total_corpus_attempts += 1
        if save_to_corpus(
            data,
            self.corpus_dir,
            self.seen_hashes,
            self.bloom,
            parent=parent,
            lineage_depth=parent_depth,
        ):
            self.corpus.append(data)
            # GA: add to population if enabled
            if self.ga:
                import hashlib as _hashlib
                from fuzzer_tool.core.ga import Individual

                seed_key = _hashlib.sha256(data).hexdigest()[:16]
                edge_count = len(self._edge_tracker.seed_edges.get(seed_key, set()))
                ind = Individual(
                    data=data,
                    edge_count=edge_count,
                    generation=self.ga.generation,
                    seed_key=seed_key,
                )
                self.ga.add_to_population(ind)
            self.seed_meta[data] = {
                "fuzz_count": 0,
                "coverage_edges": 0,
                "momentum": 0.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "added_at": time.time(),
                "lineage_depth": parent_depth + 1 if parent else 0,
                "hamming_distance": self._last_hamming_distance,
            }
            self.markov.train(data)
            self.markov_trained = self.markov.is_trained()
            # Check if markov model has plateaued (no new patterns learned)
            if self.markov.snapshot_and_check_plateau():
                log.info(
                    "Markov plateau detected (JS=%.4f) — reducing generation rate",
                    self.markov.last_js_divergence,
                )
            # Track corpus size distribution for dynamic max_len
            self._corpus_size_history.append(len(data))
            if len(self._corpus_size_history) > 1000:
                self._corpus_size_history = self._corpus_size_history[-500:]
            # Secretary-problem: track corpus discovery rate for optimal stopping
            if self._corpus_secretary:
                dr = self.discovery_rate()
                self._corpus_secretary.observe(dr)
                stop, _reason = self._corpus_secretary.should_stop()
                if stop:
                    log.info("Corpus secretary stopping: %s", _reason)
                    self._auto_minimize_corpus()
            # Auto-minimize if corpus exceeds max
            if self.max_corpus > 0 and len(self.corpus) > self.max_corpus:
                self._auto_minimize_corpus()
            # Dynamic max_len: adjust based on corpus size distribution
            if len(self._corpus_size_history) >= 100:
                sorted_sizes = sorted(self._corpus_size_history)
                p90 = sorted_sizes[-len(sorted_sizes) // 10]
                self.max_len = max(self.max_len, min(p90 * 2, 65536))
        else:
            self._duplicate_reject_count += 1

    def _trim_new_coverage(self, data: bytes, parent: bytes) -> None:
        """Trim input to minimal size that still hits the same edges.

        Single-pass: remove half the input, check edges. If it works,
        the trimmed version replaces the corpus entry. Only 1 extra
        target run per new-coverage event.
        """
        if len(data) <= 16:
            return  # too small to trim

        if self.shm_cov:
            current_edges = self.shm_cov.read_bitmap()
        elif self.ptrace_cov:
            current_edges = bytes(self.ptrace_cov.edge_map)
        else:
            return

        trimmed = data[: len(data) // 2]
        rc, _ = self._run_target(trimmed)
        if rc in (-2, -1):
            return

        if self.shm_cov:
            trimmed_edges = self.shm_cov.read_bitmap()
        elif self.ptrace_cov:
            trimmed_edges = bytes(self.ptrace_cov.edge_map)
        else:
            return

        if not self._edges_subset_of(trimmed_edges, current_edges):
            return  # lost edges — keep original

        # Trimmed version preserves all edges — replace in corpus
        seed_key = self._seed_key(data)
        if data in self.seed_meta:
            self.seed_meta.pop(data, None)
        if data in self.corpus:
            idx = self.corpus.index(data)
            self.corpus[idx] = trimmed
            self.seed_meta[trimmed] = {
                "fuzz_count": 0,
                "coverage_edges": self._edge_tracker.get_seed_edge_count(seed_key),
                "momentum": 0.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "added_at": time.time(),
                "lineage_depth": self.seed_meta.get(data, {}).get("lineage_depth", 0) + 1,
            }
            log.debug("Trimmed %d -> %d bytes", len(data), len(trimmed))

    @staticmethod
    def _edges_subset_of(candidate: bytes, reference: bytes) -> bool:
        """Check if all non-zero positions in reference are also non-zero in candidate."""
        for i in range(min(len(candidate), len(reference))):
            if reference[i] and not candidate[i]:
                return False
        return True

    def _auto_minimize_corpus(self):
        """Inline corpus minimization: hash dedup + subsumption pruning.

        Keeps inputs that discovered the most edges. Removed inputs are
        moved to ``corpus/pruned/`` by the caller (save_to_corpus).
        Triggered either by max_corpus limit or dynamically when the
        corpus accumulates too many stale seeds (high fuzz_count, zero
        new edges).
        """
        # GA mode handles population bounds via generational culling
        if self.ga:
            return
        if not self.corpus:
            return

        from fuzzer_tool.adapters.filesystem import hash_data

        # Deduplicate by content hash
        seen: set[str] = set()
        unique: list[bytes] = []
        for seed in self.corpus:
            h = hash_data(seed)
            if h not in seen:
                seen.add(h)
                unique.append(seed)

        # Dynamic trigger: if >30% of seeds have 0 edges after 50+ fuzzes,
        # the corpus is bloated — prune even if under max_corpus.
        stale_count = 0
        for seed in unique:
            meta = self.seed_meta.get(seed)
            if meta and meta["fuzz_count"] >= 50 and meta["coverage_edges"] == 0:
                stale_count += 1
        stale_ratio = stale_count / max(len(unique), 1)

        # Determine target corpus size.
        # explicit max_corpus: use it directly.
        # no max_corpus: dynamic cap = max(productive_seeds * 3, 100).
        # This keeps the corpus proportional to discovered coverage.
        if self.max_corpus > 0:
            target_size = self.max_corpus
        else:
            # Dynamic cap: based on discovered edges, not per-seed counts.
            # Keep enough seeds to cover all edges with some exploration buffer.
            edges = 0
            if self.shm_cov:
                edges = self.shm_cov.cumulative_edges
            elif self.ptrace_cov:
                edges = self.ptrace_cov.cumulative_edges
            target_size = max(edges * 2, 50)

        if stale_ratio > 0.3:
            # Stale seeds detected — prune them even if under max_corpus.
            # Reduce target by stale ratio to remove dead weight.
            if len(unique) > target_size:
                target_size = max(target_size, int(len(unique) * (1.0 - stale_ratio)))
            else:
                # Under max_corpus but many stale seeds — still prune them
                target_size = int(len(unique) * (1.0 - stale_ratio))

        # Floor: corpus cannot be smaller than the number of productive seeds
        # (seeds that discovered at least one edge). This ensures every
        # discovered edge retains at least one covering seed.
        productive = sum(
            1 for seed in unique if self.seed_meta.get(seed, {}).get("coverage_edges", 0) > 0
        )
        if productive > 0:
            target_size = max(target_size, productive)

        # Prune subsumed seeds using edge coverage + diversity scoring
        if len(unique) > target_size:
            scored = []
            for seed in unique:
                seed_key = self._seed_key(seed)
                edge_count = self._edge_tracker.get_seed_edge_count(seed_key)
                meta = self.seed_meta.get(seed)
                fuzz = meta["fuzz_count"] if meta else 0
                discovered = meta["coverage_edges"] if meta else 0

                # Edge coverage score: seeds that discovered edges are valuable
                # Penalize seeds that were fuzzed many times without discoveries
                edge_score = discovered * 10
                if fuzz > 0 and discovered == 0:
                    # Stale seed: penalize proportionally to fuzz count
                    edge_score *= max(0.01, 1.0 / (1.0 + fuzz * 0.01))
                else:
                    # Fresh or productive seed: slight boost for low fuzz count
                    edge_score += 1.0 / max(fuzz, 1)

                # Wasserstein diversity: spatially distant seeds are valuable
                wasserstein_weight = self._edge_tracker.compute_wasserstein_weight(seed_key)

                score = edge_score * wasserstein_weight
                scored.append((score, seed))
            scored.sort(key=lambda x: x[0], reverse=True)
            # Keep top target_size, but enforce floor: never prune below
            # the number of productive seeds.
            keep = min(target_size, len(scored))
            if keep < productive:
                keep = min(productive, len(scored))
            unique = [s for _, s in scored[:keep]]

        removed = len(self.corpus) - len(unique)
        if removed > 0:
            # Move pruned files to corpus/pruned/ before removing from memory
            pruned_dir = self.corpus_dir / "pruned"
            pruned_dir.mkdir(parents=True, exist_ok=True)
            from fuzzer_tool.adapters.filesystem import hash_data as _hash

            kept_set = {_hash(s) for s in unique}
            for f in self.corpus_dir.iterdir():
                if not f.is_file():
                    continue
                if f.suffix == ".json" and f.name.startswith("delta_"):
                    h = f.name[6:-5]
                elif f.name.startswith("id_"):
                    h = f.name[3:]
                else:
                    continue
                if h not in kept_set:
                    shutil.move(str(f), str(pruned_dir / f.name))

            self.corpus = unique
            # Rebuild seed_meta for kept seeds
            new_meta = {}
            for seed in unique:
                if seed in self.seed_meta:
                    new_meta[seed] = self.seed_meta[seed]
            self.seed_meta = new_meta
            # Invalidate weight caches (corpus changed)
            self._weight_cache = None
            self._cached_weights = {}
            self._last_minimize_exec = self.exec_count
            self._pruned_count += removed
            log.info(
                "Auto-minimized corpus: %d -> %d seeds -> pruned/ (stale_ratio=%.1f)",
                len(self.corpus) + removed,
                len(self.corpus),
                stale_ratio,
            )

    def _deprioritize_near_duplicates(self):
        """Find seeds with near-identical edge bitmaps and merge them.

        Uses Hamming distance on edge bitmaps (via edge_tracker) to find
        seed pairs that are coverage-redundant. When found, the seed with
        fewer new edges is removed from the corpus.

        Called periodically alongside corpus minimization.
        """
        if len(self.corpus) < 10:
            return

        near_dupes = self._edge_tracker.find_near_duplicate_seeds(max_hamming=0.05)
        if not near_dupes:
            return

        # Collect seeds to remove: keep the one with more coverage_edges
        to_remove: set[bytes] = set()
        for key_a, key_b, hdist in near_dupes:
            # Find the actual seed bytes from keys
            seed_a = None
            seed_b = None
            for s in self.corpus:
                if self._seed_key(s) == key_a:
                    seed_a = s
                elif self._seed_key(s) == key_b:
                    seed_b = s
                if seed_a and seed_b:
                    break
            if not seed_a or not seed_b:
                continue
            if seed_a in to_remove or seed_b in to_remove:
                continue

            meta_a = self.seed_meta.get(seed_a, {})
            meta_b = self.seed_meta.get(seed_b, {})
            edges_a = meta_a.get("coverage_edges", 0)
            edges_b = meta_b.get("coverage_edges", 0)

            # Remove the one with fewer discovered edges
            if edges_a <= edges_b:
                to_remove.add(seed_a)
            else:
                to_remove.add(seed_b)

        if to_remove:
            self.corpus = [s for s in self.corpus if s not in to_remove]
            for s in to_remove:
                self.seed_meta.pop(s, None)
            self._weight_cache = None
            self._cached_weights = {}
            log.info(
                "Deprioritized %d near-duplicate seeds (Hamming <= 0.05 on edge bitmaps)",
                len(to_remove),
            )

    def _pick_seed(self) -> bytes:
        # Stall recovery: pick seeds randomly
        if self._stall_recovery_active and self.corpus:
            self._seed_strategy = "random_stall"
            return random.choice(self.corpus)

        # Elo-arbitrated seed selection: pick between strategies
        if self._use_elo and self._elo:
            available = []
            if self.ga:
                available.append("ga")
            available.append("weighted")
            if self.corpus and self.seed_meta:
                available.append("pareto")
            if self._profile.format_signature:
                available.append("format")

            if len(available) >= 2:
                strategy = self._elo.select_strategy(available)
                self._seed_strategy = strategy
            elif available:
                strategy = available[0]
                self._seed_strategy = strategy
            else:
                strategy = None

            if strategy == "ga" and self.ga:
                return self.ga.pick_seed()
            elif strategy == "pareto" and self.corpus and self.seed_meta:
                return self._pick_pareto_only()
            elif strategy == "format":
                return self._format_aware_seed()
            # else: fall through to weighted

        # Non-elo paths
        if self.ga:
            return self.ga.pick_seed()
        if self.markov_generate and self.markov_trained:
            return self._pick_markov_seed()
        if self.corpus and self.seed_meta:
            return self._weighted_pick_seed()
        if self.corpus:
            return random.choice(self.corpus)
        return self._format_aware_seed()

    def _pick_markov_seed(self) -> bytes:
        """Generate a seed from the Markov chain."""
        from fuzzer_tool.core.edge_tracker import ks_significance_threshold

        plateau_threshold = ks_significance_threshold(
            max(1, self.markov._contexts_seen), alpha=0.05
        )
        gen_rate = 0.03 if self.markov.last_js_divergence < plateau_threshold else 0.15

        if not hasattr(self, "_last_corpus_pp"):
            self._last_corpus_pp = 256.0
        if self.exec_count % 500 == 0 and self.corpus:
            pp_stats = self.markov.corpus_perplexity(self.corpus)
            self._last_corpus_pp = pp_stats["mean"]
        if self._last_corpus_pp > 200:
            gen_rate = min(gen_rate * 2, 0.40)
        elif self._last_corpus_pp < 10:
            gen_rate = max(gen_rate * 0.3, 0.01)

        if random.random() < gen_rate:
            length = random.randint(1, min(256, self.max_len))
            for _ in range(3):
                candidate = self.markov.generate(length)
                pp = self.markov.perplexity(candidate)
                if pp < 512:
                    return candidate
            return candidate
        length = random.randint(1, min(256, self.max_len))
        return self.markov.generate(length)

    def _pick_pareto_only(self) -> bytes:
        """Select a seed using only the Pareto frontier (no weighted scoring)."""
        if len(self.corpus) < 3 or not self.seed_meta:
            return random.choice(self.corpus)
        now = time.time()
        # Use equal weights — pure Pareto selection
        weights = [1.0] * len(self.corpus)
        return self._pick_from_pareto_front(weights, now)

    def _format_aware_seed(self) -> bytes:
        """Generate a seed that matches the target's inferred input format."""
        fmt = getattr(self._profile, "format_signature", None)
        if fmt == "png":
            # Minimal valid PNG: signature + IHDR + IEND
            import binascii

            ihdr_data = b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"  # 1x1, 8-bit RGB
            ihdr_chunk = b"IHDR" + ihdr_data
            ihdr_crc = struct.pack(">I", binascii.crc32(ihdr_chunk))
            iend_chunk = b"IEND"
            iend_crc = struct.pack(">I", binascii.crc32(iend_chunk))
            return (
                b"\x89PNG\r\n\x1a\n"
                + struct.pack(">I", len(ihdr_data))
                + ihdr_chunk
                + ihdr_crc
                + struct.pack(">I", 0)
                + iend_chunk
                + iend_crc
            )
        if fmt == "text":
            # Text-like: common delimiters and keywords from the target
            parts = [b"GET / HTTP/1.1\r\n", b"Host: localhost\r\n"]
            if self._profile.boundary_markers:
                sep = self._profile.boundary_markers[0]
                parts.append(sep * 4)
            return b"".join(parts)
        if fmt == "json":
            return b'{"key": "value", "num": 0}'
        if fmt == "xml":
            return b'<?xml version="1.0"?><root><data/></root>'
        if fmt == "elf":
            return b"\x7fELF" + b"\x00" * 12
        if fmt == "html":
            return b"<!DOCTYPE html><html><body></body></html>"
        # Fallback: fill with boundary markers if known, else zeros
        if self._profile.boundary_markers:
            marker = self._profile.boundary_markers[0]
            return marker * min(16, self.max_len)
        return b"\x00" * min(64, self.max_len)

    def _compute_weights(self, now: float) -> list[float]:
        """Compute seed selection weights. Cached until corpus/edge-tracker changes.

        Uses five statistical signals from edge aggregation:
          1. Base: inverse fuzz count + coverage + age (exploitation)
          2. Rare edge boost: singleton/cold edge coverage (irreplaceability)
          3. Hit frequency: seeds consistently hitting edges are reliable
          4. Edge gap targeting: boost seeds near under-covered edges
          5. Edge diversity: prefer seeds whose edges don't overlap with others

        After computing base weights, applies a sliding-window Pareto
        dominance boost: seeds that are non-dominated on the
        (novelty, freshness, diversity) frontier get a 2x boost;
        dominated seeds get a 0.5x dampening. This preserves the
        information that multiplication destroys — two seeds with the
        same weight can now be distinguished by their Pareto status.
        """
        weights = []
        # Collect per-seed objective scores for Pareto analysis
        pareto_scores: list[tuple[float, float, float]] = []

        for seed in self.corpus:
            meta = self.seed_meta.get(seed)
            if meta is None:
                weights.append(1.0)
                pareto_scores.append((1.0, 1.0, 1.0))
                continue
            fuzz_count = max(meta["fuzz_count"], 1)
            coverage = meta["coverage_edges"]
            age = now - meta["added_at"]

            T = self._temperature

            # Base weight: inverse fuzz count, boosted by coverage, decayed by age
            explore_part = T * (1.0 / math.sqrt(fuzz_count))
            exploit_part = (1.0 + coverage * 0.5) / (1.0 + age * 0.01)
            w = explore_part * exploit_part

            # Momentum: recent discovery velocity boosts weight
            momentum = meta.get("momentum", 0.0)
            w *= 1.0 + momentum * 2.0

            # Energy burst: newly added seeds get up to 5x boost (decays over 60s)
            burst_factor = max(1.0, 1.0 + T * (5.0 - 1.0) - (age / 60.0) * T)

            # Stale seed detection
            staleness = fuzz_count / max(coverage + 1, 1)
            stale_threshold = 50.0 * T
            w *= 0.01 if staleness > stale_threshold else 1.0

            # Edge tracker signals: compute lazily, cache per-seed.
            seed_key = self._seed_key(seed)

            # Secretary-problem optimal stopping: dampen seeds that have
            # been explored enough (rank-based plateau detection)
            if self._secretary and seed_key in self._seed_secretary:
                stop, _reason = self._seed_secretary[seed_key].should_stop()
                if stop:
                    w *= 0.01
            if seed_key not in self._cached_weights:
                if (
                    seed_key in self._edge_tracker.seed_edges
                    and self._edge_tracker.seed_edges[seed_key]
                ):
                    sub = self._edge_tracker.compute_subsumption_weight(seed_key)
                    div = self._edge_tracker.compute_hitcount_diversity_weight(seed_key)
                    spa = self._edge_tracker.compute_wasserstein_weight(seed_key)
                    cov = self._edge_tracker.compute_coverage_proximity(seed_key)
                    self._cached_weights[seed_key] = (sub, div, spa, cov)
                else:
                    self._cached_weights[seed_key] = (1.0, 1.0, 1.0, 0.5)
            sub, div, spa, cov = self._cached_weights[seed_key]
            w *= sub * div * spa
            w *= 0.5 + cov

            # Signal 2: Rare edge boost — seeds hitting singleton/cold edges
            # are irreplaceable. High weight prevents pruning and encourages
            # re-fuzzing to find deeper paths from those rare edges.
            seed_edges = self._edge_tracker.seed_edges.get(seed_key, set())
            if seed_edges:
                rare_count = sum(
                    1 for e in seed_edges if self._edge_tracker._global_edge_hits.get(e, 0) <= 2
                )
                if rare_count > 0:
                    w *= 1.0 + rare_count * 0.5

            # Signal 3: Hit frequency — seeds that consistently hit edges
            # (high mean_hit_per_seed) are more reliable than sporadic ones.
            # A seed that hits 10 edges 50 times each is more useful than
            # one that hits 10 edges 1 time each (might be noise).
            if seed_edges:
                total_hits = 0
                for e in seed_edges:
                    total_hits += self._edge_tracker._global_edge_hits.get(e, 0)
                mean_hits = total_hits / len(seed_edges) if seed_edges else 0
                # Boost seeds with consistent hits (mean > 3), penalize sporadic (mean < 1.5)
                if mean_hits > 3:
                    w *= 1.0 + (mean_hits - 3) * 0.1
                elif mean_hits < 1.5 and fuzz_count > 10:
                    w *= 0.7

            # Signal 4: Edge gap targeting — boost seeds whose edges include
            # under-covered edges (low seed_count). This steers the fuzzer
            # toward coverage gaps.
            if seed_edges:
                gap_score = 0
                for e in seed_edges:
                    seed_count = self._edge_tracker._global_edge_hits.get(e, 0)
                    if seed_count <= 2:
                        gap_score += 1
                if gap_score > 0:
                    w *= 1.0 + gap_score * 0.3

            # Signal 5: Edge diversity — penalize seeds whose edges overlap
            # with recently-selected seeds. Encourages exploring different
            # code regions instead of re-fuzzing the same paths.
            if seed_edges and hasattr(self, "_recent_seed_edges"):
                overlap = 0
                for recent in self._recent_seed_edges:
                    overlap += len(seed_edges & recent)
                if overlap > 0:
                    # Penalize proportionally to overlap ratio
                    penalty = overlap / max(len(seed_edges), 1)
                    w *= max(0.3, 1.0 - penalty * 0.5)

            # Signal 5: Edge diversity — computed lazily in _weighted_pick_seed
            # using max边际 coverage selection (not here for performance).

            # Directed distance (cheap, always include)
            if self._distance:
                seed_dist = meta.get("avg_distance", self._distance.max_distance)
                max_d = self._distance.max_distance
                norm_dist = min(seed_dist / max_d, 1.0) if max_d > 0 else 0.5
                alpha = min(self._anneal_progress * 2, 1.0)
                dist_weight = math.exp(-norm_dist * 5.0 * alpha)
                w *= (1.0 - alpha) + alpha * dist_weight

            # Hot-function weighting: seeds exercising code paths through
            # high-branch-density functions get a proportional boost.  We
            # approximate this via coverage_edges (more edges ≈ more code
            # paths ≈ more hot functions hit).
            if self._profile.hot_functions and self._profile.functions:
                hot_density = sum(
                    self._profile.functions[f].branch_density
                    for f in self._profile.hot_functions
                    if f in self._profile.functions
                ) / max(len(self._profile.hot_functions), 1)
                all_density = sum(
                    fi.branch_density for fi in self._profile.functions.values()
                ) / max(len(self._profile.functions), 1)
                if all_density > 0:
                    hotness_ratio = hot_density / all_density
                    # Boost seeds with above-median coverage by hotness ratio
                    if coverage > 0:
                        w *= 1.0 + (hotness_ratio - 1.0) * min(coverage / 50.0, 1.0)

            # Signal 6: Mutation perturbation — seeds with very low hamming_distance
            # to their parent are minimally perturbed and unlikely to discover new
            # paths. Penalize them to avoid wasting execs on near-identical inputs.
            # hamming_distance == -1 means unknown (length-changing mutation).
            hd = meta.get("hamming_distance", -1)
            if hd == 0:
                w *= 0.1  # identical to parent — almost certainly redundant
            elif 0 < hd <= 2:
                w *= 0.5  # tiny perturbation — low novelty expected
            # hd >= 3 or hd == -1: no penalty (normal or unknown perturbation)

            weights.append(max(w, 1e-6))

            # Collect Pareto objective scores:
            # novelty (subsumption), freshness (burst), diversity (wasserstein)
            novelty = sub
            freshness = burst_factor
            diversity = spa
            pareto_scores.append((novelty, freshness, diversity))

        # Sliding-window Pareto dominance boost
        if len(pareto_scores) >= 3:
            front = self._pareto_front(pareto_scores, window=100)
            for i in range(len(weights)):
                if i in front:
                    weights[i] *= 2.0  # boost non-dominated seeds
                else:
                    weights[i] *= 0.5  # dampen dominated seeds

        return weights

    @staticmethod
    def _pareto_front(scores: list[tuple[float, float, float]], window: int = 100) -> set[int]:
        """Find indices of non-dominated points in a sliding window.

        A point is non-dominated if no other point in the window is
        better on ALL three objectives simultaneously.

        Args:
            scores: List of (novelty, freshness, diversity) per seed.
            window: Only check dominance within the most recent N seeds.

        Returns:
            Set of indices that are on the Pareto front.
        """
        n = len(scores)
        start = max(0, n - window)
        front: set[int] = set(range(start, n))

        for i in range(start, n):
            if i not in front:
                continue
            ni, fi, di = scores[i]
            for j in range(start, n):
                if j == i or j not in front:
                    continue
                nj, fj, dj = scores[j]
                # j dominates i if j is >= on all axes and > on at least one
                if nj >= ni and fj >= fi and dj >= di and (nj > ni or fj > fi or dj > di):
                    front.discard(i)
                    break

        return front

    def _pick_from_pareto_front(self, weights: list[float], now: float) -> bytes:
        """Select a seed using Pareto frontier sampling.

        When the Pareto front (non-dominated seeds on the
        novelty/freshness/diversity frontier) contains enough seeds,
        sample from the front using the existing weights. This makes
        tradeoffs visible: a seed that's moderately novel but very fresh
        beats one that's highly novel but stale, because freshness is an
        independent axis.

        Falls back to full corpus selection when:
        - Fewer than 3 seeds in the corpus
        - Fewer than 2 seeds on the Pareto front (frontier too small
          to be meaningful — likely all seeds are comparable)
        - No edge tracker data available

        Args:
            weights: Pre-computed selection weights.
            now: Current timestamp for freshness computation.

        Returns:
            Selected seed bytes.
        """
        if len(self.corpus) < 3 or not self.seed_meta:
            return random.choices(self.corpus, weights=weights, k=1)[0]

        # Compute Pareto objective scores for all seeds
        pareto_scores: list[tuple[float, float, float]] = []
        for seed in self.corpus:
            meta = self.seed_meta.get(seed)
            if meta is None:
                pareto_scores.append((1.0, 1.0, 1.0))
                continue
            seed_key = self._seed_key(seed)
            sub, div, spa, _cov = self._cached_weights.get(seed_key, (1.0, 1.0, 1.0, 0.5))
            age = now - meta["added_at"]
            burst = max(
                1.0, 1.0 + self._temperature * (5.0 - 1.0) - (age / 60.0) * self._temperature
            )
            pareto_scores.append((sub, burst, spa))

        # Find the Pareto front
        front = self._pareto_front(pareto_scores, window=100)

        if len(front) >= 2:
            # Sample from the Pareto front using weights
            front_indices = sorted(front)
            front_weights = [weights[i] for i in front_indices]
            front_seeds = [self.corpus[i] for i in front_indices]
            return random.choices(front_seeds, weights=front_weights, k=1)[0]
        else:
            # Frontier too small — fall back to full corpus
            return random.choices(self.corpus, weights=weights, k=1)[0]

    def _weighted_pick_seed(self) -> bytes:
        now = time.time()

        # Update simulated annealing temperature
        if self._anneal_budget > 0:
            self._temperature = max(0.1, 1.0 - self.exec_count / self._anneal_budget)
        else:
            self._temperature = 1.0

        # Edge diversity: track recently selected seeds' edges and penalize
        # overlap. This encourages exploring different code regions.
        if not hasattr(self, "_recent_seed_edges"):
            self._recent_seed_edges: list[set[int]] = []
            self._recent_seed_max = 20

        # Invalidate cached weights when corpus structure or edge tracker changes.
        # Two-level cache:
        #   _cached_weights: per-seed expensive signals (subsumption, diversity,
        #     spatial, coverage proximity) — only invalidated when edges change.
        #   _weight_cache: final weight vector — extended incrementally when
        #     corpus grows; fully invalidated when edges change.
        corpus_version = len(self.corpus)
        edge_version = self.shm_cov.cumulative_edges if self.shm_cov else 0
        if not hasattr(self, "_weight_cache"):
            self._weight_cache = None
            self._weight_cache_key = (-1, -1)
            self._cached_weights = {}
        cache_key = (corpus_version, edge_version)
        if cache_key != self._weight_cache_key:
            edge_changed = self._weight_cache_key[1] != edge_version
            self._weight_cache_key = cache_key
            if edge_changed:
                # Edge discovery: invalidate weight vector (recompute all weights).
                # Keep _cached_weights — expensive signals (subsumption, diversity,
                # spatial, proximity) change slowly and are safe to reuse.
                self._weight_cache = None
            elif self._weight_cache is not None and len(self._weight_cache) != corpus_version:
                # Corpus changed: fully recompute weights so fuzz_count-dependent
                # terms (explore_part, burst_factor, staleness) reflect current
                # exec counts for ALL seeds, not just new ones.  Expensive
                # per-seed signals are still cached in _cached_weights.
                self._weight_cache = None

        if self._weight_cache is not None:
            weights = self._weight_cache
        else:
            weights = self._compute_weights(now)
            self._weight_cache = weights

        # Pareto frontier selection: when enough seeds are non-dominated,
        # sample from the frontier rather than the full corpus. This makes
        # the tradeoff structure visible in selection — a seed that's
        # moderately novel but very fresh gets selected over one that's
        # highly novel but stale, because freshness is an independent axis.
        selected = self._pick_from_pareto_front(weights, now)

        # Track selected seed's edges for diversity penalty next time
        sel_key = self._seed_key(selected)
        sel_edges = self._edge_tracker.seed_edges.get(sel_key, set())
        if sel_edges:
            self._recent_seed_edges.append(sel_edges)
            if len(self._recent_seed_edges) > self._recent_seed_max:
                self._recent_seed_edges.pop(0)

        # Cache signal data for ablation logging (consumed by fuzz_one)
        if self._ablation_file:
            meta = self.seed_meta.get(selected)
            if meta:
                seed_key = self._seed_key(selected)
                cached = self._cached_weights.get(seed_key, (1.0, 1.0, 1.0))
                fuzz_count = max(meta["fuzz_count"], 1)
                coverage = meta["coverage_edges"]
                age = now - meta["added_at"]
                base_w = (1.0 / math.sqrt(fuzz_count)) * (1.0 + coverage * 0.5) / (1.0 + age * 0.01)
                burst_factor = max(1.0, 5.0 - (age / 60.0))
                staleness = fuzz_count / max(coverage + 1, 1)
                penalty = 0.01 if staleness > 50 else 1.0
                w = base_w * burst_factor * penalty * cached[0] * cached[1] * cached[2]
                # Coverage proximity weight
                w *= 0.5 + cached[3]
                mdl_weight = 1.0
                if self.markov_trained:
                    cl_ratio = self.markov.codelength_ratio(selected)
                    mdl_weight = 1.0 + min(cl_ratio / 8.0, 1.0)
                    w *= mdl_weight
                self._last_pick_signals = {
                    "seed_idx": self.corpus.index(selected),
                    "seed_hash": selected[:4].hex(),
                    "fuzz_count": fuzz_count,
                    "coverage_edges": coverage,
                    "age_s": f"{age:.1f}",
                    "base_w": f"{base_w:.4f}",
                    "burst": f"{burst_factor:.2f}",
                    "penalty": f"{penalty:.2f}",
                    "subsumption": f"{cached[0]:.4f}",
                    "diversity": f"{cached[1]:.4f}",
                    "spatial": f"{cached[2]:.4f}",
                    "mdl": f"{mdl_weight:.2f}",
                    "final_w": f"{w:.6f}",
                }

        return selected

    def fuzz_one(self, data: bytes) -> bool:
        self._last_parent_seed = data
        meta = self.seed_meta.get(data)
        if meta is not None:
            meta["fuzz_count"] += 1

        t_start = time.monotonic()
        mutated = self.mutate(data)
        returncode, stderr = self._run_target(mutated)
        t_elapsed = time.monotonic() - t_start
        self.exec_count += 1
        if self._stall_recovery_active:
            self._stall_recovery_execs += 1

        # Per-seed wall-clock cost
        if meta is not None:
            meta["total_time"] = meta.get("total_time", 0.0) + t_elapsed

        # Record execution time for adaptive timeout calibration
        self._exec_time_tracker.record(t_elapsed)

        if self.mc:
            self.mc.execs_since_refit += 1

        # Collect cmplog tokens after each execution
        cmplog_found = False
        if self._cmplog:
            new_tokens = self._cmplog.collect_tokens()
            cmplog_found = bool(new_tokens)
            if not hasattr(self, "_dict_set"):
                self._dict_set = set(self.dictionary)
                self._dict_eps_window: list[float] = []
                self._dict_last_prune = 0
            for token in new_tokens:
                if token and token not in self._dict_set:
                    self.dictionary.append(token)
                    self._dict_set.add(token)

            # Dynamic cap: scale with recent throughput.
            # High EPS → larger dictionary (more mutations explore more).
            # Low EPS → smaller dictionary (reduce overhead).
            # Window: last 500 iterations. Range: [64, 1024].
            window = 500
            if self.exec_count > 0 and self.exec_count % 100 == 0:
                elapsed = time.time() - self.start_time
                eps = self.exec_count / elapsed if elapsed > 0 else 0
                self._dict_eps_window.append(eps)
                if len(self._dict_eps_window) > 10:
                    self._dict_eps_window.pop(0)

            if self._dict_eps_window and self.exec_count - self._dict_last_prune >= window:
                avg_eps = sum(self._dict_eps_window) / len(self._dict_eps_window)
                # Map EPS to cap: 10 eps → 128, 30 eps → 256, 100+ eps → 1024
                dyn_cap = max(64, min(1024, int(avg_eps * 8)))
                if len(self.dictionary) > dyn_cap:
                    keep = max(dyn_cap // 2, 32)
                    self.dictionary = self.dictionary[-keep:]
                    self._dict_set = set(self.dictionary)
                    self._dict_last_prune = self.exec_count
            # Record redqueen matches: (offset, operand_a, operand_b)
            # for input-to-state matching during mutation
            if self._cmplog.pairs and meta is not None:
                matches = list(meta.get("redqueen_matches", []))
                seen = {(m[1], m[2]) for m in matches}  # dedup by (A, B)
                for op_a, op_b in self._cmplog.pairs:
                    if len(op_a) < 2 or (op_a, op_b) in seen:
                        continue
                    pos = 0
                    while pos <= len(mutated) - len(op_a):
                        idx = mutated.find(op_a, pos)
                        if idx == -1:
                            break
                        matches.append((idx, op_a, op_b))
                        seen.add((op_a, op_b))
                        pos = idx + 1
                        if len(matches) >= 50:
                            break
                    if len(matches) >= 50:
                        break
                meta["redqueen_matches"] = matches[:50]
                # Keep legacy field for state compat
                meta["redqueen_offsets"] = [m[0] for m in meta["redqueen_matches"]]

        if self.exec_count % 100 == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss > self._peak_rss:
                self._peak_rss = rss
            elapsed = time.time() - self.start_time
            eps = self.exec_count / elapsed if elapsed > 0 else 0
            if eps > self._peak_eps:
                self._peak_eps = eps
            self._crash_rate_history.append((self.exec_count, self.crash_count))

        for op in set(self._last_ops_used):
            self.op_counts[op] = self.op_counts.get(op, 0) + 1

        # Track cmplog as its own operator
        if cmplog_found:
            self.op_counts["cmplog"] = self.op_counts.get("cmplog", 0) + 1

        is_timeout = returncode == -1 and stderr == "timeout"
        if is_timeout:
            self.timeout_count += 1

        is_crash = self._is_crash(returncode, stderr)
        is_interesting = self._is_interesting(returncode, stderr)
        has_new_coverage = (self.ptrace_cov and self.ptrace_cov.is_new_coverage()) or (
            self.shm_cov and self.shm_cov.is_new_coverage()
        )

        # Record crash MI: I(byte_position; crash_outcome)
        if self._crash_mi:
            self._crash_mi.record(mutated, is_crash)

        # Write ablation log row: signal data + outcome
        if self._ablation_file and hasattr(self, "_last_pick_signals"):
            ps = self._last_pick_signals
            self._ablation_file.write(
                f"{self.exec_count},{ps['seed_idx']},{ps['seed_hash']},"
                f"{ps['fuzz_count']},{ps['coverage_edges']},{ps['age_s']},"
                f"{ps['base_w']},{ps['burst']},{ps['penalty']},"
                f"{ps['subsumption']},{ps['diversity']},{ps['spatial']},"
                f"{ps['mdl']},{ps['final_w']},"
                f"{1 if has_new_coverage else 0},{1 if is_crash else 0}\n"
            )
            if self.exec_count % 100 == 0:
                self._ablation_file.flush()

        # Record edges for per-seed tracking
        if has_new_coverage:
            edge_bitmap = self._get_current_edge_bitmap()
            if edge_bitmap:
                seed_key = self._seed_key(data)
                new = self._edge_tracker.record_edges(seed_key, edge_bitmap)
                if new:
                    self._last_new_edge_exec = self.exec_count
                    if self._stall_recovery_active:
                        print(f"\n[*] RECOVERED: found {len(new)} new edges at exec "
                              f"{self.exec_count}, resuming normal mode")
                        self._stall_recovery_active = False
                if meta is not None and new:
                    meta["coverage_edges"] += len(new)
                    meta["momentum"] = 0.8 * meta["momentum"] + 0.2 * 1.0
                elif meta is not None:
                    meta["momentum"] = 0.8 * meta["momentum"]
                # Secretary-problem: track seed discovery rate for optimal stopping
                if self._secretary and seed_key:
                    if seed_key not in self._seed_secretary:
                        self._seed_secretary[seed_key] = SecretaryStopping(
                            window_size=self._secretary_window,
                            exploration_frac=self._secretary_exploration,
                        )
                    fuzz_count = max(meta["fuzz_count"], 1) if meta else 1
                    discovery_rate = len(new) / fuzz_count
                    self._seed_secretary[seed_key].observe(discovery_rate)

        # Compute directed distance for targeted fuzzing
        if self._distance and meta is not None and has_new_coverage:
            edge_bitmap = self._get_current_edge_bitmap()
            if edge_bitmap:
                # Use edge bitmap positions as basic block proxies
                hit_bbs = {i for i, v in enumerate(edge_bitmap) if v > 0}
                # Record edge trace for distance computation
                seed_key = self._seed_key(data)
                edge_pairs = {(i, i) for i in hit_bbs}  # self-loops as BB proxies
                self._edge_tracker.record_edge_trace(seed_key, edge_pairs)
                # Compute average distance
                avg_dist = self._distance.seed_distance({(i, i) for i in hit_bbs})
                meta["avg_distance"] = avg_dist

        # Update annealing progress for directed mode
        if self._distance and self.exec_count > 0:
            # Anneal over first 20% of max_len-scaled iterations
            anneal_target = max(5000, self.max_len * 10)
            self._anneal_progress = min(1.0, self.exec_count / anneal_target)

        success = is_crash or is_interesting or has_new_coverage

        # Surprisal-weighted reward: discoveries in sparse regions of the
        # coverage bitmap carry more information than discoveries near
        # already-saturated areas. Weight = 1 - density so rare discoveries
        # (low density) get higher credit; saturated regions (high density)
        # get lower credit.
        if success and self._edge_tracker and self._edge_tracker.map_size:
            density = self._edge_tracker.bitmap_density()
            surprisal_weight = max(0.05, 1.0 - density)
        else:
            surprisal_weight = 1.0 if success else 0.0

        if success:
            for op in set(self._last_ops_used):
                self.op_success[op] = self.op_success.get(op, 0) + 1
            if cmplog_found:
                self.op_success["cmplog"] = self.op_success.get("cmplog", 0) + 1

        if self.mc and self.mc_bandit:
            seen = set()
            for op in self._last_ops_used:
                if op not in seen:
                    self.mc.record(op, success, weight=surprisal_weight)
                    self.mc.record_brier(op, success, weight=surprisal_weight)
                    seen.add(op)
                    # Secretary-problem: track operator quality for optimal stopping
                    if self._secretary:
                        if op not in self._op_secretary:
                            self._op_secretary[op] = SecretaryStopping(
                                window_size=self._secretary_window,
                                exploration_frac=self._secretary_exploration,
                                min_observations=50,
                            )
                        a = self.mc.arm_alpha.get(op, 1.0)
                        b = self.mc.arm_beta.get(op, 1.0)
                        rate = a / (a + b)
                        self._op_secretary[op].observe(rate)

        if self._mopt and (not self._use_elo or self._meta_strategy == "mopt"):
            seen = set()
            for op, pid in zip(self._last_ops_used, self._last_mopt_particles, strict=False):
                if op not in seen:
                    self._mopt.record(op, success, particle_id=pid, weight=surprisal_weight)
                    seen.add(op)

        if self._use_replicator and self._replicator:
            seen = set()
            for op in self._last_ops_used:
                if op not in seen:
                    self._replicator.record(op, success, weight=surprisal_weight)
                    seen.add(op)

        # Elo: record matches between operators that were used
        if self._use_elo and self._elo and len(self._last_ops_used) >= 2:
            unique_ops = list(dict.fromkeys(self._last_ops_used))  # preserve order, dedup
            winners = set(self._last_ops_used) if success else set()
            if winners:
                self._elo.record_round(unique_ops, winners, crash=is_crash)
            # Apply periodic decay
            self._elo_decay_counter = getattr(self, "_elo_decay_counter", 0) + 1
            if self._elo_decay_counter >= self._elo_decay_interval:
                self._elo_decay_counter = 0
                self._elo.apply_decay()

        # Meta-elo: record operator strategy-level match
        if self._use_elo and self._elo and self._meta_strategy:
            score = surprisal_weight if success else 0.0
            all_strategies = []
            if self._use_replicator and self._replicator:
                all_strategies.append("replicator")
            if self.mc and self.mc_bandit:
                all_strategies.append("bandit")
            if self._use_mopt and self._mopt:
                all_strategies.append("mopt")
            for other in all_strategies:
                if other != self._meta_strategy:
                    self._elo.record_strategy_match(self._meta_strategy, other, score)

        # Meta-elo: record seed strategy-level match
        if self._use_elo and self._elo and self._seed_strategy:
            score = surprisal_weight if success else 0.0
            seed_strategies = ["ga", "weighted", "pareto", "format"]
            for other in seed_strategies:
                if other != self._seed_strategy:
                    self._elo.record_strategy_match(
                        f"seed_{self._seed_strategy}", f"seed_{other}", score
                    )

        if self._use_shapley and self._shapley:
            edge_bitmap = self._get_current_edge_bitmap()
            if edge_bitmap:
                new_edges = {i for i, v in enumerate(edge_bitmap) if v > 0}
                self._shapley.record(set(self._last_ops_used), len(new_edges), new_edges)

        if self._use_mi and self._mi:
            edge_bitmap = self._get_current_edge_bitmap()
            if edge_bitmap:
                self._mi.record(data, edge_bitmap, self.map_size)

        if self._use_transfer_entropy and self._te:
            edge_bitmap = self._get_current_edge_bitmap()
            if edge_bitmap:
                self._te_input_history.append(data[:64] if len(data) > 64 else data)
                self._te_edge_history.append(edge_bitmap)
                if len(self._te_input_history) > self._te_history_max:
                    self._te_input_history = self._te_input_history[-self._te_history_max :]
                    self._te_edge_history = self._te_edge_history[-self._te_history_max :]
                # Update byte→edge causal map periodically
                if len(self._te_input_history) % 100 == 0 and len(self._te_input_history) > 50:
                    self._update_te_causal_map()

        if is_crash:
            self.crash_count += 1
            crash_name = self.save_crash(mutated, returncode, stderr)
            # Generate GDB/strace trace report if enabled
            if self._tracer and crash_name:
                report = self._tracer.trace(mutated, returncode)
                self._tracer.save_report(report, str(self.crashes_dir), crash_name)
            # Verify crash at kernel level via dmesg (supplementary to exit code)
            self._verify_kernel_crash(getattr(self, "_last_child_pid", None))
            if self.mc and self.mc_cem:
                self.mc.add_elite(mutated, 3, temperature=self._temperature)
                self.mc.maybe_refit()
            # Schedule crash replay for reproducibility check
            if self.replay_n > 0 and crash_name:
                sig = self.crash_sigs.get(crash_name, crash_name)
                if sig not in self._crash_replays:
                    self._crash_replays[sig] = []
            return True

        if is_interesting or has_new_coverage:
            self.save_to_corpus(mutated, parent=data)
            # GA: add new-coverage individual to population
            if self.ga and has_new_coverage:
                edge_count = (
                    len(self._edge_tracker.seed_edges.get(self._edge_tracker._last_seed_key, set()))
                    if hasattr(self._edge_tracker, "_last_seed_key")
                    else 0
                )
                ind = self.ga.on_fuzz_result(mutated, True, edge_count, self._edge_tracker)
                if ind is not None:
                    self.ga.add_to_population(ind)
            # Analyze byte sensitivity for seeds that found new coverage (optional)
            if has_new_coverage and self.shm_cov and self._use_sensitivity:
                try:
                    edge_bitmap = bytes(self.shm_cov._map)[: self.shm_cov.size]
                    edges = {i for i, v in enumerate(edge_bitmap) if v}
                    if edges:

                        def _exec_fn(data):
                            rc, _ = self._run_target(data)
                            if self.shm_cov:
                                bm = bytes(self.shm_cov._map)[: self.shm_cov.size]
                                return {i for i, v in enumerate(bm) if v}
                            return set()

                        self._sensitivity.analyze_seed(mutated, edges, _exec_fn)
                except Exception:
                    pass
            # Coverage-guided trimming: try to minimize inputs that hit new edges
            if has_new_coverage and len(mutated) > 10:
                self._trim_new_coverage(mutated, data)
            if self.mc and self.mc_cem:
                self.mc.add_elite(mutated, 2, temperature=self._temperature)
                self.mc.maybe_refit()
            # Periodic minimization based on edge stats
            if (
                self.minimize_every_execs > 0
                and (self.exec_count - self._exec_baseline) % self.minimize_every_execs == 0
                and len(self.corpus) > 1
            ):
                self._auto_minimize_corpus()
                self._deprioritize_near_duplicates()
            return True

        # Periodic minimization (also for non-interesting iterations)
        if (
            self.minimize_every_execs > 0
            and (self.exec_count - self._exec_baseline) % self.minimize_every_execs == 0
            and len(self.corpus) > 1
        ):
            self._auto_minimize_corpus()

        # GA: trigger generation boundary for non-coverage iterations
        if self.ga:
            self.ga.on_fuzz_result(mutated, False, 0, self._edge_tracker)

        return False

    def _record_discovery_snapshot(self):
        _record_discovery_snapshot_fn(
            self.exec_count,
            self.shm_cov,
            self.ptrace_cov,
            self._discovery_history,
        )

    def _run_calibration(self, max_execs: int = 1000) -> None:
        """Run a short calibration pass to bootstrap coverage stats.

        Replays seed corpus + cheap mutations to populate EdgeTracker
        and _discovery_history before the main fuzz loop.
        """
        from fuzzer_tool.core.mutations import byte_insert

        print(f"[*] Calibration: running {max_execs} execs to bootstrap coverage stats...")
        if not self.corpus:
            print("[*] Calibration: no seeds found, skipping")
            return

        exec_count = 0
        report_interval = max(100, max_execs // 10)
        seeds = list(self.corpus)

        # Phase 1: replay each seed as-is
        for seed in seeds:
            if exec_count >= max_execs:
                break
            self._run_target(seed)
            self.exec_count += 1
            exec_count += 1
            edge_bitmap = self._get_current_edge_bitmap()
            if edge_bitmap:
                self._edge_tracker.record_edges(self._seed_key(seed), edge_bitmap)

        # Phase 2: cheap mutations until budget exhausted
        while exec_count < max_execs:
            seed = random.choice(seeds)
            if random.random() < 0.5:
                mutated = bytearray(seed)
                if mutated:
                    mutated[random.randint(0, len(mutated) - 1)] ^= 1 << random.randint(0, 7)
                mutated = bytes(mutated)
            else:
                mutated = byte_insert(seed)
            self._run_target(mutated)
            self.exec_count += 1
            exec_count += 1
            edge_bitmap = self._get_current_edge_bitmap()
            if edge_bitmap:
                self._edge_tracker.record_edges(self._seed_key(mutated), edge_bitmap)
            if exec_count % report_interval == 0:
                edges = len(self._edge_tracker._global_edge_hits)
                print(
                    f"\r[*] Calibration: {exec_count}/{max_execs} execs, "
                    f"{edges} edges discovered",
                    end="",
                    flush=True,
                )

        self._record_discovery_snapshot()
        edges = len(self._edge_tracker._global_edge_hits)
        gt = self._edge_tracker.good_turing_estimate()
        dr = self.discovery_rate()
        print(f"\r[*] Calibration done: {exec_count} execs, {edges} edges discovered, "
              f"GT confidence={gt['confidence']}, discovery_rate={dr:.1f}/1k execs   ")

        # Crash ETA estimate
        from fuzzer_tool.core.crash_eta import estimate_execs_to_first_crash
        eta = estimate_execs_to_first_crash(self._profile, gt, dr, exec_count, self._crash_mi)
        print(f"[*] ETA to first crash: ~{eta.edges_to_crash:,} risky edges, "
              f"~{eta.point_est:,} execs "
              f"(range: {eta.low:,} - {eta.high:,}, confidence: {eta.confidence})")

    def discovery_rate(self) -> float:
        return _discovery_rate(self._discovery_history)

    def _run_crash_replays(self, budget_ms: float = 200):
        _run_crash_replays_fn(
            self.crashes_dir,
            self.target,
            self.timeout,
            self._crash_replays,
            self.replay_n,
            self._seed_key,
            budget_ms,
        )

    def _print_run_summary(self):
        """Print session-level summary statistics at run exit."""
        elapsed = time.time() - self.start_time
        eps = self.exec_count / elapsed if elapsed > 0 else 0

        print(f"\n{'=' * 60}")
        print("  RUN SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Duration:          {elapsed:.0f}s")
        print(f"  Executions:        {self.exec_count:,}")
        print(f"  Avg eps:           {eps:.1f}")
        print(f"  Peak eps:          {self._peak_eps:.1f}")

        # Corpus growth
        added = self._total_corpus_attempts
        rejected = self._duplicate_reject_count
        print(f"  Corpus:            {len(self.corpus)} entries")
        print(f"  Seeds added:       {added}")
        print(f"  Duplicates rejected: {rejected}")
        if self._pruned_count > 0:
            print(f"  Seeds pruned:      {self._pruned_count}")

        # Stall recovery stats
        if self._stall_recovery_count > 0:
            print(f"  Recovery entries:  {self._stall_recovery_count}")
            print(f"  Recovery execs:    {self._stall_recovery_execs:,} "
                  f"({self._stall_recovery_execs / max(1, self.exec_count) * 100:.1f}%)")

        # Coverage — prefer edge_tracker (authoritative), fall back to SHM/ptrace
        # SHM bitmap is source of truth for unique edge count
        shm_edges = self.shm_cov.cumulative_edges if self.shm_cov else 0
        et_edges = self._edge_tracker.get_cumulative_edge_count()
        edges = shm_edges if shm_edges else et_edges
        if self.ptrace_cov:
            edges = self.ptrace_cov.cumulative_edges
        density = self._edge_tracker.bitmap_density() * 100
        collision_risk = self._edge_tracker.birthday_collision_risk() * 100
        if shm_edges and et_edges and shm_edges != et_edges:
            print(f"  Edges discovered:  {shm_edges} (SHM unique positions)")
            print(f"  ET positions:      {et_edges} (includes stale positions after resize)")
        else:
            print(f"  Edges discovered:  {edges}")
        print(f"  Map density:       {density:.2f}%")
        print(f"  Collision risk:    {collision_risk:.2f}% (birthday paradox)")
        rec = self._edge_tracker.recommended_map_size()
        if rec:
            print(f"  Recommended map:   {rec:,} bytes (current: {self.map_size:,})")

        # Good-Turing
        gt = self._edge_tracker.good_turing_estimate()
        if gt["n"] > 0:
            print(f"  Est. remaining:    {gt['estimated_undiscovered']} edges")
            print(f"  Saturation:        {gt['saturation']:.1%} ({gt['confidence']} confidence)")

        # Lineage depth distribution
        if self.seed_meta:
            depths = [m.get("lineage_depth", 0) for m in self.seed_meta.values()]
            if depths:
                print(f"  Max lineage depth: {max(depths)}")
                avg_depth = sum(depths) / len(depths)
                print(f"  Avg lineage depth: {avg_depth:.1f}")

        # Per-seed edge discovery stats
        if self.seed_meta:
            edges_per_seed = [m.get("coverage_edges", 0) for m in self.seed_meta.values()]
            productive = sum(1 for e in edges_per_seed if e > 0)
            stale = sum(
                1
                for m in self.seed_meta.values()
                if m.get("fuzz_count", 0) >= 50 and m.get("coverage_edges", 0) == 0
            )
            total_seeds = len(self.seed_meta)
            print(f"  Productive seeds:  {productive}/{total_seeds} discovered edges")
            print(f"  Stale seeds:       {stale}/{total_seeds} (50+ fuzzes, 0 edges)")

        # Edge rarity stats
        rarity = self._edge_tracker.edge_rarity_stats()
        if rarity["total"] > 0:
            print(
                f"  Edge rarity:       {rarity['singleton']} singleton / "
                f"{rarity['cold']} cold / {rarity['warm']} warm / {rarity['hot']} hot"
            )
            print(f"  Avg seeds/edge:    {rarity['avg_seeds_per_edge']:.1f}")

            # Seed uniqueness: how many singleton edges each seed covers
            uniqueness = self._edge_tracker.seed_uniqueness()
            if uniqueness:
                irreplaceable = sum(1 for v in uniqueness.values() if v > 0)
                print(f"  Irreplaceable:     {irreplaceable} seeds cover singleton edges")

            # Top co-occurring edges
            cooccur = self._edge_tracker.edge_cooccurrence(top_k=3)
            if cooccur:
                pairs_str = ", ".join(f"e{a}↔e{b}({j:.0%})" for a, b, j in cooccur)
                print(f"  Edge co-occurrence:{pairs_str}")

        # Input size distribution
        if self._corpus_size_history:
            s = sorted(self._corpus_size_history)
            print(
                f"  Input sizes:       min={s[0]} p50={s[len(s) // 2]} p90={s[-len(s) // 10]} max={s[-1]}"
            )

        # Crash summary
        print(f"  Crashes:           {self.crash_count} ({len(self.crash_sigs)} unique)")
        if self.crash_sigs:
            for sig, count in sorted(self.crash_sigs.items(), key=lambda x: -x[1])[:5]:
                print(f"    {sig[:48]} ({count}x)")

        # Crash clustering (order-aware frame-sequence Levenshtein)
        if len(self.crash_sigs) >= 2 and self.crash_frames:
            from fuzzer_tool.core.crash_metadata import cluster_crashes

            sigs = list(self.crash_sigs.keys())
            frames = [self.crash_frames.get(s, []) for s in sigs]
            clusters = cluster_crashes(sigs, frame_lists=frames, threshold=0.6)
            multi = [c for c in clusters if len(c) > 1]
            if multi:
                print(f"  Crash clusters:    {len(multi)} group(s) from {len(sigs)} signatures")
                for cl in multi[:3]:
                    sigs_in = [sigs[i][:36] for i in cl]
                    print(f"    [{len(cl)} crashes] {' ~ '.join(sigs_in)}")

        # Operator ROI
        if self.op_counts:
            print("\n  Operator ROI:")
            print(f"    {'Operator':<22s} {'Count':>7s} {'Success':>8s} {'Rate':>7s}")
            print(f"    {'-' * 22} {'-' * 7} {'-' * 8} {'-' * 7}")
            for op, count in sorted(
                self.op_counts.items(), key=lambda x: -self.op_success.get(x[0], 0)
            )[:8]:
                succ = self.op_success.get(op, 0)
                rate = succ / count * 100 if count else 0
                print(f"    {op:<22s} {count:>7d} {succ:>8d} {rate:>6.1f}%")

        # Elo ratings
        if self._use_elo and self._elo and self._elo.ratings:
            ranking = self._elo.get_ranking()
            print(f"\n  Elo Ratings (top 10):")
            for i, (op, rating) in enumerate(ranking[:10], 1):
                matches = self._elo._match_count.get(op, 0)
                print(f"    {i:>2d}. {op:<22s} {rating:>7.0f} ({matches} matches)")

        # Duplicate rejection trend
        if self._total_corpus_attempts > 0:
            dup_rate = rejected / self._total_corpus_attempts * 100
            print(
                f"\n  Dup rejection rate: {dup_rate:.1f}% ({rejected}/{self._total_corpus_attempts})"
            )

        # Execution time
        tracker = self._exec_time_tracker
        if tracker.count > 0:
            print(f"  Exec time p50:     {tracker.p50 * 1000:.1f}ms")
            print(f"  Exec time p99:     {tracker.p99 * 1000:.1f}ms")
            print(f"  Suggested timeout: {tracker.suggested_timeout():.2f}s")

        print(f"{'=' * 60}")

    def _dump_stats(self):
        if not self.stats_file:
            return
        elapsed = time.time() - self.start_time
        eps = self.exec_count / elapsed if elapsed > 0 else 0
        stats = {
            "timestamp": time.time(),
            "exec_count": self.exec_count,
            "crash_count": self.crash_count,
            "timeout_count": self.timeout_count,
            "corpus_size": len(self.corpus),
            "unique_crash_sigs": len(self.crash_sigs),
            "eps": round(eps, 1),
            "elapsed_sec": round(elapsed, 1),
            "peak_rss_kb": self._peak_rss,
            "op_counts": dict(self.op_counts),
            "op_success": dict(self.op_success),
        }
        if self.mc and self.mc_bandit:
            stats["bandit_stats"] = {
                k: {"successes": v[0], "failures": v[1]} for k, v in self.mc.bandit_stats().items()
            }
        if self.mc and self.mc_cem:
            stats["cem_elite_size"] = len(self.mc.elite_set)
            stats["cem_fitted"] = self.mc.cem_fitted
        if self._use_replicator and self._replicator:
            stats["replicator"] = {
                "distribution": self._replicator.population_distribution(),
                "converged": self._replicator.is_converged(),
                "dominant": self._replicator.dominant_operator(),
            }
        if self._use_shapley and self._shapley:
            sv = self._shapley.shapley_values()
            stats["shapley"] = {k: round(v, 4) for k, v in sv.items()}
        if self._use_mi and self._mi:
            stats["mi"] = {
                "observations": self._mi.total_observations,
                "top_positions": [
                    {"pos": p, "mi_bits": round(v, 4)}
                    for p, v in self._mi.top_positions(k=5, input_length=self.max_len)
                ],
            }
        if self._use_renyi_weight:
            edge_hits = (
                dict(self._edge_tracker._global_edge_hits)
                if hasattr(self._edge_tracker, "_global_edge_hits")
                else {}
            )
            if edge_hits:
                from fuzzer_tool.core.renyi import RenyiEntropy

                renyi = RenyiEntropy()
                stats["renyi"] = {
                    "uniformity": round(renyi.coverage_uniformity(list(edge_hits.values())), 4),
                    "min_entropy": round(renyi.min_entropy(list(edge_hits.values())), 4),
                    "spectrum": {
                        k: round(v, 4)
                        for k, v in renyi.entropy_spectrum(list(edge_hits.values())).items()
                    },
                }
        if self._use_transfer_entropy:
            stats["transfer_entropy"] = {
                "history_len": len(self._te_input_history),
                "causal_positions": len(self._te_byte_edges),
            }
        try:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            self.stats_file.write_text(json.dumps(stats, indent=2))
        except OSError:
            log.debug("Failed to write stats to %s", self.stats_file, exc_info=True)

    def _dump_coverage_report(self):
        if not self.coverage_report:
            return
        edge_map = None
        if self.shm_cov:
            edge_map = self.shm_cov._seen
        elif self.ptrace_cov:
            edge_map = self.ptrace_cov.edge_map
        if edge_map is None:
            print("[!] No coverage data available for report")
            return

        hit_edges = []
        cumulative = 0
        for i, val in enumerate(edge_map):
            if val:
                hit_edges.append(i)
                cumulative += 1

        report = {
            "map_size": len(edge_map),
            "cumulative_edges": cumulative,
            "hit_edges": hit_edges,
            "coverage_pct": round(cumulative / len(edge_map) * 100, 4),
            "exec_count": self.exec_count,
            "corpus_size": len(self.corpus),
        }
        self.coverage_report.parent.mkdir(parents=True, exist_ok=True)
        self.coverage_report.write_text(json.dumps(report, indent=2))
        print(
            f"\n[*] Coverage report: {self.coverage_report} "
            f"({cumulative}/{len(edge_map)} edges, {report['coverage_pct']}%)"
        )

    def _append_coverage_log(self):
        if not self.coverage_log:
            return
        cumulative = 0
        if self.shm_cov:
            cumulative = self.shm_cov.cumulative_edges
        elif self.ptrace_cov:
            cumulative = self.ptrace_cov.cumulative_edges
        elif hasattr(self, "_edge_tracker"):
            cumulative = self._edge_tracker.get_cumulative_edge_count()
        elapsed = time.time() - self.start_time
        line = (
            f"{elapsed:.1f},{self.exec_count},{cumulative},{len(self.corpus)},{self.crash_count}\n"
        )
        with open(self.coverage_log, "a") as f:
            f.write(line)

    def _update_te_causal_map(self):
        update_te_causal_map(
            self._te,
            self._te_input_history,
            self._te_edge_history,
            self.map_size,
            self._te_byte_edges,
        )

    def _get_te_weighted_position(self, input_length: int) -> int | None:
        return get_te_weighted_position(self._te_byte_edges, input_length)

    def _get_current_edge_bitmap(self) -> bytes | None:
        """Get the current coverage edge bitmap."""
        if self.shm_cov:
            return bytes(self.shm_cov._map)
        if self.ptrace_cov:
            return bytes(self.ptrace_cov.edge_map)
        return None

    def _format_elapsed(self) -> str:
        return _format_elapsed_fn(self.start_time)

    def print_stats(self):
        elapsed = time.time() - self.start_time
        eps = self.exec_count / elapsed if elapsed > 0 else 0
        dict_str = f" | dict: {len(self.dictionary)}" if self.dictionary else ""
        markov_str = " | markov: trained" if self.markov_trained else ""
        if self.markov_generate:
            markov_str += "+gen"
        cov_str = ""
        if self.shm_cov:
            shm_edges = self.shm_cov.cumulative_edges
            gt = self._edge_tracker.good_turing_estimate()
            max_edges = gt["n"] + gt["estimated_undiscovered"]
            sat = gt["saturation"] * 100 if max_edges > 0 else 0
            cov_str = f" | shm: {shm_edges} max: {max_edges} sat: {sat:.0f}%"
        elif self.ptrace_cov:
            cov_str = (
                f" | edges: {self.ptrace_cov.cumulative_edges}"
                f" hits: {self.ptrace_cov.total_bp_hits}"
            )
            if self.ptrace_cov.deep_coverage:
                cov_str += f" bps:{len(self.ptrace_cov.original_bytes)}"
        mc_str = ""
        if self.mc:
            parts = []
            if self.mc_bandit:
                parts.append("bandit")
            if self.mc_cem:
                parts.append(f"cem:{len(self.mc.elite_set)}")
            if parts:
                mc_str = " | mc: " + "+".join(parts)
        sig_str = f"({len(self.crash_sigs)}sigs)" if self.crash_sigs else ""
        timeout_pct = self.timeout_count / self.exec_count * 100 if self.exec_count else 0
        timeout_str = f" | timeouts: {self.timeout_count} ({timeout_pct:.1f}%)"
        rss_kb = self._peak_rss
        rss_str = f" | rss: {rss_kb // 1024}MB" if rss_kb >= 1024 else f" | rss: {rss_kb}KB"
        ops_str = ""
        if self._last_ops_used:
            # Show last 3 operators selected (most recent first, deduped)
            recent = list(dict.fromkeys(reversed(self._last_ops_used)))[:3]
            ops_str = " | ops: " + " ".join(recent)
        div_str = ""
        if len(self._edge_tracker.seed_hit_counts) >= 2:
            diversity = self._edge_tracker.compute_corpus_diversity()
            div_str = f" | div: {diversity:.0f}"
        # Jaccard index: average pairwise overlap between seeds
        jac_str = ""
        if len(self._edge_tracker.seed_hit_counts) >= 2:
            avg_jac = self._edge_tracker.compute_average_jaccard()
            jac_str = f" | jac: {avg_jac:.2f}"
        # Discovery rate
        dr = self.discovery_rate()
        dr_str = f" | rate: {dr:.1f} ed/kexec" if self.exec_count > 100 else ""
        # Critical slowing down: observe discovery rate for phase transition signals
        if self.exec_count > 100:
            self._csd.observe(dr)
            detected, csd_reason = self._csd.is_approaching_transition()
            if detected:
                dr_str += f" [CSD: {csd_reason}]"
        # Bitmap density
        density = self._edge_tracker.bitmap_density() * 100
        collision_risk = self._edge_tracker.birthday_collision_risk() * 100
        density_str = f" | map: {density:.1f}%"
        if collision_risk > 10:
            density_str += f" (collision: {collision_risk:.0f}%)"
            # Resize at collision risk threshold, starting small and doubling
            if collision_risk > self._max_collision_risk and self.shm_cov:
                current = self.shm_cov.size
                new_size = min(1048576, current * 2)
                if new_size > current:
                    print(
                        f"\n[*] Collision risk {collision_risk:.0f}% — resizing bitmap "
                        f"{current:,} → {new_size:,} bytes"
                    )
                    self.shm_cov.resize(new_size)
                    self.map_size = new_size
                    self._edge_tracker.map_size = new_size
                    # Clear all position-based tracking — positions change after resize
                    self._edge_tracker.reset_after_resize()
                    density_str = f" | map: {self._edge_tracker.bitmap_density() * 100:.1f}% (collision: {collision_risk:.0f}%)"
        # Crash reproducibility
        repro_str = ""
        if self._crash_replays:
            done = [v for v in self._crash_replays.values() if len(v) >= self.replay_n]
            if done:
                avg_repro = (
                    sum(sum(1 for r in replays if r >= 0) / len(replays) for replays in done)
                    / len(done)
                    * 100
                )
                repro_str = f" | repro: {avg_repro:.0f}%"
        # Bandit calibration (Brier score)
        brier_str = ""
        if self.mc and self.mc_bandit and self.mc.brier_score() > 0:
            brier_str = f" | brier: {self.mc.brier_score():.3f}"
        # Exec time CRPS
        crps_str = ""
        if self._exec_time_tracker.count > 20:
            crps_str = f" | crps: {self._exec_time_tracker.mean_crps():.4f}"
        line = (
            f"[*] execs: {self.exec_count} | corpus: {len(self.corpus)} | "
            f"crashes: {self.crash_count}{sig_str}{timeout_str} | eps: {eps:.0f} | "
            f"time: {elapsed:.0f}s{rss_str}{ops_str}{dict_str}{markov_str}{cov_str}{mc_str}{div_str}{jac_str}{dr_str}{density_str}{repro_str}{brier_str}{crps_str}"
        )
        print(line, flush=True)

    def run(self, iterations=0):
        print(f"[*] Target: {self.target}")
        # Static branch density: conditional branches per KB of .text
        from fuzzer_tool.core.elf import branch_density

        bd = branch_density(self.target)
        if bd is not None:
            print(f"[*] Branch density: {bd:.1f} cond branches/KB")
        print(f"[*] Edge bitmap: {self.map_size:,} bytes (auto-sized)")
        print(f"[*] Corpus: {self.corpus_dir} ({len(self.corpus)} seeds)")
        print(f"[*] Crashes: {self.crashes_dir}")
        print(f"[*] Max input length: {self.max_len}")
        print(f"[*] Timeout: {self.timeout}s")
        print(f"[*] Seed: {self.seed}")
        # Target profile summary
        if self._profile.functions:
            print(
                f"[*] Profile: {len(self._profile.functions)} functions, "
                f"{len(self._profile.hot_functions)} hot, "
                f"format={self._profile.format_signature or 'unknown'}"
            )
        if self.grammar:
            print(f"[*] Grammar: {len(self.grammar.rules)} rules")
        if self._calibrate > 0:
            print(f"[*] Calibration: {self._calibrate} execs before main loop")
        if self.persistent:
            print("[*] Persistent mode: enabled")
        if self._inprocess_runner:
            print("[*] In-process mode: enabled")
        if self.dictionary:
            print(f"[*] Dictionary: {len(self.dictionary)} tokens")
        if self.markov_trained:
            if hasattr(self.markov, "chains"):
                orders_str = ",".join(str(o) for o in self.markov.orders)
                total_ctx = sum(c._contexts_seen for c in self.markov.chains.values())
                print(f"[*] Markov ensemble: orders=[{orders_str}], total_contexts={total_ctx}")
            else:
                print(
                    f"[*] Markov chain: order={self.markov.order}, "
                    f"transitions={len(self.markov.transitions)}"
                )
        if self.markov_generate:
            print("[*] Markov generation: enabled (15% of seeds)")
        if self.mc:
            if self.mc_bandit:
                print(f"[*] MC bandit: Thompson sampling over {len(self.mc.arm_alpha)} arms")
            if self.mc_cem:
                print(
                    f"[*] MC CEM: elite_frac={self.mc.elite_frac}, "
                    f"refit_interval={self.mc.refit_interval}"
                )
        if self.stats_file:
            print(f"[*] Stats: {self.stats_file} every {self.stats_interval} iterations")
        if self.minimize_every_execs > 0:
            print(f"[*] Minimize: every {self.minimize_every_execs} execs")
        import datetime

        epoch_start = time.time()
        boot_start = time.monotonic()
        try:
            with open("/proc/uptime") as f:
                boot_start = float(f.read().split()[0])
        except OSError:
            pass
        print(
            f"[*] Epoch start: {epoch_start:.3f} ({datetime.datetime.fromtimestamp(epoch_start).isoformat()})"
        )
        print(f"[*] Boot ticks start: {boot_start:.3f}")
        print("[*] Starting fuzzing...\n")

        i = 0
        try:
            # Run each seed as-is before mutating — catches crashes in the
            # initial corpus and gathers baseline coverage.
            for seed in list(self.corpus):
                returncode, stderr = self._run_target(seed)
                # Validate AFL shim on first execution
                if not getattr(self, "_shim_checked", False):
                    self._shim_checked = True
                    if "[shim]" in stderr:
                        log.info("AFL shim: %s", stderr.strip())
                        if "area=(nil)" in stderr and self.shm_cov:
                            log.warning(
                                "AFL shim area is NULL — SHM not attached. "
                                "Coverage data will be empty."
                            )
                self.exec_count += 1
                if self._is_crash(returncode, stderr):
                    self.crash_count += 1
                    self.save_crash(seed, returncode, stderr)
                    # Kernel crash verification (same as fuzz_one path)
                    self._verify_kernel_crash(getattr(self, "_last_child_pid", None))
            # Baseline exec_count after initial seed replay — used for
            # periodic minimization modulus so it fires at clean intervals
            # regardless of initial corpus size.
            _exec_baseline = self.exec_count
            self._exec_baseline = _exec_baseline

            # Calibration pass: bootstrap coverage stats before main loop
            if self._calibrate > 0:
                self._run_calibration(self._calibrate)

            # Initialize GA lifecycle if enabled
            if self._ga_enabled:
                from fuzzer_tool.core.ga import GALifecycle

                self.ga = GALifecycle(
                    pop_size=self._ga_pop_size,
                    elite_fraction=self._ga_elite_frac,
                    crossover_rate=self._ga_crossover_rate,
                    mutation_rate=self._ga_mutation_rate,
                    tournament_size=self._ga_tournament_size,
                    generation_size=self._ga_gen_size,
                    speciation_threshold=self._ga_speciation_threshold,
                )
                self.ga.initialize(self.corpus, self._edge_tracker)
                ga_path = self.corpus_dir / "ga.json"
                if self.resume and ga_path.exists():
                    self.ga.load(ga_path)
                    print(f"[*] GA: loaded state from {ga_path} (gen={self.ga.generation})")
                print(
                    f"[*] GA: pop_size={self.ga.pop_size}, "
                    f"gen_size={self.ga.generation_size}, "
                    f"elite={self.ga.elite_fraction:.0%}, "
                    f"crossover={self.ga.crossover_rate:.0%}, "
                    f"mutation={self.ga.mutation_rate:.0%}"
                )

            while not _shutdown:
                if iterations and i >= iterations:
                    break
                if self.continue_until_crash and self.crash_count > 0:
                    break
                seed = self._pick_seed()
                self.fuzz_one(seed)
                i += 1
                if i % 100 == 0:
                    self.print_stats()
                    self._append_coverage_log()
                    self._record_discovery_snapshot()
                    # Stall detection: no new edges in threshold execs
                    execs_since_edge = self.exec_count - self._last_new_edge_exec
                    if (
                        not self._stall_recovery_active
                        and execs_since_edge >= self._stall_threshold
                        and self.exec_count > 0
                    ):
                        self._stall_recovery_count += 1
                        print(f"\n[*] STALL #{self._stall_recovery_count}: no new edges in "
                              f"{execs_since_edge} execs, switching to random mode")
                        self._stall_recovery_active = True
                    # Periodic GC to return freed memory to OS
                    if i % 500 == 0:
                        import gc

                        gc.collect()
                if i % 500 == 0 and self.replay_n > 0:
                    self._run_crash_replays()
                if self.stats_file and i % self.stats_interval == 0:
                    self._dump_stats()
                    self._save_state()
        except (KeyboardInterrupt, SystemExit):
            pass
        except OSError as e:
            log.warning("Fuzzing interrupted by OS error: %s", e)

        self._dump_stats()
        self._dump_coverage_report()
        if self.markov.is_trained():
            self.markov.save(str(self._markov_path))
        if self._use_mi and self._mi:
            self._mi.save(str(self._mi_path))
        try:
            self._crash_mi_path.write_text(json.dumps(self._crash_mi.save(), separators=(",", ":")))
        except OSError:
            pass
        self._save_state()
        if self.ga:
            ga_path = self.corpus_dir / "ga.json"
            self.ga.save(ga_path)
            print(f"[*] GA: saved state to {ga_path} (gen={self.ga.generation})")
        if self._ablation_file:
            self._ablation_file.flush()
            self._ablation_file.close()
            self._ablation_file = None
            print(f"[*] Schedule ablation log: {self._ablation_path}")
        self._dmesg.stop_stream()
        self.print_stats()
        print(
            f"\n\n[*] Fuzzing stopped. {self.crash_count} crashes found "
            f"({len(self.crash_sigs)} unique signatures)."
        )
        if self.crash_sigs:
            print("[*] Crash signatures:")
            for sig, count in sorted(self.crash_sigs.items(), key=lambda x: -x[1]):
                print(f"    {sig} ({count}x)")
            print(f"\n[*] Crash files in: {self.crashes_dir}")
        if self._kernel_crashes:
            print(f"\n[*] Kernel-verified crashes: {len(self._kernel_crashes)}")
            by_type: dict[str, int] = {}
            for kc in self._kernel_crashes:
                by_type[kc.crash_type] = by_type.get(kc.crash_type, 0) + 1
            for ctype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                print(f"    {ctype}: {count}")
        elif self._dmesg.is_available():
            if self.crash_count > 0:
                print(
                    "\n[*] dmesg: crashes detected via exit code but not in dmesg "
                    "(likely rate-limited by kernel)"
                )
            else:
                print("\n[*] dmesg: no kernel crashes detected")
        # Show convergence stats for every active scheduler
        if self.mc and self.mc_bandit:
            print("\n[*] Bandit convergence (Thompson sampling):")
            for name, (a, b) in sorted(
                self.mc.bandit_stats_raw().items(),
                key=lambda x: -(x[1][0] / max(x[1][0] + x[1][1], 1)),
            ):
                total = a + b
                pct = a / total * 100 if total else 0
                print(f"    {name:20s}: {a:.1f}/{b:.1f} ({pct:.0f}% success)")
        if self._mopt:
            print("\n[*] MOpt convergence (PSO):")
            for p in self._mopt.particle_stats()[:5]:
                print(
                    f"    {p['name']:<20s}: fitness={p['fitness']:.4f} "
                    f"top={p['top_op']}({p['top_prob']:.1%})"
                )
        if self._replicator:
            print("\n[*] Replicator convergence:")
            for s in self._replicator.operator_stats():
                if s["window_execs"] > 0:
                    rate = s["window_successes"] / s["window_execs"] * 100
                    print(
                        f"    {s['name']:<20s}: pop={s['population']:.4f} "
                        f"({s['window_successes']}/{s['window_execs']} = {rate:.0f}%)"
                    )
        # Seed strategy convergence
        if self._use_elo and self._elo:
            seed_strategies = ["ga", "weighted", "pareto", "format"]
            has_seed_data = any(
                self._elo._strategy_match_count.get(f"seed_{s}", 0) > 0 for s in seed_strategies
            )
            if has_seed_data:
                print("\n[*] Seed strategy convergence:")
                for s in seed_strategies:
                    key = f"seed_{s}"
                    count = self._elo._strategy_match_count.get(key, 0)
                    if count > 0:
                        rating = self._elo._strategy_ratings.get(key, self._elo.default_rating)
                        delta = rating - self._elo.default_rating
                        sign = "+" if delta >= 0 else ""
                        print(f"    {s:<20s}: {rating:>7.0f} ({sign}{delta:.0f}, {count} matches)")
        self._print_run_summary()
        epoch_end = time.time()
        boot_end = time.monotonic()
        try:
            with open("/proc/uptime") as f:
                boot_end = float(f.read().split()[0])
        except OSError:
            pass
        print(
            f"\n[*] Epoch end: {epoch_end:.3f} ({datetime.datetime.fromtimestamp(epoch_end).isoformat()})"
        )
        print(f"[*] Boot ticks end: {boot_end:.3f}")
        print(f"[*] dmesg window: {boot_start:.3f} - {boot_end:.3f}")
