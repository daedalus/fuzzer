"""Fuzzer orchestration: coordinates mutations, execution, and coverage."""

import atexit
import contextlib
import json
import logging
import os
import random
import resource
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

from fuzzer_tool.adapters.process import (
    _child_pids,
)
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.markov import MarkovChain, MarkovEnsemble
from fuzzer_tool.core.mi import MutualInformationTracker
from fuzzer_tool.core.montecarlo import (
    MonteCarloScheduler,
    MOptScheduler,
    ReplicatorScheduler,
    ShapleyAttribution,
)
from fuzzer_tool.core.mutations import (
    DICT_MUTATIONS,
    FORMAT_MUTATIONS,
    MUTATIONS,
)
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.core.secretary import DEFAULT_EXPLORATION_FRAC, SecretaryStopping
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.operators import OperatorEngine
from fuzzer_tool.services.ptrace_coverage import (
    PtraceCoverage,
)
from fuzzer_tool.services.runner import TargetRunner
from fuzzer_tool.services.seed_picker import SeedPicker
from fuzzer_tool.services.stats import StatsReporter

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

        # ── Extracted modules ──────────────────────────────────────────
        self._operators = OperatorEngine(self)
        self._seed_picker = SeedPicker(self)
        self._runner = TargetRunner(self)
        self._stats = StatsReporter(self)
        self._corpus_manager = CorpusManager(self)

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

        # FrameShift: universal length-field auto-adjustment
        from fuzzer_tool.core.frameshift import FrameShift

        self._frameshift = FrameShift(max_relations=64)
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

        def _register_arms(scheduler, priors=None):
            """Register all mutation arms on a scheduler (mc, mopt, replicator, elo).

            Args:
                scheduler: Scheduler exposing init_arm(name).
                priors: Optional dict of operator name -> (prior_alpha,
                    prior_beta) overrides. Only meaningful for the
                    Beta-Bernoulli Thompson-sampling scheduler; ignored for
                    schedulers whose init_arm() doesn't accept a prior.
            """
            priors = priors or {}

            def _init(op):
                prior = priors.get(op) if getattr(scheduler, "supports_priors", False) else None
                if prior is not None and len(prior) == 2:
                    scheduler.init_arm(op, *prior)
                else:
                    scheduler.init_arm(op)

            for op in MUTATIONS:
                _init(op)
            for op in DICT_MUTATIONS:
                _init(op)
            _init("markov_bytes")
            _init("cem_bytes")
            if self.grammar:
                _init("grammar_mutate")
                _init("grammar_tree_mutate")
            for op in FORMAT_MUTATIONS:
                _init(op)

        from fuzzer_tool.core.target_profiler import format_operator_priors

        _format_priors = format_operator_priors(self._profile)

        if self.mc and self.mc_bandit:
            _register_arms(self.mc, _format_priors)
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
        return self._corpus_manager.load_corpus()

    def _init_seed_metadata(self):
        return self._corpus_manager.init_seed_metadata()

    def _seed_key(self, data: bytes):
        return self._corpus_manager.seed_key(data)

    def _save_state(self):
        return self._corpus_manager.save_state()

    def _load_state(self):
        return self._corpus_manager.load_state()

    def _run_target(self, data: bytes):
        return self._runner.run_target(data)

    def _ptrace_handle_breakpoint(self, pid: int, libc, cov: PtraceCoverage, regs_buf):
        return self._runner._ptrace_handle_breakpoint(pid, libc, cov, regs_buf)

    def _run_target_ptrace(self, data: bytes):
        return self._runner._run_target_ptrace(data)

    def _verify_kernel_crash(self, child_pid: int | None):
        return self._runner.verify_kernel_crash(child_pid)

    def _check_python_crashes(self):
        return self._runner._check_python_crashes()

    def _is_interesting(self, returncode: int, stderr: str):
        return self._runner.is_interesting(returncode, stderr)

    def _is_crash(self, returncode: int, stderr: str):
        return self._runner.is_crash(returncode, stderr)

    def mutate(self, data: bytes):
        return self._operators.mutate(data)

    def _build_ops(self, data: bytes):
        return self._operators.build_ops(data)

    def _select_op(self, ops: list[str]):
        return self._operators.select_op(ops)

    def _select_position(self, buf: bytearray, data: bytes):
        return self._operators.select_position(buf, data)

    # ── Operator handlers ──────────────────────────────────────────────
    # Each handler: (buf, byte_idx, data) -> None (in-place) or bytes (replace buf)

    def _op_bit_flip(self, buf, byte_idx, _data):
        return self._operators._op_bit_flip(buf, byte_idx, _data)

    def _op_bit_offset_flip(self, buf, _byte_idx, _data):
        return self._operators._op_bit_offset_flip(buf, _byte_idx, _data)

    def _op_bit_offset_span(self, buf, _byte_idx, _data):
        return self._operators._op_bit_offset_span(buf, _byte_idx, _data)

    def _op_byte_flip(self, buf, byte_idx, _data):
        return self._operators._op_byte_flip(buf, byte_idx, _data)

    def _op_interesting_8(self, buf, byte_idx, _data):
        return self._operators._op_interesting_8(buf, byte_idx, _data)

    def _op_interesting_16(self, buf, _byte_idx, _data):
        return self._operators._op_interesting_16(buf, _byte_idx, _data)

    def _op_interesting_32(self, buf, _byte_idx, _data):
        return self._operators._op_interesting_32(buf, _byte_idx, _data)

    def _op_arithmetic(self, buf, _byte_idx, _data):
        return self._operators._op_arithmetic(buf, _byte_idx, _data)

    def _op_random_bytes(self, buf, _byte_idx, _data):
        return self._operators._op_random_bytes(buf, _byte_idx, _data)

    def _op_block_insert(self, buf, _byte_idx, _data):
        return self._operators._op_block_insert(buf, _byte_idx, _data)

    def _op_block_delete(self, buf, _byte_idx, _data):
        return self._operators._op_block_delete(buf, _byte_idx, _data)

    def _op_block_duplicate(self, buf, _byte_idx, _data):
        return self._operators._op_block_duplicate(buf, _byte_idx, _data)

    def _op_dict_insert(self, buf, _byte_idx, _data):
        return self._operators._op_dict_insert(buf, _byte_idx, _data)

    def _op_dict_replace(self, buf, _byte_idx, _data):
        return self._operators._op_dict_replace(buf, _byte_idx, _data)

    def _op_dict_overwrite(self, buf, _byte_idx, _data):
        return self._operators._op_dict_overwrite(buf, _byte_idx, _data)

    def _op_dict_prepend(self, buf, _byte_idx, _data):
        return self._operators._op_dict_prepend(buf, _byte_idx, _data)

    def _op_dict_append(self, buf, _byte_idx, _data):
        return self._operators._op_dict_append(buf, _byte_idx, _data)

    def _op_checksum_repair(self, buf, _byte_idx, _data):
        return self._operators._op_checksum_repair(buf, _byte_idx, _data)

    def _op_token_dup(self, buf, _byte_idx, _data):
        return self._operators._op_token_dup(buf, _byte_idx, _data)

    def _op_markov_bytes(self, buf, _byte_idx, _data):
        return self._operators._op_markov_bytes(buf, _byte_idx, _data)

    def _op_cem_bytes(self, buf, _byte_idx, _data):
        return self._operators._op_cem_bytes(buf, _byte_idx, _data)

    def _op_splice(self, buf, _byte_idx, data):
        return self._operators._op_splice(buf, _byte_idx, data)

    def _op_crossover(self, buf, _byte_idx, data):
        return self._operators._op_crossover(buf, _byte_idx, data)

    def _op_type_replace(self, buf, _byte_idx, _data):
        return self._operators._op_type_replace(buf, _byte_idx, _data)

    def _op_ascii_num(self, buf, _byte_idx, _data):
        return self._operators._op_ascii_num(buf, _byte_idx, _data)

    def _op_byte_shuffle(self, buf, _byte_idx, _data):
        return self._operators._op_byte_shuffle(buf, _byte_idx, _data)

    def _op_byte_delete(self, buf, _byte_idx, _data):
        return self._operators._op_byte_delete(buf, _byte_idx, _data)

    def _op_byte_insert(self, buf, _byte_idx, _data):
        return self._operators._op_byte_insert(buf, _byte_idx, _data)

    def _op_insert_ascii_num(self, buf, _byte_idx, _data):
        return self._operators._op_insert_ascii_num(buf, _byte_idx, _data)

    def _op_transpose_16(self, buf, _byte_idx, _data):
        return self._operators._op_transpose_16(buf, _byte_idx, _data)

    def _op_transpose_32(self, buf, _byte_idx, _data):
        return self._operators._op_transpose_32(buf, _byte_idx, _data)

    def _op_transpose_64(self, buf, _byte_idx, _data):
        return self._operators._op_transpose_64(buf, _byte_idx, _data)

    def _op_bit_transpose_8(self, buf, _byte_idx, _data):
        return self._operators._op_bit_transpose_8(buf, _byte_idx, _data)

    def _op_bit_transpose_16(self, buf, _byte_idx, _data):
        return self._operators._op_bit_transpose_16(buf, _byte_idx, _data)

    def _op_bit_transpose_32(self, buf, _byte_idx, _data):
        return self._operators._op_bit_transpose_32(buf, _byte_idx, _data)

    def _op_bit_transpose_64(self, buf, _byte_idx, _data):
        return self._operators._op_bit_transpose_64(buf, _byte_idx, _data)

    def _op_length_grow(self, buf, _byte_idx, _data):
        return self._operators._op_length_grow(buf, _byte_idx, _data)

    def _op_length_shrink(self, buf, _byte_idx, _data):
        return self._operators._op_length_shrink(buf, _byte_idx, _data)

    def _op_repeat_clone(self, buf, _byte_idx, _data):
        return self._operators._op_repeat_clone(buf, _byte_idx, _data)

    def _op_truncate(self, buf, _byte_idx, _data):
        return self._operators._op_truncate(buf, _byte_idx, _data)

    def _op_swap_regions(self, buf, _byte_idx, _data):
        return self._operators._op_swap_regions(buf, _byte_idx, _data)

    def _op_swap_bytes(self, buf, _byte_idx, _data):
        return self._operators._op_swap_bytes(buf, _byte_idx, _data)

    def _op_endianness_swap(self, buf, _byte_idx, _data):
        return self._operators._op_endianness_swap(buf, _byte_idx, _data)

    def _op_grammar_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_grammar_mutate(buf, _byte_idx, _data)

    def _op_grammar_tree_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_grammar_tree_mutate(buf, _byte_idx, _data)

    def _op_png_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_png_chunk_mutate(buf, _byte_idx, _data)

    def _op_jpeg_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_jpeg_chunk_mutate(buf, _byte_idx, _data)

    def _op_jpeg_crc_fix(self, buf, _byte_idx, _data):
        return self._operators._op_jpeg_crc_fix(buf, _byte_idx, _data)

    def _op_gzip_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_gzip_chunk_mutate(buf, _byte_idx, _data)

    def _op_bmp_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_bmp_chunk_mutate(buf, _byte_idx, _data)

    def _op_zlib_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_zlib_chunk_mutate(buf, _byte_idx, _data)

    def _op_png_crc_fix(self, buf, _byte_idx, _data):
        return self._operators._op_png_crc_fix(buf, _byte_idx, _data)

    def _op_redqueen(self, buf, _byte_idx, data):
        return self._operators._op_redqueen(buf, _byte_idx, data)

    def _op_havoc(self, buf, _byte_idx, data):
        return self._operators._op_havoc(buf, _byte_idx, data)

    # ── Dispatch table: op name → handler method ───────────────────────
    def _build_dispatch(self):
        return self._operators.build_dispatch()

    def _havoc_mutate(self, buf: bytearray):
        return self._operators.havoc_mutate(buf)

    def _apply_single_mutation(self, buf: bytearray):
        return self._operators._apply_single_mutation(buf)

    def save_crash(self, data: bytes, returncode: int, stderr: str):
        return self._corpus_manager.save_crash(data, returncode, stderr)

    def save_to_corpus(self, data: bytes, parent: bytes | None = None):
        return self._corpus_manager.save_to_corpus(data, parent)

    def _trim_new_coverage(self, data: bytes, parent: bytes):
        return self._corpus_manager.trim_new_coverage(data, parent)

    @staticmethod
    def _edges_subset_of(candidate: bytes, reference: bytes):
        return CorpusManager._edges_subset_of(candidate, reference)

    def _auto_minimize_corpus(self):
        return self._corpus_manager.auto_minimize_corpus()

    def _deprioritize_near_duplicates(self):
        return self._corpus_manager.deprioritize_near_duplicates()

    def _pick_seed(self):
        return self._seed_picker.pick_seed()

    def _pick_markov_seed(self):
        return self._seed_picker._pick_markov_seed()

    def _pick_pareto_only(self):
        return self._seed_picker._pick_pareto_only()

    def _format_aware_seed(self):
        return self._seed_picker._format_aware_seed()

    def _compute_weights(self, now: float):
        return self._seed_picker._compute_weights(now)

    @staticmethod
    def _pareto_front(scores: list[tuple[float, float, float]], window: int = 100):
        return SeedPicker._pareto_front(scores, window)

    def _pick_from_pareto_front(self, weights: list[float], now: float):
        return self._seed_picker._pick_from_pareto_front(weights, now)

    def _weighted_pick_seed(self):
        return self._seed_picker.weighted_pick_seed()

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
        return self._stats.record_discovery_snapshot()

    def _run_calibration(self, max_execs: int = 1000):
        return self._stats.run_calibration(max_execs)

    def discovery_rate(self):
        return self._stats.discovery_rate()

    def _run_crash_replays(self, budget_ms: float = 200):
        return self._stats.run_crash_replays(budget_ms)

    def _print_run_summary(self):
        return self._stats.print_run_summary()

    def _dump_stats(self):
        return self._stats.dump_stats()

    def _dump_coverage_report(self):
        return self._stats.dump_coverage_report()

    def _append_coverage_log(self):
        return self._stats.append_coverage_log()

    def _update_te_causal_map(self):
        return self._stats.update_te_causal_map()

    def _get_te_weighted_position(self, input_length: int):
        return self._stats.get_te_weighted_position(input_length)

    def _get_current_edge_bitmap(self):
        return self._stats.get_current_edge_bitmap()

    def _format_elapsed(self):
        return self._stats.format_elapsed()

    def print_stats(self):
        return self._stats.print_stats()

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
        with contextlib.suppress(OSError):
            self._crash_mi_path.write_text(json.dumps(self._crash_mi.save(), separators=(",", ":")))
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
