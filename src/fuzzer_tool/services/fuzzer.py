"""Fuzzer orchestration: coordinates mutations, execution, and coverage."""

import atexit
import collections
import contextlib
import itertools
import logging
import math
import os
import random
import resource
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Collection
from typing import TYPE_CHECKING

from fuzzer_tool.core.rand_pool import RandPool

if TYPE_CHECKING:
    from fuzzer_tool.services.corpus_manager import PoissonDiskAdmission as _PoissonDiskAdmission
from array import array
from pathlib import Path

try:
    import numpy as np  # noqa: F401

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from fuzzer_tool.adapters.process import (
    _child_pids,
    disable_aslr,
)
from fuzzer_tool.adapters.shm import MAX_COUNT_GROWTH_FACTOR, ShmCoverage
from fuzzer_tool.core.analyzers.analyzer_elo import POS_STRATEGY_PREFIX, strategy_display_name
from fuzzer_tool.core.analyzers.analyzer_pll import Series as PLLSeries
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.byte_entropy import byte_entropy_pct
from fuzzer_tool.core.cadence import due
from fuzzer_tool.core.cost_ledger import cost_samples, seed_exec_us
from fuzzer_tool.core.dirichlet import AlphaMode, DirichletPicker
from fuzzer_tool.core.elf import SHM_LAYOUT_CURRENT, detect_elf_type, detect_shm_layout
from fuzzer_tool.core.format_seed_generator import FormatSeedGenerator
from fuzzer_tool.core.markov import MarkovChain, MarkovEnsemble
from fuzzer_tool.core.metropolis import accept_prob, path_energy
from fuzzer_tool.core.mi import MI_MAX_POSITIONS, MutualInformationTracker
from fuzzer_tool.core.multiple_testing import collect_and_correct
from fuzzer_tool.core.novelty_confirm import confirm
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.percolation import CoverageRegime
from fuzzer_tool.core.rng_health import quick_health_check
from fuzzer_tool.core.ro_rd import classify_operator_name
from fuzzer_tool.core.running_stats import RunningMoments
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.core.scaling_exponent import ScalingExponentDetector
from fuzzer_tool.core.schedulers import (
    BOGPUCBScheduler,
    C2UCBScheduler,
    CanaryScheduler,
    CMAESScheduler,
    ConsolidatedScheduler,
    ContextualLinUCBScheduler,
    CorralScheduler,
    CUCBScheduler,
    CUSUM_UCBScheduler,
    DUCBScheduler,
    EpsilonGreedyScheduler,
    Exp3Scheduler,
    Exp4Scheduler,
    FEWAScheduler,
    FPLScheduler,
    GPUCBScheduler,
    GradientBanditScheduler,
    HierarchicalBanditScheduler,
    KL_DUCBScheduler,
    KL_SWUCBScheduler,
    MonteCarloScheduler,
    MOptScheduler,
    MOSSScheduler,
    ReplicatorScheduler,
    RoundRobinScheduler,
    SoftmaxScheduler,
    SuccessiveEliminationScheduler,
    SWUCBScheduler,
    TopKScheduler,
    WhittleIndexScheduler,
)
from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.schedules import (
    ENTROPY_RANDOM_PCT,
    ENTROPY_SPARSE_PCT,
    SeedScorer,
    compute_mean_log_n_fuzz,
)
from fuzzer_tool.core.secretary import DEFAULT_EXPLORATION_FRAC, SecretaryStopping
from fuzzer_tool.core.seed_quality import BayesianSeedQuality
from fuzzer_tool.core.shapley import ShapleyAttribution
from fuzzer_tool.core.skipdet import SkipDetector
from fuzzer_tool.core.slopt import SloptBatchBandit
from fuzzer_tool.core.validity import Validity, ValidityChannel
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.maintenance import MaintenanceJob, MaintenanceQueue
from fuzzer_tool.services.operators import _DELOCALISED_OPS, OperatorEngine, operator_strategy_pool
from fuzzer_tool.services.position_arena import POSITION_STRATEGY_NAMES, PositionArena
from fuzzer_tool.services.ptrace_coverage import (
    PtraceCoverage,
)
from fuzzer_tool.services.runner import TargetRunner, ptrace_available
from fuzzer_tool.services.seed_picker import SeedPicker
from fuzzer_tool.services.stats import StatsReporter

log = logging.getLogger(__name__)

_shutdown = False

# CEM α when nothing is fittable yet under --dirichlet-alpha learned (Laplace).
_CEM_ALPHA_FALLBACK = 1.0

# Strategy names pre-registered with the Elo tracker (single source of truth
# for the pre-registration loop and the meta-scheduler log line).
_OPERATOR_STRATEGY_NAMES = (
    "consolidated",
    "replicator",
    "bandit",
    "mopt",
    "cem",
    "exp3",
    "exp4",
    "eps_greedy",
    "softmax",
    "topk",
    "hierarchical",
    "gp_ucb",
    "bo_gp_ucb",
    "contextual",
    "cmaes",
    "ducb",
    "swucb",
    "kl_ducb",
    "kl_swucb",
    "cucb",
    "cusum_ucb",
    "fewa",
    "moss",
    "c2ucb",
    "fpl",
    "corral",
    "invasion",
    "round_robin",
    "canary",
)
_SEED_STRATEGY_NAMES = (
    "ga",
    "qea",
    "weighted",
    "pareto",
    "format",
    "bayesian",
    "markov",
    "boltzmann",
    "ecofuzz",
    "katz",
    "mcts",
    "alphabeta",
    "tang",
    "kruskal_count",
    "entropy_kl",
    "entropy_zscore",
    "entropy_deviation",
    "entropy_gradient",
    "entropy_loo",
    "residual",
    "strata",
    "round_robin",
)


_kill_children_enabled = os.environ.get("FUZZER_DISABLE_KILL_CHILDREN", "") not in (
    "1",
    "true",
    "yes",
)

_environ_snapshot: dict[str, str] | None = None
"""os.environ as it stood before the first Fuzzer() in this process touched it.

Fuzzer.__init__ / run() write __AFL_DIST_SHM_ID, __AFL_SHM_ID, AFL_MAP_SIZE,
LD_PRELOAD (ASAN) and UBSAN_OPTIONS directly into the process environment,
because those keys have to be visible to subprocess.Popen()/os.exec* calls
made throughout the run. Only the cmplog shim's LD_PRELOAD edit was ever
restored (see the end of run()) -- the rest leaked into whatever ran next in
the same process: a second target in a multi-target session, a caller
embedding Fuzzer as a library, or the next test in a pytest run. Captured
once per process (not per Fuzzer instance) so a second Fuzzer() built while
the first is still mutating the environment doesn't re-baseline over those
mutations and adopt them as "original".
"""


def _snapshot_environ_once() -> None:
    global _environ_snapshot
    if _environ_snapshot is None:
        _environ_snapshot = dict(os.environ)


def _restore_environ() -> None:
    """Put os.environ back the way ``_snapshot_environ_once`` found it.

    Removes keys this process added and restores keys it changed. Safe to
    call more than once (idempotent) and safe to call from atexit (no
    exceptions escape).
    """
    global _environ_snapshot
    if _environ_snapshot is None:
        return
    with contextlib.suppress(Exception):
        for key in list(os.environ.keys()):
            if key not in _environ_snapshot:
                os.environ.pop(key, None)
        for key, value in _environ_snapshot.items():
            if os.environ.get(key) != value:
                os.environ[key] = value


"""Whether teardown SIGKILLs child process groups.

On by default: a fuzzer that exits leaving target processes behind will
exhaust the machine over a long campaign. It is switchable because the
teardown is destructive and not always wanted — when the fuzzer is embedded
in a larger process, driven by a supervisor that manages its own children,
or run under a debugger where killing the group would take the debugger with
it. The environment variable is read at import because the handlers install
at import; ``set_kill_children_enabled`` changes it afterwards, and the CLI
``--no-kill-children`` flag routes through that.
"""


def set_kill_children_enabled(enabled: bool) -> None:
    """Enable or disable the destructive part of teardown.

    Shutdown signalling still happens when disabled — only the SIGKILL of
    child process groups is suppressed, so the fuzzing loop still stops
    cleanly.
    """
    global _kill_children_enabled
    _kill_children_enabled = bool(enabled)


def _kill_children(sig=None, frame=None):
    global _shutdown
    _shutdown = True
    # SIGTERM/SIGINT are catchable: show where the fuzzer was executing
    # before tearing down children — a live answer to "what is it doing?".
    if sig is not None:
        try:
            import faulthandler

            faulthandler.dump_traceback()
        except Exception:
            pass
    if not _kill_children_enabled:
        return
    try:
        own_pgid = os.getpgrp()
    except OSError:
        own_pgid = None
    for pid in _child_pids():
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            pgid = os.getpgid(pid)
            # Children call os.setsid(), so a child's pgid is its own. If it
            # matches ours the pid was recorded before setsid ran, or the pid
            # has been reused — killing that group would SIGKILL the fuzzer
            # and everything sharing its group.
            if own_pgid is not None and pgid == own_pgid:
                continue
            os.killpg(pgid, signal.SIGKILL)


def install_cleanup_handlers() -> bool:
    """Register teardown on atexit, SIGTERM and SIGINT.

    Returns False if the handlers could not be installed. ``signal.signal``
    only works on the main thread, so importing this module from a worker
    thread previously raised at import time; that is now reported rather
    than fatal.
    """
    atexit.register(_kill_children)
    atexit.register(_restore_environ)
    try:
        signal.signal(signal.SIGTERM, _kill_children)
        signal.signal(signal.SIGINT, _kill_children)
    except (ValueError, OSError):
        return False
    return True


install_cleanup_handlers()

# Native-fault diagnostics. faulthandler installs a *C-level* SIGSEGV/SIGBUS/
# SIGFPE/SIGABRT handler that dumps every thread's Python stack from signal
# context and then re-raises with the default action, so the process still
# dies with the right exit status and a core is still produced.
#
# This replaces a Python-level `signal.signal(SIGSEGV, _handle_sigsegv)` that
# printed a traceback and called sys.exit(1). That could not work and made
# real crashes harder to diagnose, in three ways:
#
#  * Python signal handlers do not run from signal context. The C shim sets a
#    flag and returns, and the flag is only checked between bytecodes -- but a
#    segfault is synchronous, so the faulting instruction re-executes
#    immediately and faults again before any Python runs.
#  * A fault raised inside a native extension (z3, numpy) or on a non-main
#    thread never reaches the main interpreter loop at all, so the handler
#    produced no output whatsoever.
#  * Owning the signal *suppressed* faulthandler, which does work. An
#    intermittent suite crash was silent for exactly this reason until
#    SIGSEGV was handed back -- see
#    docs/handover/handover_done_2026-09-06.md.
#
# Also registers SIGUSR1 for on-demand live traces: `kill -USR1 <fuzzer-pid>`
# dumps every thread's Python stack to stderr without killing the process.
# SIGKILL (kill -9) is uncatchable in-process -- for that, run with
# --stack-heartbeat, whose periodic main-thread stack file survives the kill.
#
# adapters/inprocess.py installs its own SIGSEGV handler around in-process
# target execution and restores it afterwards; that one is scoped to a call
# where a fault is an expected result rather than a bug, and is unaffected.
try:
    import faulthandler as _faulthandler

    if not _faulthandler.is_enabled():
        _faulthandler.enable()
    _faulthandler.register(signal.SIGUSR1)
except (AttributeError, OSError, ValueError):  # pragma: no cover - env-dependent
    pass


def _apply_reward_shape(
    op_rewards: list[tuple[str, bool, float]], shape: float | None
) -> list[tuple[str, bool, float]]:
    """Scale the successful rounds' reward weights by *shape*.

    ``shape is None`` means "the class partition has nothing to say about this
    round" and the list is returned unchanged -- see
    :meth:`Fuzzer._credit_reward_shape` for when that is the case.

    Only the ``ok`` entries move. A failure already carries weight 0.0 (the
    reward loop passes ``surprisal_weight if ok else 0.0``), so scaling it would
    be arithmetic on a zero, but keeping the branch explicit means a future
    non-zero failure weight does not get shaped by a factor that describes a
    discovery the failure did not make.

    The shaping is applied *after* the [0, 1] cap the loop already imposes, so
    the result cannot climb back above the cap: scaling a capped weight is a
    discount, never a promotion.
    """
    if shape is None:
        return op_rewards
    return [(op, ok, w * shape if ok else w) for op, ok, w in op_rewards]


def _in_taint(taints, offset: int, length: int) -> bool:
    """True when ``[offset, offset+length)`` lies wholly inside one taint.

    A taint region is a byte range colorization proved the target does not
    read: every byte in it was replaced without moving the execution path.
    An operand occurring entirely inside one is therefore a coincidence of
    the byte values, not a value the comparison consumed -- which is the
    false-positive class colorization exists to remove.

    Partial overlap counts as *not* tainted: if any byte of the occurrence
    is path-relevant, the match is worth keeping.
    """
    if not taints:
        return False
    end = offset + length - 1
    return any(region.start <= offset and end <= region.end for region in taints)


def _cleanup_tmp_dir(path: Path) -> None:
    """Remove temp directory on exit."""

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        log.debug("Failed to clean up %s", path, exc_info=True)


# ── Entropy rate tracking constants ─────────────────────────────────
ENTROPY_HISTORY_MAX = 200  # max samples before trimming
ENTROPY_HISTORY_TRIM = 100  # keep this many after trim
ENTROPY_WINDOW = 4  # samples for rate-of-change computation
ENTROPY_FLAT_THRESHOLD = 0.001  # rate below which entropy is "flat"

# ── Memory bounds ────────────────────────────────────────────────────
CRASH_RATE_HISTORY_MAX = 500  # max entries in _crash_rate_execs/_crash_rate_counts

# ── Adaptive timeout (--adaptive-timeout) ────────────────────────────────
# suggested_timeout() reads the empirical CDF, so it needs a settled one:
# below this many observations it is tracking the warm-up, not the target.
ADAPTIVE_TIMEOUT_MIN_SAMPLES = 200
# Relative change required before a retune is applied. Without a dead band
# the value chases its own tail -- every retune changes which inputs time
# out, which changes the distribution the next suggestion is drawn from.
ADAPTIVE_TIMEOUT_HYSTERESIS = 0.25
# Minimum execs between retunes, so a drifting target cannot turn this into
# a per-100-exec handshake with the loader.
ADAPTIVE_TIMEOUT_COOLDOWN_EXECS = 1_000
# Absolute floor. Below ~5ms the deadline is measuring scheduler noise; the
# loader clamps at 1ms independently (an all-zero itimer disarms).
ADAPTIVE_TIMEOUT_FLOOR = 0.005
# Ceiling, as a multiple of the timeout the caller asked for. Retuning is
# allowed to loosen -- a target slower than the default produces false
# timeouts, which is a correctness problem, not just a throughput one --
# but not without bound, or one pathological input drags the deadline up
# and every later hang costs that much wall clock.
ADAPTIVE_TIMEOUT_MAX_GROWTH = 10.0
MAX_CRASH_SIGS = 10_000  # max unique crash signatures before pruning old entries
KERNEL_CRASHES_MAX = 500  # max kernel-verified crashes retained
SEED_SECRETARY_MAX = 500  # max per-seed SecretaryStopping entries
SEEN_HASHES_MAX = 200_000  # max unique seed hashes retained
EXEC_BLOOM_CAPACITY = 500_000  # executed-input filter capacity before generational wipe
EXEC_DEDUP_RETRIES = 3  # re-rolls of the mutation before executing a repeat anyway
# Format seed generator: every FORMAT_SEED_EVERY_EXECS execs, queue up to
# FORMAT_SEED_BUDGET field-targeted variants of the last fuzzed seed.
FORMAT_SEED_EVERY_EXECS = 5_000
FORMAT_SEED_BUDGET = 32
ELO_MATCH_WINDOW_MAX = 1_000  # max Elo match history entries
META_STRATEGY_CHOICES_MAX = 1_000  # max meta-strategy choice history entries
# ── Structure-function detector ───────────────────────────────────────────
STRUCTURE_BUFFER_POW = 8  # 2^8 = 256 samples
STRUCTURE_MIN_SAMPLES = 8  # minimum before noise_type() returns a result

# ── Continuum diagnostics ─────────────────────────────────────────────
# Stats ticks between co-occurrence graph rebuilds, and pairs kept per
# rebuild. The rebuild is the only non-O(1) part of the continuum path.
CONTINUUM_GRAPH_TICKS = 10
CONTINUUM_GRAPH_PAIRS = 64
# ── Stall reseeding (--reseed-on-stall) ───────────────────────────────
# splitmix64 constants: the derived seed must decorrelate from `self.seed`
# even though it is a small additive offset away from it.  A bare
# `seed + count` would hand adjacent stalls near-identical Mersenne
# Twister states.
SEED_MIX_GAMMA = 0x9E3779B97F4A7C15  # odd, golden-ratio derived
SEED_MIX_A = 0xBF58476D1CE4E5B9
SEED_MIX_B = 0x94D049BB133111EB
SEED_MASK_64 = 0xFFFFFFFFFFFFFFFF
SEED_MASK_32 = 0xFFFFFFFF  # np.random.seed accepts [0, 2**32)


def _current_rss_kb() -> int | None:
    """Return the process's *current* resident set size in KiB.

    ``/proc/self/statm`` field 2 is resident pages right now. Deliberately not
    ``getrusage().ru_maxrss``, which is the monotonic high-water mark: a
    threshold check against a peak can only ever latch on.

    Returns:
        Current RSS in KiB, or None if /proc is unavailable or unparseable.
    """
    try:
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return resident_pages * (os.sysconf("SC_PAGE_SIZE") // 1024)


# Symbols that mean the target can populate an edge bitmap: the shim's own
# (linked or preloaded) and clang's sancov callbacks.
_AFL_SYMS = ("__afl_area", "__afl_map_shm", "__sanitizer_cov")


def afl_instrumentation_status(target_path: str) -> str:
    """Classify *target_path*'s edge instrumentation as present/absent/unknown.

    The third state is the point.  ``nm`` reports nothing at all for a
    stripped binary, so a plain boolean cannot tell "this target has no
    instrumentation" from "this target's symbol table was removed" — and a
    stripped, fully-instrumented target is a normal thing to be handed.
    Treating that as absent would fire the no-instrumentation warning on a
    run that is working perfectly, which is the fastest way to teach someone
    to ignore the warning.

    The rule, checked against every target shape in ``targets/``:

    - the static symbol table has entries and one of :data:`_AFL_SYMS` is
      among them (or among the dynamic ones)     -> ``"present"``
    - the static symbol table has entries and none match  -> ``"absent"``
    - the static symbol table is empty, or ``nm`` is unusable -> ``"unknown"``

    The dynamic table alone is not enough to rule instrumentation *out*:
    ``__afl_area`` lives in the static symtab, so a stripped target keeps
    thousands of dynamic symbols and none of the ones looked for here.

    Returns:
        One of ``"present"``, ``"absent"``, ``"unknown"``.
    """
    import subprocess

    def _nm(*flags: str) -> str | None:
        try:
            r = subprocess.run(
                ["nm", *flags, target_path],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout

    static = _nm()
    if static is None or not static.strip():
        # Stripped, or no nm on this box. Cannot distinguish; say so.
        return "unknown"
    dynamic = _nm("-D") or ""
    haystack = static + dynamic
    return "present" if any(s in haystack for s in _AFL_SYMS) else "absent"


def _detect_afl(target_path: str) -> bool:
    """True when *target_path* has AFL edge coverage instrumentation.

    Kept as the boolean face of :func:`afl_instrumentation_status` for the
    call sites that only need "should I print [AFL]".  Anything that decides
    whether to *warn* must use the tri-state instead — see the docstring
    there for why ``unknown`` must not collapse into ``False``.
    """
    return afl_instrumentation_status(target_path) == "present"


def _detect_distance(target_path: str) -> bool:
    """Check if a binary has the AFLGo distance channel compiled in.

    Distance builds define __afl_dist_flush (the shim under
    __AFL_DISTANCE_MODE) and define __sanitizer_cov_trace_pc (trace-pc
    instrumentation); the bare trace_pc symbol distinguishes them from
    plain trace-pc-guard builds, which only carry the *_guard callbacks.
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["nm", target_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "__afl_dist_flush" in result.stdout:
            return True
        return bool(re.search(r"^[tT] __sanitizer_cov_trace_pc$", result.stdout, re.MULTILINE))
    except (OSError, subprocess.TimeoutExpired):
        return False


def _detect_cmplog(target_path: str) -> bool:
    """Check if a binary has cmplog or tracecmp built in.

    Recognizes either the symbol-based shim (__cmplog_reset) or the
    compiler-IR shim (__tracecmp_reset).
    """
    import subprocess

    cmplog_symbols = ("__cmplog_reset", "__tracecmp_reset")
    try:
        for flags in [[], ["-D"]]:
            result = subprocess.run(
                ["nm"] + flags + [target_path], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for sym in cmplog_symbols:
                    if sym in result.stdout:
                        return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    return False


def _detect_tracecmp_target(target_path: str) -> bool:
    """Check if a binary was compiled with -fsanitize-coverage=trace-cmp.

    Targets compiled with trace-cmp have undefined (U) references to
    __sanitizer_cov_trace_cmp{1,2,4,8} that must be resolved at runtime
    by tracecmp_shim.so or LD_PRELOAD.
    """
    import subprocess

    target_syms = ("__sanitizer_cov_trace_cmp1",)
    try:
        for flags in [[], ["-D"]]:
            result = subprocess.run(
                ["nm"] + flags + [target_path], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for sym in target_syms:
                    if sym in result.stdout:
                        return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    return False


def _detect_asan(target_path: str) -> bool:
    """Detect if a binary is ASAN-instrumented by checking for __asan_init symbol."""
    import subprocess

    for flags in [[], ["-D"]]:
        try:
            r = subprocess.run(["nm"] + flags + [target_path], capture_output=True, timeout=10)
            if r.returncode == 0 and (
                b"__asan_init" in r.stdout or b"__asan_register_globals" in r.stdout
            ):
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return False


def _detect_ubsan(target_path: str) -> bool:
    """Detect if a binary is UBSAN-instrumented by checking for __ubsan_handle_* symbols."""
    import subprocess

    for flags in [[], ["-D"]]:
        try:
            r = subprocess.run(["nm"] + flags + [target_path], capture_output=True, timeout=10)
            if r.returncode == 0 and b"__ubsan_handle" in r.stdout:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return False


# Attributes on Fuzzer whose scheduler may expose last_selection_probs().
# Every name here must be a real attribute -- asserted by
# tests/test_regression_fluctuation_probs.py. A name that is not silently
# disables the feature instead of failing, which is how W = L*log(L) shipped.
_SELECTION_PROB_SOURCES: tuple[str, ...] = ("_exp3",)


class Fuzzer:
    def _warn_no_coverage(self) -> None:
        """Warn that an in-process target is running without coverage.

        Emitted once per run. Without an SHM segment nothing populates the
        edge bitmap, so every coverage-guided subsystem downstream — seed
        scheduling, MI/TE/sensitivity position weighting, Elo/bandit operator
        scheduling, stall detection, corpus admission — runs on a
        constant-zero signal. The run looks fast and healthy while
        discovering nothing, so the failure is otherwise entirely silent.
        """
        if getattr(self, "_no_cov_warned", False):
            return
        self._no_cov_warned = True
        msg = (
            "No coverage enabled: running blind. Edge discovery, "
            "coverage-guided scheduling and corpus growth are all inactive. "
            "Pass -c/--coverage to enable the AFL SHM bitmap."
        )
        log.warning(msg)
        print(f"[!] WARNING: {msg}")

    def _report_instrumentation(self) -> None:
        """Print the target's instrumentation state, and warn if it has none.

        Coverage is on by default, so the common failure is no longer "the
        user forgot -c" but "the target was never built with instrumentation".
        Both produce the same symptom — healthy throughput, an empty bitmap,
        a corpus that never grows — and until this warning existed only the
        first one was ever reported.
        """
        status = afl_instrumentation_status(self.target)
        if status == "present":
            print("[*] AFL instrumentation: detected")
            self._warn_no_compiler_coverage(self.target)
            self._check_shm_layout(self.target)
        elif status == "absent":
            self._warn_uninstrumented([self.target])
        # "unknown" (stripped binary, or no nm): say nothing rather than
        # guess. A false alarm here trains people to ignore the real one.

    def _report_map_cache_residency(self) -> None:
        """Informational: the coverage map against the host cache hierarchy.

        Silent when the topology is unknown (non-Linux, a container without
        sysfs mounted) rather than guessing, and silent with no segment.

        Phrased as capacity, not speed, on purpose. "Fits L2" is the obvious
        thing to print here and the obvious reading of it is wrong: the edge
        table is a sparsely-touched hash table, so only the lines an
        execution actually reaches are ever resident and the size of the
        allocation does not determine that. Measured with the touched set
        held at 32 KiB and only the segment varied, 32 KiB to 32 MiB across
        two cache boundaries, per-fire cost went 5.36, 5.28, 5.18, 5.21,
        5.18, 5.25 ns -- no trend, the smallest segment fractionally
        slowest. Holding the segment at 8 MiB and varying distinct edges
        instead does move it, 5.18 ns at 512 distinct to 5.98 at 524,288.

        So the line quotes how many distinct edges stay resident per level,
        which is the quantity that actually moves, and leaves the operator
        to compare it against the edge count they are seeing.
        """
        if self.shm_cov is None:
            return
        from fuzzer_tool.core.cpu_cache import describe_map_residency

        line = describe_map_residency(self.shm_cov.shm_bytes, self.shm_cov.num_entries)
        if line:
            print(line)

    def _check_shm_layout(self, target: str) -> None:
        """Refuse a target built against an incompatible SHM layout.

        Fatal, unlike the uninstrumented warning, because there is no
        degraded-but-usable mode to fall back to. The layouts disagree about
        where the edge table starts and what the word at offset 4 means, so
        a stale target writes every entry at the wrong offset: the ids read
        back are halves of two adjacent entries spliced together, and our own
        edge_count header is read as an edge. That produces a
        plausible-looking stream of never-before-seen edges -- a corpus that
        grows on garbage, which is worse than a run reporting nothing.

        Only reached when the target is known to carry shim instrumentation,
        so an uninstrumented or stripped binary cannot trip it.
        """
        if not self.use_coverage or getattr(self, "shm_cov", None) is None:
            return

        found = detect_shm_layout(target)
        if found == SHM_LAYOUT_CURRENT:
            return
        raise RuntimeError(
            f"{target} was built against SHM layout {found}, this fuzzer speaks "
            f"layout {SHM_LAYOUT_CURRENT} — the two disagree about the segment "
            "layout, so coverage would be read from the wrong offsets rather than "
            "simply missing. Rebuild the target against the current "
            "adapters/afl_shim.c (tools/build_targets.sh), or run with "
            "--no-coverage to fuzz it blind."
        )

    def _warn_uninstrumented(self, targets: list[str]) -> None:
        """Warn that coverage is on but the target(s) cannot report edges.

        Not fatal: crash and timeout detection still work, so a blind run
        against an uninstrumented binary is a legitimate thing to want. It
        just must not look like a coverage-guided one.
        """
        if not self.use_coverage or getattr(self, "_uninstrumented_warned", False):
            return
        if getattr(self, "ptrace_cov", None) is not None:
            # ptrace derives edges from breakpoints on the binary itself and
            # needs no build-time instrumentation, so the premise of this
            # warning does not hold. Caught by running --no-shm against an
            # uninstrumented target: coverage was working (5 breakpoints,
            # edges accumulating) while this told the user the bitmap would
            # stay empty and offered --no-coverage as the fix.
            return
        if getattr(self, "shm_cov", None) is None:
            # Not on the SHM path at all (in-process modes); those have their
            # own warning in _warn_no_coverage.
            return
        self._uninstrumented_warned = True
        which = targets[0] if len(targets) == 1 else f"{len(targets)} targets"
        msg = (
            f"No edge instrumentation found in {which}: the coverage bitmap "
            "will stay empty, so edge discovery, coverage-guided scheduling "
            "and corpus growth are all inactive. Rebuild with "
            "tools/build_targets.sh, or pass --no-coverage to run blind on "
            "purpose (crash detection is unaffected)."
        )
        log.warning(msg)
        print(f"[!] WARNING: {msg}")

    def _warn_no_compiler_coverage(self, target: str) -> None:
        """Warn when the target carries the shim but no instrumented call sites.

        ``afl_instrumentation_status`` cannot catch this. It looks for
        ``__afl_area``/``__afl_map_shm``/``__sanitizer_cov``, all of which are
        the shim's own definitions, and the shim is ``-include``'d into every
        target -- so a binary the compiler never instrumented still reports
        "present" and this function's caller still prints "detected".

        Measured on a default ``tools/build_targets.sh`` run: 4 of 20
        binaries carried a guard section, all 20 were classified "present",
        and 300 execs against one of the other 16 in ``--inprocess-direct``
        reported ``shm: 2 max: 2 sat: 100%``, ``Edges discovered: 2``,
        ``Total richness: 2 - 2 (95% CI, Chao2)`` and
        ``P(new code next): 0.00%``. The two edges were the harness's own
        ``__afl_map_edge`` calls. Nothing in that output distinguishes it
        from a genuinely exhausted target.

        Warn-only, and deliberately narrow:

        - silent under ptrace, which sets breakpoints on the binary and needs
          no build-time instrumentation (the same carve-out
          ``_warn_uninstrumented`` makes, for the same reason)
        - silent with coverage off, since the premise does not hold
        - silent on ``"unknown"``: the bound symbols are static-only, so a
          stripped target cannot be judged here
        - hand-written ``__afl_map_edge`` calls still register, which is why
          the wording says "only those" rather than "no edges"

        The verdict is also recorded on ``self._coverage_trusted``, so a
        scheduler can decline to draw conclusions from per-edge statistics
        instead of confidently scoring the harness's own edges. Read it with
        ``getattr(f, "_coverage_trusted", True)``: the check runs only when
        the shim is detected at all, so the attribute may be absent.
        """
        from fuzzer_tool.core.scheduler_substrate import coverage_trust

        if getattr(self, "_no_scov_warned", False):
            return
        trusted, reason = coverage_trust(
            target,
            use_coverage=self.use_coverage,
            ptrace=getattr(self, "ptrace_cov", None) is not None or self.use_ptrace,
        )
        self._coverage_trusted = trusted
        if trusted:
            return
        self._no_scov_warned = True
        msg = f"{reason}. Rebuild with tools/build_targets.sh --clang-scov."
        log.warning(msg)
        print(f"[!] WARNING: {msg}")

    def _ensure_ctx_ids_are_exec_stable(self, targets: list[str]) -> None:
        """Put a context-sensitive target in base-relative mode if ASLR survived.

        ``afl_shim.c``'s ``__afl_get_caller_ctx`` hashes the return address
        of the frame above the edge. With ASLR on and
        ``__AFL_CTX_SENSITIVE=1`` that address moves in every process, so
        the same input reports a disjoint edge set every execution:
        measured on fuzzgoat, six runs of one input shared 2 ids out of a
        295-id union (Jaccard 0.007), and the union over 250 seeds inflated
        445 -> 1513, every phantom id owned by exactly one seed and so
        maximally "rare" to the seed picker. See F1 in docs/handover/
        handover_edge_id_axis_2026-09-18.md.

        The shim now resolves that address to a load-base-relative offset,
        which is exec-stable -- but it switches on ``FUZZER_KEEP_ASLR=1``,
        read in the *target* process, rather than on whether ASLR is
        actually on. Those two are not the same condition.
        ``disable_aslr()`` also returns False when ``personality()`` is
        refused -- seccomp, some container runtimes, a non-Linux host --
        and there the variable is unset, so the shim stays in raw mode
        while ASLR is on. That is the whole remaining gap, and the user
        cannot be expected to find it: the fix is to set a variable named
        for keeping ASLR, on a host that never disabled it in the first
        place.

        So set it here instead of telling them to. ``os.environ`` is what
        ``services/runner`` copies into every child, the value is only read
        by the shim (and by ``disable_aslr()``, which has already run and
        cached its answer), and for a target built before the shim grew
        relative mode it is simply ignored. Measured on a gcc build of the
        test driver, six executions of one input with ASLR on: raw mode
        gives a 76-id union with an empty intersection, relative mode gives
        22 ids reproduced exactly.

        A target built against a shim that predates relative mode ignores
        the variable, so setting it would fix nothing and say it had. That
        case is read off ``__afl_ctx_relative_capable`` and warned about by
        name instead, with the rebuild as the only escape. The measurement
        in :meth:`_report_edge_id_stability` still runs either way: a marker
        says what the binary can do, not what it did.
        """
        if self._aslr_disabled or not self.use_coverage:
            return
        if getattr(self, "shm_cov", None) is None and not self._target_shm_covs:
            # ptrace and the in-process modes do not go through the shim's
            # context hash, so the premise does not hold for them.
            return
        from fuzzer_tool.core.elf import detect_ctx_bits, detect_ctx_relative_capable

        capable, stale = [], []
        for t in targets:
            if not t:
                continue
            # None means "no marker, unknown shim", which is not evidence of
            # context sensitivity; only a positive width is.
            if not detect_ctx_bits(t):
                continue
            # None here is "could not read the symbol table", which is not
            # evidence that the shim is old -- leave those to the probe.
            (stale if detect_ctx_relative_capable(t) is False else capable).append(t)
        if stale:
            which = os.path.basename(stale[0]) if len(stale) == 1 else f"{len(stale)} targets"
            msg = (
                f"ASLR is enabled and {which} was built against a shim without "
                "base-relative caller context (no __afl_ctx_relative_capable "
                "marker): edge ids are hashed from a raw return address, so "
                "every execution of the same input reports a different edge "
                "set and coverage feedback is noise. Rebuild the target "
                "(tools/build_targets.sh), or build it with "
                "-D__AFL_CTX_SENSITIVE=0."
            )
            log.warning(msg)
            print(f"[!] WARNING: {msg}")
        if not capable or os.environ.get("FUZZER_KEEP_ASLR") == "1":
            return  # nothing to switch, or already switched by the user
        os.environ["FUZZER_KEEP_ASLR"] = "1"
        which = os.path.basename(capable[0]) if len(capable) == 1 else f"{len(capable)} targets"
        print(
            f"[*] ASLR survived startup and {which} is a context-sensitive build: "
            "setting FUZZER_KEEP_ASLR=1 for the target so the shim hashes "
            "load-base-relative return addresses. Without it every execution "
            "would report a different edge set."
        )

    def _report_rng_health(self) -> None:
        """Quick PRNG sanity check, printed once on the startup banner.

        Runs monobit + chi-squared smoke tests (see core/rng_health.py) on
        a small sample from ``self._rng``. Not fatal -- a suspect RNG is a
        warning, never a reason to abort a campaign that was otherwise
        ready to start -- but a genuinely broken stream (stuck seed,
        degenerate bit-generator) silently produces near-duplicate mutants
        for the whole run, which is a much more expensive way to find out.
        """
        try:
            result = quick_health_check(self._rng)
        except Exception as exc:  # noqa: BLE001 -- never let a smoke test sink startup
            log.warning("RNG health check errored: %s", exc)
            return
        if result.ok:
            print(f"[*] RNG health: {result.summary()}")
        else:
            msg = f"PRNG health check: {result.summary()}"
            log.warning(msg)
            print(f"[!] WARNING: {msg}")

    @staticmethod
    def _probe_so_function(target):
        """Probe a shared object for the best fuzz entry point.

        Uses nm -D to scan symbols without loading the library (avoids
        ASAN issues with ctypes.CDLL loading order).
        Falls back to fuzz_shm_run if nothing is found.
        """
        import subprocess

        try:
            result = subprocess.run(
                ["nm", "-D", target],
                capture_output=True,
                text=True,
                timeout=5,
            )
            symbols = result.stdout
        except (OSError, subprocess.TimeoutExpired):
            return "fuzz_shm_run"

        # Prefer the standard wrapper
        if "fuzz_shm_run" in symbols:
            return "fuzz_shm_run"

        # Scan for any fuzz_* symbol
        for line in symbols.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                name = parts[-1]
                if name.startswith("fuzz_"):
                    return name

        return "fuzz_shm_run"

    def __init__(
        self,
        target,
        corpus_dir,
        crashes_dir,
        max_len=4096,
        timeout=1,
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
        dirichlet_alpha=AlphaMode.FIXED,
        mc_bandit=False,
        mc_cem=False,
        mc_cycle_detect=False,
        mopt=False,
        mopt_mc_stop_multiplier=None,
        cmaes=False,
        cmaes_pop_size=8,
        cmaes_generation_size=200,
        cmaes_step_size=0.3,
        cmaes_elite_frac=0.5,
        targets=None,
        anneal_budget=0,
        boltzmann=False,
        tang=False,
        tang_rank=10,
        tang_refit_interval=2000,
        ecofuzz=False,
        ecofuzz_mc_penalty_multiplier=None,
        metropolis=False,
        mc_elite_frac=0.1,
        mc_refit_interval=1000,
        mc_decay_interval=100,
        pairwise_blend=0.0,
        stats_file=None,
        stats_interval=1000,
        coverage_report=None,
        coverage_log=None,
        stack_heartbeat=None,
        grammar=None,
        persistent=False,
        net_host=None,
        net_port=None,
        net_proto="tcp",
        net_keepalive=False,
        net_settle_ms=10,
        inprocess=False,
        inprocess_direct=False,
        inprocess_func="LLVMFuzzerTestOneInput",
        calibrate_stability=0,
        # cmplog is always on by default; --no-cmplog-fifo-sink disables
        # the FIFO drain mode but not the comparison tracing itself.
        cmplog=True,
        cmplog_max_tokens=0,
        cmplog_max_pairs=0,
        cmplog_workdir=None,
        cmplog_fifo_sink=True,
        cmplog_fifo_sink_size=None,
        asan_target=None,
        ubsan_target=None,
        max_corpus=0,
        max_corpus_bytes=0,
        minimize_every_execs=0,
        prune_corpus_max_memory=80,
        no_shm=False,
        use_ptrace=False,
        adaptive_havoc=True,
        use_cfg_cache=True,
        # Opt-in control-dependence discount for directed-mode distance
        # (core/distance.py::TargetDistance). 0.0 (default) reproduces
        # exact prior BFS-only distances; see --gate-bonus help text.
        gate_bonus=0.0,
        adaptive_timeout=False,
        resume=False,
        trace_crashes=True,
        learn_format=False,
        corpus_ppmd=False,
        corpus_quasiperiodicity=False,
        seed=42,
        extra_crash_codes=None,
        replay_n=0,
        crash_blocklist=None,
        crash_allowlist=None,
        save_smaller=False,
        honggfuzz=False,
        hw_perf=False,
        intel_pt=False,
        intel_pt_mode="block",
        lbr=False,
        lbr_period=0,
        schedule_ablation=None,
        schedule="base",
        aflgo_cooling="exp",
        t_x_minutes=60.0,
        differential_target=None,
        replicator=False,
        replicator_mc_stop_multiplier=None,
        shapley=False,
        bayesian=False,
        mi_guided=False,
        renyi_weight=False,
        transfer_entropy=False,
        occupation=False,
        causal_sector=False,
        secretary=False,
        secretary_window=500,
        secretary_exploration=None,
        elo=False,
        invasion=False,
        round_robin=False,
        canary_scheduler=False,
        garch=False,
        continuum=False,
        pll=False,
        temp_control=False,
        temp_setpoint_fraction=0.5,
        temp_reference_rate=None,
        exp3=False,
        exp3_gamma=0.1,
        exp4=False,
        exp4_gamma=0.1,
        slopt=False,
        eps_greedy=False,
        eps_greedy_epsilon0=1.0,
        eps_greedy_decay=0.9995,
        softmax=False,
        softmax_tau=1.0,
        use_topk=False,
        topk_k=1,
        hierarchical_bandit=False,
        gp_ucb=False,
        gp_length_scale=1.0,
        gp_beta=2.0,
        bo_gp_ucb=False,
        bo_gp_length_scale=1.0,
        bo_gp_noise=0.01,
        ducb=False,
        ducb_gamma=0.9999,
        kl_ducb=False,
        kl_ducb_gamma=0.9999,
        swucb=False,
        swucb_window=4000,
        kl_swucb=False,
        kl_swucb_window=4000,
        cucb=False,
        cucb_gamma=0.9995,
        cusum_ucb=False,
        cusum_ucb_m=30,
        cusum_ucb_epsilon=0.1,
        cusum_ucb_h=40.0,
        cusum_ucb_xi=0.6,
        fewa=False,
        fewa_alpha=0.5,
        fewa_max_window=512,
        fpl=False,
        fpl_epsilon=1.0,
        gradient=False,
        gradient_alpha=0.05,
        gradient_temperature=1.0,
        gradient_temp_decay=0.9995,
        gradient_min_temperature=0.05,
        gradient_floor=0.05,
        corral=False,
        corral_eta=0.6,
        whittle=False,
        whittle_n_states=5,
        whittle_gamma=0.95,
        whittle_passive_decay=0.0,
        whittle_floor=0.05,
        whittle_recompute_batch=25,
        op_katz=False,
        op_katz_alpha_fraction=0.85,
        op_kuramoto=False,
        op_kuramoto_k=1.0,
        op_kuramoto_omega_scale=1.0,
        op_kuramoto_dt=0.05,
        op_kuramoto_steps_per_batch=5,
        op_kuramoto_recompute_batch=25,
        op_tang=False,
        op_tang_rank=10,
        op_tang_refit_interval=2000,
        op_kruskal_count=False,
        op_credit=False,
        op_tpe=False,
        op_strata=False,
        shaped_reward=False,
        shaped_reward_floor=0.0,
        continuum_reward=False,
        continuum_reward_floor=0.0,
        consolidated=False,
        moss=False,
        moss_gamma=1.0,
        contextual=False,
        contextual_alpha=1.0,
        contextual_lambda=1.0,
        c2ucb=False,
        c2ucb_alpha=1.0,
        c2ucb_lambda=1.0,
        c2ucb_min_out_rounds=30.0,
        overlap_density=False,
        overlap_density_mode="modifier",
        overlap_min_jaccard=0.25,
        overlap_density_blend=0.5,
        fractal_diversity=False,
        fractal_diversity_depth=3,
        fractal_diversity_bonus=1.3,
        lineage=False,
        lineage_backtrack=False,
        mds_select=False,
        mcts=False,
        alphabeta=False,
        sensitivity=False,
        ga=False,
        qea=False,
        wfc=False,
        ga_pop_size=200,
        ga_gen_size=500,
        ga_elite_frac=0.1,
        ga_crossover_rate=0.7,
        ga_mutation_rate=0.3,
        ga_tournament_size=3,
        ga_speciation_threshold=0.3,
        qea_rotation_angle=0.05,
        qea_strong_bias=None,
        qea_elite_reset=0,
        qea_correlation=False,
        qea_correlation_delta=0.02,
        qea_correlation_max=2.0,
        qea_correlation_sweeps=3,
        qea_cooling=False,
        qea_cooling_decay=0.98,
        qea_cooling_min_angle=0.005,
        calibrate=0,
        stall_threshold=1000,
        stall_release_edges=1,
        resize_map_on_stall=True,
        reseed_on_stall=False,
        job_scheduler=False,
        map_size=0,
        max_collision_risk=30,
        continue_until_crash=False,
        multi_targets=None,
        debug=False,
        enable_regex_bomb=False,
        colorize=False,
        colorize_max_execs=512,
        enable_x86_mutator=False,
        enable_arm_mutator=False,
        enable_smt_z3=False,
        path_negation=False,
        mod_solving="concolic",
        corpus_boost=0,
        boost_mean=None,
        boost_std=None,
        boost_pad="repeat",
        refresh_profile=False,
        chi2_operator_interval=0,
        quiet_stats=False,
        no_save_state=False,
        dedup_execs=True,
        seed_calibration=True,
        # Exec-dedup backend.  "bloom" is the historic default (a
        # BloomFilter with generational reset); "cuckoo" swaps in a
        # CuckooFilter, which supports deletions and has a lower realised
        # FP rate per bit.  Both expose the same update_bytes(key,
        # reset_on_full) contract that _dedup_mutate drives, so the
        # branch lives in one place: here, at construction.
        exec_dedup_backend="bloom",
        fluctuation=False,
        fluctuation_beta=1.0,
        fluctuation_window=1000,
        # Appended rather than grouped with the other mutation-targeting
        # flags: this signature is positional, so inserting a parameter
        # mid-list silently shifts every caller argument after it.
        region_profile=False,
        deterministic=True,
        forkserver=True,
        seed_skip_size=0,
        seed_truncate_size=0,
        seed_slide_size=0,
        seed_slide_max_seeds=0,
        perf_novelty=True,
        reject_code=None,
        sharpe_kelly_blend=0.0,
        bootstrap=False,
        bootstrap_k=1,
        # Weizz structure tags (P1 collector + P2 operators). Appended at the
        # end so positional callers of Fuzzer() are not shifted.
        weizz_tags=False,
        weizz_tags_max_len=8192,
        email_on_crash=None,
        # Poisson-disk admission (proactive corpus diversity via MinHash LSH).
        # Appended at the end so positional callers of Fuzzer() are not shifted.
        poisson_disk_admission=False,
        poisson_disk_min_jaccard=0.25,
        # TSP neighbourhood operators (Phase 1 / C2). Appended at the end.
        op_span_reverse=False,
        op_span_relocate=False,
        # AFL deterministic sweep as an arbitrated arm (T1-1, core/mutations/afl_det.py)
        op_afl_det=False,
        # FormatFuzzer structural mutators (see handover_formatfuzzer_integration).
        formatfuzzer=False,
        ff_bin_dir=None,
        ff_templates=None,
        # Successive-elimination / racing bandit for operator scheduling.
        successive_elim=False,
        successive_elim_delta=0.1,
        successive_elim_min_pulls=3,
        successive_elim_reopen=0,
        kruskal_count=False,
        # Byte-entropy seed arms (see handover_entropy_seed_schedulers).
        entropy_kl=False,
        entropy_zscore=False,
        entropy_zscore_target=0.0,
        entropy_deviation=False,
        entropy_gradient=False,
        entropy_gradient_decay=0.98,
        entropy_loo=False,
        seed_residual=False,
        strata=False,
        pool_drift=False,
        dict_thompson=False,
        # Seed arena's argmin floor (see core/schedulers/seed_canary.py).
        # The op_canary counterpart for the seed-selection Elo pool.
        confirm_novelty=False,
        seed_canary_scheduler=False,
        # Seed arena's deterministic baseline (see
        # core/schedulers/seed_round_robin.py). The seed_ counterpart of
        # --round-robin: reachable both as an Elo arm and, needing no
        # arbiter, directly in pick_seed()'s no-elo fallback chain --
        # the same standalone treatment round_robin already gets on the
        # operator side.
        seed_round_robin_scheduler=False,
        # Least-slack revisit bound in seconds (P3-3 step 6); 0 disables.
        # See SeedPicker._pick_lst_seed.
        lst_revisit=0.0,
        # Position arena (see services/position_arena.py): Elo arbitrates the
        # position proposers, uniform included. Needs --elo. The arena always
        # fields the BurnFrontPositionScheduler arm; --burn-front alone adds
        # it as one more candidate in select_position's uniform pick.
        burn_front=False,
        position_arena=False,
    ):
        # Snapshot os.environ before anything below (or later in run()) can
        # write __AFL_DIST_SHM_ID / __AFL_SHM_ID / AFL_MAP_SIZE / LD_PRELOAD /
        # UBSAN_OPTIONS into it, so run() can hand the process environment
        # back afterwards. See _restore_environ()/finding #10.
        _snapshot_environ_once()
        self.target = target
        self.debug = debug
        # Persistent-loader ptrace self-trace for fault-address/register
        # capture (PTRACE_TRACEME on every forked call). Off by default —
        # it adds per-exec overhead and can be blocked by yama ptrace_scope.
        # Crash triage (_run_triage_ptrace) still fires a one-off re-run on
        # each crash regardless of this flag; this only controls the
        # always-on per-iteration trace in the persistent loader.
        self.use_ptrace = use_ptrace
        # Weight havoc's 11 inline sub-mutations by their measured hit rate
        # instead of drawing them uniformly. On by default: the priors start
        # uniform, the per-draw cost is a bisect over 11 floats, and the
        # branch mix is otherwise the one part of the mutation stack that
        # ignores the feedback every layer above it collects. --no-adaptive-
        # havoc restores the flat `r[0] % 11` split for A/B runs.
        self._adaptive_havoc = adaptive_havoc
        self.refresh_profile = refresh_profile
        self.quiet_stats = quiet_stats
        # Multi-target support: list of target binaries to fuzz with shared corpus
        self.multi_targets = multi_targets  # None for single-target
        self._active_target_idx = 0  # round-robin index
        self._target_shm_covs = {}  # target_path -> ShmCoverage (per-target)
        self._target_profiles = {}  # target_path -> TargetProfile
        # Pin the address-space layout BEFORE anything spawns, dlopens, or
        # profiles a target. personality(ADDR_NO_RANDOMIZE) is inherited by
        # every child through fork and survives execve, so this one call
        # covers posix_spawn (which has no preexec_fn hook), Popen, the
        # in-process subprocess loader, and the forkserver.
        #
        # Required for correctness, not just reproducibility: afl_shim.c's
        # caller-context edge hashing derives edge_id from a runtime return
        # address, and _seen_edge_ids is compared across target processes.
        # With ASLR on, every exec of a CTX build reports a fresh edge set.
        # See adapters/process.disable_aslr.
        self._aslr_disabled = disable_aslr()

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
        self._resize_map_on_stall = resize_map_on_stall
        self._reseed_on_stall = reseed_on_stall
        self._max_collision_risk = max_collision_risk
        self._last_new_edge_exec = 0
        # Effective edges of the executions between discoveries, for the
        # stall reason (P2-1).  Sparse SHM path only; empty elsewhere.
        from fuzzer_tool.core.scheduler_substrate import ExecutionPerplexity

        self._exec_perplexity = ExecutionPerplexity()
        self._novel_input_count = 0  # execs where record_edges found ≥1 new edge
        self._stall_recovery_active = False
        self._stall_recovery_count = 0  # times recovery was activated
        # Drop-driven resize bookkeeping. The interval is in executions and
        # only rate-limits the check; the stats interval already gates how
        # often the surrounding block runs at all.
        self._drop_resize_checked_at = 0
        self._drop_resize_interval = 1000
        self._drop_cap_warned = False
        self._stall_recovery_execs = 0  # execs spent in recovery mode
        self._stall_reseed_count = 0  # times the RNG was reseeded on stall
        self._last_stall_seed = None  # seed applied by the most recent reseed
        # Relay telemetry (see _stall_relay_stats). The stall mechanism is a
        # relay: engage after `stall_threshold` execs of silence, release on
        # new coverage. Its period and amplitude are exactly what a relay
        # auto-tuning experiment measures, and the campaign runs the
        # experiment whether or not anyone reads it -- so read it.
        self._stall_release_edges = max(1, int(stall_release_edges))
        self._stall_edges_in_recovery = 0  # edges since the current engage
        self._stall_edges_active = 0  # cumulative edges found while engaged
        self._stall_engaged_at = None  # exec of the current/last engage
        self._stall_last_engage_exec = None  # exec of the PREVIOUS engage
        self._stall_cycles: collections.deque = collections.deque(maxlen=256)
        self._stall_cycle_edges = 0  # edges in the cycle being accumulated
        self._stall_cycle_execs = 0  # execs engaged in that cycle
        self._stall_release_reason = None
        self.extra_crash_codes = set(extra_crash_codes) if extra_crash_codes else set()
        self.max_len = max_len
        # Floor for the adaptive max_len in corpus_manager: that value
        # tracks the corpus size distribution and must be allowed to
        # fall again when the corpus shrinks, but never below what the
        # caller asked for.
        self._max_len_floor = max_len
        self.timeout = timeout
        # Adaptive timeout: retune self.timeout from the live
        # ExecutionTimeTracker rather than leaving it fixed at construction.
        # Opt-in, because suggested_timeout() is derived from one target's
        # observed distribution and is not a safe global -- the default
        # stays exactly where the caller put it.
        self._adaptive_timeout = adaptive_timeout
        self._timeout_initial = timeout
        # (exec_count, old, new) per applied retune; reported at the end.
        self._timeout_retunes: list[tuple[int, float, float]] = []
        self._last_timeout_retune_exec = 0
        self.mutations_per_input = mutations_per_input
        self.use_coverage = use_coverage
        self.dictionary = dictionary or []
        self.file_mode = file_mode
        self.target_args = target_args or []
        self.max_corpus = max_corpus
        self.max_corpus_bytes = max_corpus_bytes
        self.minimize_every_execs = minimize_every_execs
        self.prune_corpus_max_memory = prune_corpus_max_memory
        self._last_corpus_prune_exec = 0
        self._last_bloat_warn_exec = 0
        # P3-3 step 4/6, gated behind --job-scheduler (excluded from
        # --hail-mary: this changes maintenance-tick *cadence*, not a
        # fuzzing strategy, and the change should be opted into
        # deliberately rather than force-enabled alongside everything
        # else). False (the default) keeps the original three independent
        # ad-hoc gates byte-for-byte -- see _legacy_memory_prune_tick and
        # the tick loop below. True replaces them with one MaintenanceQueue
        # (services/maintenance.py) that sequences due jobs via Lawler's
        # algorithm instead of running them in fixed program order, and
        # folds crash/sanitizer replays and gc.collect into the same
        # stats-interval cadence memory pruning already had (see
        # maintenance.py's module docstring for the cadence-change caveat
        # this implies for fast targets).
        self.job_scheduler = job_scheduler
        self._last_memory_prune_exec = 0  # legacy path only; queue owns its own bookkeeping
        self._maintenance = MaintenanceQueue(
            [
                MaintenanceJob("gc", interval_execs=500, action=self._gc_collect),
                MaintenanceJob(
                    "crash_replays",
                    interval_execs=500,
                    action=self._run_crash_replays,
                    active=lambda: self.replay_n > 0,
                ),
                MaintenanceJob(
                    "sanitizer_replays",
                    interval_execs=500,
                    action=self._run_sanitizer_replays,
                    active=lambda: bool(self.asan_target or self.ubsan_target),
                ),
                MaintenanceJob(
                    "memory_prune",
                    interval_execs=1000,
                    action=self._check_memory_and_prune,
                    active=lambda: self.prune_corpus_max_memory > 0,
                ),
            ]
        )
        self._minimize_pending = False
        # Set by run()'s broad handler when the loop dies on an unexpected
        # exception. State is still persisted; this only marks the run as
        # incomplete so the summary does not read like a clean stop.
        self._aborted_by_error = False
        self.coverage_report = Path(coverage_report) if coverage_report else None
        self.coverage_log = Path(coverage_log) if coverage_log else None
        if self.coverage_log:
            self.coverage_log.parent.mkdir(parents=True, exist_ok=True)
        self.grammar = grammar
        self.persistent = persistent
        self.net_host = net_host
        self.net_port = net_port
        self.net_proto = net_proto
        self.net_keepalive = net_keepalive
        self.net_settle_ms = net_settle_ms
        self.enable_regex_bomb = enable_regex_bomb
        # Colorization (--colorize): off by default. Costs executions and
        # buys redqueen precision, so it needs a per-target A/B first.
        self.colorize = colorize
        self.colorize_max_execs = colorize_max_execs
        self._colorize_taint_cache: dict[int, object] = {}
        self._colorize_execs = 0
        # Weizz structure tags (--weizz-tags): off by default. Passive
        # collector consumes existing cmplog pairs; field/chunk operators
        # only fire when a seed carries a StructureMap in seed_meta.
        self.weizz_tags = weizz_tags
        self.weizz_tags_max_len = weizz_tags_max_len
        self._weizz_tags_collected = 0
        # FormatFuzzer structural mutators (--formatfuzzer): off by default.
        # Operators self-register on import; is_available gates on this flag
        # plus binary presence. Also enabled by --hail-mary.
        self.formatfuzzer = formatfuzzer
        self.ff_bin_dir = ff_bin_dir
        self.ff_templates = ff_templates
        if formatfuzzer:
            try:
                from fuzzer_tool.core.mutations.formatfuzzer import (
                    register_formatfuzzer_mutators,
                    report_availability,
                )

                templates = None
                if ff_templates:
                    templates = [t.strip() for t in ff_templates.split(",") if t.strip()]
                muts = register_formatfuzzer_mutators(templates=templates, bin_dir=ff_bin_dir)
                # Registration is silent by design (it runs on every import).
                # Say something here, or --formatfuzzer with nothing installed
                # yields operators that never fire and no message at all.
                report_availability(muts)
            except Exception as exc:  # noqa: BLE001
                import logging

                logging.getLogger(__name__).warning("FormatFuzzer registration failed: %s", exc)
        # TSP neighbourhood operators (Phase 1 / C2) — gated availability.
        self.op_span_reverse = op_span_reverse
        self.op_span_relocate = op_span_relocate
        self.op_afl_det = op_afl_det
        # MailConfig | None — novel-crash email notification (see services/sendmail.py)
        self.email_on_crash = email_on_crash
        self.enable_x86_mutator = enable_x86_mutator
        self.enable_arm_mutator = enable_arm_mutator
        self.seed = seed
        random.seed(seed)
        # ── Vectorized random number pool for mutation hotpath ────────
        # Generates random values in batches (one numpy C-level call per
        # batch) instead of per-call Python-level random() invocations.
        #
        # Built here rather than further down with the schedulers: every
        # scheduler that follows Hard Rule 16 takes it as its `rng`, and the
        # MCTS seed schedulers are constructed ~600 lines above where it used
        # to be assigned. It belongs next to `random.seed` anyway -- the three
        # streams start together, the same invariant `_reseed_after_stall`
        # maintains.
        self._rng = RandPool(seed=seed)
        # RandPool holds its OWN np.random.default_rng(seed) Generator, which
        # shares no state with the legacy global np.random.* functions. Nothing
        # in src/ seeded that global, so every np.random draw outside RandPool
        # — qea.py:267,361,364 (observe/mutate amplitudes) and
        # schedulers/op_monte_carlo.py:778,895 (spectral probe vectors) — ran off
        # OS entropy and made --seed non-reproducible whenever QEA or the
        # Monte-Carlo scheduler was active. Seed it here, next to random.seed,
        # so the three streams start together.
        self._seed_global_numpy(seed)
        # GA lifecycle parameters
        self._ga_enabled = ga
        self._ga_pop_size = ga_pop_size
        self._ga_gen_size = ga_gen_size
        self._ga_elite_frac = ga_elite_frac
        self._ga_crossover_rate = ga_crossover_rate
        self._ga_mutation_rate = ga_mutation_rate
        self._ga_tournament_size = ga_tournament_size
        self._ga_speciation_threshold = ga_speciation_threshold
        # QEA coupling magnitudes, exposed so the zero endpoint is an arm.
        self._qea_rotation_angle = qea_rotation_angle
        self._qea_strong_bias = qea_strong_bias
        self._qea_elite_reset = qea_elite_reset
        self._qea_correlation = qea_correlation
        self._qea_correlation_delta = qea_correlation_delta
        self._qea_correlation_max = qea_correlation_max
        self._qea_correlation_sweeps = qea_correlation_sweeps
        self._qea_cooling = qea_cooling
        self._qea_cooling_decay = qea_cooling_decay
        self._qea_cooling_min_angle = qea_cooling_min_angle
        self.ga = None  # Initialized in run() when --ga is set

        # QEA lifecycle
        self._qea_enabled = qea
        self.qea = None  # Initialized in run() when --qea is set

        # Differential fuzzing
        self._diff_target = differential_target
        self._diff_tracker = None
        self._diff_divergences = 0

        # WFC structural generation mode
        self._wfc_enabled = wfc
        from fuzzer_tool.core.wfc_chunks import WFC_MUTATOR

        WFC_MUTATOR.use_wfc = wfc

        # Corpus byte drift vs seeds (core/pool_drift.py); read by init_seed_metadata
        self._use_pool_drift = pool_drift

        # Dictionary token Thompson sampling (core/dirichlet.py): drawn in
        # OperatorEngine.mutate, credited in fuzz_one on new coverage
        self._dict_picker = DirichletPicker(self._rng) if dict_thompson else None

        # Corpus size boost: normal-distribution seed resizing
        self._corpus_boost = corpus_boost
        self._boost_mean = boost_mean
        self._boost_std = boost_std
        self._boost_pad = boost_pad

        # Seed preprocessing: skip/truncate/slide
        self._seed_skip_size = seed_skip_size
        self._seed_truncate_size = seed_truncate_size
        self._seed_slide_size = seed_slide_size
        self._seed_slide_max_seeds = seed_slide_max_seeds

        # Static analysis: profile target for string extraction, function
        # boundaries, input format hints, and call graph structure.
        # Run this BEFORE estimate_map_size so it can reuse the decoded data.
        from fuzzer_tool.core.target_profiler import TargetProfiler

        self._profile = TargetProfiler(target).profile_cached(refresh=self.refresh_profile)

        # Edge bitmap size: use provided value or auto-size from branch density.
        # Pass the profile to skip the redundant full-text disassembly when
        # the profile already has text_size and total_branches.
        if map_size > 0:
            self.map_size = map_size
        else:
            from fuzzer_tool.core.elf import estimate_map_size

            self.map_size = estimate_map_size(target, profile=self._profile)

        # Auto-populate dictionary from the target profile (strings, magic
        # bytes, disassembly constants, literal data-section words, parser
        # token tables). Channel order is stable so corpus discovery does not
        # depend on hash iteration order.
        self._merge_profile_dictionary()

        # Cmplog: comparison tracing via LD_PRELOAD
        self._cmplog = None
        # The most recent execution's own comparison vector, refilled by the
        # drain in fuzz_one and empty whenever cmplog is off.
        self._last_cmp_fired: dict[str, int] = {}
        self._last_cmp_asserted: dict[str, int] = {}
        self._cmplog_skip_counter = 0  # adaptive cmplog collection skip
        # Tri-state: None = auto-detect, True = forced on, False = forced off.
        # Auto-detect resolves here rather than at the direct_lite decision
        # further down, because that site only runs when self._cmplog is
        # already non-None -- i.e. it could refine how cmplog runs, never
        # whether it runs at all.
        # Seed stability calibration (handover item D). n_runs per accepted
        # seed; 0 disables. Opt-in: see _calibrate_seed_stability.
        self._calibrate_stability = int(calibrate_stability or 0)
        # F2: rerun an execution that reported new coverage and keep only what
        # reproduces. See _confirm_new_coverage.
        self._confirm_novelty = bool(confirm_novelty)
        self._confirmed_edges: frozenset[int] | None = None
        self._confirm_stats = {"reruns": 0, "withdrawn": 0, "phantom_ids": 0}
        self._unstable_edges: set[int] = set()
        self._stability_calibrations = 0
        self._cmplog_auto = True  # always auto-detect; no tri-state any more
        # Cmplog is always on by default; detection runs unconditionally to
        # drive the confirmation message and decide whether to build the shim.
        # cmplog=False is accepted for programmatic callers that need it off.
        if cmplog:
            has_cmplog = _detect_cmplog(self.target)
            if has_cmplog:
                print("[*] Cmplog: target is instrumented, enabling comparison tracing")
            else:
                print(
                    "[!] Cmplog: target does not appear to be instrumented; "
                    "shim compilation will be attempted"
                )
            from fuzzer_tool.core.cmplog import CmplogCollector

            self._cmplog = CmplogCollector(
                max_tokens=cmplog_max_tokens,
                max_pairs=cmplog_max_pairs,
                workdir=cmplog_workdir,
                fifo_sink=cmplog_fifo_sink,
                fifo_max_buffered=cmplog_fifo_sink_size,
                debug=self.debug,
                # Per-PC-site counters feed _record_cmp_progress so growth is
                # measured per comparison site, not folded across a family
                # (P0-3).  The shim cost is one hash probe per comparison.
                site_counts=True,
            )
            if self._cmplog.start():
                from fuzzer_tool.core.elf import detect_cmplog_functions

                funcs = detect_cmplog_functions(self.target)
                funcs_str = ",".join(funcs)
                print(f"[*] Cmplog: comparison tracing enabled ({funcs_str})")
                from fuzzer_tool.core.rq_encodings import encoders_summary

                encoders = encoders_summary()
                print(
                    f"[*]   Redqueen encoders: {len(encoders)} ({', '.join(e['name'] for e in encoders)})"
                )
            else:
                print("[!] Cmplog: failed to compile shim, disabling")
                self._cmplog = None

        # self.checksum_learner: constructed by analyzer_registry.wire_all()
        # below (swallow_errors=True there reproduces this analyzer's
        # original try/except-on-construction fail-open behaviour).

        # SMT solver: arithmetic constraint solving on cmplog pairs
        self._smt_solver = None
        self._enable_smt_z3 = enable_smt_z3

        # Path-condition negation. Independent of --enable-smt-z3: that flag
        # selects the modulo-solving strategy for cmplog pairs, whereas this
        # solves for inputs that flip a branch outright.
        self._path_solver = None
        if path_negation:
            from fuzzer_tool.core.path_constraints import PathConstraintSolver, _z3

            if _z3() is None:
                print("[!] Path negation requested but z3 is unavailable — disabled")
            else:
                self._path_solver = PathConstraintSolver()
                print("[*] Path negation: solving for branch-flipping inputs")

        self._mod_solving = mod_solving if enable_smt_z3 else "heuristic"
        if enable_smt_z3:
            from fuzzer_tool.core.smt_solver import Z3Solver

            self._smt_solver = Z3Solver(mod_solving_mode=mod_solving)
            if self._smt_solver._available:
                print(f"[*] SMT solver: modulo solving mode '{mod_solving}'")

                # Trace mode: pre-compute PC→divisor map from static analysis
                if mod_solving == "trace":
                    from fuzzer_tool.core.elf import extract_div_constants

                    targets = multi_targets or [target]
                    div_map: dict[int, int] = {}
                    weak_set: set[int] = set()
                    for t in targets:
                        try:
                            d, w = extract_div_constants(t)
                            div_map.update(d)
                            weak_set.update(w)
                        except Exception:
                            log.debug("Failed to extract DIV constants from %s", t)
                    if div_map or weak_set:
                        from fuzzer_tool.core.smt_solver import (
                            set_pc_divisor_map,
                            set_weak_mod_set,
                        )

                        set_pc_divisor_map(div_map)
                        set_weak_mod_set(weak_set)
                        print(
                            f"[*] SMT solver: loaded {len(div_map)} PC→divisor mappings"
                            f" + {len(weak_set)} weak modulus PCs"
                        )
                    else:
                        print("[!] SMT solver: no DIV/IDIV with known divisor in target(s)")
            else:
                print(
                    "[!] SMT solver: z3-solver not installed — install with: pip install z3-solver"
                )
                self._smt_solver = None

        # Gated on the *request* (enable_smt_z3), not on self._smt_solver.
        # It used to test `self._smt_solver is not None`, but the z3-missing
        # branch above has already set that to None -- so on a machine
        # without z3 this whole block was skipped and _enable_smt_z3 stayed
        # True, leaving the flag claiming an SMT path that has no solver
        # *and* no cmplog behind it. The two conditions are independent and
        # both have to clear the flag.
        if enable_smt_z3 and self._cmplog is None:
            print("[!] SMT solver: --enable-smt-z3 requires --cmplog; disabling SMT path")
            self._smt_solver = None
            self._enable_smt_z3 = False

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

        # Per-target SHM for multi-target mode
        if self.multi_targets and self.use_coverage and not no_shm:
            for t in self.multi_targets:
                try:
                    self._target_shm_covs[t] = ShmCoverage(size=self.map_size)
                except OSError:
                    log.warning("Failed to create SHM for %s, using shared SHM", t)
            if self._target_shm_covs:
                print(f"[*] Multi-target: {len(self._target_shm_covs)} per-target SHM regions")

        # Here rather than beside the disable_aslr() call above: that runs
        # before self.use_coverage is assigned and before the coverage
        # backend is chosen, so acting there would touch --no-coverage and
        # --no-shm runs, neither of which reads a shim edge id. Still before
        # anything executes a target, which is what matters -- the variable
        # is read once per target process.
        self._ensure_ctx_ids_are_exec_stable(list(self.multi_targets or [self.target]))

        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir.mkdir(parents=True, exist_ok=True)

        # Single-file state store (replaces per-component JSON files).
        # Loaded eagerly so all components can fetch their section via get().
        from fuzzer_tool.core.state_store import StateStore

        self._state_store = StateStore(self.corpus_dir, enabled=not no_save_state)
        if self.resume:
            self._state_store.load()
        else:
            # Not merely "don't load": get() lazy-loads on first access, so
            # skipping load() here deferred the read instead of preventing it.
            self._state_store.start_empty()

        self._fluctuation_beta = fluctuation_beta
        self._fluctuation_window = fluctuation_window
        self._fluctuation_requested = fluctuation

        self.corpus: list[bytes] = []
        self.seen_hashes: set[str] = set()
        self.irreplaceable_hashes: set[str] = set()
        self.bloom = BloomFilter(capacity=100_000)
        self.bloom.init_fuzzy(max_recent=200)
        # Executed-input filter.  The mutation space is not uniform: stall
        # recovery, dictionary ops and the deterministic stages collapse onto
        # very short buffers, so the same mutant is handed to the target
        # hundreds of times.  A membership test costs ~1us against a ~1.5ms
        # exec, so re-rolling the mutation on a hit is close to free.
        # Generational: wiped once `capacity` inputs are absorbed, which keeps
        # the realised FP rate at 1e-3 over an unbounded exec stream.
        self._dedup_execs = dedup_execs
        self._seed_calibration = seed_calibration
        self._exec_dedup_backend = exec_dedup_backend
        if exec_dedup_backend == "cuckoo":
            from fuzzer_tool.core.cuckoo import CuckooFilter

            self._exec_bloom = CuckooFilter(capacity=EXEC_BLOOM_CAPACITY)
        elif exec_dedup_backend == "bloom":
            self._exec_bloom = BloomFilter(capacity=EXEC_BLOOM_CAPACITY, error_rate=1e-3)
        else:
            raise ValueError(
                f"unknown exec_dedup_backend {exec_dedup_backend!r}; expected 'bloom' or 'cuckoo'"
            )
        self._dedup_hits = 0
        self._dedup_gaveup = 0
        # Performance novelty (per-edge max hit count). Separate from the
        # coverage signal on purpose -- see ShmCoverage._update_max_counts.
        self._perf_novelty = perf_novelty
        self._perf_novelty_hits = 0
        # Zest validity channel: the harness reports parser rejection with
        # an exit code, and coverage reached on accepted inputs gets its own
        # map. Inert without --reject-code -- see core/validity.py.
        self._validity = ValidityChannel(reject_code)
        self._validity_admits = 0
        # Comparison progress (per-callback max asserted count in a single
        # execution). The same shape as the per-edge maxima above, one
        # channel over: an input that satisfies more comparisons of some
        # family than any input before it got further into the parser, and
        # says so even when it flipped no branch the map had not already
        # seen. That is the regime the cmplog-band operators
        # (magic_byte_search, climb_hill, gradient_descent, condstmt_solve)
        # work in, and the only reward they had was the edge that arrives
        # once the comparison is fully solved -- so seven of eight correct
        # bytes paid exactly nothing.
        # Keyed by callback name and, when the shim's site counters are
        # available, by (callback, pc).  The two key spaces are disjoint.
        self._cmp_max_asserted: dict[str | tuple[str, int], int] = {}
        self._cmp_novelty_hits = 0
        self.crash_hashes: set[str] = set()
        self.crash_sigs: dict[str, int] = {}
        self.crash_frames: dict[str, list[str]] = {}  # sig -> frames for clustering
        # Crash-stop state captured by the ptrace runner (fault address from
        # PTRACE_GETSIGINFO, registers from PTRACE_GETREGS); consumed by
        # corpus_manager.save_crash for the sidecar + signature.
        self._last_fault_addr: int | None = None
        self._last_regs: dict[str, int] = {}
        # Lazy probe: whether ptrace crash triage (re-running direct_lite
        # crashes through the ptrace-attached loader) is usable here.
        self._triage_ok: bool | None = None
        self.exec_count = 0
        self.crash_count = 0
        self.timeout_count = 0
        self.start_time = time.time()
        self.last_report: SanitizerReport | None = None
        self.op_counts: dict[str, int] = {}
        self.op_success: dict[str, int] = {}
        # Selections where the operator could actually have fired -- i.e.
        # its own format sniffer matched the input it was handed. Equal to
        # op_counts for every ungated operator; strictly smaller for a
        # sniffer-gated one on a corpus that does not contain its format.
        # This, not op_counts, is the honest denominator for a success rate
        # in that regime -- paired with op_success_applicable below, which
        # is the numerator over the same selections.
        self.op_applicable: dict[str, int] = {}
        self.op_success_applicable: dict[str, int] = {}
        self.op_edges: dict[str, int] = {}
        # RO/RD classification counts (Du orientation vs history-reversal
        # distinction, docs/handover/handover_RoRd.md Phase C1). Metadata
        # only -- never read back to gate operator selection. Keyed by
        # OrientationClass.value ("ro" / "rd" / "neutral").
        self._ro_rd_edge_counts: dict[str, float] = {"ro": 0.0, "rd": 0.0, "neutral": 0.0}
        self._peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._discovery_execs: array = array("Q")  # exec_count per discovery snapshot
        self._discovery_edges: array = array("Q")  # cumulative edges per snapshot
        self._discovery_timestamps: array = array("d")  # wall-clock timestamp per snapshot
        self._crash_rate_execs: array = array("Q")  # exec_count per crash-rate sample
        self._crash_rate_counts: array = array("Q")  # crash_count per sample
        self._duplicate_reject_count = 0
        self._total_corpus_attempts = 0
        self._pruned_count = 0
        self._exec_baseline = 0
        self._peak_eps = 0.0
        self._total_exec_time = 0.0
        self._replay_budget_ms: float = 0.2  # max 200ms per batch for crash replay
        self._crash_replays: dict[str, list[int]] = {}  # sig -> list of replay return codes
        # sig -> crash base name on disk, so the replay scheduler can open the
        # right file instead of guessing at it by filename prefix (finding #22).
        self._crash_files: dict[str, str] = {}
        # Signature the most recent save_crash() counted its crash under, or
        # None when that crash was not written. Set by CorpusManager.save_crash.
        self._last_crash_signature: str | None = None
        self.replay_n: int = replay_n  # --replay-N: replay each crash N times
        self.asan_target: str | None = asan_target  # --asan-target: ASAN-instrumented variant
        self.ubsan_target: str | None = ubsan_target  # --ubsan-target: UBSAN-instrumented variant
        self._crash_sanitizer_replays: dict[str, dict] = {}  # sig -> {data, asan, ubsan}
        self.crash_blocklist: set[str] = crash_blocklist or set()
        self.crash_allowlist: set[str] = crash_allowlist or set()
        self.save_smaller: bool = save_smaller
        self.honggfuzz: bool = honggfuzz
        self.hw_perf: bool = hw_perf
        self.intel_pt: bool = intel_pt
        self.intel_pt_mode: str = intel_pt_mode
        self.lbr: bool = lbr
        self.lbr_period: int = int(lbr_period or 0)
        self.crash_min_sizes: dict[str, int] = {}  # stack_hash -> min trigger size
        # Honggfuzz power factor stats (for display)
        self._hf_novelty_boosts: int = 0
        self._hf_freshness_boosts: int = 0
        self._hf_fertility_boosts: int = 0
        self._hf_density_boosts: int = 0
        self._hf_entropy_penalties: int = 0
        self._hf_timeout_penalties: int = 0
        self._favored: set[str] = set()

        # self._exec_time_tracker / self._exec_time_anomaly: constructed by
        # analyzer_registry.wire_all() below.

        self._last_child_pid: int | None = None

        self.stats_file = Path(stats_file) if stats_file else None
        self.stats_interval = stats_interval
        # Optional stack heartbeat: a daemon thread writes the main-thread
        # stack to this file every few seconds, so a SIGKILL (uncatchable)
        # still leaves the last executing location on disk.
        self._stack_heartbeat_path = Path(stack_heartbeat) if stack_heartbeat else None
        self._last_stats_exec = 0
        self._eps = 0.0
        # Kalman filter for denoised EPS tracking.
        # Uses 2D RobustKF to handle scheduling jitter / GC pauses / bursty
        # throughput.  The adaptive-R variant learns the actual measurement
        # noise online.  Filtered rate replaces the raw sliding-window in
        # dict pruning, stats-interval calc, etc.
        self._eps_kf = None  # lazy-initialized after first stats tick
        self._last_eps_count = 0  # exec_count at last EPS KF update
        self._last_eps_time = 0.0  # monotonic time at last EPS KF update
        # exec_count at the start of this process (0 for fresh runs, loaded
        # value for --resume).  EPS display and interval math subtract it so
        # resumed runs don't divide the cumulative count by fresh wall time.
        self._resume_baseline_exec = 0

        # Rolling avg-eps samples (one per stats tick). The first ticks often
        # show inflated EPS (bursty warm-up, startup time in the denominator),
        # so the stats line is held until the window fills and the effective
        # stats interval is driven by the window mean, not a single reading.
        self._eps_history: array = array("d")
        self._eps_history_max = 10  # stabilization window (in stats ticks)

        # Bayesian seed quality estimation
        self._seed_quality = BayesianSeedQuality(rng=self._rng)

        # Schedule ablation: per-iteration CSV log of signal contributions
        self._ablation_path = Path(schedule_ablation) if schedule_ablation else None
        self._ablation_file = None
        if self._ablation_path:
            self._ablation_path.parent.mkdir(parents=True, exist_ok=True)
            self._ablation_file = open(self._ablation_path, "w")  # noqa: SIM115
            self._ablation_file.write(
                "iter,seed_idx,seed_hash,fuzz_count,coverage_edges,age_s,"
                "temperature,base_w,burst,penalty,subsumption,diversity,"
                "spatial,mdl,final_w,new_coverage,new_crash,operator\n"
            )
            self._ablation_file.flush()

        # Support multiple markov orders via comma-separated list or single int
        if isinstance(markov_order, str):
            orders = [int(o.strip()) for o in markov_order.split(",")]
        elif isinstance(markov_order, list):
            orders = markov_order
        else:
            orders = [markov_order]
        # LEARNED: Markov smoothing and CEM α refit as Dirichlet MLEs (core/dirichlet.py)
        self._dirichlet_alpha = dirichlet_alpha
        if len(orders) > 1:
            self.markov = MarkovEnsemble(
                orders=orders, blend=markov_blend, rng=self._rng, alpha_mode=dirichlet_alpha
            )
        else:
            self.markov = MarkovChain(order=orders[0], rng=self._rng, alpha_mode=dirichlet_alpha)
        self.markov_generate = markov_generate
        self.markov_trained = False

        # ── Extracted modules ──────────────────────────────────────────
        self._operators = OperatorEngine(self)
        self._seed_picker = SeedPicker(self)
        self._runner = TargetRunner(self)
        self._stats = StatsReporter(self, rng=self._rng)
        self._corpus_manager = CorpusManager(self)
        self._poisson_admission = None  # lazy-init; created on first save_to_corpus()
        # PoissonDiskAdmission holds a reference to _edge_tracker._minhash,
        # which is only fully initialized after corpus init; lazily create it
        # on first admission check so the reference is valid.

        # Intel PT coverage (optional, uninstrumented binaries).  Built
        # before the counters so the banner order matches the flag order.
        self.pt_cov = None
        self._pt_session = None
        if self.intel_pt:
            self._setup_intel_pt()

        # Sampled branch-record coverage (AMD BRS/LbrExtV2, Intel LBR).
        self.branch_cov = None
        self._lbr_session = None
        if self.lbr:
            self._setup_lbr()

        # Hardware performance counters (optional, requires CAP_PERFMON)
        self._perf_counters = None
        self._last_perf_deltas: dict[str, int] = {}
        if self.hw_perf:
            from fuzzer_tool.adapters.perf_event import PerfCounters

            self._perf_counters = PerfCounters()
            if not self._perf_counters.available:
                log.warning("Hardware perf counters not available (needs CAP_PERFMON or root)")
                self._perf_counters = None
                self.hw_perf = False

        # Seed-level energy multiplier: scales mutations_per_input per seed
        self._seed_scorer = SeedScorer(
            schedule=schedule or "base",
            aflgo_cooling=aflgo_cooling,
            t_x_minutes=t_x_minutes,
        )
        self._power_schedule = schedule
        self._last_perf_score = 100.0  # default multiplier (1x)

        # Seed key: computed on demand from corpus manager.  Caching the
        # full bytes object as a dict key pinned every unique mutation in
        # memory between minimization passes, which is the primary cause of
        # the 10 GB RSS growth during long stalls.

        # Per-byte sensitivity tracker (Lyapunov exponent). Constructed here
        # (via the "early" analyzer_registry phase, before
        # _init_seed_metadata) so load_state (resume) can restore
        # sensitivity.json — it crashed with AttributeError otherwise.
        self._use_sensitivity = sensitivity
        from fuzzer_tool.core.analyzer_registry import REGISTRY as _ANALYZER_REGISTRY

        _ANALYZER_REGISTRY.wire_all(self, phase="early")

        # Statistical region profiling (randomness.profile_buffer): labels
        # each window of a seed incompressible / tabular / textual /
        # repetitive and weights byte selection accordingly, so mutation
        # effort moves off compressed payloads and onto offset and length
        # tables. Off by default -- the profile costs ~1 ms per 4 KiB, which
        # is only worth paying on structured targets, and it is cached per
        # seed in OperatorEngine rather than recomputed per mutation.
        self._use_region_profile = region_profile

        # Weighted mutation lineage tree (parent/ops/sites/new-edge weight
        # per seed). Initialised early so the post-metadata rebuild can
        # consume it; rebuilt from persisted seed_meta after metadata init.
        self._use_lineage = lineage
        # Backtracking widens exploration: when a lineage branch stops
        # producing edges, its seeds are penalised geometrically by depth so
        # weight shifts back toward shallow seeds with unexplored siblings.
        # Implies --lineage (needs the tree to know what a branch is).
        self._use_lineage_backtrack = bool(lineage_backtrack and lineage)
        self._lineage_backtrack_decay = 0.7
        self._lineage_backtrack_min_fuzz = 8
        # Weighted local-search Maximum Disjoint Set selection in
        # auto_minimize_corpus: replaces flat top-K-by-score with a
        # value-weighted packing over Jaccard-signature space (see
        # core/mds_local_search.py). Only affects the count-budget path
        # (max_corpus_bytes uses its own knapsack).
        self._use_mds_select = mds_select
        # MCTS seed scheduling walks the lineage genealogy, so it is
        # meaningless without the tree; --mcts implies --lineage.
        self._use_mcts = bool(mcts)
        self._mcts = None
        if self._use_mcts and not lineage:
            lineage = True
            log.info("--mcts implies --lineage (MCTS schedules over the lineage tree)")

        # "alphabeta" arm: same lineage dependency, Thompson-sampling descent (not minimax).
        self._use_alphabeta = bool(alphabeta)
        self._alphabeta = None
        if self._use_alphabeta and not lineage:
            lineage = True
            log.info("--alphabeta implies --lineage (the arm schedules over the lineage tree)")

        self._lineage = None
        if lineage:
            from fuzzer_tool.core.lineage import LineageTree

            self._lineage = LineageTree()
            log.info("Mutation lineage tree enabled")

        if self._use_mcts and self._lineage is not None:
            from fuzzer_tool.core.schedulers.seed_mcts import MCTSSeedScheduler

            self._mcts = MCTSSeedScheduler(rng=self._rng)
            log.info("MCTS seed scheduling enabled")

        if self._use_alphabeta and self._lineage is not None:
            from fuzzer_tool.core.schedulers.seed_mcts import AlphaBetaMCTSSeedScheduler

            self._alphabeta = AlphaBetaMCTSSeedScheduler(rng=self._rng)
            log.info("Alpha-beta (Thompson descent) seed scheduling enabled")

        self._load_corpus()
        self._apply_seed_transforms()
        if self._corpus_boost > 0 and self.corpus:
            self._boost_corpus_sizes()
        self._init_seed_metadata()

        # Rebuild the lineage tree from persisted seed_meta (single source
        # of truth; never re-derived from runs to avoid double-counting).
        if self._use_lineage and self._lineage is not None:
            self._lineage.rebuild_from_meta(self.seed_meta, self._seed_key)
        # Load persisted Markov state from state store; skip retrain if loaded
        markov_data = self._state_store.get("markov")
        if markov_data is not None:
            loaded = True
            self.markov.from_dict(markov_data)
        else:
            loaded = False
        if self.corpus and not loaded:
            self.markov.train_corpus(self.corpus)
        self.markov_trained = self.markov.is_trained()

        # Seed aggregate cache: compute from initial seed_meta
        self._refresh_agg_cache()

        # SkipDet: initialize the skip detector for deterministic stage
        # gating. Default on to match already-shipped behavior; opt out
        # with --no-deterministic if the exec-budget cost (bitflip 1/1
        # alone is 8*len(seed) execs per favored seed) isn't wanted.
        self._skip_detector = (
            SkipDetector(map_size=getattr(self._edge_tracker, "map_size", 65536))
            if deterministic
            else None
        )
        self._det_execs: int = 0

        self.mc_bandit = mc_bandit
        self._sharpe_kelly_blend = sharpe_kelly_blend
        # Bootstrap percolation corpus minimization
        self._use_bootstrap = bootstrap
        self._bootstrap_k = bootstrap_k
        self.mc_cem = mc_cem
        # Opt-in Floyd cycle detection on the MC operator-transition
        # stationary distribution (see MonteCarloScheduler.stationary_distribution).
        # Off by default -- costs extra computation on top of an already
        # non-converged power iteration. Only meaningful alongside mc_bandit
        # or mc_cem, since it's the transition chain those build that gets
        # checked.
        self.mc_cycle_detect = mc_cycle_detect
        self._use_mopt = mopt
        self.mc = (
            MonteCarloScheduler(
                elite_frac=mc_elite_frac,
                refit_interval=mc_refit_interval,
                pairwise_blend=pairwise_blend,
                decay_interval=mc_decay_interval,
                cem_dirichlet_concentration=(
                    _CEM_ALPHA_FALLBACK if dirichlet_alpha is AlphaMode.LEARNED else 0.0
                ),
                rng=self._rng,
            )
            if (mc_bandit or mc_cem or mopt)
            else None
        )
        if self.mc is not None and sharpe_kelly_blend > 0:
            self.mc.set_sharpe_kelly_blend(sharpe_kelly_blend)
        self._mopt = None
        if mopt:
            self._mopt = MOptScheduler(
                n_particles=5,
                window_size=200,
                rng=self._rng,
                marginal_cost_stop_multiplier=mopt_mc_stop_multiplier,
            )
            if mopt_mc_stop_multiplier is not None:
                log.info(
                    "MOpt PSO scheduling enabled (5 particles, window=200, "
                    "mc_stop_multiplier=%.2f)",
                    mopt_mc_stop_multiplier,
                )
            else:
                log.info("MOpt PSO scheduling enabled (5 particles, window=200)")
        self._use_cmaes = cmaes
        self._cmaes = None
        if cmaes:
            self._cmaes = CMAESScheduler(
                # Without an explicit rng, CMAESScheduler now falls back to
                # the shared get_default_rand_pool() singleton (deterministic,
                # no OS entropy -- see core/rand_pool.py), but that fallback
                # is still a *different* stream from the campaign's own
                # self._rng, so a crash found under CMA-ES scheduling would
                # replay identically only if --seed also happened to match
                # the fallback's fixed constant. Passing self._rng explicitly
                # keeps CMA-ES on the one campaign-seeded stream, same as
                # every other scheduler since the Hard Rule 16 migration
                # (8312b15).
                rng=self._rng,
                pop_size=cmaes_pop_size,
                generation_size=cmaes_generation_size,
                step_size=cmaes_step_size,
                elite_frac=cmaes_elite_frac,
            )
            log.info(
                "CMA-ES scheduling enabled (pop=%d, gen=%d, sigma=%.3f)",
                cmaes_pop_size,
                cmaes_generation_size,
                cmaes_step_size,
            )
        self._use_replicator = replicator
        self._seed_strategy = None
        # Operator scheduler that made the last select_op choice (None when
        # the random fallback or stall recovery chose). Set in select_op.
        self._op_selector: str | None = None
        self._seed_strategy_pool: list[str] = []
        self._seed_strategies_used: set[str] = set()
        self._use_boltzmann = boltzmann
        self._tang = None
        if tang:
            from fuzzer_tool.core.schedulers.seed_tang import TangRecommendationScheduler

            self._tang = TangRecommendationScheduler(
                self._rng, rank=tang_rank, refit_interval=tang_refit_interval
            )
        self._kruskal_count = None
        if kruskal_count:
            from fuzzer_tool.core.schedulers.seed_kruskal_count import KruskalCountSeedStrategy

            self._kruskal_count = KruskalCountSeedStrategy(self._rng, self._profile)
        # Byte-content entropy seed arms. Both score the corpus the picker
        # hands them and keep no target state, so they cost nothing when the
        # Elo pool does not select them.
        self._entropy_kl = None
        if entropy_kl:
            from fuzzer_tool.core.schedulers.seed_entropy_kl import EntropyKLSeedStrategy

            self._entropy_kl = EntropyKLSeedStrategy(self._rng)
        self._entropy_zscore = None
        if entropy_zscore:
            from fuzzer_tool.core.schedulers.seed_entropy_zscore import (
                EntropyZScoreSeedStrategy,
            )

            self._entropy_zscore = EntropyZScoreSeedStrategy(
                self._rng, target_z=entropy_zscore_target
            )
        self._entropy_deviation = None
        if entropy_deviation:
            from fuzzer_tool.core.schedulers.seed_entropy_deviation import (
                EntropyDeviationSeedStrategy,
            )

            self._entropy_deviation = EntropyDeviationSeedStrategy(self._rng)
        # Leave-one-out pooled-entropy arm (entropy §5, first step)
        self._entropy_loo = None
        if entropy_loo:
            from fuzzer_tool.core.schedulers.seed_entropy_loo import EntropyLOOSeedStrategy

            self._entropy_loo = EntropyLOOSeedStrategy(self._rng)
        # Unlike the three siblings above, this one has to observe every
        # corpus admission to assign credit (see the module docstring), not
        # just the ones where it happens to be picked -- so it costs a small
        # O(1) update per admission even when the Elo pool never selects it.
        self._entropy_gradient = None
        if entropy_gradient:
            from fuzzer_tool.core.schedulers.seed_entropy_gradient import (
                EntropyGradientSeedStrategy,
            )

            self._entropy_gradient = EntropyGradientSeedStrategy(
                self._rng, decay=entropy_gradient_decay
            )
        # Matrix arms (seed_residual + op_credit) share one canonical edge space,
        # one refit cadence and one preflight gate: see core/edge_matrix.py and
        # docs/handover/handover_edge_id_axis_2026-09-18.md (P3-3, P3-4). Both are
        # off by default and gated on a paired benchmark, not on this wiring.
        self._matrix_substrate = None
        self._seed_residual = None
        if seed_residual or op_credit or shaped_reward:
            from fuzzer_tool.core.edge_matrix import MatrixSubstrate

            self._matrix_substrate = MatrixSubstrate(
                target=getattr(self, "target", None),
                use_coverage=getattr(self, "use_coverage", True),
                ptrace=getattr(self, "ptrace_cov", None) is not None
                or bool(getattr(self, "use_ptrace", False)),
            )
        if seed_residual:
            from fuzzer_tool.core.schedulers.seed_residual import ResidualSeedScheduler

            self._seed_residual = ResidualSeedScheduler(
                self._rng, self._matrix_substrate, outcome_fn=self._seed_residual_outcomes
            )
            log.info("seed_residual enabled")
        # Strata arms (docs/handover/handover_strata_schedulers_2026-09-19.md
        # §3.2-3.4): one EdgeLedger of confirmed ids folded into families,
        # shared by the seed arm and the op arm. Off by default; unmeasured.
        self._edge_ledger = None
        self._seed_strata = None
        self._strata_bytes: dict[str, bytes] = {}
        self._strata_live_len = -1
        if strata or op_strata:
            self._build_strata(strata)
        # Seed-arena canary: deliberately worst-in-class seed scheduler, the
        # _pick_seed_elo counterpart of op_canary (see
        # core/schedulers/seed_canary.py). Only meaningful alongside --elo,
        # which is what ranks it against the rest of the seed-strategy pool.
        self._use_seed_canary = seed_canary_scheduler
        self._seed_canary = None
        if seed_canary_scheduler:
            from fuzzer_tool.core.schedulers.seed_canary import SeedCanaryScheduler

            self._seed_canary = SeedCanaryScheduler()
            log.info("Seed-canary scheduling enabled (deliberately worst-in-class)")
        # Seed-arena round robin: deterministic cycling over the corpus, the
        # _pick_seed_elo counterpart of op_round_robin (see
        # core/schedulers/seed_round_robin.py). Unlike seed_canary it needs
        # no arbiter, so it is also reachable directly from pick_seed()'s
        # no-elo fallback chain -- the same standalone treatment round_robin
        # already gets on the operator side.
        self._use_seed_round_robin = seed_round_robin_scheduler
        self._seed_round_robin = None
        if seed_round_robin_scheduler:
            from fuzzer_tool.core.schedulers.seed_round_robin import SeedRoundRobinScheduler

            self._seed_round_robin = SeedRoundRobinScheduler()
            log.info("Seed round-robin scheduling enabled")
        # LST override: no seed waits more than lst_revisit seconds between
        # picks (SeedPicker._pick_lst_seed); last_picked is stamped in _pick_seed.
        self._lst_revisit = max(0.0, float(lst_revisit))
        self._lst_next_check = 0.0
        # Position selection (core/schedulers/pos_*.py): burn-front proposer
        # and the Elo arena that arbitrates it against the other proposers.
        self._burn_front = None
        # The arena always fields burn-front: it is the one proposer that
        # exists only as a position scheduler, and the arena is where its
        # rating against uniform gets measured.
        if burn_front or position_arena:
            from fuzzer_tool.core.schedulers.pos_burn_front import BurnFrontPositionScheduler

            self._burn_front = BurnFrontPositionScheduler(self._rng)
            log.info("Burn-front position scheduling enabled")
        self._use_position_arena = position_arena
        self._position_arena = None
        if position_arena:
            # The ctor parameter, not self._use_elo: that attribute is only
            # assigned ~500 lines further down, so reading it here raised
            # AttributeError and --position-arena could never construct.
            if not elo:
                log.warning("--position-arena has no effect without --elo")
            self._position_arena = PositionArena(
                self,
                region_fn=self._operators._region_weighted_position,
                burn_front=self._burn_front,
            )
            log.info("Position arena enabled (Elo over pos_ strategies)")
        self._use_ecofuzz = ecofuzz
        self._ecofuzz_mc_penalty_multiplier = ecofuzz_mc_penalty_multiplier
        self._metropolis = metropolis
        self._op_dispatch = self._build_dispatch()
        self._replicator = None
        if replicator:
            self._replicator = ReplicatorScheduler(
                window_size=200,
                learning_rate=0.1,
                rng=self._rng,
                marginal_cost_stop_multiplier=replicator_mc_stop_multiplier,
            )
            if replicator_mc_stop_multiplier is not None:
                log.info(
                    "Replicator dynamics scheduling enabled (window=200, eta=0.1, "
                    "mc_stop_multiplier=%.2f)",
                    replicator_mc_stop_multiplier,
                )
            else:
                log.info("Replicator dynamics scheduling enabled (window=200, eta=0.1)")
        # EXP3 adversarial bandit
        self._use_exp3 = exp3
        self._exp3 = None
        if exp3:
            self._exp3 = Exp3Scheduler(gamma=exp3_gamma, rng=self._rng)
            log.info("EXP3 adversarial bandit enabled (gamma=%.2f)", exp3_gamma)
        # EXP4 expert-advice bandit over operator categories
        self._use_exp4 = exp4
        self._exp4 = None
        if exp4:
            self._exp4 = Exp4Scheduler(gamma=exp4_gamma, rng=self._rng)
            log.info("EXP4 expert-advice bandit enabled (gamma=%.2f)", exp4_gamma)
        # SLOPT (core/slopt.py): one operator per round, applied 2**t times,
        # t learned per (seed-size group, operator).
        self._use_slopt = slopt
        self._slopt = SloptBatchBandit(rng=self._rng) if slopt else None
        # (op, seed_len, exponent) drawn this round; None when no arm was
        # pulled (deterministic stage, SLOPT off).
        self._last_slopt_arm = None
        # Epsilon-greedy with annealing
        self._use_eps_greedy = eps_greedy
        self._eps_greedy = None
        if eps_greedy:
            self._eps_greedy = EpsilonGreedyScheduler(
                epsilon_0=eps_greedy_epsilon0, decay=eps_greedy_decay, rng=self._rng
            )
            log.info(
                "Epsilon-greedy enabled (epsilon0=%.2f, decay=%.4f)",
                eps_greedy_epsilon0,
                eps_greedy_decay,
            )
        # Softmax and TopK schedulers
        self._use_softmax = softmax
        self._use_topk = use_topk
        self._softmax = None
        self._topk = None
        if softmax:
            self._softmax = SoftmaxScheduler(tau=softmax_tau, rng=self._rng)
            log.info(
                "Softmax scheduler enabled (tau=%.2f)",
                softmax_tau,
            )
        if use_topk:
            self._topk = TopKScheduler(k=topk_k, rng=self._rng)
            log.info("TopK scheduler enabled (k=%d)", topk_k)
        # Hierarchical bandit
        self._use_hierarchical = hierarchical_bandit
        self._hierarchical = None
        if hierarchical_bandit:
            self._hierarchical = HierarchicalBanditScheduler(rng=self._rng)
            log.info(
                "Hierarchical bandit enabled (%d categories)",
                len(HierarchicalBanditScheduler.CATEGORIES),
            )
        # GP-UCB bandit
        self._use_gp_ucb = gp_ucb
        self._gp_ucb = None
        if gp_ucb:
            self._gp_ucb = GPUCBScheduler(length_scale=gp_length_scale, beta=gp_beta)
            log.info("GP-UCB enabled (l=%.2f, beta=%.2f)", gp_length_scale, gp_beta)

        # BO-GP-UCB bandit (Expected Improvement version)
        self._use_bo_gp_ucb = bo_gp_ucb
        self._bo_gp_ucb = None
        if bo_gp_ucb:
            self._bo_gp_ucb = BOGPUCBScheduler(
                length_scale=bo_gp_length_scale,
                noise=bo_gp_noise,
                rng=self._rng,
            )
            log.info("BO-GP-UCB enabled (l=%.2f, noise=%.4f)", bo_gp_length_scale, bo_gp_noise)

        # Recency-weighted UCB pair (Garivier & Moulines). Both take the
        # shared RandPool per Hard Rule 16 so --seed reproduces the campaign.
        self._use_ducb = ducb
        self._ducb = None
        if ducb:
            self._ducb = DUCBScheduler(gamma=ducb_gamma, rng=self._rng)
            log.info("D-UCB enabled (gamma=%.5f)", ducb_gamma)

        self._use_swucb = swucb
        self._swucb = None
        if swucb:
            self._swucb = SWUCBScheduler(window=swucb_window, rng=self._rng)
            log.info("SW-UCB enabled (window=%d)", swucb_window)

        # KL-UCB variants: same structure as D-UCB/SW-UCB but with
        # the Bernoulli KL upper bound as the confidence width,
        # unconditionally.
        self._use_kl_ducb = kl_ducb
        self._kl_ducb = None
        if kl_ducb:
            self._kl_ducb = KL_DUCBScheduler(gamma=kl_ducb_gamma, rng=self._rng)
            log.info("KL-D-UCB enabled (gamma=%.5f)", kl_ducb_gamma)

        self._use_kl_swucb = kl_swucb
        self._kl_swucb = None
        if kl_swucb:
            self._kl_swucb = KL_SWUCBScheduler(window=kl_swucb_window, rng=self._rng)
            log.info("KL-SW-UCB enabled (window=%d)", kl_swucb_window)

        # Combinatorial UCB: the only scheduler here that models the round's
        # operator stack as one superarm rather than N independent pulls.
        self._use_cucb = cucb
        self._cucb = None
        if cucb:
            self._cucb = CUCBScheduler(gamma=cucb_gamma, rng=self._rng)
            log.info("CUCB enabled (gamma=%.5f)", cucb_gamma)

        # CUSUM-UCB (Liu, Lee & Shroff 2018): change-point detection instead
        # of D-UCB/SW-UCB's continuous forgetting -- see op_cusum_ucb.py for why
        # both approaches earn a place here.
        self._use_cusum_ucb = cusum_ucb
        self._cusum_ucb = None
        if cusum_ucb:
            self._cusum_ucb = CUSUM_UCBScheduler(
                m=cusum_ucb_m,
                epsilon=cusum_ucb_epsilon,
                h=cusum_ucb_h,
                xi=cusum_ucb_xi,
                rng=self._rng,
            )
            log.info(
                "CUSUM-UCB enabled (m=%d, epsilon=%.3f, h=%.2f, xi=%.2f)",
                cusum_ucb_m,
                cusum_ucb_epsilon,
                cusum_ucb_h,
                cusum_ucb_xi,
            )

        # FEWA (Seznec et al. 2019): windowed elimination for arms whose own
        # yield rots with their own pull count, instead of an environment-wide
        # shift -- see op_fewa.py for how this differs from D-UCB/CUSUM-UCB.
        self._use_fewa = fewa
        self._fewa = None
        if fewa:
            self._fewa = FEWAScheduler(
                alpha=fewa_alpha,
                max_window=fewa_max_window,
                rng=self._rng,
            )
            log.info(
                "FEWA enabled (alpha=%.3f, max_window=%d)",
                fewa_alpha,
                fewa_max_window,
            )

        # Follow Perturbed Leader: perturb-and-select bandit with decaying
        # perturbation schedule for stochastic bandit convergence.
        self._use_fpl = fpl
        self._fpl = None
        if fpl:
            self._fpl = FPLScheduler(epsilon=fpl_epsilon, rng=self._rng)
            log.info("FPL enabled (epsilon=%.2f)", fpl_epsilon)

        # Gradient / softmax bandit (Boltzmann exploration). Preference
        # weights updated by the classic REINFORCE-style rule with optional
        # baseline and temperature annealing. Off by default and Elo-only
        # (see core/schedulers/op_gradient.py's module docstring): it is a
        # legitimate stationary control arm but has no non-stationary
        # forgetting mechanism, and the tuning that keeps it converging
        # reliably even in the stationary case (alpha, floor) was found by
        # testing, not derived -- same discipline as op_katz/op_tang below.
        self._use_gradient = gradient
        self._gradient = None
        if gradient:
            self._gradient = GradientBanditScheduler(
                alpha=gradient_alpha,
                temperature=gradient_temperature,
                temp_decay=gradient_temp_decay,
                min_temperature=gradient_min_temperature,
                floor=gradient_floor,
                rng=self._rng,
            )
            log.info(
                "Gradient bandit enabled (alpha=%.3f, temp=%.2f, decay=%.4f, floor=%.3f)",
                gradient_alpha,
                gradient_temperature,
                gradient_temp_decay,
                gradient_floor,
            )

        # Corral: log-barrier OMD over the operator arms with
        # importance-weighted losses. Off by default and Elo-only, for a
        # measured reason rather than by analogy -- see
        # core/schedulers/op_corral.py. On the stationary convergence harness it
        # is solid (best-arm tail share min 0.924 over 40 seeds, regret slope
        # max 0.563), but on DecayingBest its recovery is seed-fragile
        # (best_late share min 0.006, median 0.777 over 12 seeds): the
        # doubling trick raises a starved arm's learning rate without getting
        # it drawn again. That is the same profile that moved gradient back
        # out of the fallback chain, so it must not become a campaign's
        # silent selector without --elo.
        self._use_corral = corral
        self._corral = None
        if corral:
            self._corral = CorralScheduler(eta=corral_eta, rng=self._rng)
            log.info("Corral (log-barrier OMD) enabled (eta=%.2f)", corral_eta)

        # Whittle index (restless-bandit index policy). Off by default and
        # Elo-only (see core/schedulers/op_whittle.py's module docstring): the
        # passive_decay restless-drift assumption is an unmeasured guess
        # pending measurement via the ablation CSV's operator column
        # (docs/handover/handover_non_ucb_schedulers_2026-09-13.md §6), and
        # this has not been run against the convergence harness yet --
        # same discipline as op_katz/op_tang/gradient above.
        self._use_whittle = whittle
        self._whittle = None
        if whittle:
            self._whittle = WhittleIndexScheduler(
                n_states=whittle_n_states,
                gamma=whittle_gamma,
                passive_decay=whittle_passive_decay,
                floor=whittle_floor,
                recompute_batch=whittle_recompute_batch,
                rng=self._rng,
            )
            log.info(
                "Whittle index scheduler enabled (n_states=%d, gamma=%.2f, "
                "passive_decay=%.3f, floor=%.3f, recompute_batch=%d)",
                whittle_n_states,
                whittle_gamma,
                whittle_passive_decay,
                whittle_floor,
                whittle_recompute_batch,
            )

        # Successive elimination / racing: prune arms whose UCB falls
        # below the best LCB. Deterministic given the observation stream.
        self._use_successive_elim = successive_elim
        self._successive_elim = None
        if successive_elim:
            self._successive_elim = SuccessiveEliminationScheduler(
                delta=successive_elim_delta,
                min_pulls=successive_elim_min_pulls,
                reopen_interval=successive_elim_reopen,
                rng=self._rng,
            )
            log.info(
                "Successive elimination enabled (delta=%.3f, min_pulls=%d, reopen=%d)",
                successive_elim_delta,
                successive_elim_min_pulls,
                successive_elim_reopen,
            )

        # Katz centrality over the operator discovery-transition graph.
        # Off by default: see core/schedulers/op_katz.py's module docstring
        # for the empirical caveat before enabling this on a real campaign.
        self._use_op_katz = op_katz
        self._op_katz = None
        if op_katz:
            from fuzzer_tool.core.schedulers.op_katz import OpKatzScheduler

            self._op_katz = OpKatzScheduler(
                rng=self._rng,
                alpha_fraction=op_katz_alpha_fraction,
                # Badness-indexed exploration floor (P1 of
                # docs/action_plan_compositional_stability.md): when the
                # corpus is SUBCRITICAL (stalled), the floor rises toward
                # max_explore_floor instead of staying pinned at the
                # static default. See _current_scheduling_badness and
                # core/badness_floor.py.
                badness_fn=self._current_scheduling_badness,
            )
            log.info("op_katz enabled (alpha_fraction=%.2f)", op_katz_alpha_fraction)

        # Kuramoto phase-coherence bandit over the operator discovery-
        # transition graph. Off by default: same unproven-exploratory-arm
        # posture as op_katz/op_tang above -- see
        # core/schedulers/op_kuramoto.py's module docstring for what is and
        # isn't established empirically before enabling this on a real
        # campaign.
        self._use_op_kuramoto = op_kuramoto
        self._op_kuramoto = None
        if op_kuramoto:
            from fuzzer_tool.core.schedulers.op_kuramoto import OpKuramotoScheduler

            self._op_kuramoto = OpKuramotoScheduler(
                rng=self._rng,
                k=op_kuramoto_k,
                omega_scale=op_kuramoto_omega_scale,
                dt=op_kuramoto_dt,
                steps_per_batch=op_kuramoto_steps_per_batch,
                recompute_batch=op_kuramoto_recompute_batch,
                # Same badness-indexed floor as op_katz above.
                badness_fn=self._current_scheduling_badness,
            )
            log.info(
                "op_kuramoto enabled (k=%.2f, omega_scale=%.2f, dt=%.3f, "
                "steps_per_batch=%d, recompute_batch=%d)",
                op_kuramoto_k,
                op_kuramoto_omega_scale,
                op_kuramoto_dt,
                op_kuramoto_steps_per_batch,
                op_kuramoto_recompute_batch,
            )

        # Tang's low-rank recommender over the operator x edge matrix. Off
        # by default: see core/op_edge_tracker.py's module docstring for
        # the empirical caveat before enabling this on a real campaign.
        self._use_op_tang = op_tang
        self._op_tang = None
        if op_tang:
            from fuzzer_tool.core.schedulers.op_tang import OpTangScheduler

            self._op_tang = OpTangScheduler(
                rng=self._rng, rank=op_tang_rank, refit_interval=op_tang_refit_interval
            )
            log.info(
                "op_tang enabled (rank=%d, refit_interval=%d)",
                op_tang_rank,
                op_tang_refit_interval,
            )

        # Kruskal-count coupling over the operator jump graph. Off by
        # default: see core/schedulers/op_kruskal_count.py's module
        # docstring -- unproven exploratory arm, same posture as op_katz
        # and op_tang above.
        self._use_op_kruskal_count = op_kruskal_count
        self._op_kruskal_count = None
        if op_kruskal_count:
            from fuzzer_tool.core.schedulers.op_kruskal_count import (
                OpKruskalCountScheduler,
            )

            self._op_kruskal_count = OpKruskalCountScheduler(rng=self._rng)
            log.info("op_kruskal_count enabled")

        # Stratified Thompson over (op, family) cells (strata §3.4). Off by
        # default, Elo-only; see core/schedulers/op_strata.py.
        self._use_op_strata = op_strata
        self._op_strata = None
        if op_strata:
            from fuzzer_tool.core.schedulers.op_strata import OpStrataScheduler

            self._op_strata = OpStrataScheduler(rng=self._rng)
            log.info("op_strata enabled")

        # Categorical TPE (BO-3): l/g density ratio over operators. Off by
        # default, Elo-only; see core/schedulers/op_tpe.py.
        self._use_op_tpe = op_tpe
        self._op_tpe = None
        if op_tpe:
            from fuzzer_tool.core.schedulers.op_tpe import OpTPEScheduler

            self._op_tpe = OpTPEScheduler(rng=self._rng)
            log.info("op_tpe enabled")

        # Operator credit on canonical edge classes (P3-3): the reward is the
        # change, the selector is a stock Thompson. Off by default; leaves the
        # ballot while the preflight gate is closed. Same unproven-arm posture as
        # op_tang above.
        self._use_op_credit = op_credit
        self._op_credit = None
        if op_credit:
            from fuzzer_tool.core.schedulers.op_credit import OpCreditScheduler

            self._op_credit = OpCreditScheduler(self._rng, self._matrix_substrate)
            log.info("op_credit enabled")

        # Reward shaping on the same canonical classes, but for EVERY scheduler's
        # reward rather than for one arm's posterior: a round whose new edges are
        # one duplicate chain pays 1/n instead of n (F10). Independent of
        # --op-credit on purpose -- it is the cheapest single-variable A/B the
        # edge-id handover names, and mixing it with a selector change would make
        # the paired run answer two questions at once.
        self._shaped_reward = bool(shaped_reward)
        # Lower clamp on the factor (--shaped-reward-floor). 0.0 is the faithful
        # form; it is a knob because the two ways the factor collapses (a long
        # duplicate chain paying 1/n, a derived-only round paying 0 were `derived`
        # ever filled -- P1-2 says it is not) are a design bet the handover states
        # and nothing has measured.
        self._shaped_reward_floor = float(shaped_reward_floor)
        self._shaped_reward_rounds = 0
        self._shaped_reward_gated = 0
        self._shaped_reward_factor_sum = 0.0
        if shaped_reward:
            log.info("shaped_reward enabled (op_rewards scaled by class credit)")

        # Reward shaping from the continuum's pressure field: a discovery's
        # reward is scaled by how scarce the territory it landed in was
        # (mean pressure of the edges co-hit alongside it), rather than the
        # constant surprisal_weight every round gets today. Composes with
        # --shaped-reward (both are pure multiplicative factors on the same
        # op_rewards list) but is its own paired A/B question -- see
        # Fuzzer._continuum_reward_shape.
        self._continuum_reward = bool(continuum_reward)
        self._continuum_reward_floor = float(continuum_reward_floor)
        self._continuum_reward_rounds = 0
        self._continuum_reward_neutral = 0
        self._continuum_reward_factor_sum = 0.0
        if continuum_reward:
            log.info("continuum_reward enabled (op_rewards scaled by frontier pressure)")

        # Consolidated: flat Thompson with a category-shrunk prior and capped
        # evidence -- the single learner meant to replace the Elo portfolio
        # (see core/schedulers/op_consolidated.py for the measurements).
        self._use_consolidated = consolidated
        self._consolidated = None
        if consolidated:
            self._consolidated = ConsolidatedScheduler(rng=self._rng)
            log.info("Consolidated operator scheduler enabled")

        # MOSS: UCB whose exploration bonus ends at an arm's fair share t/K,
        # built for many low-yield operators (see core/schedulers/op_moss.py).
        self._use_moss = moss
        self._moss = None
        if moss:
            self._moss = MOSSScheduler(gamma=moss_gamma, rng=self._rng)
            log.info("MOSS enabled (gamma=%.5f)", moss_gamma)

        # Round-robin: deterministic baseline. --seed should reproduce
        # exactly, so no RandPool is used here -- the cycling order is
        # the registration order, fully driven by operator init.
        self._use_round_robin = round_robin
        self._round_robin = None
        if round_robin:
            self._round_robin = RoundRobinScheduler()
            log.info("Round-robin operator scheduling enabled")

        # Canary: deliberately worst-in-class operator scheduler. Fed the
        # same record(op, success, weight) signal as every real scheduler
        # in the pool, it always argmin-selects instead of argmax-selects
        # (see core/schedulers/op_canary.py). Only meaningful alongside --elo,
        # which is what actually ranks it against the rest of the pool.
        self._use_canary = canary_scheduler
        self._canary = None
        if canary_scheduler:
            self._canary = CanaryScheduler()
            log.info("Canary operator scheduling enabled (deliberately worst-in-class)")

        self._use_contextual = contextual
        self._contextual = None
        if contextual:
            from fuzzer_tool.services.operators import CONTEXT_DIM

            self._contextual = ContextualLinUCBScheduler(
                dim=CONTEXT_DIM, alpha=contextual_alpha, lambda_reg=contextual_lambda
            )
            log.info(
                "Contextual LinUCB enabled (dim=%d, alpha=%.2f, lambda=%.2f)",
                CONTEXT_DIM,
                contextual_alpha,
                contextual_lambda,
            )

        # C2UCB (Qin, Chen & Zhu 2014): CUCB's superarm/semi-bandit credit
        # assignment fused with LinUCB's per-arm context -- see op_c2ucb.py for
        # why neither half alone is enough, and for the documented context-
        # dilution limitation when _track_op_effect is off.
        self._use_c2ucb = c2ucb
        self._c2ucb = None
        if c2ucb:
            from fuzzer_tool.services.operators import CONTEXT_DIM

            self._c2ucb = C2UCBScheduler(
                dim=CONTEXT_DIM,
                alpha=c2ucb_alpha,
                lambda_reg=c2ucb_lambda,
                min_out_rounds=c2ucb_min_out_rounds,
            )
            log.info(
                "C2UCB enabled (dim=%d, alpha=%.2f, lambda=%.2f, min_out_rounds=%.1f)",
                CONTEXT_DIM,
                c2ucb_alpha,
                c2ucb_lambda,
                c2ucb_min_out_rounds,
            )
        # Running mean/stddev of log1p(seed size), updated in
        # corpus_manager.save_to_corpus(). Feeds the contextual scheduler's
        # "position in corpus size distribution" feature via the normal CDF
        # of that log axis, instead of sorting the whole corpus on every
        # mutation. On the *log* axis: seed sizes are right-skewed, so a
        # Gaussian percentile taken on raw bytes is wrong by ~0.12 on
        # average against the empirical percentile, and by ~0.001 here.
        # Distinct from corpus_manager's _seed_size_moments, which tracks
        # raw bytes precisely because its consumer (the bloat warning)
        # wants the raw right-tail skewness a log would flatten away.
        self._corpus_log_size_stats = RunningMoments()

        self._use_shapley = shapley
        self._shapley = ShapleyAttribution(n_samples=100, window_size=500) if shapley else None
        self._use_bayesian = bayesian
        self._use_mi = mi_guided
        self._mi = (
            # Cap tracked positions: max_len auto-grows to 65536 and the MI
            # joint is positions x 256 byte values x MAX_EDGES_PER_CELL cells —
            # unbounded positions is a multi-GB memory blowup.
            MutualInformationTracker(
                max_positions=min(max_len, MI_MAX_POSITIONS), min_observations=50
            )
            if mi_guided
            else None
        )
        # Load persisted MI state from state store (resume-gated — an oversized
        # mi.json otherwise becomes a multi-GB object tree at every startup)
        if self._use_mi and self._mi and self.resume:
            mi_data = self._state_store.get("mi")
            if mi_data is not None:
                self._mi.from_dict(mi_data)
                log.info(
                    "MI tracker loaded from state store (%d positions)", self._mi.max_positions
                )

        self._use_renyi_weight = renyi_weight
        self._use_transfer_entropy = transfer_entropy
        self._te_byte_edges: dict[int, dict[int, int]] = {}  # pos → {edge: count}
        self._te_causal_version = 0  # bumped per causal-map update; keys the phase-lock memo
        self._use_occupation = occupation
        self._use_causal_sector = causal_sector

        # Gating flags read by analyzer_registry specs below (elo, garch,
        # continuum, format_learner, corpus_compression, distance, trace all
        # look these up off self via `available(f)` rather than taking them
        # as factory arguments -- see the registry module docstring).
        self._use_elo = elo
        self._use_garch = garch
        self._use_continuum = continuum
        self._use_pll = pll
        # Closed-loop temperature control (--temperature-control). Read by
        # the analyzer registry's temperature_control spec, which builds
        # self._temp_controller. Off by default: the sign and magnitude of
        # d(discovery rate)/d(temperature) are unmeasured, and if that
        # derivative is near zero the loop cannot work at all -- see
        # docs/handover/handover_control_theory_loops_2026-09-12.md §5.
        self._use_temp_control = temp_control
        self._temp_setpoint_fraction = temp_setpoint_fraction
        self._temp_reference_rate = temp_reference_rate
        self._learn_format_requested = learn_format
        self._corpus_ppmd_requested = corpus_ppmd
        self._corpus_quasiperiodicity_requested = corpus_quasiperiodicity
        self._distance_targets = targets
        self._use_cfg_cache = use_cfg_cache
        self._gate_bonus = gate_bonus
        self._trace_crashes_requested = trace_crashes

        # Crash MI tracker, length-edge tracker, transfer entropy, occupation,
        # causal-sector, structure function, fluctuation tracking, execution-time
        # tracking, frameshift, the coverage-regime cluster (csd /
        # coverage_homogeneity / garch / continuum / coverage_regime), format
        # learner, corpus PPMD compression, Elo, directed-distance, crash
        # tracing, and the checksum learner are all constructed here in one
        # pass — see core/analyzer_registry.py, the single source of truth
        # for which analyzers exist and what gates each one.
        from fuzzer_tool.core.analyzer_registry import REGISTRY as _ANALYZER_REGISTRY

        _ANALYZER_REGISTRY.wire_all(self)

        # self._frameshift: constructed by analyzer_registry.wire_all() above.
        self._last_ops_used: list[str] = []
        # Subset of _last_ops_used that actually changed the buffer. Set by
        # OperatorEngine.mutate() when _track_op_effect is on; consumed by
        # _record_outcome() to keep no-op operators out of the winner set.
        self._last_ops_effective: set[str] = set()
        # Bitmask of havoc sub-mutations applied this round, set by
        # OperatorEngine._apply_single_mutation and consumed by
        # _record_outcome(). Havoc's inner loop has no visibility into the
        # coverage verdict at mutation time, so credit is deferred here the
        # same way it is for top-level operators.
        self._last_havoc_subops: int = 0
        self._last_ops_with_sites: list[tuple[str, int]] = []
        self._last_op_costs: dict[str, float] = {}
        # EMA of wall-clock seconds per call, per operator. Populated in
        # OperatorMixin.mutate(). Used to convert bandit rewards from
        # edges-per-selection to edges-per-unit-time so expensive operators
        # (gradient_descent, condstmt_solve, path_negate, crc_learn) don't
        # get rated on the same scale as bit_flip.
        self._op_time_ema: dict[str, float] = {}
        # Per-operator attempt and decline counts, campaign-cumulative (not
        # reset per round like _last_op_costs). A decline is an operator that
        # was selected and had nothing to work on: input did not parse, no
        # constraint solved, no candidate site. They used to fall through to
        # havoc under the declining operator's name, which made the
        # effectiveness signal credit them for havoc's work -- see
        # OperatorMixin._op_declined. The ratio is what `--stats` reports and
        # what tells you a format-aware operator is never actually reaching
        # its format.
        self._op_attempts: dict[str, int] = {}
        self._op_declines: dict[str, int] = {}
        self._last_new_edge_count = 0
        self._last_hamming_distance: int = -1
        self._last_mutation_offset: int = 0

        # self._csd / self._homogeneity / self._homogeneity_col_cumulative /
        # self._garch / self._continuum / self._continuum_adjacency /
        # self._continuum_graph_tick / self._regime: constructed by
        # analyzer_registry.wire_all() above (csd, coverage_homogeneity,
        # garch, continuum, coverage_regime specs, in that dependency order).

        # self._structure_fn / self._last_structure_edge_count: constructed by
        # analyzer_registry.wire_all() above, alongside crash_mi,
        # length_tracker, transfer_entropy, and fluctuation.

        # BH-corrected view across the dispersion/Ljung-Box tests that all
        # run on the same per-tick delta series -- see
        # core/multiple_testing.py. Display-only; populated each stats
        # tick, empty until the first one.
        self._last_dispersion_corrections: list = []

        # ── Running aggregate cache for seed metadata ──────────────────
        # Avoids O(n·m) recomputation of corpus-wide sums every iteration.
        # Updated by delta in fuzz_one() and invalidated when the corpus
        # structure changes (add/remove/replace seeds).
        self._cached_total_time: float = 0.0
        self._cached_total_fuzz: int = 0
        # Executions credited to _cached_total_time.  Not _cached_total_fuzz:
        # the initial seed replay in run() bumps fuzz_count without timing
        # anything, so the two diverge by the corpus size on every campaign.
        self._cached_cost_samples: int = 0
        self._cached_total_edges: int = 0
        self._cached_mean_log_n_fuzz: float = 0.0
        self._agg_cache_valid: bool = False

        # ── Dictionary scratch buffer (vectorized choice) ─────────────
        # Refilled in mutate() via one randint_list call; consumed by
        # dict-aware operators instead of calling random.choice(f.dictionary).
        self._dict_scratch: list[int] = []
        self._dict_scratch_idx = 0

        # FormatSeedGenerator: built on the first _refill_format_seeds when
        # format learning is active. Its queue is drained one seed per round
        # by OperatorEngine.mutate(); its stats feed the report.
        self._format_seed_generator: FormatSeedGenerator | None = None
        self._format_seed_queue: collections.deque[bytes] = collections.deque()
        self._format_seed_exec = 0

        # self._format_learner / self._ppmd: constructed by
        # analyzer_registry.wire_all() above (format_learner,
        # corpus_compression specs).

        # Per-operator buffer-change tracking costs one xxh3 digest per
        # mutation (~3.4us at 64KiB, no copy). Only pay it when something
        # actually consumes the credit assignment.
        self._track_op_effect = bool(
            elo
            or (self.mc and self.mc_bandit)
            or self._mopt
            or self._replicator
            or self._exp3
            or self._exp4
            or self._eps_greedy
            or self._hierarchical
            or self._gp_ucb
            # bo_gp_ucb was missing here the same way: with only
            # --bo-gp-ucb enabled, _track_op_effect stayed False and
            # no-op operators were credited with the round's success.
            or self._bo_gp_ucb
            # _cmaes was absent here while its dispatch branch in
            # operators.py was live: with only --cma-es enabled,
            # _track_op_effect stayed False, `effective` stayed None, and
            # every no-op operator in the round was credited with the
            # round's success exactly like the operator that did the work.
            or self._cmaes
            or self._contextual
            or self._c2ucb
            or self._ducb
            or self._swucb
            # kl_ducb/kl_swucb were missing here the same way cmaes was:
            # with only --kl-ducb or --kl-swucb enabled, no-op operators
            # were credited with the round's success.
            or self._kl_ducb
            or self._corral
            or self._kl_swucb
            or self._consolidated
            or self._moss
            or self._cucb
            or self._cusum_ucb
            or self._fewa
            or self._fpl
            or self._gradient
            or self._whittle
            or self._successive_elim
            or self._canary
            or self._use_shapley
        )

        # self._elo (+ its decay/match-window state): constructed by
        # analyzer_registry.wire_all() above (elo spec). self._use_elo is
        # set earlier, alongside the registry's other gating flags.

        # Invasion percolation operator selection (percolation handover
        # Module 4): an additional Elo-arbitrated strategy, not a bandit
        # family of its own -- it reads f.mc's existing bandit_stats() as
        # its resistance signal rather than tracking separate arm state, so
        # it also requires f.mc/mc_bandit to be enabled.
        self._use_invasion = invasion
        if invasion and not (self.mc and self.mc_bandit):
            log.warning(
                "--invasion has no effect without --mc-bandit (invasion_select "
                "reads operator success/failure stats from the MC bandit tracker)"
            )

        if continuum and not (self.mc and self.mc_bandit):
            log.warning(
                "--continuum has no effect on operator ranking without --mc-bandit "
                "(the flux map is built from the MC bandit's success/failure stats); "
                "regime diagnostics still record"
            )

        # Chi-squared operator heterogeneity test interval
        self._chi2_operator_interval = chi2_operator_interval
        if chi2_operator_interval > 0:
            print(f"[*] Chi-squared operator test: every {chi2_operator_interval} execs")

        # Elo arbitrates between all available strategies when enabled
        self._meta_strategy: str | None = None
        # Per-exec cache: resolved once in mutate(), reused for all mutations
        self._meta_strategy_cached: str | None = None
        # Operator schedulers actually selected this run (for the convergence
        # report, which must show only used schedulers)
        self._meta_strategy_used: set[str] = set()
        if self._use_elo:
            log.info(
                "Meta-scheduler enabled: Elo arbitrating across %d operator and %d seed strategies",
                len(_OPERATOR_STRATEGY_NAMES),
                len(_SEED_STRATEGY_NAMES),
            )

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

        # FMM-clustered pairwise overlap density
        self._use_overlap_density = overlap_density
        self._overlap_mode = overlap_density_mode
        self._overlap_min_jaccard = overlap_min_jaccard
        self._overlap_density_blend = overlap_density_blend
        self._overlap_density_cache: dict[str, float] = {}

        # Poisson-disk admission (proactive corpus diversity).
        # Bridges Bridson/Mitchell Poisson-disk sampling into the fuzzer's
        # corpus save path: query MinHashLSH.find_similar() at admission
        # time and reject seeds whose edge signature is closer than the
        # Jaccard radius to any already-admitted corpus member, with a
        # rare-edge safety valve so seeds contributing new edges are kept.
        self._use_poisson_disk_admission = poisson_disk_admission
        self._poisson_disk_min_jaccard = poisson_disk_min_jaccard
        self._admitted_keys: set[str] = set()
        self._redundant_admission_count = 0
        self._poisson_reject_count = 0
        self._poisson_near_dup_admit_count = 0
        self._poisson_admission: _PoissonDiskAdmission | None = None
        # Track distinct LSH buckets touched for maximality signal.
        self._poisson_occupied_buckets: set[tuple[int, int]] = set()
        self._poisson_last_new_bucket_exec = 0

        # Fractal Voronoi corpus-diversity bonus (see
        # core/fractal_partition.py; applied within one corpus)
        self._use_fractal_diversity = fractal_diversity
        self._fractal_diversity_depth = fractal_diversity_depth
        self._fractal_diversity_bonus = fractal_diversity_bonus

        # Entropy rate tracking: (exec_count, shannon_entropy) samples
        self._entropy_execs: array = array("Q")  # exec_count per entropy sample
        self._entropy_vals: array = array("d")  # shannon entropy per sample

        # self._distance / self._dist_table_shm: constructed by
        # analyzer_registry.wire_all() above (distance spec).
        # self._distance_targets is set earlier, alongside the registry's
        # other gating flags.
        self._anneal_progress = 0.0  # 0.0 = pure coverage, 1.0 = pure distance
        # Running min/max of observed per-seed distances (AFLGo queue
        # normalization); the no-data sentinel (20.0) is excluded.
        self._dist_min_observed: float | None = None
        self._dist_max_observed: float | None = None
        # Most recent avg_distance reading (either source below), sampled
        # once per stats tick into a ScalingExponentDetector to classify
        # whether directed scheduling is producing ballistic (directed),
        # diffusive (Brownian/no-better-than-random), or trapped progress.
        # See core/scaling_exponent.py (P3-T5 -- restored on request; the
        # handover's gating questions on this proposal are still open) and
        # docs/handover/handover_thermo_stochastic_concepts_2026-09-12.md.
        self._dist_last_value: float | None = None
        self._distance_trend = ScalingExponentDetector()

        # K-Scheduler node channel: mutually exclusive with directed mode
        # (both upload __AFL_DIST_SHM_ID; evaluation campaigns are not
        # directed).
        self._katz_channel = None
        if not targets:
            try:
                from fuzzer_tool.services.katz_channel import KatzChannel

                ch = KatzChannel.build(target, use_cfg_cache=use_cfg_cache, debug=self.debug)
                if ch is not None and ch.upload():
                    self._katz_channel = ch
                    print(
                        f"[*] K-Scheduler node channel: {len(ch.node_of)} probe sites, "
                        f"{ch.n_nodes} ICFG nodes"
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("Katz channel setup failed: %s", e)

        # Simulated annealing temperature schedule
        self._anneal_budget = anneal_budget  # 0 = no annealing (temperature always 1.0)
        self._temperature = 1.0

        # self._tracer: constructed by analyzer_registry.wire_all() above
        # (trace spec). self._trace_crashes_requested is set earlier,
        # alongside the registry's other gating flags.

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

            for op in REGISTRY.names():
                _init(op)

        from fuzzer_tool.core.target_profiler import format_operator_priors

        _format_priors = format_operator_priors(self._profile)

        if self.mc and self.mc_bandit:
            _register_arms(self.mc, _format_priors)
        if self._mopt:
            _register_arms(self._mopt)
        if self._cmaes:
            _register_arms(self._cmaes)
        if self._replicator:
            _register_arms(self._replicator)
        if self._exp3:
            _register_arms(self._exp3)
        if self._exp4:
            _register_arms(self._exp4)
        if self._eps_greedy:
            _register_arms(self._eps_greedy)
        if self._hierarchical:
            _register_arms(self._hierarchical)
        if self._gp_ucb:
            _register_arms(self._gp_ucb)
        if self._bo_gp_ucb:
            _register_arms(self._bo_gp_ucb, _format_priors)
        if self._ducb:
            _register_arms(self._ducb)
        if self._swucb:
            _register_arms(self._swucb)
        if self._kl_ducb:
            _register_arms(self._kl_ducb)
        if self._kl_swucb:
            _register_arms(self._kl_swucb)
        if self._cucb:
            _register_arms(self._cucb)
        if self._cusum_ucb:
            _register_arms(self._cusum_ucb)
        if self._fewa:
            _register_arms(self._fewa)
        if self._fpl:
            _register_arms(self._fpl)
        if self._corral:
            _register_arms(self._corral)
        if self._gradient:
            _register_arms(self._gradient)
        if self._whittle:
            _register_arms(self._whittle)
        if self._successive_elim:
            _register_arms(self._successive_elim)
        if self._op_kuramoto:
            _register_arms(self._op_kuramoto)
        if self._consolidated:
            _register_arms(self._consolidated, _format_priors)
        if self._moss:
            _register_arms(self._moss)
        if self._contextual:
            _register_arms(self._contextual)
        if self._c2ucb:
            _register_arms(self._c2ucb)
        if self._round_robin:
            _register_arms(self._round_robin)
        if self._canary:
            _register_arms(self._canary)
        if self._op_tpe:
            _register_arms(self._op_tpe, _format_priors)
        if self._op_strata:
            _register_arms(self._op_strata)
        if self._elo:
            _register_arms(self._elo)
        del _format_priors  # free priors dict after arm registration

        self._persistent_runner = None
        if self.persistent:
            from fuzzer_tool.adapters.persistent_signal import PersistentRunner

            self._persistent_runner = PersistentRunner(target=self.target, timeout=self.timeout)
            if self._persistent_runner.start():
                print("[*] Persistent mode: target started")
            else:
                print("[!] Persistent mode: failed to start target, falling back to fork")
                self._persistent_runner = None

        self._network_runner = None
        if getattr(self, "net_host", None) and getattr(self, "net_port", None):
            from fuzzer_tool.adapters.network import NetworkRunner
            from fuzzer_tool.core.kalman import RobustKF

            # Create a settle KF to adapt the per-iteration settle time.
            # After each run_one() the runner records wall-clock duration
            # as an observation; the KF's estimate smooths jitter and
            # provides a filtered settle time via _settle().
            initial_settle = getattr(self, "net_settle_ms", 10) / 1000
            settle_kf = RobustKF(
                dim=1,
                process_noise=initial_settle * 0.05,
                measurement_noise=initial_settle * 0.5,
                huber_threshold=3.0,
                adaptive_r_gain=0.02,
            )
            settle_kf.update(initial_settle)

            self._network_runner = NetworkRunner(
                host=self.net_host,
                port=self.net_port,
                proto=getattr(self, "net_proto", "tcp"),
                keepalive=getattr(self, "net_keepalive", False),
                settle=initial_settle,
                settle_kf=settle_kf,
            )
            print(
                f"[*] Network mode: fuzzing {self._network_runner.proto}://"
                f"{self.net_host}:{self.net_port} "
                f"(keepalive={self._network_runner.keepalive}, no reply read)"
            )

        self._inprocess_runner = None
        # Detect ASAN and set LD_PRELOAD BEFORE probing/loading (ctypes.CDLL) for any
        # .so/.dylib/.dll target — needed by both the auto-detect path below and the
        # explicit --inprocess/--inprocess-direct path, or ASAN aborts on first call
        # with "ASan runtime does not come first" instead of running the target.
        target_is_asan = False
        if self.target.lower().endswith((".so", ".dylib", ".dll")):
            target_is_asan = _detect_asan(self.target)
            if target_is_asan:
                libasan = "/usr/lib/x86_64-linux-gnu/libasan.so.8"
                if not os.path.exists(libasan):
                    import ctypes.util

                    libasan = ctypes.util.find_library("asan") or libasan
                # Read original LD_PRELOAD from process-start environment
                # (/proc/self/environ), not os.environ which may have been
                # modified by commands.py's ASAN detection before we get here.
                _original_ld_preload = ""
                try:
                    with open("/proc/self/environ", "rb") as _f:
                        for _entry in _f.read().split(b"\0"):
                            if _entry.startswith(b"LD_PRELOAD="):
                                _original_ld_preload = _entry[len(b"LD_PRELOAD=") :].decode()
                                break
                except OSError:
                    _original_ld_preload = os.environ.get("LD_PRELOAD", "")
                _asan_was_preloaded = libasan in _original_ld_preload
                if not _asan_was_preloaded:
                    existing = os.environ.get("LD_PRELOAD", "")
                    os.environ["LD_PRELOAD"] = f"{libasan}:{existing}" if existing else libasan
            # Preload ASAN runtime via ctypes for in-process loading (both
            # auto-detect and --inprocess-direct paths). This loads the
            # verify_asan_link_order=0 shim so ASAN's "does not come first"
            # check is suppressed, then loads libasan via RTLD_GLOBAL so the
            # target .so's DT_NEEDED libasan.so.8 is satisfied at dlopen time.
            _asan_ctypes_loaded = False
            if target_is_asan and not _asan_was_preloaded:
                import ctypes as _ctypes
                import subprocess as _subprocess
                import tempfile as _tempfile

                from fuzzer_tool.adapters.shim_factory import _find_compiler

                _asan_opts_shim_src = (
                    b"const char *__asan_default_options() {  "
                    b'return "halt_on_error=0:abort_on_error=0:verify_asan_link_order=0";}'
                )
                _fd, _shim_path = _tempfile.mkstemp(suffix=".so", prefix="asan_opts_")
                os.close(_fd)
                try:
                    _compiler = _find_compiler()
                    # Strip ASAN from compiler subprocess (clang/gcc aren't
                    # built with ASAN; libasan's LeakSanitizer causes false
                    # leak reports that make the compiler exit non-zero).
                    _env = os.environ.copy()
                    _env.pop("ASAN_OPTIONS", None)
                    _env.pop("LSAN_OPTIONS", None)
                    _ld_preload = _env.get("LD_PRELOAD", "")
                    if _ld_preload:
                        _parts = [p for p in _ld_preload.split(":") if "libasan" not in p]
                        _env["LD_PRELOAD"] = ":".join(_parts) if _parts else ""
                    _r = _subprocess.run(
                        [_compiler, "-shared", "-fPIC", "-O2", "-o", _shim_path, "-xc", "-"],
                        input=_asan_opts_shim_src,
                        capture_output=True,
                        timeout=30,
                        env=_env,
                    )
                    if _r.returncode != 0:
                        raise OSError(f"compiler failed: {_r.stderr.decode(errors='replace')}")
                    _ctypes.CDLL(_shim_path, mode=_ctypes.RTLD_GLOBAL)
                    _ctypes.CDLL(libasan, mode=_ctypes.RTLD_GLOBAL)
                    _asan_ctypes_loaded = True
                    print(f"[*] ASAN preloaded for in-process: {libasan}")

                    # With halt_on_error=0 (set via __asan_default_options shim
                    # above), ASAN reports bugs to stderr but does not abort().
                    # The target function returns normally (rc=0) and stderr
                    # contains the full ASAN report. The existing crash detection
                    # pipeline (SanitizerReport.parse() in runner.py) detects
                    # the crash from captured stderr. No death callback is
                    # needed — ASAN only fires death callbacks in the fatal
                    # path (halt_on_error=1).
                except OSError as e:
                    print(f"[!] ASAN ctypes preload failed: {e}")
                finally:
                    with contextlib.suppress(OSError):
                        os.unlink(_shim_path)
            # UBSAN detection — set UBSAN_OPTIONS so errors abort the
            # target for crash detection.  The UBSAN runtime was already
            # preloaded via LD_PRELOAD by ldpreload_wrapper.py.
            target_is_ubsan = _detect_ubsan(self.target)
            if target_is_ubsan:
                ubsan_opts = os.environ.get("UBSAN_OPTIONS", "")
                opt_parts = [p for p in ubsan_opts.split(":") if p] if ubsan_opts else []
                seen = {p.split("=")[0] for p in opt_parts}
                for opt in ("halt_on_error=1", "abort_on_error=1", "print_stacktrace=1"):
                    key = opt.split("=")[0]
                    if key not in seen:
                        opt_parts.append(opt)
                        seen.add(key)
                os.environ["UBSAN_OPTIONS"] = ":".join(opt_parts)
        # Auto-detect .so targets and use in-process mode
        if not inprocess and self.target.lower().endswith((".so", ".dylib", ".dll")):
            from fuzzer_tool.adapters.inprocess import InProcessRunner

            cov_env_id = self.shm_cov.env_id if self.shm_cov else None
            if not cov_env_id:
                # Without an SHM segment nothing populates the edge bitmap, so
                # every coverage-guided subsystem downstream (seed scheduling,
                # MI/TE/sensitivity position weighting, Elo/bandit operator
                # scheduling, stall detection, corpus admission) runs on a
                # constant-zero signal. That degrades silently — the run looks
                # healthy and fast while discovering nothing — so say so.
                self._warn_no_coverage()
            # Probe the shared object for a fuzz function name
            auto_func = self._probe_so_function(self.target)
            # Decide whether to use direct_lite (in-process ctypes) mode.
            # ASAN-instrumented .so targets need the ASAN runtime loaded
            # before the target. If LD_PRELOAD already contained libasan
            # at process start (external wrapper), it's already available.
            # Otherwise, load a tiny shim that exports __asan_default_options
            # (returning "verify_asan_link_order=0") before libasan.so, so
            # ASAN skips the post-startup first-load check. Safe for fuzzing:
            # ASAN only needs target-side bug detection, not Python-side.
            use_direct_lite = True  # NEVER EVER CHANGE THIS!!!
            # ASAN ctypes preloading was done above (before the branch). If it
            # failed, fall back to persistent mode where LD_PRELOAD handles it.
            # Even if ctypes preloading succeeds, ASAN detection does NOT work
            # in direct_lite mode when loaded mid-process: the compiled-in
            # shadow offset (0x7fff8000) doesn't match the runtime ASAN shadow
            # mapping for mid-process-loaded libasan (the ASAN heap is placed
            # at addresses whose shadow lands outside the mapped shadow region
            # on 48-bit systems). See docs/ASAN-LIMITATION.md §Layer 1.
            # Subprocess mode with LD_PRELOAD (set above at line 1027) is the
            # reliable path: ASAN initializes at process start in the child,
            # the shadow mapping is correct, and halt_on_error=0 (from the
            # ctypes-loaded shim) prevents ASAN from aborting the child.
            if target_is_asan and not _asan_was_preloaded:
                use_direct_lite = False
            # Cmplog: if the .so has cmplog compiled in, direct_lite works
            # because the shim is part of the .so itself. If the shim is
            # externally LD_PRELOAD'd, that also works. Otherwise we need
            # a process boundary (or preload the shim via ctypes).
            if self._cmplog is not None:
                has_cmplog = _detect_cmplog(self.target)
                has_tracecmp = _detect_tracecmp_target(self.target)
                if has_cmplog or has_tracecmp:
                    if has_cmplog:
                        print("[*] Cmplog: compiled into target .so (direct_lite compatible)")
                    else:
                        print(
                            "[*] Trace-cmp: compiled into target .so (direct_lite compatible, preloading shim)"
                        )
                else:
                    # Matches the artifact FILENAME, not the source file. The
                    # preload shim is still cached as
                    # fuzz_cmplog_shim.<digest>.so; it is built from
                    # afl_shim.c -D__AFL_PRELOAD_ONLY now that cmplog_shim.c
                    # is gone. tracecmp_shim is a name from an older split
                    # that external wrappers may still preload.
                    ld_preload = os.environ.get("LD_PRELOAD", "")
                    shim_in_preload = "cmplog_shim" in ld_preload or "tracecmp_shim" in ld_preload
                    if not shim_in_preload:
                        use_direct_lite = False
                    else:
                        print("[*] Cmplog: externally LD_PRELOAD'd (direct_lite compatible)")
            # Set _CMPLOG_OUT in os.environ so the cmplog constructor can
            # open the log file. Must happen regardless of execution mode:
            # direct_lite loads the .so in-process, persistent mode loads
            # it in a subprocess that inherits os.environ.
            if self._cmplog is not None:
                self._cmplog.setup_env_for_run()
                if use_direct_lite:
                    self._cmplog.preload_shims()
            self._inprocess_runner = InProcessRunner(
                target=self.target,
                function_name=auto_func,
                timeout=self.timeout,
                shm_size=self.map_size,
                direct_lite=use_direct_lite,
                coverage_env_id=cov_env_id,
                cov=bool(cov_env_id),
                debug=self.debug,
                capture_stderr=target_is_asan or target_is_ubsan,
                use_ptrace=self.use_ptrace,
            )
            if use_direct_lite:
                mode = "direct_lite"
            elif self._inprocess_runner._persistent:
                mode = "persistent"
            else:
                mode = "subprocess loader"
            print(f"[*] Auto-detected .so target: in-process mode ({mode}) with {auto_func}")
        elif inprocess:
            from fuzzer_tool.adapters.inprocess import InProcessRunner

            cov_env_id = self.shm_cov.env_id if self.shm_cov else None
            if not cov_env_id:
                # Without an SHM segment nothing populates the edge bitmap, so
                # every coverage-guided subsystem downstream (seed scheduling,
                # MI/TE/sensitivity position weighting, Elo/bandit operator
                # scheduling, stall detection, corpus admission) runs on a
                # constant-zero signal. That degrades silently — the run looks
                # healthy and fast while discovering nothing — so say so.
                self._warn_no_coverage()
            # For .so targets, probe for the correct fuzz function name
            # when the user didn't explicitly specify one.
            func = inprocess_func
            if (
                self.target.lower().endswith((".so", ".dylib", ".dll"))
                and func == "LLVMFuzzerTestOneInput"
            ):
                func = self._probe_so_function(self.target)
            # ASAN ctypes preloading was done above (before the branch). If the
            # verify_asan_link_order=0 shim was loaded successfully, direct mode
            # works even for ASAN .so targets. The user explicitly requested
            # --inprocess-direct, so try direct regardless — ASAN-detected bugs
            # may abort the process, but that's the user's accepted tradeoff.
            direct_ok = inprocess_direct
            # Refuse PIE executables in direct ctypes mode: the OS refuses to
            # dlopen a position-independent executable with the same cryptic
            # OSError, but the failure is silent and easy to miss. Check the
            # ELF type up front so the error message names the real problem
            # instead of leaking the OS errno through.
            if (
                direct_ok
                and not self.target.lower().endswith((".so", ".dylib", ".dll"))
                and detect_elf_type(self.target) == 3
            ):  # ET_DYN
                raise RuntimeError(
                    f"target {self.target!r} is a PIE executable (ET_DYN), "
                    "which cannot be loaded via ctypes.CDLL in direct mode. "
                    "Use a shared library (.so/.dylib/.dll) target, or drop "
                    "--inprocess-direct and let the subprocess loader handle it."
                )
            # Cmplog: mirror the auto-detect .so branch's env/shim setup
            # (see above). This branch is taken whenever --inprocess is
            # explicit -- including via --hail-mary, which force-enables
            # both --inprocess and --inprocess-direct. Without this block,
            # _CMPLOG_OUT never gets set before InProcessRunner loads the
            # target below, so a compiled-in cmplog shim has nowhere to
            # write and cmplog silently collects nothing even though
            # _detect_cmplog() reports the target as instrumented.
            if self._cmplog is not None:
                has_cmplog = _detect_cmplog(self.target)
                has_tracecmp = _detect_tracecmp_target(self.target)
                if has_cmplog or has_tracecmp:
                    if has_cmplog:
                        print("[*] Cmplog: compiled into target .so (direct_lite compatible)")
                    else:
                        print(
                            "[*] Trace-cmp: compiled into target .so "
                            "(direct_lite compatible, preloading shim)"
                        )
                elif direct_ok:
                    # Not compiled in -- direct ctypes mode needs either an
                    # externally LD_PRELOAD'd shim (LD_PRELOAD is fixed at
                    # process start, so it still applies to a ctypes-loaded
                    # .so) or falls back to the subprocess loader, which
                    # picks up the shim via preload_shims()/_CMPLOG_OUT
                    # like any other cmplog-off-by-default path.
                    ld_preload = os.environ.get("LD_PRELOAD", "")
                    shim_in_preload = "cmplog_shim" in ld_preload or "tracecmp_shim" in ld_preload
                    if not shim_in_preload:
                        direct_ok = False
                    else:
                        print("[*] Cmplog: externally LD_PRELOAD'd (direct_lite compatible)")
                # Must happen regardless of execution mode: direct ctypes
                # loads the .so in-process (needs the env var set before
                # CDLL below), subprocess loader inherits os.environ.
                self._cmplog.setup_env_for_run()
                if direct_ok:
                    self._cmplog.preload_shims()
            self._inprocess_runner = InProcessRunner(
                target=self.target,
                function_name=func,
                timeout=self.timeout,
                shm_size=self.map_size,
                direct=direct_ok,
                coverage_env_id=cov_env_id,
                cov=bool(cov_env_id),
                debug=self.debug,
                use_ptrace=self.use_ptrace,
            )
            mode = "direct ctypes" if direct_ok else "subprocess loader"
            cov_note = f", SHM cov id={cov_env_id}" if cov_env_id else ""
            print(f"[*] In-process mode ({mode}{cov_note}): {self.target}::{func}")
            if self._inprocess_runner._persistent:
                print("[*] Persistent loader: enabled (1 process, many calls)")

        if forkserver:
            self._setup_forkserver()

    def _setup_forkserver(self) -> None:
        """Start the C fuzz_loader for the default (spawn-per-exec) path.

        Replaces `run_target_fast`'s posix_spawn + ELF load + dynamic linker
        + libc init per execution with a fork+exec from an already-loaded
        process. The loader is spawned holding __AFL_SHM_ID / AFL_MAP_SIZE,
        so its children attach to the fuzzer's own coverage segment through
        afl_shim.c's constructor — no bitmap is round-tripped.

        Only claims the exact set of runs `run_target_fast` handles today:
        every other mode either owns the child itself (in-process,
        persistent, network, ptrace) or needs per-execution setup the
        loader's environment is fixed against (cmplog truncates its log per
        run; perf counters must be opened on the child pid we never see;
        file_mode/target_args build their own argv; multi-target needs more
        than one binary).
        """
        if (
            self._inprocess_runner
            or self._persistent_runner
            or self._network_runner
            or self.ptrace_cov
            or self.multi_targets
            or self.file_mode
            or self.target_args
            or self._cmplog
            or self._perf_counters
            or self._pt_session
            or self._lbr_session
        ):
            return

        from fuzzer_tool.adapters.forkserver import ForkserverRunner

        env: dict[str, str] = {}
        if self.use_coverage:
            env["AFL_MAP_SIZE"] = str(self.map_size)
        if self.shm_cov:
            env["__AFL_SHM_ID"] = self.shm_cov.env_id

        runner = ForkserverRunner(self.target, timeout=self.timeout, env=env)
        if runner.start():
            self._forkserver = runner
            print(f"[*] Forkserver: fork+exec from a loaded process ({self.target})")
        else:
            log.warning("Forkserver unavailable, falling back to spawn-per-exec")

    def _setup_lbr(self) -> bool:
        """Build the sampled branch-record map and the session feeding it.

        Degrades like --intel-pt: no usable PMU leaves the configured
        backend untouched.  Unlike PT this opens on AMD and on Intel, since
        PERF_SAMPLE_BRANCH_STACK is vendor-neutral.
        """
        from fuzzer_tool.adapters.lbr_trace import DEFAULT_PERIOD, LbrSession
        from fuzzer_tool.core.branch_record import BranchCoverage

        cov = BranchCoverage()
        period = self.lbr_period or DEFAULT_PERIOD
        session = LbrSession(period=period, sink=cov)
        if not session.available:
            log.warning(
                "Branch-record sampling unavailable (no branch-stack event); "
                "coverage unchanged. Needs AMD Zen 3+ with BRS/LbrExtV2 or "
                "Intel LBR, and perf_event_paranoid low enough to sample."
            )
            self.lbr = False
            return False

        self.branch_cov = cov
        self._lbr_session = session
        # Sampled, so absence of an edge is a fact about the period, not the
        # input. Said out loud because the number looks like edge coverage.
        print(f"[*] Coverage: branch records (sampled, period={period})")
        return True

    def _setup_intel_pt(self) -> bool:
        """Build the PT coverage map and the AUX session that feeds it.

        Degrades instead of failing: a host without the PMU keeps whatever
        coverage backend was already configured, because --intel-pt is an
        additional source rather than a replacement for one.
        """
        from fuzzer_tool.adapters.pt_trace import PT_SYSFS_ROOT, PtTraceSession
        from fuzzer_tool.core.intel_pt import PtCoverage, PtMapMode

        mode = PtMapMode.EDGE if self.intel_pt_mode == "edge" else PtMapMode.BLOCK
        # Not self.map_size: that is derived from the target's instrumentation
        # via ELF symbol scanning, and a PT target has none.
        cov = PtCoverage(mode=mode)
        session = PtTraceSession(sink=cov)
        if not session.available:
            log.warning(
                "Intel PT unavailable (%s absent); coverage unchanged. "
                "PT needs an Intel CPU exposing the PMU, and most VMs do not.",
                PT_SYSFS_ROOT,
            )
            self.intel_pt = False
            return False

        self.pt_cov = cov
        self._pt_session = session
        print(f"[*] Coverage: Intel PT ({mode.value}), aux={session.aux_size >> 10} KiB")
        return True

    def _setup_ptrace(self, target, deep_coverage, max_bps, fallback_hint=False):
        from fuzzer_tool.core.elf import detect_ngram_k

        cov = PtraceCoverage(
            target,
            deep_coverage=deep_coverage,
            max_bps=max_bps,
            ngram_k=detect_ngram_k(target),
        )
        if cov.bb_addrs:
            self.ptrace_cov = cov
            mode = "deep (pure decoder)" if cov.deep_coverage else "function-entry"
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

    def _build_strata(self, seed_arm: bool) -> None:
        """EdgeLedger at the target's ctx width; the seed arm if asked."""
        from fuzzer_tool.core import elf
        from fuzzer_tool.core.edge_ledger import EdgeLedger
        from fuzzer_tool.core.schedulers.seed_strata import Guard, StrataSeedScheduler

        target = getattr(self, "target", None)
        self._edge_ledger = EdgeLedger(elf.detect_ctx_bits(target) if target else None)
        if not seed_arm:
            return

        guard = Guard(elf.sancov_guard_status(target)) if target else Guard.UNKNOWN
        self._seed_strata = StrataSeedScheduler(self._rng, self._edge_ledger, guard)
        log.info("strata enabled (guard %s, family shift %d)", guard.value, self._edge_ledger.shift)

    def _strata_observe(self, seed: bytes, edges) -> None:
        """Mirror one record_edges call into the ledger; credit the strata pick."""
        # getattr: __new__-built test fuzzers reach record_edges without __init__.
        led = getattr(self, "_edge_ledger", None)
        if led is None or not edges or isinstance(edges, (bytes, bytearray)):
            return

        key = self._seed_key(seed)
        self._strata_bytes[key] = seed
        nov = led.observe(key, frozenset(edges))
        if self._seed_strata is not None and nov.families:
            self._seed_strata.credit(nov.families, key)

    def _strata_set_stability(self, jaccard: float) -> None:
        """Edge id probe verdict -> ledger trust (F1: moving ids -> family resolution)."""
        from fuzzer_tool.core.edge_ledger import Trust

        if self._edge_ledger is not None:
            self._edge_ledger.set_trust(Trust.STABLE if jaccard == 1.0 else Trust.UNSTABLE)

    def _strata_stratum(self, seed: bytes) -> int | None:
        """op_strata's stratum: the strata arm's phi for its own pick, else the rarest family."""
        led = self._edge_ledger
        if led is None:
            return None

        key = self._seed_key(seed)
        arm = self._seed_strata
        if arm is not None and arm.last_phi is not None and arm.last_key == key:
            return arm.last_phi
        return led.rarest_family(key)

    def _save_strata(self) -> None:
        if self._edge_ledger is None:
            return
        state = {"ledger": self._edge_ledger.to_dict(), "bytes": dict(self._strata_bytes)}
        if self._seed_strata is not None:
            state["seed"] = self._seed_strata.to_dict()
        self._state_store.set("strata", state)

    def _save_learned(self) -> None:
        """Persist op_credit, burn-front, PLL and WFC tables for ``--resume``."""
        from fuzzer_tool.core.wfc_chunks import WFC_MUTATOR

        if self._op_credit is not None:
            self._state_store.set("op_credit", self._op_credit.to_dict())
        if self._burn_front is not None:
            self._state_store.set("burn_front", self._burn_front.to_dict())
        pll = getattr(self, "_pll", None)
        if pll is not None:
            self._state_store.set("pll", pll.save())
        if self._wfc_enabled:
            self._state_store.set("wfc_tables", WFC_MUTATOR.store.to_dict())

    def _load_learned(self) -> None:
        """Restore :meth:`_save_learned` state; fresh runs reset the shared WFC tables.

        PLL is restored at analyzer activation (``_activate_pll``).
        """
        from fuzzer_tool.core.wfc_chunks import WFC_MUTATOR, WfcChunkTableStore

        # WFC_MUTATOR is process-global: without the reset a second campaign
        # in one process would inherit the first one's tables.
        if not self.resume:
            WFC_MUTATOR.store = WfcChunkTableStore()
            return

        if self._op_credit is not None:
            self._op_credit.from_dict(self._state_store.get("op_credit", {}))
        if self._burn_front is not None:
            self._burn_front.from_dict(self._state_store.get("burn_front", {}))
        if self._wfc_enabled:
            WFC_MUTATOR.store.from_dict(self._state_store.get("wfc_tables", {}))

    def _load_strata(self) -> None:
        """Restore ledger + seed arm on resume; malformed payloads start fresh."""
        from fuzzer_tool.core.edge_ledger import EdgeLedger
        from fuzzer_tool.core.schedulers.seed_strata import StrataSeedScheduler

        data = self._state_store.get("strata")
        if not self.resume or data is None:
            return
        try:
            ledger = EdgeLedger.from_dict(data["ledger"])
            arm = self._seed_strata
            if arm is not None and "seed" in data:
                arm = StrataSeedScheduler.from_dict(data["seed"], self._rng, ledger, arm._guard)
        except (KeyError, TypeError, ValueError) as e:
            log.warning("strata state unreadable, starting fresh: %s", e)
            return
        self._edge_ledger = ledger
        self._seed_strata = arm
        self._strata_bytes = dict(data.get("bytes", {}))
        print(
            f"[*] Strata: loaded ledger ({ledger.n_seeds} seeds, {len(ledger.frontier())} frontier)"
        )

    def _seed_residual_outcomes(self) -> dict[str, float]:
        """Seed key -> edges its descendants found: the falsification log's outcome.

        Read only by ``ResidualSeedScheduler.falsification`` at refit; never by the score.
        """
        return {
            self._seed_key(d): float(m.get("coverage_edges", 0)) for d, m in self.seed_meta.items()
        }

    def _seed_key(self, data: bytes) -> str:
        """Return content hash for *data*."""
        return self._corpus_manager.seed_key(data)

    def _boost_corpus_sizes(self) -> None:
        """Resize each corpus seed to a target size drawn from N(boost_mean, boost_std),
        clamped to [1, corpus_boost]. Target sizes are shuffled to avoid ordering bias
        (e.g. all small seeds paired with small targets)."""
        if not self._corpus_boost:
            return
        n = len(self.corpus)
        if n == 0:
            return
        mean = self._boost_mean if self._boost_mean is not None else self._corpus_boost / 2.0
        std = self._boost_std if self._boost_std is not None else self._corpus_boost / 6.0
        std = max(std, 1.0)
        target_sizes = [
            max(1, min(int(self._rng.gauss(mean, std)), self._corpus_boost)) for _ in range(n)
        ]
        self._rng.shuffle(target_sizes)
        self.corpus = [
            self._resize_seed(s, t) for s, t in zip(self.corpus, target_sizes, strict=False)
        ]

    def _apply_seed_transforms(self) -> None:
        """Apply skip/truncate/slide transforms to the loaded corpus.

        If any transform flag is set, seeds are not used as-is. The order is:
        skip -> truncate -> slide. Sliding replaces each seed with a set of
        fixed-size windows.
        """
        if not self.corpus:
            return
        if not (self._seed_skip_size or self._seed_truncate_size or self._seed_slide_size):
            return

        original = list(self.corpus)
        transformed: list[bytes] = []
        # None means "uncapped" (0, or slide disabled) — every `cap is not
        # None` check below is then simply skipped.
        cap = (
            self._seed_slide_max_seeds
            if self._seed_slide_size and self._seed_slide_max_seeds
            else None
        )

        for seed in original:
            if cap is not None and len(transformed) >= cap:
                break
            if self._seed_skip_size and len(seed) > self._seed_skip_size:
                continue
            if self._seed_slide_size:
                win = max(1, self._seed_slide_size)
                if len(seed) <= win:
                    transformed.append(seed)
                else:
                    # bytes slicing is already a single C-level memcpy, so
                    # there is nothing for memoryview to save here — it just
                    # adds an extra allocation on the way to the same bytes.
                    for i in range(len(seed) - win + 1):
                        if cap is not None and len(transformed) >= cap:
                            break
                        transformed.append(seed[i : i + win])
            else:
                if self._seed_truncate_size and len(seed) > self._seed_truncate_size:
                    seed = seed[: self._seed_truncate_size]
                transformed.append(seed)

        self.corpus = transformed

    def _resize_seed(self, seed: bytes, target_size: int) -> bytes:
        """Truncate or pad *seed* to *target_size* bytes.

        Padding modes (controlled by ``self._boost_pad``):
          * repeat — cycle the existing bytes (AFL-style, default)
          * zero   — zero-pad
          * random — fill with random bytes
        """
        if len(seed) == target_size:
            return seed
        if len(seed) > target_size:
            return seed[:target_size]
        need = target_size - len(seed)
        if self._boost_pad == "zero":
            return seed + b"\x00" * need
        if self._boost_pad == "random":
            return seed + bytes(self._rng.randrange(256) for _ in range(need))
        # "repeat" (default): AFL-style cyclic padding
        if len(seed) == 0:
            return b"\x00" * target_size
        repeats = (need // len(seed)) + 1
        return seed + (seed * repeats)[:need]

    def _save_state(self):
        return self._corpus_manager.save_state()

    def _load_state(self):
        return self._corpus_manager.load_state()

    def _selection_distribution(self) -> dict[str, float] | None:
        """Return the active scheduler's last normalised selection distribution.

        Returns None when the scheduler has none to give. That is the common
        case and not a failure: the UCB family selects by deterministic argmax,
        so there is no distribution to report, and a fabricated uniform one
        would make the work functional a function of trajectory length alone
        (W = L*log(L)) while still being labelled an entropy.

        A scheduler opts in by exposing ``last_selection_probs()`` returning a
        mapping over the operators it was offered, normalised to 1. ``exp3`` is
        the one that does today -- it already keeps that mixture for its own
        importance-weighted estimator.
        """
        for attr in _SELECTION_PROB_SOURCES:
            sched = getattr(self, attr, None)
            getter = getattr(sched, "last_selection_probs", None)
            if getter is None:
                continue
            try:
                probs = getter()
            except Exception:
                continue
            if not probs:
                continue
            total = sum(probs.values())
            if total <= 0 or not math.isfinite(total):
                continue
            # Normalise defensively: the identity needs a probability vector,
            # and a scheduler that drifts off 1.0 would bias the estimate
            # silently rather than loudly.
            return {k: v / total for k, v in probs.items()}
        return None

    def _record_fluctuation_observation(self, outcome: str, hit_edges: set[int]) -> None:
        """Ingest the current round's operator trajectory into the fluctuation tracker."""
        f = self._fluctuation
        if f is None or not self._last_ops_used:
            return
        # `self._operators._available` used to be consulted here. It is not an
        # attribute of the operators service and never was, so the hasattr
        # guard always fell through to the trajectory itself, every step got
        # probability 1/L, and the recorded work was exactly L*log(L) -- a
        # function of the mutation-stack depth carrying no operator, scheduler
        # or coverage information. Measured on a live run: 5,697 samples, 7
        # distinct values, all n*log(n). See the thermo handover, P2-T4.
        dist = self._selection_distribution()
        ops = tuple(self._last_ops_used)
        if dist is None:
            probs = tuple(1.0 / max(len(ops), 1) for _ in ops)
            probs_are_true = False
        else:
            probs = tuple(dist.get(op, 0.0) for op in ops)
            # A missing operator means the distribution does not cover the
            # trajectory, so it is not the law the trajectory was drawn from.
            probs_are_true = all(p > 0.0 for p in probs)
        from fuzzer_tool.core.analyzers.analyzer_fluctuation import TrajectoryRecord

        record = TrajectoryRecord(
            ops=ops,
            probs=probs,
            outcome=outcome,
            hit_edges=frozenset(hit_edges),
            new_edges=self._last_new_edge_count,
            probs_are_true=probs_are_true,
        )
        f.observe(record)

    def _run_target(self, data: bytes):
        return self._runner.run_target(data)

    def _check_differential(self, data: bytes):
        """Run data on differential target and track divergence.

        Both levels of the comparison run off one execution pair. The per-input
        verdict is logged; the observations behind it feed the statistical
        drift tracker.

        This used to call diff_run(), which returns only the verdict, and then
        record constants -- ``record(0, "", 0, "", ...)``. Both counters filled
        with the same single category, so KL(B || A) was 0 by construction and
        drift could never be detected. The comment claimed "diff_run already
        logs", which it does not; the verdict was dropped too.
        """
        if not self._diff_tracker or not self._diff_target:
            return
        from fuzzer_tool.services.differential import diff_run_detailed

        outcome = diff_run_detailed(self.target, self._diff_target, data)
        if outcome.diverged:
            log.warning("Differential divergence: %s", outcome.description)
            self._diff_divergences += 1

        was_drifting = self._diff_tracker.drift_detected
        self._diff_tracker.record(
            outcome.rc_a,
            outcome.stderr_a,
            outcome.rc_b,
            outcome.stderr_b,
            outcome.time_a,
            outcome.time_b,
        )
        # Report on the transition only. The tracker recomputes every 10 inputs
        # and stays latched once it trips, so logging on the flag itself would
        # emit the same line for the rest of the campaign.
        if self._diff_tracker.drift_detected and not was_drifting:
            log.warning(
                "Differential drift detected after %d inputs: %s",
                self._diff_tracker.total_inputs,
                self._diff_tracker.drift_description,
            )

    def _ptrace_handle_breakpoint(self, pid: int, libc, cov: PtraceCoverage, regs_buf):
        return self._runner._ptrace_handle_breakpoint(pid, libc, cov, regs_buf)

    def _run_target_ptrace(self, data: bytes):
        return self._runner._run_target_ptrace(data)

    def _is_interesting(self, returncode: int, stderr: str):
        return self._runner.is_interesting(returncode, stderr)

    def _is_crash(self, returncode: int, stderr: str):
        return self._runner.is_crash(returncode, stderr)

    def mutate(self, data: bytes):
        return self._operators.mutate(data)

    def _dedup_mutate(self, data: bytes) -> bytes:
        """Mutate *data*, re-rolling mutants the exec bloom has already seen.

        A hit means the mutant was almost certainly executed before, so the
        exec would buy no coverage and no bandit signal.  Re-rolling is close
        to free: ``mutate()`` costs ~2 orders of magnitude less than a target
        run.  After ``EXEC_DEDUP_RETRIES`` consecutive hits the last mutant is
        executed anyway, so a saturated filter degrades to the old behaviour
        rather than spinning.

        A bloom false positive (rate 1e-3) discards a genuinely novel mutant.
        That is harmless — it stays reachable on later iterations — and the
        filter never yields a false negative, so nothing already executed
        slips through as new.
        """
        mutated = self.mutate(data)
        if not self._dedup_execs:
            return mutated
        for _ in range(EXEC_DEDUP_RETRIES):
            if not self._exec_bloom.update_bytes(bytes(mutated), reset_on_full=True):
                return mutated
            self._dedup_hits += 1
            mutated = self.mutate(data)
        self._dedup_gaveup += 1
        return mutated

    def _seed_entropy_pct(self, seed: bytes, meta: dict | None) -> float:
        """Byte entropy of ``seed`` on the 0-100 scale, memoised in seed_meta.

        Entropy is a pure function of the seed bytes and seeds are immutable,
        so this is computed once per seed and cached alongside ``input_size``.
        Caching in ``seed_meta`` rather than a companion dict is deliberate:
        ``seed_meta`` is already rebuilt from survivors when the corpus is
        pruned (corpus_manager._maybe_prune), so the cache cannot outlive its
        seed. A separate seed-keyed map would need its own eviction and is
        exactly the shape that has leaked stale entries here before.
        """
        if meta is None:
            return byte_entropy_pct(seed)
        cached = meta.get("input_entropy")
        if cached is None:
            cached = byte_entropy_pct(seed)
            meta["input_entropy"] = cached
        return cached

    def _cost_adjusted_weight(self, op: str, base_weight: float) -> float:
        """Scale a bandit reward by the inverse of what the round cost.

        Converts "edges per selection" into an "edges per unit time"
        proxy. The unit of time is one iteration, and an iteration pays the
        target execution whichever operator ran, so the ratio is

            (t_exec + median_op_cost) / (t_exec + this_op_cost)

        with t_exec the median execution time. An operator costing the
        median is unaffected (1.0); an expensive one (gradient_descent,
        condstmt_solve, path_negate, crc_learn, ~ms-s) is penalized in
        proportion to how many ordinary iterations fit in its budget; a
        cheap one gains only the time it actually saves, which next to the
        execution is little.

        This used to be median_op_cost / this_op_cost, with t_exec left
        out. _op_time_ema times the operator call alone, so a 1us operator
        was paid 10x the reward of a 10us one for a difference of 9us
        against an execution of hundreds -- and since that ratio was
        clamped at 20, not 1, rewards reached 20 while every scheduler
        documents them as [0, 1]. The bounded scale is enforced where the
        per-op bandit rewards are built in fuzz_one; the Elo edge_counts
        caller wants proportions, not a bounded scale, and is unclamped.

        Falls back to the unscaled weight until at least a few operators
        have timing data, and clamps the ratio so a single outlier can't
        zero out the reward.
        """
        if base_weight <= 0.0 or len(self._op_time_ema) < 2:
            return base_weight
        cost = self._op_time_ema.get(op)
        if cost is None or cost <= 0.0:
            return base_weight
        costs = sorted(c for c in self._op_time_ema.values() if c > 0.0)
        if not costs:
            return base_weight
        median_cost = costs[len(costs) // 2]
        tracker = getattr(self, "_exec_time_tracker", None)
        t_exec = tracker.p50 if tracker is not None else 0.0
        ratio = (t_exec + median_cost) / (t_exec + cost)
        ratio = max(0.05, min(ratio, 20.0))
        return base_weight * ratio

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

    def _op_bit_swap_8(self, buf, _byte_idx, _data):
        return self._operators._op_bit_swap_8(buf, _byte_idx, _data)

    def _op_bit_swap_16(self, buf, _byte_idx, _data):
        return self._operators._op_bit_swap_16(buf, _byte_idx, _data)

    def _op_bit_swap_32(self, buf, _byte_idx, _data):
        return self._operators._op_bit_swap_32(buf, _byte_idx, _data)

    def _op_bit_swap_64(self, buf, _byte_idx, _data):
        return self._operators._op_bit_swap_64(buf, _byte_idx, _data)

    def _op_length_grow(self, buf, _byte_idx, _data):
        return self._operators._op_length_grow(buf, _byte_idx, _data)

    def _op_length_shrink(self, buf, _byte_idx, _data):
        return self._operators._op_length_shrink(buf, _byte_idx, _data)

    def _op_repeat_clone(self, buf, _byte_idx, _data):
        return self._operators._op_repeat_clone(buf, _byte_idx, _data)

    def _op_truncate(self, buf, _byte_idx, _data):
        return self._operators._op_truncate(buf, _byte_idx, _data)

    def _op_length_boundary(self, buf, _byte_idx, _data):
        return self._operators._op_length_boundary(buf, _byte_idx, _data)

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

    def _op_magicyuv_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_magicyuv_chunk_mutate(buf, _byte_idx, _data)

    def _op_jpeg2000_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_jpeg2000_chunk_mutate(buf, _byte_idx, _data)

    def _op_av1_rtp_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_av1_rtp_chunk_mutate(buf, _byte_idx, _data)

    def _op_rasc_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_rasc_chunk_mutate(buf, _byte_idx, _data)

    def _op_ffconcat_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_ffconcat_chunk_mutate(buf, _byte_idx, _data)

    def _op_tiff_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_tiff_chunk_mutate(buf, _byte_idx, _data)

    def _op_dvbsub_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_dvbsub_chunk_mutate(buf, _byte_idx, _data)

    def _op_cfhd_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_cfhd_chunk_mutate(buf, _byte_idx, _data)

    def _op_shorten_chunk_mutate(self, buf, _byte_idx, _data):
        return self._operators._op_shorten_chunk_mutate(buf, _byte_idx, _data)

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

    def _prune_crash_data(self) -> None:
        """Trim crash structures when they exceed MAX_CRASH_SIGS.

        Keeps the most frequent crash signatures and evicts all data for
        the least frequent ones. Leaves crash_hashes intact (small memory
        footprint, prevents duplicate disk writes).
        """
        if len(self.crash_sigs) <= MAX_CRASH_SIGS:
            return
        # Keep top 75% of signatures sorted by frequency descending
        keep_count = max(MAX_CRASH_SIGS * 3 // 4, 1)
        sorted_sigs = sorted(self.crash_sigs.items(), key=lambda x: -x[1])
        kept = {sig for sig, _ in sorted_sigs[:keep_count]}
        evicted = set(self.crash_sigs) - kept
        self.crash_sigs = dict(sorted_sigs[:keep_count])
        # Evict associated data for dropped signatures
        for sig in evicted:
            self.crash_frames.pop(sig, None)
            self.crash_min_sizes.pop(sig, None)
            self._crash_replays.pop(sig, None)
            self._crash_files.pop(sig, None)

    def save_to_corpus(self, data: bytes, parent: bytes | None = None):
        return self._corpus_manager.save_to_corpus(data, parent)

    def _trim_new_coverage(self, data: bytes, parent: bytes):
        return self._corpus_manager.trim_new_coverage(data, parent)

    def _auto_minimize_corpus(self):
        return self._corpus_manager.auto_minimize_corpus()

    def _defer_minimize(self):
        """Schedule auto_minimize_corpus for the next main-loop iteration.
        This avoids pruning seeds that were just added but not yet fuzzed."""
        self._minimize_pending = True

    def _record_lineage_insert(self, child: bytes, parent: bytes | None, corpus_len_before: int):
        """Insert *child* into the lineage tree when it joined the corpus.

        Gated on the flag; no-op when the seed was rejected as a duplicate
        (corpus length unchanged) or under qea where seeds bypass f.corpus.
        Node weight = new coverage edges contributed by this iteration.
        """
        if not self._use_lineage or self._lineage is None:
            return
        if len(self.corpus) <= corpus_len_before:
            return
        ops = [op for op, _ in self._last_ops_with_sites]
        sites = [s for _, s in self._last_ops_with_sites]
        parent_key = self._seed_key(parent) if parent is not None else None
        self._lineage.insert(
            parent_key, self._seed_key(child), ops, sites, self._last_new_edge_count
        )

    def _record_entropy_gradient_credit(
        self, child: bytes, parent: bytes | None, corpus_len_before: int
    ):
        """Feed the entropy-gradient seed arm one admission event, if enabled.

        Unconditional on which seed strategy actually picked *parent* --
        this arm has to observe every admission to assign credit, not just
        the ones from rounds where Elo happened to pick it (see
        core/schedulers/seed_entropy_gradient.py). Gated the same way
        _record_lineage_insert is: a no-op when the seed was rejected as a
        duplicate (corpus length unchanged).
        """
        strategy = getattr(self, "_entropy_gradient", None)
        if strategy is None:
            return
        if len(self.corpus) <= corpus_len_before:
            return
        strategy.record_child(parent, child, self.corpus)

    def _write_ablation_row(self, has_new_coverage: bool, is_crash: bool) -> None:
        """Append one --schedule-ablation row: pick signals, outcome, op stack.

        ``operator`` is the round's op stack joined by ``+`` (e.g.
        ``bit_flip+havoc``), so reward vs own-pull-count is measurable.
        """
        ps = self._last_pick_signals
        self._ablation_file.write(
            f"{self.exec_count},{ps['seed_idx']},{ps['seed_hash']},"
            f"{ps['fuzz_count']},{ps['coverage_edges']},{ps['age_s']},"
            f"{ps['temperature']},{ps['base_w']},{ps['burst']},{ps['penalty']},"
            f"{ps['subsumption']},{ps['diversity']},{ps['spatial']},"
            f"{ps['mdl']},{ps['final_w']},"
            f"{1 if has_new_coverage else 0},{1 if is_crash else 0},"
            f"{'+'.join(self._last_ops_used)}\n"
        )
        if self.exec_count % 100 == 0:
            self._ablation_file.flush()

    def _flush_pending_minimize(self):
        """Run deferred minimize if one is pending."""
        if self._minimize_pending:
            self._minimize_pending = False
            self._auto_minimize_corpus()

    def _deprioritize_near_duplicates(self):
        return self._corpus_manager.deprioritize_near_duplicates()

    def _gc_collect(self):
        """Periodic GC to return freed memory to OS.

        Gating (the old ``i % 500``) now lives in ``self._maintenance``
        (P3-3 step 4); this is just the action.
        """
        import gc

        gc.collect()

    def _legacy_memory_prune_tick(self):
        """Pre-P3-3 gating for ``_check_memory_and_prune``, byte-for-byte.

        Used only when ``self.job_scheduler`` is False (the default): the
        off-switch and the once-per-1000-execs throttle both used to live
        inside ``_check_memory_and_prune`` itself. They moved out to
        ``self._maintenance``'s ``active``/``interval_execs`` when
        ``--job-scheduler`` is on; this wrapper keeps the untouched
        original behavior available for everyone who hasn't opted in.
        """
        if self.prune_corpus_max_memory <= 0:
            return
        if self.exec_count - self._last_memory_prune_exec < 1000:
            return
        self._last_memory_prune_exec = self.exec_count
        self._check_memory_and_prune()

    def _check_memory_and_prune(self):
        """Check RSS against total RAM and prune corpus if threshold exceeded.

        Uses /proc/meminfo for total RAM and /proc/self/statm for current RSS.

        The "once per 1000 execs" throttle and the
        ``self.prune_corpus_max_memory <= 0`` off-switch used to live here as
        an internal early return; both now live in ``self._maintenance``
        (P3-3 step 4), which is the only place tracking "when did this last
        run" across all absorbed jobs.

        NOT ``getrusage(RUSAGE_SELF).ru_maxrss``, which this used to read: that
        is the high-water mark and never decreases, so a single transient spike
        past the threshold armed the pruner permanently and every later check
        re-pruned an already-small corpus while printing the stale peak as if it
        were current usage.
        """
        try:
            rss_kb = _current_rss_kb()
            if rss_kb is None:
                return
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                        break
                else:
                    return
        except (OSError, ValueError):
            return

        used_pct = (rss_kb / total_kb) * 100
        if used_pct >= self.prune_corpus_max_memory:
            before = len(self.corpus)
            log.warning(
                "Memory usage %.1f%% exceeds %d%% threshold — pruning corpus",
                used_pct,
                self.prune_corpus_max_memory,
            )
            self._auto_minimize_corpus()
            after = len(self.corpus)
            if after < before:
                print(
                    f"\n[*] MEMORY PRUNE: {before} → {after} seeds "
                    f"(RSS {rss_kb // 1024}MB / {total_kb // 1024}MB, {used_pct:.1f}%)"
                )

    def _check_corpus_size_and_prune(self):
        """Auto-minimize corpus when it far exceeds the edge-derived target size.

        This runs independently of ``--minimize-every-execs`` so a large initial
        corpus does not stay bloated for the entire campaign. The trigger is
        throttled to avoid thrashing.
        """
        if self.exec_count - self._last_corpus_prune_exec < 1000:
            return
        self._last_corpus_prune_exec = self.exec_count

        if len(self.corpus) <= 1:
            return

        if self.max_corpus > 0:
            target_size = self.max_corpus
        else:
            edges = 0
            if self.shm_cov:
                edges = self.shm_cov.cumulative_edges
            elif self.ptrace_cov:
                edges = self.ptrace_cov.cumulative_edges
            target_size = min(max(edges, 50), 5000)

        if len(self.corpus) > max(target_size * 2, 5000):
            before = len(self.corpus)
            self._auto_minimize_corpus()
            after = len(self.corpus)
            if after < before:
                print(f"\n[*] CORPUS PRUNE: {before} → {after} seeds (target_size={target_size})")

    def _select_next_target(self):
        """Select the next target for multi-target round-robin fuzzing."""
        if not self.multi_targets:
            return
        # Weighted round-robin: prefer targets with fewer total edges discovered
        if len(self.multi_targets) > 1 and self.exec_count > 100:
            # Weight by inverse of cumulative edges (less-covered targets get more execs)
            weights = []
            for t in self.multi_targets:
                shm = self._target_shm_covs.get(t)
                edges = shm.cumulative_edges if shm else 0
                weights.append(1.0 / max(edges, 1))
            total = sum(weights)
            r = self._rng.random() * total
            cumulative = 0.0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    self._active_target_idx = idx
                    break
        else:
            self._active_target_idx = (self._active_target_idx + 1) % len(self.multi_targets)
        self.target = self.multi_targets[self._active_target_idx]

    def _pick_seed(self):
        seed = self._seed_picker.pick_seed()

        # last_picked: revisit clock for the --lst-revisit override (P3-3 step 6)
        meta = self.seed_meta.get(seed)
        if meta is not None:
            meta["last_picked"] = time.time()
        return seed

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

    def _refresh_agg_cache(self) -> None:
        """Recompute running aggregates from seed_meta."""
        self._cached_total_time = sum(m.get("total_time", 0.0) for m in self.seed_meta.values())
        self._cached_total_fuzz = sum(m.get("fuzz_count", 1) for m in self.seed_meta.values())
        self._cached_cost_samples = sum(cost_samples(m) for m in self.seed_meta.values())
        self._cached_total_edges = sum(m.get("coverage_edges", 0) for m in self.seed_meta.values())
        n_fuzz_vals = [m.get("fuzz_count", 0) for m in self.seed_meta.values()]
        self._cached_mean_log_n_fuzz = compute_mean_log_n_fuzz(n_fuzz_vals)
        self._agg_cache_valid = True

    def mean_exec_time(self) -> float:
        """Corpus-wide mean target time per execution, in seconds.

        Zero until something has been timed.  This is the stand-in the cost
        ledger uses for seeds it has no samples for — a resumed seed or one
        that has never been fuzzed is *unmeasured*, not free, and giving it
        the 1 microsecond floor made it beat every timed seed in the favored
        set (see core/cost_ledger.py).
        """
        if not self._agg_cache_valid:
            self._refresh_agg_cache()
        if self._cached_cost_samples <= 0:
            return 0.0
        return self._cached_total_time / self._cached_cost_samples

    def _cull_queue(self) -> None:
        """Compute AFL-style top_rated / favored minimal-set-cover.

        For each edge, pick the cheapest seed that covers it, then greedily
        build a favored set that covers all edges. Seeds in the favored set
        receive energy bonuses in FAST/COE schedules.
        """
        top_rated: dict[int, tuple[str, float]] = {}
        mean_us = self.mean_exec_time() * 1_000_000
        for key, edges in self._edge_tracker.seed_edges.items():
            m = self.seed_meta.get(key) or {}
            exec_us = seed_exec_us(m, mean_us)
            input_size = max(1, m.get("input_size", 1))
            cost = exec_us * input_size
            for e in edges:
                cur = top_rated.get(e)
                if cur is None or cost < cur[1]:
                    top_rated[e] = (key, cost)

        covered: set[int] = set()
        favored: set[str] = set()
        # Cover the rarest edges first: an edge reached by one seed forces that
        # seed into the favored set, while an edge reached by many is likely to
        # be picked up for free along the way.
        #
        # This used to sort by ``rare_edge_count(e)``, which is keyed by *seed*,
        # not by edge. Passing an edge id looked up an absent seed and returned
        # 0 for every edge, so the key was constant and ``sorted`` -- being
        # stable -- left the edges in dict insertion order. The greedy cover ran
        # in an arbitrary order and the rarity prioritisation this loop exists
        # for never happened. The edge id breaks ties so the favored set is a
        # function of the coverage data alone, not of insertion history.
        et = self._edge_tracker
        for e in sorted(top_rated, key=lambda e: (et.edge_owner_count(e), e)):
            if e in covered:
                continue
            k = top_rated[e][0]
            favored.add(k)
            covered |= self._edge_tracker.seed_edges[k]
        self._favored = favored

    def _maybe_retune_timeout(self) -> None:
        """Feed ``suggested_timeout()`` back into the live timeout.

        The tracker has always computed this; nothing ever consumed it, so
        the timeout stayed at whatever it was constructed with and the
        report printed a suggestion nobody acted on. Two things had to be
        true before it could be applied: fractional timeouts had to survive
        the loader handshake (c99bb27), and the forkserver had to be able to
        accept a new deadline without a re-handshake (the TIMEOUT command).

        The forkserver is updated *first* and the rest only follows if it
        succeeded. Every other consumer reads ``self.timeout`` per exec, so
        they cannot disagree with it; the loader is the one that is told
        once and then believed. Updating ``self.timeout`` on a loader that
        rejected the change would leave the reported deadline and the
        enforced deadline different -- which is the entire defect class the
        E2 timeout work was about, reintroduced from the other end.
        """
        if not self._adaptive_timeout:
            return
        tracker = self._exec_time_tracker
        if tracker.count < ADAPTIVE_TIMEOUT_MIN_SAMPLES:
            return
        if self.exec_count - self._last_timeout_retune_exec < ADAPTIVE_TIMEOUT_COOLDOWN_EXECS:
            return

        proposed = tracker.suggested_timeout()
        ceiling = self._timeout_initial * ADAPTIVE_TIMEOUT_MAX_GROWTH
        proposed = max(ADAPTIVE_TIMEOUT_FLOOR, min(proposed, ceiling))

        current = self.timeout
        if current > 0 and abs(proposed - current) / current < ADAPTIVE_TIMEOUT_HYSTERESIS:
            return

        if self._forkserver is not None and not self._forkserver.set_timeout(proposed):
            # No 'retune' capability (a stale loader binary), or the loader
            # did not answer. Either way the deadline it enforces is not
            # moving, so nothing else may move either. Disable rather than
            # retry every cooldown: the capability will not appear mid-run.
            log.warning(
                "Adaptive timeout: loader would not accept %.3fs; disabling retuning", proposed
            )
            self._adaptive_timeout = False
            return

        applied = self._forkserver.timeout if self._forkserver is not None else proposed
        for runner in (self._inprocess_runner, self._persistent_runner):
            if runner is not None:
                runner.timeout = applied

        self._timeout_retunes.append((self.exec_count, current, applied))
        self._last_timeout_retune_exec = self.exec_count
        self.timeout = applied
        log.info(
            "Adaptive timeout: %.3fs -> %.3fs at exec %d (p99=%.3fs, n=%d)",
            current,
            applied,
            self.exec_count,
            tracker.p99,
            tracker.count,
        )

    def _reset_cmplog(self):
        """Flush cmplog buffer to disk/FIFO before collecting tokens.

        In direct_lite mode with cmplog compiled into the target .so,
        the shim buffers CMP lines in a 256KB internal buffer. This flushes
        that buffer to disk/FIFO so collect_tokens() can read the data.

        Does NOT truncate the file — collect_tokens() handles truncation
        after reading. In fifo_sink mode the flush still happens (the shim
        buffer must be emptied into the pipe); only the file-truncate is
        skipped.
        """
        if self._cmplog is None:
            return

        # Flush tracecmp shim's internal buffer to disk
        self._cmplog.flush_shims()

        # Flush the target .so's compiled-in shim buffer
        runner = self._inprocess_runner
        if runner and runner.direct_lite and runner._lib:
            try:
                if hasattr(runner._lib, "__tracecmp_flush"):
                    runner._lib.__tracecmp_flush()
            except (AttributeError, OSError):
                pass

    def _record_cmp_progress(self, *vectors: dict) -> bool:
        """Fold one execution's asserted counts into the per-site maxima.

        Args:
            *vectors: One or more ``{key: satisfied count}`` maps for this
                execution.  ``{(callback, pc): count}`` from
                ``CmplogCollector.last_site_asserted`` is the fine-grained
                axis (P0-3): folding by callback family mixes progress at
                one comparison site with stagnation at another.
                ``{callback: count}`` is the coarse axis.  Both may be
                passed; str and tuple keys never collide, so the two
                high-water maps live side by side in one dict.  All empty
                whenever cmplog is off, which makes the whole channel inert
                rather than needing a flag of its own.

        Returns:
            True if some site's (or callback's) maximum grew enough to report.

        The high-water mark is updated on every increase; only increases
        past the growth threshold are *reported*. Separating the two is the
        point: a climb of one extra satisfied comparison per input would
        otherwise report on every step of the climb, and each report admits
        an input to the corpus.

        Every vector is walked before returning -- an early exit on the
        first report would leave the remaining maxima stale and turn the
        next execution's flat counts into a phantom climb.
        """
        reported = False
        for asserted in vectors:
            for key, count in asserted.items():
                prev = self._cmp_max_asserted.get(key, 0)
                if count <= prev:
                    continue
                if prev == 0 or count >= max(prev + 1, int(prev * MAX_COUNT_GROWTH_FACTOR)):
                    reported = True
                self._cmp_max_asserted[key] = count
        if reported:
            self._cmp_novelty_hits += 1
        return reported

    def _credit_reward_shape(self) -> float | None:
        """Scale factor for this round's shared operator reward, or None.

        None means "leave the reward alone", which is the answer in every case
        where the class partition cannot speak to this round:

        * ``--shaped-reward`` off, or no substrate built;
        * the round discovered no edges -- ``success`` is a disjunction (crash,
          interesting, slow, new max, cmp progress, new valid coverage), so most
          successful rounds have nothing to deduplicate and a factor of 0.0 here
          would silently delete every non-coverage reward in the fuzzer;
        * the preflight gate is closed (F1 per-process ids, F11 uninstrumented
          target), where every id is a singleton owned by one seed and the classes
          are noise. Counted separately so a bench run can tell "shaping was off"
          from "shaping was on and neutral".
        """
        if not self._shaped_reward or self._matrix_substrate is None:
            return None
        if not self._last_new_edge_ids:
            return None
        if not self._matrix_substrate.trusted:
            self._shaped_reward_gated += 1
            return None
        from fuzzer_tool.core.schedulers.op_credit import shaped_weight

        factor = shaped_weight(
            self._matrix_substrate, self._last_new_edge_ids, self._shaped_reward_floor
        )
        self._shaped_reward_rounds += 1
        self._shaped_reward_factor_sum += factor
        return factor

    def shaped_reward_stats(self) -> dict:
        """Did the shaping bite? Instrument for the paired A/B, not a decision."""
        n = self._shaped_reward_rounds
        return {
            "shaped_reward": self._shaped_reward,
            "shaped_reward_floor": self._shaped_reward_floor,
            "shaped_reward_rounds": n,
            "shaped_reward_gated_rounds": self._shaped_reward_gated,
            "shaped_reward_mean_factor": (self._shaped_reward_factor_sum / n) if n else None,
        }

    def _continuum_reward_shape(self) -> float | None:
        """Scale factor for this round's shared operator reward, or None.

        None means "leave the reward alone", same contract as
        ``_credit_reward_shape``, and for the same three reasons:

        * ``--continuum-reward`` off;
        * the round discovered no edges -- a factor here would zero every
          non-coverage reward (crash, hang, new max, cmp progress) the same
          way an unconditional class-shaping factor would;
        * the discovery has no neighbourhood to price -- no corpus seeds
          yet, or the trace is nothing but the new edges themselves.
          Counted apart (``_continuum_reward_neutral``) so a bench run can
          tell "off" from "on and the frontier had nothing to say".

        Independent of ``--shaped-reward``/``--op-credit``: both call sites
        multiply into the same ``op_rewards`` list (``_apply_reward_shape``
        composes), so combining them would move two variables in one paired
        run. This one prices *where* a discovery landed; that one prices
        *how duplicated* it was.
        """
        if not self._continuum_reward:
            return None
        if not self._last_new_edge_ids:
            return None
        if self._edge_tracker is None:
            return None

        from fuzzer_tool.core.analyzers.analyzer_navier_stokes import frontier_weight

        factor = frontier_weight(
            self._last_new_edge_ids,
            self._last_trace_edges,
            self._edge_tracker.edge_owner_count,
            len(self._edge_tracker.seed_edges),
            self._continuum_reward_floor,
        )
        if factor is None:
            self._continuum_reward_neutral += 1
            return None

        self._continuum_reward_rounds += 1
        self._continuum_reward_factor_sum += factor
        return factor

    def continuum_reward_stats(self) -> dict:
        """Did the shaping bite? Instrument for the paired A/B, not a decision."""
        n = self._continuum_reward_rounds
        return {
            "continuum_reward": self._continuum_reward,
            "continuum_reward_floor": self._continuum_reward_floor,
            "continuum_reward_rounds": n,
            "continuum_reward_neutral_rounds": self._continuum_reward_neutral,
            "continuum_reward_mean_factor": (
                (self._continuum_reward_factor_sum / n) if n else None
            ),
        }

    def _note_det_effector(self) -> None:
        """Tell the operator engine whether the byteflip just run moved the trace.

        Silence is the safe answer: every branch that cannot establish both
        a baseline and a current trace hash returns without recording, which
        leaves the byte position UNKNOWN and keeps its full deterministic
        schedule. Only a positive "this byte was flipped and nothing moved"
        removes the 24 arithmetic and interesting-value mutants at that
        position.

        The baseline is the seed's own path hash from ``EdgeTracker``, which
        ``_calibrate_seed_baselines`` records by executing every starting
        seed verbatim in this process, and which later seeds get from the
        execution that discovered them. It is therefore same-process, which
        matters: ``__afl_get_caller_ctx`` hashes a raw return address, so
        under ``FUZZER_KEEP_ASLR=1`` edge ids -- and the rolling hash over
        them -- are not stable across processes. A stale baseline makes every
        byte look live, so the gate switches itself off rather than
        mis-skipping.
        """
        engine = self._operators
        seed_key = engine.pending_det_seed_key()
        if seed_key is None:
            return
        if self.shm_cov is None:
            return
        baseline = self._edge_tracker.get_seed_path_hash(seed_key)
        if baseline == 0:
            return
        current = self.shm_cov.read_path_hash()
        if current == 0:
            # Shim built without the rolling hash. The edge-set fallback used
            # elsewhere costs a full SHM scan, and the memo in shm._scan is
            # keyed partly on the path hash and bypassed when it is zero --
            # exactly this case -- so it would be a second real scan on every
            # byteflip. Not worth it to gate one pass; leave the map unfilled.
            return
        engine.note_deterministic_result(current != baseline)

    def fuzz_one(self, data: bytes) -> bool:
        # Invalidate Elo K-factor cache at the start of each iteration
        # so record_strategy_match calls recompute K from the current
        # prediction errors if record_match hasn't been called yet.
        if self._use_elo and self._elo:
            self._elo._eff_k_cache = None
        self._last_parent_seed = data
        if self._op_strata is not None:
            self._op_strata.set_stratum(self._strata_stratum(data))
        self._last_new_edge_count = 0  # reset; set when record_edges finds new edges
        # The ids themselves, not just how many: the shaped reward needs the
        # identities to ask the canonical partition how many classes they are.
        # Reset every round so a round with no discovery cannot be shaped by the
        # previous round's edges.
        self._last_new_edge_ids: list[int] = []
        # This round's full hit-edge trace, for _continuum_reward_shape's
        # neighbourhood. Reset for the same reason: a round with no
        # discovery must not be priced against a stale trace.
        self._last_trace_edges: Collection[int] = ()
        meta = self.seed_meta.get(data)
        if meta is not None:
            meta["fuzz_count"] += 1
            self._cached_total_fuzz += 1

        self._cov_before_fuzz = (
            len(self._edge_tracker._global_edge_hits)
            if hasattr(self._edge_tracker, "_global_edge_hits")
            else 0
        )
        # The timing window covers _run_target only. It used to open before
        # _dedup_mutate, which folded Python-side mutation cost into every
        # consumer of t_elapsed, and those consumers all want target time:
        #
        #   * _exec_time_anomaly -> is_slow -> `success` (see below), which
        #     is credited to the operators that ran this iteration. Ops whose
        #     own cost is milliseconds-to-seconds (gradient_descent,
        #     condstmt_solve, path_negate, crc_learn -- see
        #     _cost_adjusted_weight) pushed t_elapsed over the anomaly
        #     threshold by running at all, were credited for it, and were
        #     therefore selected more often. The contaminant was correlated
        #     with the arm being rewarded, so it compounded.
        #   * meta["total_time"] -> exec_us -> Schedule._speed_factor and the
        #     _cull_queue favored-set cost, making the favored set partly a
        #     function of which operators happened to produce each seed.
        #   * _exec_time_tracker -> suggested_timeout(), inflated by mutation.
        #
        # _dedup_mutate also calls mutate() up to EXEC_DEDUP_RETRIES + 1
        # times, so the contamination carried a multiplier driven by bloom
        # saturation rather than by anything the target did.
        mutated = self._dedup_mutate(data)
        t_start = time.monotonic()
        returncode, stderr = self._run_target(mutated)
        t_elapsed = time.monotonic() - t_start
        self.exec_count += 1
        # Effector map: read the trace hash while it is still this
        # execution's. One ctypes word read, and only when the mutant just
        # executed came from the byteflip 8/8 pass.
        self._note_det_effector()
        if self._stall_recovery_active:
            self._stall_recovery_execs += 1

        # Per-seed wall-clock cost
        if meta is not None:
            meta["total_time"] = meta.get("total_time", 0.0) + t_elapsed
            meta["cost_samples"] = meta.get("cost_samples", 0) + 1
            self._cached_total_time += t_elapsed
            self._cached_cost_samples += 1

        # Record execution time for adaptive timeout calibration
        self._exec_time_tracker.record(t_elapsed)
        # PLL observation (--pll): getattr since __new__-built test fuzzers skip wire_all.
        pll = getattr(self, "_pll", None)
        if pll is not None:
            pll.push(PLLSeries.EXEC_TIME, t_elapsed)

        # Feed the anomaly calibrator for slow-but-completed detection.
        self._exec_time_anomaly.observe(t_elapsed)

        if self.mc:
            self.mc.execs_since_refit += 1

        # Flush tracecmp buffer before collecting tokens (direct_lite mode)
        self._reset_cmplog()

        # Drain the comparison counters here, on the execution boundary, and
        # NOT on the token-collection schedule below.
        #
        # collect_counts() was only ever reached through collect_tokens(),
        # which throttles itself to every 5th and then every 20th iteration
        # once the pair pool saturates. That is right for tokens -- parsing
        # the record stream costs 14-23ms -- and wrong for the counters,
        # which are a handful of short lines read from a saved offset. On
        # the throttled schedule a delta is the sum over up to twenty
        # executions, so it says nothing about which input produced it; on
        # this schedule it is the executed input's own comparison vector.
        #
        # The reset above is what makes the shim dump: __tracecmp_flush and
        # __cmplog_reset both call __afl_cmp_dump_counts, and in subprocess
        # mode the exiting child has already dumped from __afl_cmplog_fini.
        #
        # Caveat: trimming re-executes the target on admission iterations,
        # so those vectors carry the trim runs too. Those iterations found
        # coverage by definition, which is the stronger signal anyway.
        self._last_cmp_fired = {}
        self._last_cmp_asserted = {}
        if self._cmplog:
            self._last_cmp_fired, self._last_cmp_asserted = self._cmplog.collect_counts()

        # Collect cmplog tokens — periodic sampling once pool is saturated.
        # collect_tokens() reads + parses the whole cmplog file (~14-23ms
        # with 5000 pairs); running it every iteration when the pool is
        # already saturated destroys throughput.  Adaptive: eager while
        # building the pool, then sample every N iterations.
        cmplog_found = False
        smt_found = False
        if self._cmplog:
            pr_count = len(self._cmplog.pairs)
            _interval = 1 if pr_count < 500 else (5 if pr_count < 2000 else 20)
            self._cmplog_skip_counter += 1
            _collect_now = self._cmplog_skip_counter >= _interval
            if _collect_now:
                self._cmplog_skip_counter = 0
                new_tokens = self._cmplog.collect_tokens()
            else:
                new_tokens = []
            cmplog_found = bool(new_tokens)
            # In direct_lite mode the compiled-in shim keeps the cmplog file
            # open with O_APPEND. collect_tokens() truncates the file
            # externally, but the shim's internal file offset is not reset by
            # that truncation. Call __cmplog_reset() so the next execution
            # writes at offset 0 instead of a stale position, which would
            # create a sparse file and inflate RSS.
            runner = self._inprocess_runner
            if runner and runner.direct_lite and runner._lib:
                try:
                    if hasattr(runner._lib, "__cmplog_reset"):
                        runner._lib.__cmplog_reset()
                except (AttributeError, OSError):
                    pass
            if not hasattr(self, "_dict_set"):
                self._dict_set = set(self.dictionary)
                self._dict_eps_window: list[float] = []
                self._dict_last_prune = 0
            for token in new_tokens:
                if token and token not in self._dict_set:
                    self.dictionary.append(token)
                    self._dict_set.add(token)

            # Feed checksum learner: format-aware pairs from current input
            # + cmplog heuristic pairs (gated on _collect_now to avoid
            # repeated work when the pool is saturated).
            if self.checksum_learner and _collect_now:
                fmt_pairs = self.checksum_learner.extract_format_pairs(data)
                if fmt_pairs:
                    self.checksum_learner.add_pairs(fmt_pairs)
                cmp_pairs = self.checksum_learner.extract_cmplog_pairs(data)
                if cmp_pairs:
                    self.checksum_learner.add_pairs(cmp_pairs)

            # Feed PRNG state learner: same cmplog pool, mirror-image
            # extraction (operands absent from the input rather than
            # present in it -- see core/prng_state_learner.py). Same
            # _collect_now gate as checksum_learner, for the same reason.
            if self.prng_state_learner and _collect_now:
                self.prng_state_learner.observe_execution(data)

            # Dynamic cap: scale with recent throughput.
            # High EPS → larger dictionary (more mutations explore more).
            # Low EPS → smaller dictionary (reduce overhead).
            # Window: last 500 iterations. Range: [64, 1024].
            window = 500
            if self.exec_count > 0 and due(self.exec_count, 100, "fuzzer.dict_eps"):
                elapsed = time.time() - self.start_time
                eps = (self.exec_count - self._resume_baseline_exec) / elapsed if elapsed > 0 else 0
                self._dict_eps_window.append(eps)
                if len(self._dict_eps_window) > 10:
                    self._dict_eps_window.pop(0)

            if self._dict_eps_window and self.exec_count - self._dict_last_prune >= window:
                # Use Kalman-filtered EPS if available, fall back to window avg.
                if (
                    hasattr(self, "_eps_filtered")
                    and self._eps_filtered is not None
                    and self._eps_filtered > 0
                ):
                    avg_eps = self._eps_filtered
                else:
                    avg_eps = sum(self._dict_eps_window) / len(self._dict_eps_window)
                # Map EPS to cap: 10 eps → 128, 30 eps → 256, 100+ eps → 1024
                dyn_cap = max(64, min(1024, int(avg_eps * 8)))
                if len(self.dictionary) > dyn_cap:
                    keep = max(dyn_cap // 2, 32)
                    self.dictionary = self.dictionary[-keep:]
                    self._dict_set = set(self.dictionary)
                    self._dict_last_prune = self.exec_count
            # Record redqueen matches: (offset, operand_a, operand_b)
            # for input-to-state matching during mutation.
            # Only scan new pairs (not yet seen) to avoid O(5000) per iteration.
            matches = list(meta.get("redqueen_matches", [])) if meta is not None else []
            seen = {(m[1], m[2]) for m in matches}  # dedup by (A, B)
            # Colorization taints for this seed, if the pass is enabled. Bytes
            # inside a taint can be replaced without changing the execution
            # path, so an operand found there is coincidence, not
            # input-to-state. See _colorize_seed().
            taints = self._colorize_seed(mutated)
            _pending = self._cmplog.pending_new_pairs() if meta is not None else []
            if _pending:
                _consumed = 0
                for op_a, op_b in _pending:
                    _consumed += 1
                    if len(op_a) < 2 or (op_a, op_b) in seen:
                        continue

                    # Pass 1: find op_a literally in mutated input (original redqueen)
                    pos = 0
                    matched = False
                    while pos <= len(mutated) - len(op_a):
                        idx = mutated.find(op_a, pos)
                        if idx == -1:
                            break
                        if _in_taint(taints, idx, len(op_a)):
                            # Every byte of this occurrence can be replaced
                            # without changing the path, so the target never
                            # read it: a coincidental match, not the operand
                            # the comparison actually consumed.
                            pos = idx + 1
                            continue
                        matches.append((idx, op_a, op_b))
                        seen.add((op_a, op_b))
                        matched = True
                        pos = idx + 1
                        if len(matches) >= 50:
                            break
                    if matched or len(matches) >= 50:
                        if len(matches) >= 50:
                            break
                        continue

                    # Pass 2: try finding op_b instead (reverse direction)
                    if len(op_b) >= 2:
                        pos = 0
                        while pos <= len(mutated) - len(op_b):
                            idx = mutated.find(op_b, pos)
                            if idx == -1:
                                break
                            matches.append((idx, op_b, op_a))  # swap: replace op_b with op_a
                            seen.add((op_b, op_a))
                            matched = True
                            pos = idx + 1
                            if len(matches) >= 50:
                                break
                    if matched or len(matches) >= 50:
                        if len(matches) >= 50:
                            break
                        continue

                # Only what the loop actually reached. Breaking at the match
                # cap above leaves the rest queued for the next iteration
                # instead of dropping it on the floor.
                self._cmplog.consume_new_pairs(_consumed)

            # SMT sampling pass: runs every iteration regardless of redqueen gate.
            # Adaptive sample size based on historical solve rate.
            if self._smt_solver is not None and self._cmplog.pairs:
                self._smt_solver.reset_batch()
                # Tune sample budget: if solve rate is sustained >50% we can
                # invest more; if <10% we're mostly wasting time on that input.
                _smt_q = self._smt_solver.queries_attempted
                if _smt_q > 50:
                    _rate = self._smt_solver.queries_solved / _smt_q
                    if _rate < 0.1:
                        _budget = 2
                    elif _rate > 0.5:
                        _budget = 10
                    else:
                        _budget = 5
                else:
                    _budget = 5
                sample = self._cmplog.pairs[:]
                self._rng.shuffle(sample)
                smt_counter = 0
                for op_a, op_b in sample:
                    if smt_counter >= _budget:
                        break
                    if (op_a, op_b) in seen:
                        continue
                    smt_counter += 1
                    pc = self._cmplog.pair_pc(op_a, op_b)
                    result = self._smt_solver.solve_cmplog_pair(op_a, op_b, pc=pc)
                    smt_found = False
                    if result is not None:
                        solved = result["solved_bytes"]
                        for candidate, target in [(op_a, solved), (op_b, solved)]:
                            if len(candidate) < 2:
                                continue
                            pos = 0
                            while pos <= len(mutated) - len(candidate):
                                idx = mutated.find(candidate, pos)
                                if idx == -1:
                                    break
                                matches.append((idx, candidate, target))
                                seen.add((candidate, target))
                                smt_found = True
                                pos = idx + 1
                                if len(matches) >= 50:
                                    break
                            if smt_found or len(matches) >= 50:
                                break
                        if len(matches) >= 50:
                            break
            # Concolic mode: after accumulating trace entries, solve and inject
            if (
                self._smt_solver is not None
                and self._smt_solver.mod_solving_mode == "concolic"
                and self._smt_solver.concolic_trace is not None
                and self._smt_solver.concolic_trace.has_entries()
            ):
                concolic_result = self._smt_solver.solve_concolic(mutated)
                if concolic_result is not None and concolic_result != mutated:
                    self._smt_solver.queries_solved += 1
                    self._smt_solver.batch_solved += 1
                    # Inject the concolic solution as a replacement mutation
                    matches.append((0, mutated, concolic_result))
                    smt_found = True
                else:
                    self._smt_solver.queries_failed += 1

            # Path negation: solve for an input that takes the opposite side
            # of a branch this run actually took. Unlike the concolic block
            # above — which pins every byte to a literal, giving a fully
            # determined system that reproduces the observed operands — this
            # leaves the operand window symbolic and asserts the *negated*
            # predicate, so z3 searches for a value reaching the sibling
            # branch rather than replaying one already seen.
            if self._path_solver is not None and self._cmplog is not None:
                from fuzzer_tool.core.path_constraints import records_from_collector

                records = records_from_collector(self._cmplog)
                if records:
                    negated = self._path_solver.solve_first(records, mutated)
                    if negated is not None and negated != mutated:
                        matches.append((0, mutated, negated))
                        smt_found = True
            if meta is not None:
                meta["redqueen_matches"] = matches[:50]
                # Keep legacy field for state compat
                meta["redqueen_offsets"] = [m[0] for m in meta["redqueen_matches"]]

        if due(self.exec_count, 100, "fuzzer.rss_eps"):
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss > self._peak_rss:
                self._peak_rss = rss
            elapsed = time.time() - self.start_time
            eps = (self.exec_count - self._resume_baseline_exec) / elapsed if elapsed > 0 else 0
            if eps > self._peak_eps:
                self._peak_eps = eps
            self._crash_rate_execs.append(self.exec_count)
            self._crash_rate_counts.append(self.crash_count)
            if len(self._crash_rate_execs) > CRASH_RATE_HISTORY_MAX:
                del self._crash_rate_execs[:250]
                del self._crash_rate_counts[:250]
            # Same cadence as the other periodic bookkeeping; the method's
            # own cooldown and hysteresis decide whether anything happens.
            self._maybe_retune_timeout()

        # Selections split by regime. A sniffer-gated operator has two of
        # them: it mutates a file of its own format, or -- on input that is
        # not that format -- synthesises a fresh one from scratch
        # (_op_png_chunk_mutate's parse_png_chunks() else
        # _generate_random_png() branch, and the same shape in every other
        # format op). Pooling the two makes the reported rate a function of
        # how much of the corpus happens to be that format, which is the
        # distortion the bootstrap trickle and the live-format short
        # circuit both feed.
        #
        # setdefault, so the key exists even at zero: absent has to keep
        # meaning "unknown" (a state file predating this) rather than
        # "never", or the report cannot tell them apart and falls back to
        # the raw count -- printing exactly the inflated rate this removes.
        applicable_now = getattr(self, "_last_ops_applicable", set())
        for op in set(self._last_ops_used):
            self.op_counts[op] = self.op_counts.get(op, 0) + 1
            self.op_applicable.setdefault(op, 0)
            if op in applicable_now:
                self.op_applicable[op] += 1

        # Track cmplog as its own operator
        if cmplog_found:
            self.op_counts["cmplog"] = self.op_counts.get("cmplog", 0) + 1

        # Track SMT solver as its own operator
        if smt_found:
            self.op_counts["smt_solver"] = self.op_counts.get("smt_solver", 0) + 1

        # -1 is the cross-backend timeout sentinel. stderr is not part of the
        # contract: forkserver reports loader hangs as (-1, "") after its
        # restart retry, so keying on the stderr text missed every one.
        is_timeout = returncode == -1
        if is_timeout:
            self.timeout_count += 1
            # Mark the parent seed as timeout-causing for power schedule
            parent_meta = self.seed_meta.get(self._last_parent_seed)
            if parent_meta is not None:
                parent_meta["timed_out"] = True
            self._corpus_manager.save_timeout(mutated)

        is_crash = self._is_crash(returncode, stderr)
        is_interesting = self._is_interesting(returncode, stderr)
        is_slow = False
        if not is_timeout and not is_crash:
            thresh = self._exec_time_anomaly.threshold()
            if thresh is not None and t_elapsed > thresh:
                is_slow = True
        # Check new coverage (per-target SHM in multi-target mode).
        # Use is_new_coverage_with_edges() on SHM to get both the boolean
        # and the edge set in one buffer scan, avoiding redundant scans.
        self._current_edges_cache = None  # will be set below if SHM scanned
        # Set from whichever ShmCoverage was actually scanned this iteration.
        # Only the sparse SHM path maintains per-edge maxima; the ptrace
        # bitmap has no counts to take a maximum of, so it stays 0 there.
        scanned_shm = None
        if self.multi_targets:
            active_shm = self._target_shm_covs.get(self.target)
            if active_shm:
                has_new, edge_ids = active_shm.is_new_coverage_with_edges()
                self._current_edges_cache = edge_ids
                has_new_coverage = has_new
                scanned_shm = active_shm
            else:
                # bool(): `x and x.f()` yields None (not False) when x is
                # None, and that None propagates into `success`, which
                # MonteCarloScheduler.record() feeds to float().
                has_new_coverage = bool(
                    (self.ptrace_cov and self.ptrace_cov.is_new_coverage())
                    or (self.shm_cov and self.shm_cov.is_new_coverage())
                    or (self.pt_cov and self.pt_cov.is_new_coverage())
                    or (self.branch_cov and self.branch_cov.is_new_coverage())
                )
        elif self.shm_cov:
            has_new, edge_ids = self.shm_cov.is_new_coverage_with_edges()
            self._current_edges_cache = edge_ids
            has_new_coverage = has_new
            scanned_shm = self.shm_cov
        else:
            has_new_coverage = bool(
                (self.ptrace_cov and self.ptrace_cov.is_new_coverage())
                or (self.pt_cov and self.pt_cov.is_new_coverage())
                or (self.branch_cov and self.branch_cov.is_new_coverage())
            )

        has_new_coverage, self._current_edges_cache = self._confirm_new_coverage(
            mutated,
            scanned_shm,
            has_new_coverage,
            self._current_edges_cache,
            skip=is_crash or is_timeout,
        )

        # Sampled per-execution hit mass for the stall reason's effective-edge
        # trend.  Every executed input, not only admitted ones -- a stall is
        # precisely a run of inputs that are never admitted.  Crashes and
        # timeouts are skipped: their counts are truncated executions.
        if (
            self._exec_perplexity.due()
            and scanned_shm is not None
            and not is_crash
            and not is_timeout
        ):
            self._exec_perplexity.observe(scanned_shm.get_edge_counts())

        # Performance novelty: an edge whose trip count grew substantially
        # past anything seen before. The hit-count buckets saturate (129 and
        # 10^6 are the same bucket), so this is the only signal that stays
        # live once a loop is merely being spun harder -- which is the
        # algorithmic-complexity bug class the timing channel is actually
        # good for. Suppressed on timeout and crash: a partial execution's
        # counts are truncated, not extreme.
        new_max_edges = 0
        if self._perf_novelty and scanned_shm is not None and not is_timeout and not is_crash:
            new_max_edges = scanned_shm.new_max_edges
        is_new_max = new_max_edges > 0
        if is_new_max:
            self._perf_novelty_hits += 1

        # Comparison progress: a callback family this input satisfied more
        # times, in one execution, than any input before it. Suppressed on
        # timeout and crash for the same reason as above -- a truncated
        # execution's counts are short, not extreme, and the crash handler's
        # dump would be attributed to the wrong boundary.
        #
        # Growth rather than strict `>`, exactly as _update_max_counts
        # argues: each report both rewards operators and admits an input, so
        # the number of times one callback can report over a whole campaign
        # has to be bounded. A first-ever assert reports unconditionally --
        # unlike a first-seen edge it is not already covered by another
        # signal, and there are only twenty-seven callbacks, so the
        # unbounded-looking case is bounded at twenty-seven.
        is_cmp_progress = False
        if not is_timeout and not is_crash:
            # Per-PC-site asserted counts are the fine-grained axis; folding
            # by callback merges progress at one site with stagnation at
            # another (measured memcmp (4,3) vs sites (3,3)+(1,0)).  Feed
            # BOTH axes rather than preferring one: the shim's site table is
            # fixed-size and never evicts, so once it saturates the per-site
            # view is a subset and the sites it refused would silently lose
            # their progress signal.  Keys cannot collide -- sites are
            # (callback, pc) tuples, callbacks are strings.
            site_asserted = (
                getattr(self._cmplog, "last_site_asserted", None) if self._cmplog else None
            )
            is_cmp_progress = self._record_cmp_progress(
                self._last_cmp_asserted, site_asserted or {}
            )

        # Zest validity channel: coverage reached while the target ACCEPTED
        # the input, tracked in its own map. An input that is valid and
        # covers something no valid input covered before is worth keeping
        # even when the main map has seen those edges already -- reached
        # from the parser's error path, they lead nowhere; reached from an
        # accepted input, they are the semantic stages behind the syntax
        # check. Inert unless --reject-code gave the harness a way to say
        # "rejected".
        is_new_valid_coverage = False
        validity = Validity.UNKNOWN
        if self._validity.enabled and not is_timeout and not is_crash:
            validity = self._validity.classify(returncode)
            valid_edges = (
                self._current_edges_cache
                if self._current_edges_cache is not None
                else self._get_current_edge_set()
            )
            is_new_valid_coverage = self._validity.record(validity, valid_edges)
            if is_new_valid_coverage:
                self._validity_admits += 1

        # Region liveness (item 4, handover_skittercreek_tailslayer_port.md):
        # fold this exec's coverage diff into the per-region
        # LiveBitMaskEstimator for whichever byte the mutation touched.
        # Deliberately unconditional on has_new_coverage above -- that flag
        # only means "globally new edge", which is rare; the liveness
        # estimator needs the far more common "no new edges, but still an
        # observation" samples to ever reach convergence at all. Cheap and
        # skipped outright when there's no edge data or no known parent
        # baseline to diff against.
        _liveness_parent = getattr(self, "_last_parent_seed", None)
        _liveness_offset = getattr(self, "_last_mutation_offset", None)
        if (
            self._current_edges_cache is not None
            and _liveness_parent is not None
            and _liveness_offset is not None
        ):
            parent_key = self._seed_key(_liveness_parent)
            baseline_edges = self._edge_tracker.seed_edges.get(parent_key)
            if baseline_edges:
                newly_dead = self._operators.record_coverage_diff(
                    _liveness_parent,
                    _liveness_offset,
                    baseline_edges,
                    self._current_edges_cache,
                )
                if newly_dead is not None and self._format_learner:
                    region_offset, region_width = newly_dead
                    self._format_learner.record_liveness(
                        region_offset, region_width, confirmed_dead=True, input_bytes=mutated
                    )

        # Bayesian seed quality feedback: record whether this parent seed
        # produced new coverage (Thompson sampling posterior update).
        #
        # The outcome weight is proportional to discovery rarity, per
        # seed_quality.record_outcome's contract.  When the F0 estimator is
        # wired (edge_tracker.enable_f0) it supplies that rarity signal
        # directly: in an unsaturated corpus a new edge is common, so the
        # weight is small; as the estimate saturates toward the exact count
        # each remaining discovery is rare, so the weight rises toward 1.
        # The weight is bounded above by 1.0, so the F0 signal never inflates
        # a posterior beyond the default -- it only ever re-weights.
        if self._seed_quality or self._seed_canary or self._seed_round_robin:
            parent_key = self._seed_key(data)
            weight = 1.0
            f0_est = self._edge_tracker.estimate_distinct_edges_f0()
            if f0_est is not None:
                observed = self._edge_tracker.get_cumulative_edge_count()
                # Fraction of the estimated total that has been discovered.
                # Near 1 = saturated -> each remaining discovery is rare ->
                # weight rises toward 1.  Near 0 = lots undiscovered ->
                # discoveries are common -> weight shrinks.  Bounded in
                # (0, 1], so the F0 signal never inflates a posterior
                # beyond the default -- it only ever re-weights.
                weight = observed / max(1.0, f0_est)
            if self._seed_quality:
                self._seed_quality.init_seed(parent_key)
                self._seed_quality.record_outcome(
                    parent_key, discovered=bool(has_new_coverage), weight=weight
                )
            # Same off-policy signal, fed to the seed-arena canary floor
            # (see core/schedulers/seed_canary.py) regardless of which seed
            # strategy actually picked this parent, and independent of
            # whether --bayesian is on -- canary does not need
            # BayesianSeedQuality enabled to track its own posterior.
            if self._seed_canary:
                self._seed_canary.record(parent_key, success=bool(has_new_coverage), weight=weight)
            # Elo-compatibility signal only -- round-robin's own selection
            # ignores it entirely (deterministic cycling), same as its
            # operator-side counterpart's record().
            if self._seed_round_robin:
                self._seed_round_robin.record(
                    parent_key, success=bool(has_new_coverage), weight=weight
                )

        # Credit the cmplog operands this gain is attributable to: the
        # input-to-state matches found in the input, which are the operands
        # the mutators actually had to work with. Crediting the whole
        # resident pool instead (what this did before) cannot rank anything
        # and froze the pool at the first gain -- see mark_coverage_gain.
        if has_new_coverage and self._cmplog:
            _credit = meta.get("redqueen_matches", ()) if meta is not None else ()
            self._cmplog.mark_coverage_gain(
                pairs=[(m[1], m[2]) for m in _credit],
                tokens=[m[1] for m in _credit],
            )

        # Dirichlet token posterior: credit the tokens this round inserted
        if self._dict_picker is not None:
            if has_new_coverage:
                self._dict_picker.reward(self._dict_scratch_idx)
            else:
                self._dict_picker.clear()

        # Weizz structure tags: once-per-lineage passive collection after a
        # coverage gain, gated by --weizz-tags and max_len. Uses existing
        # cmplog pairs (+ optional colorize taints); no second tracer.
        if (
            has_new_coverage
            and self.weizz_tags
            and self._cmplog is not None
            and len(mutated) <= self.weizz_tags_max_len
        ):
            self._maybe_collect_weizz_tags(mutated)

        # Record crash MI: I(byte_position; crash_outcome)
        if self._crash_mi:
            self._crash_mi.record(mutated, is_crash)

        # Write ablation log row: signal data + outcome
        if self._ablation_file and hasattr(self, "_last_pick_signals"):
            self._write_ablation_row(has_new_coverage, is_crash)

        # K-Scheduler bitmap sampling runs EVERY exec (beta needs R_i over
        # all mutations); per-seed mask attribution only for corpus-worthy
        # inputs, keyed like EdgeTracker.
        if getattr(self, "_katz_channel", None) is not None:
            bits = self._katz_channel.sample()
            if bits is not None and bits.any():
                katz_key = self._seed_key(data) if has_new_coverage else None
                self._katz_channel.record(bits, seed_key=katz_key)

        # Tang low-rank refit. Gated on the interval inside maybe_refit, and
        # placed here rather than on the pick path because the SVD plus the
        # matrix build is 100ms+ at ffmpeg scale -- three orders of magnitude
        # past the per-pick budget the seed-picker profiling established.
        if self._tang is not None:
            self._tang.maybe_refit(self._edge_tracker, self.exec_count)

        # Record edges for per-seed tracking
        if has_new_coverage:
            seed_key = self._seed_key(data)
            # Prefer sparse edge set with counts (SHM), fall back to byte bitmap (ptrace)
            # Must read from `scanned_shm` (the segment actually scanned above —
            # the per-target one in multi-target mode), not unconditionally
            # from `self.shm_cov`, which in multi-target mode is a separate,
            # unscanned shared segment.
            if scanned_shm is not None and not self.ptrace_cov:
                hit_counts = self._only_confirmed(scanned_shm.get_edge_counts())
                hit_edges = set(hit_counts.keys())
            else:
                hit_edges = (
                    self._current_edges_cache
                    if self._current_edges_cache is not None
                    else self._get_current_edge_set()
                )
                hit_counts = None
            if getattr(self, "_occupation_rarity", None) is not None and hit_counts:
                # Finite-time occupation (Du, Sec. 3): the run's own edge-visit
                # counts as a longitudinal statistical support, distinct from
                # EdgeTracker's horizontal (across-seeds) hit frequencies.
                from fuzzer_tool.core.analyzers.analyzer_occupation import OccupationMeasure

                occ = OccupationMeasure.from_counts(hit_counts)
                self._last_occupation = occ.sparse_snapshot(self._occupation_max_edges)
                self._occupation_rarity.observe(occ)
            if hit_edges:
                # Read stack depth and path hash from SHM metadata (if available)
                stack_depth = 0
                path_hash = 0
                if scanned_shm is not None:
                    stack_depth = scanned_shm.read_stack_depth()
                    path_hash = scanned_shm.read_path_hash()
                # Fallback: compute path hash from edge IDs in Python
                if path_hash == 0 and hit_edges and not isinstance(hit_edges, bytes):
                    path_hash = (
                        scanned_shm.compute_path_hash_from_edges(hit_edges)
                        if scanned_shm is not None
                        else 0
                    )
                new = self._edge_tracker.record_edges(
                    seed_key,
                    hit_edges,
                    target_name=os.path.basename(self.target) if self.multi_targets else "",
                    hit_counts=hit_counts,
                    stack_depth=stack_depth,
                    path_hash=path_hash,
                    hw_instructions=self._last_perf_deltas.get("instructions", 0),
                    hw_branches=self._last_perf_deltas.get("branches", 0),
                    hw_branch_misses=self._last_perf_deltas.get("branch_misses", 0),
                )
                self._strata_observe(data, hit_edges)
                if new:
                    self._last_new_edge_exec = self.exec_count
                    self._exec_perplexity.note_new_edge()
                    self._last_new_edge_count = len(new)
                    self._last_new_edge_ids = list(new)
                    self._last_trace_edges = hit_edges
                    self._novel_input_count += 1
                    self._saturation = None  # invalidate cached saturation
                    # Attribute new edges to the operators that ran this iteration.
                    # Proportional split: edges ÷ unique ops in _last_ops_used.
                    unique_ops = list(dict.fromkeys(self._last_ops_used))
                    if unique_ops:
                        share = len(new) / len(unique_ops)
                        for op in unique_ops:
                            self.op_edges[op] = self.op_edges.get(op, 0.0) + share
                            orientation = classify_operator_name(op).value
                            self._ro_rd_edge_counts[orientation] = (
                                self._ro_rd_edge_counts.get(orientation, 0.0) + share
                            )
                        # op_tang: attribute the actual new edge ids (not
                        # just a scalar share) to every contributing op --
                        # duplication across ops is fine, Tang's math
                        # doesn't require exclusive ownership. Kept at this
                        # call site (rather than the shared op_rewards loop
                        # below) because Tang needs the edge identities
                        # themselves, which only exist here; op_katz only
                        # needs a success flag, so it's fed from the
                        # canonical op_rewards loop instead, alongside
                        # replicator/exp3/etc, so it gets the same
                        # effective-op-filtered success signal they do
                        # rather than a locally-reinvented one.
                        if self._op_tang is not None:
                            for op in unique_ops:
                                self._op_tang.observe_new_edges(op, new)
                            self._op_tang.maybe_refit(self.exec_count)
                        # op_credit needs the edge identities too: it stores what an
                        # operator found and settles the credit against the canonical
                        # classes at read time, so a class that splits later is right.
                        if self._op_credit is not None:
                            for op in unique_ops:
                                self._op_credit.observe_new_edges(op, new)
                    # The fold follows the tracker: refit on the arms' own cadence
                    # whenever new edges (hence new seed rows) have appeared.
                    if self._matrix_substrate is not None:
                        self._matrix_substrate.maybe_refit(self._edge_tracker, self.exec_count)
                    # Separate counter for cmplog-involved edge discoveries
                    # (cumulative with the op attribution above — cmplog is a
                    #  signal source, not a mutation op, so it can overlap).
                    if cmplog_found:
                        self.op_edges["cmplog"] = self.op_edges.get("cmplog", 0.0) + len(new)
                    if smt_found:
                        self.op_edges["smt_solver"] = self.op_edges.get("smt_solver", 0.0) + len(
                            new
                        )
                    self._stall_note_coverage(len(new))
                if meta is not None and new:
                    meta["coverage_edges"] += len(new)
                    self._cached_total_edges += len(new)
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
                        if len(self._seed_secretary) > SEED_SECRETARY_MAX:
                            # Evict oldest 100 entries (dict preserves insertion order)
                            for k in list(self._seed_secretary)[:100]:
                                del self._seed_secretary[k]
                    fuzz_count = max(meta["fuzz_count"], 1) if meta else 1
                    discovery_rate = len(new) / fuzz_count
                    self._seed_secretary[seed_key].observe(discovery_rate)

        # Format learner. Runs after the record_edges block above on purpose:
        # coverage_after is len(_global_edge_hits), which only record_edges
        # grows, and _cov_before_fuzz was sampled from the same dict at the
        # top of the round. Recorded before record_edges (where this block
        # used to sit) the two were always equal, so every TimelineEntry
        # carried delta == 0 and the learner's delta statistics never saw a
        # discovery. Only record when coverage actually changes.
        if self._format_learner and self._last_ops_used and has_new_coverage:
            current_edges = (
                self._current_edges_cache
                if self._current_edges_cache is not None
                else self._get_current_edge_set()
            )
            new_edges = set()
            lost_edges = set()
            if hasattr(self, "_prev_edge_set"):
                new_edges = current_edges - self._prev_edge_set
                lost_edges = self._prev_edge_set - current_edges
            self._prev_edge_set = current_edges

            cov_after = (
                len(self._edge_tracker._global_edge_hits)
                if hasattr(self._edge_tracker, "_global_edge_hits")
                else 0
            )
            parent_meta = self.seed_meta.get(self._last_parent_seed)
            stride = parent_meta.get("record_stride") if parent_meta else None
            # Set per-format-cluster (not globally): different formats in a
            # multi-format target can have different record strides, so this
            # is routed by `mutated`'s own signature rather than compared
            # against whatever the primary cluster's stride happens to be.
            if stride is not None:
                self._format_learner.set_record_stride(stride, input_bytes=mutated)
            self._format_learner.record_transition(
                input_bytes=mutated,
                mutation_op=self._last_ops_used[0] if self._last_ops_used else "unknown",
                mutation_offset=self._last_mutation_offset,
                mutation_width=len(mutated),
                coverage_before=self._cov_before_fuzz,
                coverage_after=cov_after,
                new_edges=new_edges,
                lost_edges=lost_edges,
            )
        elif self._format_learner:
            self._prev_edge_set = (
                self._current_edges_cache
                if self._current_edges_cache is not None
                else self._get_current_edge_set()
            )

        # Update edge lifetime tracking for every execution
        if self._inprocess_runner or self.ptrace_cov or self.shm_cov:
            current_edges = (
                self._current_edges_cache
                if self._current_edges_cache is not None
                else self._get_current_edge_set()
            )
            if current_edges:
                self._edge_tracker.record_edge_lifetimes(current_edges, self.exec_count)

        # Track input-length → edge discovery correlation
        if has_new_coverage and self._length_tracker:
            new_edges = (
                self._current_edges_cache
                if self._current_edges_cache is not None
                else self._get_current_edge_set()
            )
            if new_edges:
                self._length_tracker.record(len(mutated), new_edges)

        # Compute directed distance for targeted fuzzing.  Prefer the
        # runtime average from the SHM tail (AFLGo channel, exact per-BB
        # distances accumulated in the target) when the target carries
        # the distance table; otherwise derive it in Python from the
        # edge trace.
        if self._distance and meta is not None:
            runtime_avg = self._read_runtime_avg_distance()
            if runtime_avg is not None:
                meta["avg_distance"] = runtime_avg
                self._dist_last_value = runtime_avg
                if self._dist_min_observed is None or runtime_avg < self._dist_min_observed:
                    self._dist_min_observed = runtime_avg
                if self._dist_max_observed is None or runtime_avg > self._dist_max_observed:
                    self._dist_max_observed = runtime_avg
            elif has_new_coverage:
                hit_bbs = (
                    self._current_edges_cache
                    if self._current_edges_cache is not None
                    else self._get_current_edge_set()
                )
                if hit_bbs:
                    # Record edge trace for distance computation
                    seed_key = self._seed_key(data)
                    edge_pairs = {(i, i) for i in hit_bbs}  # self-loops as BB proxies
                    self._edge_tracker.record_edge_trace(seed_key, edge_pairs)
                    # Compute average distance
                    avg_dist = self._distance.seed_distance({(i, i) for i in hit_bbs})
                    meta["avg_distance"] = avg_dist
                    if avg_dist < 20.0:  # exclude the no-valued-blocks sentinel
                        self._dist_last_value = avg_dist
                        if self._dist_min_observed is None or avg_dist < self._dist_min_observed:
                            self._dist_min_observed = avg_dist
                        if self._dist_max_observed is None or avg_dist > self._dist_max_observed:
                            self._dist_max_observed = avg_dist

        # Update annealing progress for directed mode
        if self._distance and self.exec_count > 0:
            # Anneal over first 20% of max_len-scaled iterations
            anneal_target = max(5000, self.max_len * 10)
            self._anneal_progress = min(1.0, self.exec_count / anneal_target)

        # is_cmp_progress joins the disjunction rather than replacing any
        # part of it: it is a weaker event than an edge (a comparison can be
        # satisfied more often without the branch it guards ever flipping),
        # but it arrives during exactly the stretches where the edge signal
        # is silent, which is when the cmplog-band operators are doing their
        # work and getting paid nothing for it.
        success = bool(
            is_crash
            or is_interesting
            or has_new_coverage
            or is_slow
            or is_new_max
            or is_cmp_progress
            or is_new_valid_coverage
        )

        # Per-operator credit. An operator that was selected but left the
        # buffer unchanged cannot have caused this round's outcome, so it
        # must not be recorded as a success -- but it must still be recorded
        # as a failure, or an operator that no-ops forever would never be
        # deprioritised and would keep consuming selection slots.
        effective = self._last_ops_effective if self._track_op_effect else None

        def _op_success(op: str) -> bool:
            return success and (effective is None or op in effective)

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

        if success and self._last_havoc_subops:
            # Havoc's inner branches, credited on the same signal the outer
            # bandits use. Trials are counted at application time, so a
            # branch whose guard fails accrues trials without hits and
            # decays -- the same treatment no-op operators get above.
            self._operators.credit_havoc_subops(self._last_havoc_subops)

        if success:
            # Same rule as the bandits: no-op operators didn't earn this.
            for op in effective if effective is not None else set(self._last_ops_used):
                self.op_success[op] = self.op_success.get(op, 0) + 1
                # Numerator for the mutate-regime rate. It has to be
                # restricted to the same selections as op_applicable or the
                # two do not divide: a format op that synthesised a file
                # from scratch and found an edge that way is a success on a
                # selection op_applicable never counted, which produced
                # successes against a zero denominator in the first cut of
                # this change.
                if op in applicable_now:
                    self.op_success_applicable[op] = self.op_success_applicable.get(op, 0) + 1
            if cmplog_found:
                self.op_success["cmplog"] = self.op_success.get("cmplog", 0) + 1
            if smt_found:
                self.op_success["smt_solver"] = self.op_success.get("smt_solver", 0) + 1

        # Per-operator outcome and reward, computed once. Every scheduler
        # below scored the same operator identically, so this was seven
        # copies of the same dedup-and-weight loop.
        # Bounded to [0, 1]: the contract every consumer below documents --
        # KL-UCB's Bernoulli divergence and Exp3's exponent assume it, the
        # UCB widths take b=1.0 as the range, and the Beta posteriors (mc,
        # hierarchical) add the weight as pseudo-successes, so a weight of
        # 15 was fifteen discoveries.
        op_rewards = []
        for op in dict.fromkeys(self._last_ops_used):
            ok = _op_success(op)
            w = self._cost_adjusted_weight(op, surprisal_weight if ok else 0.0)
            op_rewards.append((op, ok, min(1.0, w)))

        # Class-deduplicated shaping, applied once to the finished list because
        # the partition does not depend on which operator ran. Applying it here
        # rather than inside each scheduler is the whole point: every consumer of
        # op_rewards below sees the same shaped number, so a paired run moves one
        # variable. Kept as a pure transform (`_apply_reward_shape`) so it can be
        # driven by a test without a live campaign.
        op_rewards = _apply_reward_shape(op_rewards, self._credit_reward_shape())
        op_rewards = _apply_reward_shape(op_rewards, self._continuum_reward_shape())

        # SLOPT: credit the batch exponent drawn for this round's operator
        # with that operator's outcome. The scheme applies one operator per
        # round, so the round's reward is the arm's reward.
        if self._slopt is not None and self._last_slopt_arm is not None:
            s_op, s_len, s_exp = self._last_slopt_arm
            for op, ok, w in op_rewards:
                if op == s_op:
                    self._slopt.record(s_op, s_len, s_exp, ok, weight=w)
                    break

        if self.mc and self.mc_bandit:
            for op, ok, w in op_rewards:
                self.mc.record(op, ok, weight=w)
                self.mc.record_brier(op, ok, weight=w)
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
                    self._op_secretary[op].observe(a / (a + b))

        # On-policy learners: their update is only valid for operators they
        # drew themselves, so they learn from the rounds they selected and
        # from nothing else. Everything else below learns from every round,
        # which is sound for them -- a sample mean or Beta posterior does not
        # care who pulled the arm -- and measurably helps: fed only their own
        # draws, a 4-scheduler Elo portfolio lost ~8% of discoveries on a
        # 150-arm synthetic campaign.
        #
        # - MOpt credits the particle that drew each op; another scheduler's
        #   draw has no particle, and record(particle_id=None) spreads it over
        #   every particle. This was already gated under Elo; with Elo off it
        #   still recorded, whichever scheduler had precedence.
        # - Exp3's estimate r / p_i is unbiased only when p_i is the
        #   probability Exp3 itself drew i with. Fed another scheduler's draw
        #   it divided by a _last_probs left over from the last round Exp3
        #   selected -- possibly thousands of rounds stale.
        # - CMA-ES closes a generation after generation_size records. Counting
        #   every scheduler's records closed it after ~generation_size / N of
        #   its own evaluations, and credited its current candidate whenever
        #   another scheduler happened to pick the op that candidate had last
        #   drawn.
        selector = self._op_selector
        if self._mopt and selector == "mopt":
            # MOpt is separate: it needs the particle each operator was drawn
            # from, so it pairs each op with its first particle rather than
            # iterating the deduped list.
            rewards_by_op = {op: (ok, w) for op, ok, w in op_rewards}
            seen = set()
            for op, pid in zip(self._last_ops_used, self._last_mopt_particles, strict=False):
                if op not in seen and op in rewards_by_op:
                    ok, w = rewards_by_op[op]
                    self._mopt.record(op, ok, particle_id=pid, weight=w)
                    seen.add(op)

        # Schedulers sharing the record(op, success, weight=...) signature.
        for scheduler in (
            self._replicator if self._use_replicator else None,
            self._exp3 if selector == "exp3" else None,
            self._exp4 if selector == "exp4" else None,
            self._eps_greedy,
            self._hierarchical,
            self._gp_ucb,
            self._bo_gp_ucb,
            self._cmaes if selector == "cmaes" else None,
            self._ducb,
            self._swucb,
            self._kl_ducb,
            self._kl_swucb,
            self._cucb,
            self._cusum_ucb,
            self._fewa,
            self._fpl,
            # On-policy, like exp3/cmaes above: the importance weight is only
            # unbiased against the distribution that produced the draw, so a
            # round another scheduler selected must not reach it.
            self._corral if selector == "corral" else None,
            self._gradient,
            self._whittle,
            self._successive_elim,
            self._consolidated,
            self._moss,
            self._canary,
            self._op_katz,
            self._op_kuramoto,
            self._op_tang,
            self._op_kruskal_count,
            self._op_credit,
            self._op_tpe,
            self._op_strata,
        ):
            if scheduler is None:
                continue
            for op, ok, w in op_rewards:
                scheduler.record(op, ok, weight=w)

        # CUCB batches the round rather than updating per operator, so the
        # superarm is only complete once the loop above has run.
        if self._cucb:
            self._cucb.settle_round()

        if self._contextual:
            # LinUCB takes a feature vector rather than a success flag.
            for op, ok, w in op_rewards:
                self._contextual.record(op, self._operators._context_vector(op), w if ok else 0.0)

        if self._c2ucb:
            # Stage every operator's outcome+context into the open round;
            # settle_round() computes the actual per-arm credit once the
            # whole round's membership is known (see op_c2ucb.py's module
            # docstring for why this can't happen per-record like
            # ContextualLinUCBScheduler above).
            for op, ok, w in op_rewards:
                self._c2ucb.record(op, self._operators._context_vector(op), ok, weight=w)
            # When _track_op_effect is on, `ok` above is already per-op
            # attributed truth (see _op_success), not a broadcast outcome --
            # bypass C2UCB's own inclusion-contrast entirely and hand it
            # that truth directly. This is the documented difference
            # between C2UCB actually working and merely running; see
            # "Context dilution" in op_c2ucb.py.
            c2ucb_credits = (
                {op: (w if ok else 0.0) for op, ok, w in op_rewards}
                if self._track_op_effect
                else None
            )
            self._c2ucb.settle_round(credits=c2ucb_credits)

        # Chi-squared operator heterogeneity test
        if (
            self._chi2_operator_interval > 0
            and self.exec_count > 0
            and self.exec_count % self._chi2_operator_interval == 0
        ):
            try:
                self._run_chi2_operator_test()
            except Exception as ex:
                log.debug("Chi-squared operator test failed: %s", ex)

        # Elo: record matches between operators that were used
        # `>= 1`, not `>= 2`: a SLOPT round applies one operator 2**t times,
        # so its deduplicated set always has one member. The old guard
        # skipped every such round, and record_round's cross-round path is
        # the one that can still score it (see there).
        if self._use_elo and self._elo and self._last_ops_used:
            unique_ops = list(dict.fromkeys(self._last_ops_used))  # preserve order, dedup
            # Winners are the operators that actually changed the buffer. This
            # used to be `set(self._last_ops_used)`, which is by construction
            # the same set as `unique_ops` -- so `losers` in record_round() was
            # always empty and the entire winners-beat-losers branch, including
            # the proportional edge_counts path, was unreachable from here.
            # Measured: 1000/1000 rounds fell through to the cross-iteration
            # fallback. Crediting no-op operators is also wrong on its own
            # terms: on a single-seed corpus, splice and crossover change
            # nothing 100% of the time yet were scored as full winners on every
            # successful round.
            #
            # Buffer-change is necessary, not sufficient -- an operator can
            # change a byte the target never reads. But it strictly dominates
            # "everyone wins", and it produces a usable split in 56.5% of
            # rounds (measured over 2500 execs); the rest fall through to the
            # cross-iteration path as before.
            winners = set(self._last_ops_effective) if success else set()
            # Record unconditionally, including rounds where nothing found
            # coverage. Guarding on `if winners:` meant Elo only ever saw
            # successful iterations -- a systematic positive bias, and no
            # learning at all during a stall (measured: 4000 execs sitting
            # in random_stall produced zero recorded matches across all 106
            # arms). record_round() handles the empty-winners case via the
            # cross-iteration comparison against the previous round.
            #
            # Failed rounds vastly outnumber successes and the
            # cross-iteration path is quadratic in operators per round, so
            # sampling them was tried. It is not worth it: direct
            # instrumentation puts record_round at 0.69s of a 48.7s,
            # 2500-exec run (1.4%), while sampling 1-in-4 narrowed the
            # rating spread from 49.4 to 38.1 points. Paying 1.4% for the
            # full signal is the better trade. Note wall-clock A/B is
            # useless for judging this -- run-to-run variance on the same
            # build spanned 23s-54s, which is why the figure above comes
            # from timing record_round itself rather than total runtime.
            # Cost-aware edge_counts: previously always None, so record_round
            # took the flat score_a=1.0 path for every winner regardless of
            # how expensive it was to run. Feeding a per-op edges/time proxy
            # activates record_round's existing proportional-scoring branch
            # (edges[op] / max_edges among winners) so an expensive winner
            # that merely tied a cheap winner's edge count scores lower.
            edge_counts: dict[str, float] | None = None
            if winners and self._last_new_edge_count:
                # Split across winners, not all selected operators: a no-op
                # operator contributed no edges, and including it in the
                # denominator diluted everyone else's share.
                raw_share = self._last_new_edge_count / len(winners)
                edge_counts = {op: self._cost_adjusted_weight(op, raw_share) for op in winners}
            self._elo.record_round(unique_ops, winners, edge_counts=edge_counts, crash=is_crash)
            # Apply periodic decay
            self._elo_decay_counter += 1
            if self._elo_decay_counter >= self._elo_decay_interval:
                self._elo_decay_counter = 0
                self._elo.apply_decay()
                if self._use_canary and self._canary:
                    self._check_canary_inspection()

        # Meta-elo: record operator strategy-level match. Only when the current
        # strategy is a real selectable scheduler (random_stall is excluded, so
        # stall recovery never accrues phantom matches)
        if self._use_elo and self._elo and self._meta_strategy:
            self._record_operator_strategy_matches(surprisal_weight if success else 0.0)

        # Meta-elo: record seed strategy-level match
        if self._use_elo and self._elo and self._seed_strategy:
            score = surprisal_weight if success else 0.0
            self._record_seed_strategy_matches(score)

        # Position arena matches and burn-front credit
        self._settle_positions(Outcome.GAIN if success else Outcome.MISS, surprisal_weight)

        if self._use_shapley and self._shapley:
            new_edges = self._get_current_edge_set()
            if new_edges:
                self._shapley.record(
                    set(effective) if effective is not None else set(self._last_ops_used),
                    len(new_edges),
                    new_edges,
                )
            elif self.exec_count > 0 and self._last_ops_used:
                # Even with no edges, record a zero to track operator impact
                self._shapley.record(set(self._last_ops_used), 0, set())

        if self._use_mi and self._mi:
            current_edges = self._get_current_edge_set()
            if current_edges:
                self._mi.record(data, current_edges, self.map_size)

        if self._use_transfer_entropy and self._te:
            current_edges = self._get_current_edge_set()
            if current_edges:
                self._te_input_history.append(data[:64] if len(data) > 64 else data)
                self._te_edge_history.append(current_edges)
                if len(self._te_input_history) > self._te_history_max:
                    self._te_input_history = self._te_input_history[-self._te_history_max :]
                    self._te_edge_history = self._te_edge_history[-self._te_history_max :]
                # Update byte→edge causal map periodically
                if len(self._te_input_history) % 100 == 0 and len(self._te_input_history) > 50:
                    self._update_te_causal_map()
                    # Same cadence: feed the causal-sector graph from the TE
                    # edge history already being maintained above, rather
                    # than recomputing full pairwise TE every iteration.
                    if getattr(self, "_causal_sector", None) is not None:
                        self._update_causal_sector()

        if is_crash:
            self.crash_count += 1
            # direct_lite crashes carry no fault address (no ptrace, and the
            # guarded call reports only the signal). Re-run the input once
            # through the ptrace-attached loader to capture si_addr + regs.
            if (
                self._last_fault_addr is None
                and self._triage_ok is not False
                and self._inprocess_runner is not None
                and self._inprocess_runner.direct_lite
                and str(self.target).lower().endswith((".so", ".dylib", ".dll"))
            ):
                if self._triage_ok is None:
                    self._triage_ok = ptrace_available()
                if self._triage_ok:
                    try:
                        TargetRunner(self)._run_triage_ptrace(mutated)
                    except Exception as e:
                        log.debug("crash triage failed: %s", e)
            crash_name = self.save_crash(mutated, returncode, stderr)
            self._prune_crash_data()
            # Generate GDB/strace trace report if enabled
            if self._tracer and crash_name:
                report = self._tracer.trace(mutated, returncode)
                self._tracer.save_report(report, str(self.crashes_dir), crash_name)
            if self.mc and self.mc_cem:
                self.mc.add_elite(mutated, 3, temperature=self._temperature)
                self.mc.maybe_refit()
            # Schedule crash replay for reproducibility check.
            #
            # The key is the signature save_crash() counted this crash under,
            # published on the fuzzer rather than re-derived here. The old
            # `self.crash_sigs.get(crash_name, crash_name)` looked a FILENAME
            # up in a signature-keyed dict: it always missed, so the key
            # became the filename, and every downstream consumer that keys by
            # signature (_prune_crash_data, the reproducibility report) was
            # working against a different key space (finding #22).
            sig = self._last_crash_signature
            if self.replay_n > 0 and sig and sig not in self._crash_replays:
                self._crash_replays[sig] = []
            # Schedule sanitizer replay: re-run crash on ASAN/UBSAN targets
            if (
                (self.asan_target or self.ubsan_target)
                and sig
                and sig not in self._crash_sanitizer_replays
            ):
                self._crash_sanitizer_replays[sig] = {
                    "data": mutated,
                    "asan": None,
                    "ubsan": None,
                }
            self._record_fluctuation_observation("crash", self._get_current_edge_set())
            return True

        # is_new_max admits too. Without admission the signal cannot compound:
        # amplifying a loop is incremental, and the input that first doubled a
        # trip count is the only useful parent for the one that doubles it
        # again. Bounded by the growth factor -- an edge can report at most
        # log_1.5(2^24) < 40 times over a whole run -- so this cannot run away
        # the way PerfFuzz's strict `>` would on a counted loop.
        # is_cmp_progress admits for the reason is_new_max does: the signal
        # cannot compound without it. Getting one comparison further into a
        # length-prefixed header is only useful if the input that managed it
        # becomes the parent for the input that gets one further still --
        # and during a magic-bytes plateau, which is precisely when this
        # fires, no other criterion is admitting anything at all.
        if (
            is_interesting
            or has_new_coverage
            or is_new_max
            or is_cmp_progress
            or is_new_valid_coverage
        ):
            _corpus_len_before = len(self.corpus)
            self.save_to_corpus(mutated, parent=data)
            # Validity is a property of this execution, so it is recorded
            # here rather than reconstructed later: the seed picker reads it
            # off the metadata and nothing re-runs the input to ask again.
            if validity is not Validity.UNKNOWN:
                meta = self.seed_meta.get(mutated)
                if meta is not None:
                    meta["valid"] = validity is Validity.VALID
            self._record_lineage_insert(mutated, data, _corpus_len_before)
            self._record_entropy_gradient_credit(mutated, data, _corpus_len_before)
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
            # QEA: amplitude rotation feedback + new individual on coverage
            if self.qea:
                edge_count = (
                    len(self._edge_tracker.seed_edges.get(self._edge_tracker._last_seed_key, set()))
                    if hasattr(self._edge_tracker, "_last_seed_key")
                    else 0
                )
                qea_ind = self.qea.on_fuzz_result(
                    mutated, has_new_coverage, edge_count, self._edge_tracker
                )
                if qea_ind is not None:
                    self.qea.add_to_population(qea_ind)
            # Analyze byte sensitivity for seeds that found new coverage (optional)
            if has_new_coverage and self.shm_cov and self._use_sensitivity:
                try:
                    edges = self.shm_cov.get_edge_ids()
                    if edges:

                        def _exec_fn(data):
                            rc, _ = self._run_target(data)
                            if self.shm_cov:
                                return self.shm_cov.get_edge_ids()
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
            self._record_fluctuation_observation("success", self._get_current_edge_set())
            return True

        # ── Metropolis acceptance for non-improving / non-crashing inputs ──
        if self._metropolis and self._anneal_budget > 0 and not is_timeout:
            mutant_edges = self._get_current_edge_set()
            p_accept = self._metropolis_accept_p(data, mutant_edges)
            if self._rng.random() < p_accept:
                _corpus_len_before = len(self.corpus)
                self.save_to_corpus(mutated, parent=data)
                self._record_lineage_insert(mutated, data, _corpus_len_before)
                self._record_entropy_gradient_credit(mutated, data, _corpus_len_before)
                if self.mc and self.mc_cem:
                    self.mc.add_elite(mutated, 1, temperature=self._temperature)
                    self.mc.maybe_refit()
                self._record_fluctuation_observation("success", mutant_edges)
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

        # QEA: trigger generation boundary and rotation for non-coverage
        if self.qea:
            self.qea.on_fuzz_result(mutated, False, 0, self._edge_tracker)

        self._record_fluctuation_observation("boring", self._get_current_edge_set())
        return False

    def _record_discovery_snapshot(self):
        return self._stats.record_discovery_snapshot()

    def _run_calibration(self, max_execs: int = 1000):
        return self._stats.run_calibration(max_execs)

    def _repeat_edge_sets(self, data: bytes, n_runs: int):
        """Execute *data* ``n_runs`` times; return what each run reported.

        Returns ``(edge_sets, path_hashes, dropped)`` or None when no
        measurement could be taken (no SHM coverage, fewer than two runs, or
        an input that will not re-run). The caller decides what the numbers
        mean; this only collects them, so that the two consumers --
        :meth:`_calibrate_seed_stability`, which masks, and
        :meth:`_report_edge_id_stability`, which only reports -- cannot drift
        apart on how the measurement is taken.

        The drop counter is drained before the first run on purpose: it is
        cumulative for the segment, so without this one drop early in a
        campaign would poison every later measurement. What the accumulated
        total means afterwards is the caller's problem; both callers happen
        to treat any drop as invalidating, for the reason spelled out in
        :meth:`_calibrate_seed_stability`.
        """
        shm = self.shm_cov
        if shm is None or n_runs < 2:
            return None
        edge_sets: list[set[int]] = []
        hashes: set[int] = set()
        dropped = 0
        shm.dropped_edges_delta()  # discard drops from before this measurement
        for _ in range(n_runs):
            try:
                self._run_target(data)
            except Exception:
                return None
            edge_sets.append(shm.get_edge_ids())
            hashes.add(shm.read_path_hash())
            dropped += shm.dropped_edges_delta()
        if not edge_sets:
            return None
        return edge_sets, hashes, dropped

    def _confirm_new_coverage(self, data: bytes, shm, has_new: bool, edge_ids, *, skip=False):
        """Rerun *data* once if it reported new coverage; keep what reproduces.

        F2: an execution can report ids no later execution of the same input
        reproduces, and they were 12-18% of the "new coverage" successes on
        the default path (``edge_diagnostic.py phantom``). Nothing else catches
        them: ``_calibrate_seed_stability`` compares reruns with each other and
        the phantoms are only in the original run. Withdrawing them here, before
        ``record_edges``, the corpus and the reward fan-out read ``has_new``,
        means no consumer needs a purge API.

        Returns ``(has_new, edge_ids)`` after confirmation. Costs one execution
        per new-coverage event and nothing otherwise. A crash or timeout is
        never rerun (its ids are short, not phantom), a rerun that raises leaves
        the verdict alone, and the flag off is a pass-through.
        """
        self._confirmed_edges = None
        if not (self._confirm_novelty and has_new and shm is not None) or skip:
            return has_new, edge_ids
        new_ids = frozenset(shm.last_new_ids)
        old_bucket = bool(shm.last_old_bucket_novel)
        try:
            self._run_target(data)
        except Exception:
            return has_new, edge_ids
        result = confirm(has_new, new_ids, old_bucket, edge_ids, shm.get_edge_ids())
        self._confirm_stats["reruns"] += 1
        if result.phantoms:
            shm.reject_phantoms(result.phantoms)
            self._confirm_stats["phantom_ids"] += len(result.phantoms)
        if has_new and not result.has_new:
            self._confirm_stats["withdrawn"] += 1
        self._confirmed_edges = result.edge_ids
        return result.has_new, set(result.edge_ids)

    def _only_confirmed(self, hit_counts):
        """Restrict per-edge counts to the ids the last confirmation kept."""
        keep = self._confirmed_edges
        if keep is None or hit_counts is None:
            return hit_counts
        return {e: c for e, c in hit_counts.items() if e in keep}

    def _calibrate_seed_stability(self, data: bytes, n_runs: int = 3) -> set[int]:
        """Re-run *data* and mask edges that don't reproduce.

        AFL-style per-seed stability calibration. An identical input run
        several times should produce an identical edge set; any edge that
        appears in some runs and not others is nondeterministic — ASLR-,
        time-, thread-, or uninitialized-memory-dependent. Left alone those
        edges read as an endless supply of new coverage and permanently
        absorb mutation energy, because every re-execution "discovers" them
        again.

        Returns the set of unstable edge ids found (already masked). Empty
        set means the seed was stable, calibration was disabled, or there
        was no SHM coverage to read.

        Deliberately *not* using ``read_path_hash()`` alone as the verdict.
        The hash is order- and multiplicity-sensitive, so it diverges when
        the same edges fire in a different order or a different number of
        times — which is the common case for a loop whose trip count depends
        on nothing but scheduling, and which is not what we want to mask.
        The hash is used only as a cheap screen: identical hashes across all
        runs means definitely stable, skip the set comparison entirely.
        Divergence sends us to the per-edge set-diff, which is what actually
        decides.

        Cost is ``n_runs`` extra executions per accepted seed. That is a
        real throughput tax on a corpus that accepts often, which is why
        this is opt-in via ``--calibrate-stability`` rather than on by
        default: no A/B against a real target has been run, and the repo's
        standing rule is that an unmeasured throughput change does not
        become the default.
        """
        shm = self.shm_cov
        measured = self._repeat_edge_sets(data, n_runs)
        if measured is None:
            # No SHM, too few runs, or a seed that will not re-run: none of
            # those tell us anything about stability, so leave it unmasked
            # rather than guessing.
            return set()
        edge_sets, hashes, dropped = measured

        if dropped:
            # The verdict this function reaches is "these edges did not
            # reproduce, so they are nondeterministic", and masking is
            # permanent. A saturated table makes that inference invalid: an
            # edge whose probe window is full is discarded, and WHICH edge
            # loses the slot depends on arrival order against whatever the
            # previous execution left behind (reset_edge_map bumps a tag, it
            # does not clear the table). The same input, firing the same
            # edges in the same order, can therefore report different edge
            # sets across runs with no nondeterminism in the target at all.
            #
            # Measured on an 8192-entry table with a fixed 4000-guard
            # sequence and only the prior occupancy differing: three runs
            # shared 819 edges out of a 2,507-edge union. This function would
            # have masked the other 1,688 -- every one of them perfectly
            # reproducible -- and never unmasked them.
            #
            # Abstaining rather than masking a subset is the only available
            # answer: from the edge sets alone there is no way to tell which
            # divergences were drops. One drop is enough, because one drop is
            # one edge this run will never see; there is no threshold below
            # which the set diff becomes trustworthy again.
            self._stability_calibrations += 1
            log.info(
                "Stability calibration skipped: %d edge(s) dropped to a full map "
                "during %d runs — set divergence is not evidence of instability "
                "while coverage is being discarded",
                dropped,
                n_runs,
            )
            return set()

        # Cheap screen: one distinct hash across every run means the same
        # edges fired the same number of times in the same order.
        if len(hashes) == 1:
            self._stability_calibrations += 1
            return set()

        stable = set.intersection(*edge_sets)
        seen = set.union(*edge_sets)
        unstable = seen - stable

        self._stability_calibrations += 1
        if not unstable:
            # Hashes diverged but the edge *sets* agree: ordering or trip
            # counts moved, not which code ran. Not an unstable-edge case.
            return set()

        newly = shm.mask_edges(unstable)
        self._unstable_edges |= unstable
        if newly:
            log.info(
                "Stability calibration: %d unstable edge(s) masked (%d total) after %d runs",
                newly,
                len(self._unstable_edges),
                n_runs,
            )
        return unstable

    def discovery_rate(self):
        return self._stats.discovery_rate()

    def _run_crash_replays(self, budget_ms: float = 200):
        return self._stats.run_crash_replays(budget_ms)

    def _run_sanitizer_replays(self, budget_ms: float = 200):
        """Replay crashes on ASAN/UBSAN targets for sanitizer reports.

        Runs in subprocess mode (fork+exec with LD_PRELOAD) because
        ASAN detection doesn't work in-process via ctypes/direct_lite.
        """
        from fuzzer_tool.adapters.process import run_target_stdin
        from fuzzer_tool.core.sanitizer import SanitizerReport

        t0 = time.monotonic()
        pending = [
            (sig, info)
            for sig, info in self._crash_sanitizer_replays.items()
            if info["asan"] is None or info["ubsan"] is None
        ]
        for sig, info in pending:
            if (time.monotonic() - t0) * 1000 > budget_ms:
                break

            data = info["data"]

            # Replay on ASAN target
            if self.asan_target and info["asan"] is None:
                env = os.environ.copy()
                self._setup_asan_env(env)
                try:
                    rc, stderr, _ = run_target_stdin(self.asan_target, data, self.timeout, env=env)
                    report = SanitizerReport.parse(stderr)
                    info["asan"] = {
                        "rc": rc,
                        "report": report.to_dict() if report and report.is_valid() else None,
                        "stderr": stderr[:4096],
                    }
                except Exception as e:
                    info["asan"] = {"rc": -2, "error": str(e)}

            # Replay on UBSAN target
            if self.ubsan_target and info["ubsan"] is None:
                env = os.environ.copy()
                self._setup_ubsan_env(env)
                try:
                    rc, stderr, _ = run_target_stdin(self.ubsan_target, data, self.timeout, env=env)
                    report = SanitizerReport.parse(stderr)
                    info["ubsan"] = {
                        "rc": rc,
                        "report": report.to_dict() if report and report.is_valid() else None,
                        "stderr": stderr[:4096],
                    }
                except Exception as e:
                    info["ubsan"] = {"rc": -2, "error": str(e)}

            # Save reports when both are done (or one is done and the other is absent)
            if (
                info["asan"] is not None
                and info["ubsan"] is not None
                or self.asan_target
                and info["asan"] is not None
                and not self.ubsan_target
                or self.ubsan_target
                and info["ubsan"] is not None
                and not self.asan_target
            ):
                self._save_sanitizer_reports(sig, info)

    @staticmethod
    def _setup_asan_env(env: dict) -> None:
        """Set LD_PRELOAD and ASAN_OPTIONS for an ASAN-instrumented target."""
        from fuzzer_tool.cli.ldpreload_wrapper import ASAN_RELEASE_TO_OS, _resolve_asan

        libasan = _resolve_asan()
        if libasan:
            ld_preload = env.get("LD_PRELOAD", "")
            parts = [p for p in ld_preload.split(":") if p] if ld_preload else []
            parts.insert(0, libasan)
            env["LD_PRELOAD"] = ":".join(parts)
        asan_opts = env.get("ASAN_OPTIONS", "")
        opt_parts = [p for p in asan_opts.split(":") if p] if asan_opts else []
        seen = {p.split("=")[0] for p in opt_parts}
        for opt in ("halt_on_error=0", "abort_on_error=0", "detect_leaks=0", ASAN_RELEASE_TO_OS):
            key = opt.split("=")[0]
            if key not in seen:
                opt_parts.append(opt)
                seen.add(key)
        env["ASAN_OPTIONS"] = ":".join(opt_parts)

    @staticmethod
    def _setup_ubsan_env(env: dict) -> None:
        """Set UBSAN_OPTIONS for an UBSAN-instrumented target."""
        ubsan_opts = env.get("UBSAN_OPTIONS", "")
        opt_parts = [p for p in ubsan_opts.split(":") if p] if ubsan_opts else []
        seen = {p.split("=")[0] for p in opt_parts}
        for opt in ("halt_on_error=1", "abort_on_error=1", "print_stacktrace=1"):
            key = opt.split("=")[0]
            if key not in seen:
                opt_parts.append(opt)
                seen.add(key)
        env["UBSAN_OPTIONS"] = ":".join(opt_parts)

    def _save_sanitizer_reports(self, sig: str, info: dict) -> None:
        """Write ASAN/UBSAN reports as JSON alongside the crash file."""
        import json as _json

        report_path = self.crashes_dir / f"{sig[:16]}_sanitizer_report.json"
        try:
            existing = {}
            if report_path.exists():
                existing = _json.loads(report_path.read_text())
            existing.update(
                {
                    "asan_result": info.get("asan"),
                    "ubsan_result": info.get("ubsan"),
                }
            )
            report_path.write_text(_json.dumps(existing, indent=2))
        except Exception:
            log.warning("Failed to save sanitizer report for sig=%s", sig)

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

    def _update_causal_sector(self):
        return self._stats.update_causal_sector()

    def _get_te_weighted_position(self, input_length: int):
        return self._stats.get_te_weighted_position(input_length)

    def _get_phase_weighted_position(self, input_length: int, stride: int | None):
        return self._stats.get_phase_weighted_position(input_length, stride)

    def _colorize_seed(self, data: bytes):
        """Colorization taints for ``data``, or ``None`` when disabled.

        AFL++/Redqueen colorization replaces every byte it can while holding
        the execution path fixed. What survives is the set of bytes the
        target actually reads, and what gets replaced is a *taint region* --
        provably path-irrelevant.

        The redqueen match loop needs this because it accepts every literal
        occurrence of a comparison operand as an input-to-state candidate,
        filtered only by ``len(op_a) >= 2``. A two-byte operand hits a 4 KiB
        input roughly sixteen times by chance, each coincidence becomes an
        entry in a list capped at 50, and the operator then substitutes at a
        random one. Colorization is what tells the two apart.

        Off by default (``--colorize``). It costs real executions -- up to
        ``2 * len(data)``, bounded here to keep a large seed from stalling
        the loop -- and buys precision, not throughput, so the trade needs
        measuring per target before it becomes a default. Results are cached
        per seed; a seed is colorized at most once.

        Returns None when disabled, unavailable, or the budget was spent
        without a usable answer. Callers treat None as "no filtering",
        which is the pre-existing behaviour.
        """
        if not getattr(self, "colorize", False):
            return None
        if not data or self.shm_cov is None:
            return None

        cache = self._colorize_taint_cache
        key = hash(data)
        if key in cache:
            return cache[key]

        def exec_fn(candidate: bytes) -> int:
            """Path checksum for one execution. 0 means 'unknown'."""
            try:
                self._runner.run_target(candidate)
            except Exception:
                return 0
            path_hash = self.shm_cov.read_path_hash()
            if path_hash == 0:
                # Shim without the rolling hash: fall back to the edge set,
                # which is coarser (order- and multiplicity-insensitive) but
                # still separates different paths.
                edges = self._get_current_edge_set()
                path_hash = self.shm_cov.compute_path_hash_from_edges(edges) if edges else 0
            return path_hash

        try:
            from fuzzer_tool.core.colorization import colorize

            result = colorize(
                bytes(data),
                exec_fn,
                use_type_aware=True,
                max_execs=min(2 * len(data), self.colorize_max_execs),
                rng=self._rng,
            )
        except Exception:
            log.debug("colorization failed for a seed; continuing unfiltered", exc_info=True)
            cache[key] = None
            return None

        self.exec_count += result.exec_count
        self._colorize_execs += result.exec_count
        taints = result.taints or None

        # Bounded cache: colorization is per-seed and the corpus grows.
        if len(cache) > 512:
            cache.clear()
        cache[key] = taints
        return taints

    def _maybe_collect_weizz_tags(self, data: bytes) -> None:
        """Passive Weizz structure-tag collection for one coverage-gaining seed.

        Once-per-lineage: if ``seed_meta`` already carries a non-dirty RLE
        map for this exact byte string, skip. Size-gated by
        ``weizz_tags_max_len``. Does not run the differential path (that
        stays available via ``get_deps`` for callers that need causality).
        """
        if not data or self._cmplog is None:
            return
        meta = self.seed_meta.get(data)
        if meta is not None:
            existing = meta.get("weizz_tags_rle")
            if existing and not meta.get("weizz_tags_dirty", False):
                return
        try:
            from fuzzer_tool.core.weizz_tags import (
                TagCollectorConfig,
                attach_tags_to_meta,
                collect_structure_map,
            )

            taints = None
            if self.colorize:
                taints = self._colorize_seed(data)
            smap = collect_structure_map(
                data,
                self._cmplog,
                colorization_result=taints,
                config=TagCollectorConfig(max_input_len=self.weizz_tags_max_len),
            )
            if smap is None or smap.ntypes == 0:
                return
            if meta is None:
                meta = {
                    "fuzz_count": 0,
                    "coverage_edges": 0,
                    "momentum": 0.0,
                    "edge_bitmap": bytearray(0),
                    "redqueen_offsets": [],
                    "added_at": time.time(),
                    "record_stride": None,
                    "seed_passed_det": False,
                }
                self.seed_meta[data] = meta
            meta = attach_tags_to_meta(meta, smap)
            self._weizz_tags_collected += 1
        except Exception:  # noqa: BLE001 — never take down the fuzz loop
            log.debug("weizz tag collection failed", exc_info=True)

    def _metropolis_accept_p(self, parent: bytes, mutant_edges: set[int]) -> float:
        """Admission probability for a non-improving mutant of *parent*.

        ``ΔE = |P| / |M ∪ P|`` against the parent's recorded path (see
        core/metropolis.py). ``.get`` so an untracked parent is not inserted.
        """
        parent_edges = self._edge_tracker.seed_edges.get(self._seed_key(parent), set())
        delta_e = path_energy(mutant_edges, parent_edges)
        return accept_prob(delta_e, self._temperature)

    def _get_current_edge_set(self) -> set[int]:
        """Return the set of currently-active edge IDs.

        Works for sparse-entry SHM (edge_id from struct entries) and
        byte-bitmap ptrace coverage (non-zero byte positions).
        """
        return self._stats.get_current_edge_set()

    def _current_scheduling_badness(self) -> float:
        """Badness score in [0, 1] for badness-indexed scheduler floors.

        Reads the coverage-regime classifier already maintained by
        `_regime` (``core/analyzers/analyzer_coverage_regime.py`` --
        SUBCRITICAL/CRITICAL/SUPERCRITICAL) and maps it to a scalar via
        ``core.badness_floor.badness_from_regime``. No new signal: this
        only translates an existing one into a form ``OpKatzScheduler``'s
        (and, potentially, other schedulers') exploration floor can key
        off of. See docs/handover/handover_badness_indexed_floor_2026-09-21.md.

        `self._regime` may not exist yet the first time this is called
        (`_op_katz` is constructed earlier in `__init__` than
        `_regime`, and its own `badness_fn` is only invoked later, during
        `select_op` calls in the fuzzing loop, by which point `__init__`
        has finished -- the `getattr` default is defensive insurance
        against that ordering changing, not insurance this is expected to
        need at runtime).
        """
        from fuzzer_tool.core.badness_floor import badness_from_regime

        regime_detector = getattr(self, "_regime", None)
        regime = getattr(regime_detector, "regime", None)
        return badness_from_regime(regime)

    def _read_runtime_avg_distance(self) -> float | None:
        """Read the per-execution average distance from the SHM tail.

        Returns the unscaled average when the target reported valued
        blocks (dist_count > 0), else None (fall back to Python-side
        distance computation).
        """
        shm = (
            self._target_shm_covs.get(self.target, self.shm_cov)
            if self.multi_targets
            else self.shm_cov
        )
        if shm is None:
            return None
        try:
            dist_sum, dist_count = shm.read_distance_tail()
        except (AttributeError, OSError):
            return None
        if dist_count <= 0:
            return None
        return dist_sum / dist_count / 100.0

    def _format_elapsed(self):
        return self._stats.format_elapsed()

    def print_stats(self):
        return self._stats.print_stats()

    def _last_avg_eps(self) -> float:
        """Mean of the last `_eps_history_max` avg-eps samples.

        Returns 0 until the window is full, so callers fall back to a fixed
        interval instead of chasing the unstable first EPS readings.
        """
        if len(self._eps_history) < self._eps_history_max:
            return 0.0
        return sum(self._eps_history) / len(self._eps_history)

    def _stats_effective_interval(self) -> int:
        """Stats-tick spacing in execs.

        The first tick uses ~1 second of work (1x EPS) so the first [*] execs
        line appears promptly; subsequent ticks space at ~10 seconds of work
        using 10x the mean of the last 10 avg-eps samples rather than a single
        raw reading.  Before the window fills, the fixed stats interval is
        used (the first samples are too unstable to trust).
        """
        if not self._eps_history:
            elapsed = max(time.time() - self.start_time, 1e-9)
            eps_now = (self.exec_count - self._resume_baseline_exec) / elapsed
            if eps_now <= 0:
                return self.stats_interval
            return max(1, int(eps_now))
        last_avg_eps = self._last_avg_eps()
        if last_avg_eps <= 0:
            return self.stats_interval
        return max(1, int(10 * last_avg_eps))

    def _continuum_graph(self) -> dict:
        """Frontier adjacency from edge co-occurrence, refreshed on a budget.

        ``edge_cooccurrence`` walks every seed's edge set to build its
        edge->seeds map, so it is O(sum of seed edge counts) per call --
        affordable occasionally, not every tick (Hard Rule 41).  The cached
        graph is reused in between; a stale graph only softens the pressure
        gradient, it cannot select an operator the caller did not offer.
        """
        self._continuum_graph_tick += 1
        stale = self._continuum_graph_tick % CONTINUUM_GRAPH_TICKS == 1
        if self._continuum_adjacency and not stale:
            return self._continuum_adjacency

        adjacency: dict[int, set[int]] = {}
        for edge_a, edge_b, _jaccard in self._edge_tracker.edge_cooccurrence(
            top_k=CONTINUUM_GRAPH_PAIRS
        ):
            adjacency.setdefault(edge_a, set()).add(edge_b)
            adjacency.setdefault(edge_b, set()).add(edge_a)

        self._continuum_adjacency = adjacency
        return adjacency

    def _observe_continuum(self, delta: int, hit_counts: dict) -> None:
        """Recompute the steady continuum fields for this tick."""
        if self._continuum is None:
            return

        stats = self.mc.bandit_stats() if (self.mc and self.mc_bandit) else {}
        wins = sum(s for s, _f in stats.values())
        losses = sum(f for _s, f in stats.values())
        total = wins + losses
        failure_rate = losses / total if total > 0 else 0.0

        self._continuum.observe(
            occupancy=hit_counts,
            adjacency=self._continuum_graph(),
            velocity=float(delta),
            failure_rate=failure_rate,
        )

    def _record_entropy_sample(self, sh):
        """Append a Shannon-entropy sample and trim history to a bounded size."""
        self._entropy_execs.append(self.exec_count)
        self._entropy_vals.append(sh)
        if len(self._entropy_execs) > ENTROPY_HISTORY_MAX:
            self._entropy_execs = self._entropy_execs[-ENTROPY_HISTORY_TRIM:]
            self._entropy_vals = self._entropy_vals[-ENTROPY_HISTORY_TRIM:]

    def _compute_entropy_flat(self):
        """Return whether the recent Shannon-entropy rate of change is flat.

        Returns True if entropy has been flat over the last N samples,
        False if it is still changing (redistribution, not stagnation),
        or None if there aren't enough samples yet to measure the rate.
        """
        if len(self._entropy_execs) < ENTROPY_WINDOW:
            return None
        recent_execs = self._entropy_execs[-ENTROPY_WINDOW:]
        recent_vals = self._entropy_vals[-ENTROPY_WINDOW:]
        dt = recent_execs[-1] - recent_execs[0]
        if dt <= 0:
            return None
        dS = abs(recent_vals[-1] - recent_vals[0])
        entropy_rate = dS / dt
        return entropy_rate < ENTROPY_FLAT_THRESHOLD

    @staticmethod
    def _seed_global_numpy(seed: int | None) -> None:
        """Seed the legacy global ``np.random`` state.

        Separate from ``RandPool``, which owns an independent
        ``default_rng`` Generator. ``np.random.seed`` accepts only
        ``[0, 2**32)``, so a wider seed is folded rather than raising;
        ``None`` reseeds from OS entropy, matching ``random.seed(None)``.

        Args:
            seed: The run seed, or None for an unseeded run.
        """
        if not _HAS_NUMPY:
            return
        np.random.seed(None if seed is None else seed & SEED_MASK_32)

    def _derive_stall_seed(self) -> int:
        """Return the seed to apply for the current stall reseed.

        When the run was given a seed, the new seed is a pure function of
        ``(self.seed, self._stall_reseed_count)`` — not of the draw history,
        which varies with target execution timing — so a seeded run stays
        reproducible across the reseed. Without a seed there is nothing to
        be reproducible against, so OS entropy is used.

        Returns:
            A seed in ``[0, 2**32)``, suitable for ``np.random.seed``.
        """
        if self.seed is None:
            return int.from_bytes(os.urandom(4), "little")
        x = (self.seed + self._stall_reseed_count * SEED_MIX_GAMMA) & SEED_MASK_64
        x = ((x ^ (x >> 30)) * SEED_MIX_A) & SEED_MASK_64
        x = ((x ^ (x >> 27)) * SEED_MIX_B) & SEED_MASK_64
        x ^= x >> 31
        return x & SEED_MASK_32

    def _reseed_after_stall(self) -> int:
        """Reseed every RNG the mutation hotpath draws from.

        A stall means the current mutation trajectory has stopped producing
        edges. Recovery already widens *what* is mutated; reseeding changes
        *which* stream those choices come from, so a resumed run does not
        replay the same exhausted sequence.

        All three streams are reseeded together, matching how ``__init__``
        seeds them: ``random`` drives the non-hotpath choices, ``RandPool``
        owns its own ``default_rng`` Generator and backs the mutation
        hotpath, and the global ``np.random`` state backs QEA and the
        Monte-Carlo scheduler. ``RandPool`` is NOT backed by global
        ``np.random`` — an earlier version of this docstring said it was,
        which is why the global went unseeded. ``RandPool.reseed`` also drops
        the pre-fetched pool, which would otherwise keep dispensing
        old-stream values for another ``_POOL_ENTRIES`` draws.

        Returns:
            The seed that was applied.
        """
        self._stall_reseed_count += 1
        new_seed = self._derive_stall_seed()
        random.seed(new_seed)
        self._seed_global_numpy(new_seed)
        if _HAS_NUMPY:
            self._rng.reseed(new_seed)
        self._last_stall_seed = new_seed
        print(f"[*] Reseeded RNG → {new_seed} (stall reseed #{self._stall_reseed_count})")
        return new_seed

    def _stall_recovery_enter(self, reason: str, execs_since_edge: int) -> None:
        """Engage the stall relay and open a new telemetry cycle."""
        prev = self._stall_last_engage_exec
        self._stall_recovery_count += 1
        self._stall_recovery_active = True
        self._stall_engaged_at = self.exec_count
        self._stall_last_engage_exec = self.exec_count
        self._stall_edges_in_recovery = 0
        print(
            f"\n[*] STALL #{self._stall_recovery_count}: {reason} in "
            f"{execs_since_edge} execs, switching to random mode"
        )
        if prev is not None:
            # Engage-to-engage spacing is the relay period. Recorded on
            # engage rather than on release because a campaign that ends
            # mid-recovery would otherwise drop its last period silently.
            self._stall_cycles.append(
                {
                    "engage_exec": prev,
                    "period": self.exec_count - prev,
                    "edges_in_recovery": self._stall_cycle_edges,
                    "recovery_execs": self._stall_cycle_execs,
                }
            )
        self._stall_cycle_edges = 0
        self._stall_cycle_execs = 0

    def _stall_note_coverage(self, n_new: int) -> bool:
        """Credit ``n_new`` edges to the relay; return True if it released.

        Deliberately a method and not an inline block in the fuzz loop. The
        release condition is what ``--stall-release-edges`` tunes, so a test
        that mirrors it inline cannot detect the condition changing -- the
        first version of tests/test_regression_stall_relay.py did mirror it,
        and stripping the dwell from production left all 13 cases passing.

        The dwell is counted in EDGES, not execs: a burst of ``n`` arriving
        in one exec satisfies a dwell of ``n`` immediately, because the dwell
        asks for evidence of renewed discovery, not for elapsed time.
        """
        if not self._stall_recovery_active or n_new <= 0:
            return False
        self._stall_edges_active += n_new
        self._stall_edges_in_recovery += n_new
        if self._stall_edges_in_recovery < self._stall_release_edges:
            return False
        print(
            f"\n[*] RECOVERED: found {self._stall_edges_in_recovery} new edges "
            f"by exec {self.exec_count}, resuming normal mode"
        )
        self._stall_recovery_exit("new coverage")
        return True

    def _stall_recovery_exit(self, reason: str) -> None:
        """Release the stall relay. The single definition of the exit.

        Three call sites used to assign ``_stall_recovery_active = False``
        directly (new coverage, the SUPERCRITICAL regime clear, and the
        release dwell), which is how a release condition ends up with three
        slightly different meanings. Everything that must happen on release
        -- cycle accounting, dwell reset -- happens here or nowhere.
        """
        if not self._stall_recovery_active:
            return
        self._stall_recovery_active = False
        self._stall_cycle_edges = self._stall_edges_in_recovery
        if self._stall_engaged_at is not None:
            self._stall_cycle_execs = self.exec_count - self._stall_engaged_at
        self._stall_edges_in_recovery = 0
        self._stall_release_reason = reason

    def _stall_relay_stats(self) -> dict:
        """Relay period and amplitude, for tuning anything built on top.

        ``amplitude`` is the ratio of discovery rate while engaged to
        discovery rate while released -- the open question Tier 1.2 of the
        control-theory handover cannot answer without it, since adding
        release dwell trades switching frequency for duty cycle and which
        side of that trade is better depends entirely on whether recovery
        mode is more or less productive per exec than normal mode.

        A ratio above 1.0 says recovery earns its execs and more dwell is
        probably right; below 1.0 says the relay should be released sooner,
        not held longer. Returns ``None`` for fields with no data rather
        than 0.0: never-measured and measured-zero are different answers.
        """
        idle_execs = max(0, self.exec_count - self._stall_recovery_execs)
        total_edges = self._edge_tracker.get_cumulative_edge_count()
        idle_edges = max(0, total_edges - self._stall_edges_active)
        active_rate = (
            self._stall_edges_active / self._stall_recovery_execs
            if self._stall_recovery_execs > 0
            else None
        )
        idle_rate = idle_edges / idle_execs if idle_execs > 0 else None
        periods = [c["period"] for c in self._stall_cycles if c.get("period")]
        return {
            "cycles": len(self._stall_cycles),
            "engagements": self._stall_recovery_count,
            "release_edges": self._stall_release_edges,
            "period_mean": sum(periods) / len(periods) if periods else None,
            "period_min": min(periods) if periods else None,
            "period_max": max(periods) if periods else None,
            "duty": (self._stall_recovery_execs / self.exec_count if self.exec_count > 0 else None),
            "edges_active": self._stall_edges_active,
            "edges_idle": idle_edges,
            "rate_active": active_rate,
            "rate_idle": idle_rate,
            "amplitude": (
                active_rate / idle_rate
                if active_rate is not None and idle_rate not in (None, 0.0)
                else None
            ),
        }

    def _refill_format_seeds(self) -> None:
        """Queue field-targeted variants of the last fuzzed seed.

        Runs on the stats tick, at most every FORMAT_SEED_EVERY_EXECS execs
        and only once the previous batch is drained. mutate() hands out one
        per round, so each goes through the normal exec/coverage/save path.
        """
        learner = self._format_learner
        if learner is None or self._format_seed_queue:
            return
        if self.exec_count - self._format_seed_exec < FORMAT_SEED_EVERY_EXECS:
            return

        base = getattr(self, "_last_parent_seed", None)
        if not base or not learner.hypotheses:
            return
        self._format_seed_exec = self.exec_count

        # One generator for the whole run so its stats accumulate.
        gen = self._format_seed_generator
        if gen is None:
            gen = FormatSeedGenerator(learner.hypotheses, rng=self._rng)
            self._format_seed_generator = gen
        else:
            gen.set_fields(learner.hypotheses)

        limit = self.max_len
        for s in gen.generate(base, n_seeds=FORMAT_SEED_BUDGET):
            self._format_seed_queue.append(s.data[:limit] if limit > 0 else s.data)

    def _maybe_trigger_stall_recovery(self, execs_since_edge):
        """Activate stall recovery unless entropy shows active redistribution.

        Entropy rate confirmation: if entropy is still changing, this is
        redistribution among existing edges, not genuine stagnation, so
        recovery is skipped this round. If there aren't enough samples yet
        to measure the rate (`entropy_flat is None`), fall back to the
        no-new-edges signal alone.

        structure function confirmation: the noise type of the incremental edge
        discovery rate provides a leading indicator. White noise = normal
        exploration (skip stall). Flicker noise = correlated discoveries
        approaching saturation (reduce threshold). Random walk = integrated
        signal, likely genuine stall (bypass entropy gate).

        Dispersion index override: complements structure function by resolving
        a blind spot — a buffer dominated by zeros (genuine stall) and a
        buffer with rare bursts of discoveries (bursty exploration) both
        produce near-zero structure-function deviation, but dispersion index D tells
        them apart: D › 1.5 = bursty (override stall), D « 0.3 = stall.
        """
        entropy_flat = self._compute_entropy_flat()
        if entropy_flat is False:
            return False

        # Consult Structure-function detector for noise-type signal
        noise = self._structure_fn.noise_type()
        structure_slope = self._structure_fn.noise_slope()
        dispersion = self._structure_fn.dispersion()

        reason = "no new edges"
        if entropy_flat:
            reason += " + flat entropy"
        if noise != "unknown":
            reason += (
                f" + {noise} noise (slope={structure_slope:+.2f})"
                if structure_slope is not None
                else f" + {noise} noise"
            )
        if dispersion is not None:
            reason += f" + D={dispersion:.2f}"

        # What the edge signal cannot distinguish: a stall the campaign is
        # still driving into a comparison it never passes, versus a stall
        # where it stopped reaching that comparison at all. Both are "no new
        # edges for N execs"; the first wants budget on the wall, the second
        # wants the parent seeds back.
        if self._cmplog is not None:
            walls = self._cmplog.wall_summary()
            if walls:
                reason += f" + {walls}"

        # Dispersion index override: a *significantly* overdispersed D
        # (chi-squared dispersion test, not a fixed cutoff — see
        # StructureFunctionDetector.is_overdispersed) means clusters of
        # discoveries with gaps — NOT a stall even if structure-function says stalled.
        if self._structure_fn.is_overdispersed():
            return False

        # Noise-type gating and threshold adjustment
        if noise == "active":
            # Random exploration is normal — don't stall even if entropy is flat
            return False

        effective_threshold = self._stall_threshold
        if noise == "fatiguing":
            # Pre-stall: discovery rate is trending down.
            # Halve the threshold to catch it earlier.
            effective_threshold = max(self._stall_threshold // 2, 100)
            if execs_since_edge < effective_threshold:
                return False

        if noise == "stalled":
            # Near-zero variance confirms genuine stall.
            # Bypass entropy gate and use minimal threshold.
            # Significantly underdispersed D (chi-squared test) further
            # confirms stall — use the most aggressive threshold.
            if self._structure_fn.is_underdispersed():
                effective_threshold = max(self._stall_threshold // 8, 25)
            else:
                effective_threshold = max(self._stall_threshold // 4, 50)

        # Without any detector signal (structure-function unknown + entropy unknown),
        # fall through to original behavior (trigger on no-new-edges alone).
        if noise not in ("fatiguing", "stalled", "unknown") and entropy_flat is None:
            return False

        # Check coverage growth model for saturation
        growth = self._edge_tracker.coverage_growth_model()
        if growth["confidence"] > 0.3 and growth["current_rate"] < 0.001:
            reason += " + near-saturation"

        # Where the executions' hits went since coverage last grew, against
        # the window that ended in that discovery.  A fall with the raw edge
        # count flat is the collapse this reason had no equivalent of.
        reason += self._exec_perplexity.reason_suffix()

        self._stall_recovery_enter(reason, execs_since_edge)

        # Optionally reseed the RNGs so recovery explores a different
        # mutation stream rather than continuing the exhausted one.
        if self._reseed_on_stall:
            self._reseed_after_stall()

        # Optionally resize the coverage bitmap to reduce hash collision risk
        if self._resize_map_on_stall and self.shm_cov:
            # Ask the shim how many edges it had to throw away. Without this
            # the load factor is computed only from edges that survived, so
            # a saturated table looks under-full and never triggers a resize
            # — the exact case this recovery path exists for.
            dropped = self.shm_cov.read_dropped_edges()
            new_size = self._edge_tracker.recommended_map_size(dropped_edges=dropped)
            if dropped:
                pinned = " (shim counter pinned)" if self.shm_cov.drop_counter_saturated() else ""
                print(
                    f"[!] Coverage map saturated: {dropped:,} edges dropped{pinned} — "
                    f"coverage was lost, not merely delayed"
                )
            if new_size > self.shm_cov.size:
                current = self.shm_cov.size
                print(
                    f"[*] Resizing SHM {current:,} → {new_size:,} entries "
                    f"(stall-triggered, n={len(self._edge_tracker._global_edge_hits)}, "
                    f"dropped={dropped:,})"
                )
                self.shm_cov.resize(new_size)
                self.map_size = new_size
                # on_resize() sets map_size and clears only what a resize
                # actually invalidates — nothing, on the SHM path, since
                # edge IDs do not depend on map_size.
                self._edge_tracker.on_resize(new_size)
                # Drops recorded against the old table say nothing about the
                # new one; clearing keeps the next decision on fresh evidence.
                self.shm_cov.reset_dropped_edges()
                self._drop_resize_checked_at = self.exec_count
                os.environ["__AFL_SHM_ID"] = self.shm_cov.env_id
                os.environ["AFL_MAP_SIZE"] = str(new_size)
                if self._inprocess_runner:
                    self._inprocess_runner.update_shm_after_resize(
                        self.shm_cov._ptr, new_size, self.shm_cov.env_id
                    )
                if self._forkserver:
                    self._forkserver.update_shm_after_resize(self.shm_cov.env_id, new_size)
        return True

    def _maybe_resize_on_drops(self) -> None:
        """Grow the map when the shim reports drops, without waiting for a stall.

        Drops were already consumed by ``_maybe_trigger_stall_recovery``, but
        only there -- and that path does not run until ``--stall`` executions
        have passed with no new edge (default 1,000), and only when
        ``--resize-map-on-stall`` is set. So the one honest saturation signal
        in the system was read late, conditionally, and (while it was a
        16-bit packed field) pinned: measured at 1,953 drops per execution it
        reached its 65,535 ceiling after 34 executions, meaning every value
        that consumer ever read on a saturating target was the ceiling.

        A drop is not a symptom awaiting confirmation. It is the fuzzer being
        told, by the only component that can know, that coverage it will
        never see has already been discarded. Waiting for a stall to act on
        it inverts cause and effect: the stall is downstream of the lost
        coverage.

        Cheap and rate-limited on purpose -- one word read and a decision
        that returns 0 in the common case. The rate limit also stops a fresh
        table from being re-proposed on consecutive ticks before it has
        produced any evidence of its own.
        """
        if not self._resize_map_on_stall or self.shm_cov is None:
            return
        if self.exec_count - self._drop_resize_checked_at < self._drop_resize_interval:
            return
        self._drop_resize_checked_at = self.exec_count
        dropped = self.shm_cov.read_dropped_edges()
        if not dropped:
            return
        new_size = self._edge_tracker.recommended_map_size(dropped_edges=dropped)
        if new_size <= self.shm_cov.size:
            # Already at the cap, or the recommendation does not beat the
            # current size. Say so once rather than silently doing nothing: a
            # run losing coverage it cannot size its way out of is something
            # the operator should know about, and the stall path was the only
            # other place this was ever reported.
            if not self._drop_cap_warned:
                self._drop_cap_warned = True
                print(
                    f"[!] Coverage map saturated at {self.shm_cov.size:,} entries: "
                    f"{dropped:,} edges dropped and no larger map available "
                    f"(AFL_MAP_SIZE_MAX) — consider lowering __AFL_CTX_BITS"
                )
            return
        print(
            f"[*] Resizing SHM {self.shm_cov.size:,} → {new_size:,} entries "
            f"(drop-triggered, dropped={dropped:,})"
        )
        self.shm_cov.resize(new_size)
        self.map_size = new_size
        self._edge_tracker.on_resize(new_size)
        self.shm_cov.reset_dropped_edges()
        os.environ["__AFL_SHM_ID"] = self.shm_cov.env_id
        os.environ["AFL_MAP_SIZE"] = str(new_size)
        if self._inprocess_runner:
            self._inprocess_runner.update_shm_after_resize(
                self.shm_cov._ptr, new_size, self.shm_cov.env_id
            )
        if self._forkserver:
            self._forkserver.update_shm_after_resize(self.shm_cov.env_id, new_size)

    def _run_chi2_operator_test(self) -> None:
        """Chi-squared test: do operators have different success rates?

        Builds a 2×K contingency table (operators × success/failure) and
        tests the null hypothesis that all operators share the same success
        probability.  Results are logged at ``info`` when significant.
        """
        from fuzzer_tool.core.chi_squared import chi_squared_independence, cramers_v

        ops = sorted(set(self.op_counts.keys()) | set(self.op_success.keys()))
        if len(ops) < 2:
            return

        table: list[list[float]] = []
        for op in ops:
            total = self.op_counts.get(op, 0)
            success = self.op_success.get(op, 0)
            if total < 1:
                continue
            table.append([float(success), float(total - success)])

        if len(table) < 2:
            return
        if not any(row[1] > 0 for row in table):
            return

        try:
            chi2, p, dof = chi_squared_independence(table)
            n = sum(sum(r) for r in table)
            v = cramers_v(chi2, n, len(table), 2)

            if p < 0.05:
                log.info(
                    "χ² op heterogeneity: χ²=%.2f, p=%.4f, V=%.3f, "
                    "%d operators — significant (p<0.05)",
                    chi2,
                    p,
                    v,
                    len(table),
                )
            else:
                log.debug(
                    "χ² op heterogeneity: χ²=%.2f, p=%.4f, V=%.3f, %d operators — not significant",
                    chi2,
                    p,
                    v,
                    len(table),
                )
        except Exception as ex:
            log.debug("Chi-squared test failed: %s", ex)

    def _record_seed_strategy_matches(self, score: float) -> None:
        """Record the active seed strategy's Elo match against every OTHER
        eligible strategy in the current pool. Only the strategies that were
        actually selectable at pick time participate, so never-enabled
        strategies do not accrue phantom matches.
        """
        if not (self._use_elo and self._elo and self._seed_strategy):
            return
        seed_strategies = getattr(self, "_seed_strategy_pool", [])
        if self._seed_strategy not in seed_strategies:
            return
        for other in seed_strategies:
            if other != self._seed_strategy:
                self._elo.record_strategy_match(
                    f"seed_{self._seed_strategy}", f"seed_{other}", score
                )

    def _settle_positions(self, outcome: Outcome, weight: float) -> None:
        """Close the round for position schedulers: burn-front credit and,
        with the arena on, the Elo matches. Delocalised operators have no
        true offset and are not credited (see _DELOCALISED_OPS).
        """
        arena = getattr(self, "_position_arena", None)
        front = getattr(self, "_burn_front", None)
        if arena is None and front is None:
            return

        sites = [s for op, s in self._last_ops_with_sites if op not in _DELOCALISED_OPS]
        if arena is None:
            front.record(self._last_parent_seed, sites, outcome, weight)
            return

        score = weight if outcome is Outcome.GAIN else 0.0
        arena.settle(self._last_parent_seed, sites, outcome, weight, score)

    def _record_operator_strategy_matches(self, score: float) -> None:
        """Record the active operator scheduler's Elo match against every other
        enabled scheduler. Only schedulers actually selected this run
        participate (random_stall is never recorded), so enabled-but-unused
        schedulers do not accrue phantom matches.
        """
        if not (self._use_elo and self._elo and self._meta_strategy):
            return
        if self._meta_strategy not in self._meta_strategy_used:
            return
        # Same list select_op offered, from the same function: an opponent
        # list kept by hand here drifted from the selection side twice (cmaes,
        # then kl_ducb/kl_swucb in one direction and fpl in the other).
        all_strategies = operator_strategy_pool(self)
        for other in all_strategies:
            if other != self._meta_strategy:
                self._elo.record_strategy_match(self._meta_strategy, other, score)

    def _check_canary_inspection(self) -> None:
        """Warn when a real scheduler ranks at or below the canary floor.

        Called on the same cadence as ``apply_decay`` (every
        ``_elo_decay_interval`` recorded rounds), not every round --
        canary and its rivals both need ``min_matches`` accumulated for
        ``strategies_below_canary`` to say anything, so checking more
        often just repeats the same "not enough data yet" empty result.
        """
        # Only flag operator (non-seed) strategies below the canary floor.
        # Seed strategies have their own separate seed-canary floor check below.
        flagged = self._elo.strategies_below_canary()
        for strategy, mu, canary_mu in flagged:
            log.warning(
                "Elo meta-scheduler: %r rated %.1f, at or below the canary "
                "floor (%.1f) -- this scheduler needs inspection",
                strategy_display_name(strategy),
                mu,
                canary_mu,
            )
        # Position arena: uniform is the floor. A proposer rated at or below
        # it is no better than picking offsets blindly.
        if getattr(self, "_position_arena", None) is not None:
            for strategy, mu, floor_mu in self._elo.strategies_below_canary("pos_uniform"):
                log.warning(
                    "Elo meta-scheduler: position strategy %r rated %.1f, at or "
                    "below uniform (%.1f) -- this proposer needs inspection",
                    strategy_display_name(strategy),
                    mu,
                    floor_mu,
                )
        # Same check for the seed arena's own floor (see
        # core/schedulers/seed_canary.py) -- a separate tournament under
        # seed_-prefixed keys, so it needs its own canary_name.
        if getattr(self, "_use_seed_canary", False) and self._seed_canary:
            seed_flagged = self._elo.strategies_below_canary("seed_canary")
            for strategy, mu, canary_mu in seed_flagged:
                log.warning(
                    "Elo meta-scheduler: seed strategy %r rated %.1f, at or "
                    "below the seed-canary floor (%.1f) -- this strategy "
                    "needs inspection",
                    strategy_display_name(strategy),
                    mu,
                    canary_mu,
                )

    def _seed_convergence_rows(self) -> list[tuple[str, float, float, int]]:
        """(name, rating, delta, matches) for every seed strategy actually used
        this run. Strategies that were never selected (only ever recorded as
        phantom opponents) are excluded from the convergence report. Sorted by
        rating descending, same convention as the bandit/mopt/replicator
        convergence tables above.
        """
        if not (self._use_elo and self._elo):
            return []
        rows = []
        for s in getattr(self, "_seed_strategies_used", set()):
            key = f"seed_{s}"
            count = self._elo._strategy_match_count.get(key, 0)
            if count > 0:
                rating = self._elo._strategy_mu.get(key, self._elo.initial_mu)
                rows.append((s, rating, rating - self._elo.initial_mu, count))
        return sorted(rows, key=lambda r: -r[1])

    def _position_convergence_rows(self) -> list[tuple[str, float, float, int]]:
        """(name, rating, delta, matches) for every pos_ strategy with matches."""
        if not (self._use_elo and self._elo and self._position_arena is not None):
            return []
        rows = []
        for s in POSITION_STRATEGY_NAMES:
            key = POS_STRATEGY_PREFIX + s
            count = self._elo._strategy_match_count.get(key, 0)
            if count > 0:
                rating = self._elo._strategy_mu.get(key, self._elo.initial_mu)
                rows.append((s, rating, rating - self._elo.initial_mu, count))
        return sorted(rows, key=lambda r: -r[1])

    def _operator_convergence_rows(self) -> list[tuple[str, float, float, int]]:
        """(name, rating, delta, matches) for every operator scheduler actually
        selected this run, plus ``canary`` whenever it is enabled and rated.
        Schedulers that were enabled but never selected are otherwise excluded
        from the convergence report.

        canary is a deliberate exception: it is designed (see
        ``core/schedulers/op_canary.py``) to be selected the *least* of any
        scheduler in the pool, so gating it on "was ever selected" the same
        way as real schedulers means it disappears from exactly the report
        that exists to show it as a floor. It still accrues a real rating and
        match count every round as the opponent side of
        ``record_strategy_match`` regardless of whether it was picked, so it
        is included here on rated-with-matches rather than
        ``_meta_strategy_used`` membership. Sorted by rating descending, same
        convention as the bandit/mopt/replicator convergence tables above.
        """
        if not (self._use_elo and self._elo):
            return []
        used = set(getattr(self, "_meta_strategy_used", set()))
        if getattr(self, "_use_canary", False) and self._canary:
            used.add("canary")
        rows = []
        for s in used:
            count = self._elo._strategy_match_count.get(s, 0)
            if count > 0:
                rating = self._elo._strategy_mu.get(s, self._elo.initial_mu)
                rows.append((s, rating, rating - self._elo.initial_mu, count))
        return sorted(rows, key=lambda r: -r[1])

    def _load_kruskal_count(self) -> None:
        """Restore Kruskal-count counters on resume; malformed payloads start fresh."""
        from fuzzer_tool.core.schedulers.seed_kruskal_count import KruskalCountSeedStrategy

        data = self._state_store.get("kruskal_count")
        if self.resume and data is not None:
            self._kruskal_count = KruskalCountSeedStrategy.from_dict(data, self._rng, self._profile)
            print(
                f"[*] Kruskal count: loaded state ({self._kruskal_count.stats()['scored']} scored)"
            )
        print("[*] Kruskal-count seed scheduling enabled")

    def _load_entropy_kl(self) -> None:
        """Restore entropy-KL counters on resume; malformed payloads start fresh."""
        from fuzzer_tool.core.schedulers.seed_entropy_kl import EntropyKLSeedStrategy

        data = self._state_store.get("entropy_kl")
        if self.resume and data is not None:
            self._entropy_kl = EntropyKLSeedStrategy.from_dict(data, self._rng)
        print("[*] Entropy-KL seed scheduling enabled")

    def _load_entropy_zscore(self) -> None:
        """Restore entropy z-score counters on resume; the moments rebuild
        from the corpus on the first scoring pass (see the module docstring)."""
        from fuzzer_tool.core.schedulers.seed_entropy_zscore import EntropyZScoreSeedStrategy

        data = self._state_store.get("entropy_zscore")
        if self.resume and data is not None:
            self._entropy_zscore = EntropyZScoreSeedStrategy.from_dict(data, self._rng)
        print(
            "[*] Entropy z-score seed scheduling enabled "
            f"(target_z={self._entropy_zscore.stats()['target_z']:.2f})"
        )

    def _selected_schedulers_str(self) -> str:
        """One-line summary of the active scheduling stack (startup banner)."""
        parts = []
        if getattr(self, "_power_schedule", "base") != "base":
            parts.append(f"power={self._power_schedule}")

        ops = []
        if getattr(self, "_consolidated", False):
            ops.append("consolidated")
        if self.mc_bandit:
            ops.append("bandit")
        if self.mc_cem:
            ops.append("cem")
        if getattr(self, "_use_mopt", False):
            ops.append("mopt")
        if getattr(self, "_use_replicator", False):
            ops.append("replicator")
        if getattr(self, "_use_exp3", False):
            ops.append("exp3")
        if getattr(self, "_use_exp4", False) and self._exp4:
            ops.append("exp4")
        if getattr(self, "_eps_greedy", False):
            ops.append("eps_greedy")
        # Was _hierarchical_bandit, an attribute that has never existed --
        # the constructor stores _use_hierarchical -- so the banner silently
        # omitted the hierarchical bandit on every run it was enabled.
        if getattr(self, "_use_hierarchical", False):
            ops.append("hierarchical")
        if getattr(self, "_gp_ucb", False):
            ops.append("gp_ucb")
        if getattr(self, "_bo_gp_ucb", False):
            ops.append("bo_gp_ucb")
        if getattr(self, "_cmaes", False):
            ops.append("cmaes")
        if getattr(self, "_contextual", False):
            ops.append("contextual")
        if getattr(self, "_c2ucb", False):
            ops.append("c2ucb")
        if getattr(self, "_ducb", False):
            ops.append("ducb")
        if getattr(self, "_swucb", False):
            ops.append("swucb")
        if getattr(self, "_cucb", False):
            ops.append("cucb")
        if getattr(self, "_cusum_ucb", False):
            ops.append("cusum_ucb")
        if getattr(self, "_fewa", False):
            ops.append("fewa")
        if getattr(self, "_moss", False):
            ops.append("moss")
        if getattr(self, "_fpl", False):
            ops.append("fpl")
        # corral is deliberately absent from this banner too, same reason,
        # see core/schedulers/op_corral.py.
        # gradient is deliberately absent from this banner, matching
        # op_katz/op_tang: it is Elo-only, see core/schedulers/op_gradient.py.
        # whittle is deliberately absent from this banner too, same
        # reason, see core/schedulers/op_whittle.py.
        if getattr(self, "_use_successive_elim", False) and self._successive_elim:
            ops.append("successive_elim")
        if getattr(self, "_use_invasion", False) and self.mc_bandit:
            ops.append("invasion")
        if getattr(self, "_use_round_robin", False) and self._round_robin:
            ops.append("round_robin")
        if getattr(self, "_use_canary", False) and self._canary:
            ops.append("canary")
        if getattr(self, "_use_shapley", False):
            ops.append("shapley")
        if ops:
            parts.append("ops=" + "+".join(ops))

        seeds = []
        if self.ga:
            seeds.append("ga")
        if self.qea:
            seeds.append("qea")
        if getattr(self, "_use_bayesian", False):
            seeds.append("bayesian")
        if getattr(self, "_use_boltzmann", False):
            seeds.append("boltzmann")
        if getattr(self, "_use_ecofuzz", False):
            seeds.append("ecofuzz")
        if getattr(self, "_distance", None) is not None:
            seeds.append("aflgo")
        if getattr(self, "_kruskal_count", None) is not None:
            seeds.append("kruskal-count")
        if getattr(self, "_entropy_kl", None) is not None:
            seeds.append("entropy-kl")
        if getattr(self, "_entropy_zscore", None) is not None:
            seeds.append("entropy-zscore")
        # entropy_deviation was missing here too (same bug commit 18d2b051
        # fixed for _print_enabled_features's "Seed selection" group; this
        # sibling banner was not covered by that fix).
        if getattr(self, "_entropy_deviation", None) is not None:
            seeds.append("entropy-deviation")
        if getattr(self, "_entropy_gradient", None) is not None:
            seeds.append("entropy-gradient")
        if getattr(self, "_entropy_loo", None) is not None:
            seeds.append("entropy-loo")
        if getattr(self, "_seed_residual", None) is not None:
            seeds.append("residual")
        if getattr(self, "_seed_strata", None) is not None:
            seeds.append("strata")
        if getattr(self, "_use_seed_canary", False) and self._seed_canary:
            seeds.append("canary")
        if getattr(self, "_use_seed_round_robin", False) and self._seed_round_robin:
            seeds.append("round-robin")
        if seeds:
            parts.append("seeds=" + "+".join(seeds))

        positions = []
        if getattr(self, "_burn_front", None) is not None:
            positions.append("burn-front")
        if getattr(self, "_position_arena", None) is not None:
            positions.append("arena")
        if positions:
            parts.append("positions=" + "+".join(positions))

        if getattr(self, "_use_elo", False):
            parts.append("elo")
        if self.markov_generate:
            parts.append("markov-gen")

        return " | ".join(parts) if parts else "base"

    def _start_stack_heartbeat(self, interval: float = 3.0) -> None:
        """Daemon thread: periodically write the main-thread Python stack.

        SIGKILL cannot be handled in-process, so `kill -9` leaves nothing.
        This thread writes where the main thread is executing (only when the
        top frame moves) to a small file, giving the last known location
        after a hard kill.
        """
        if self._stack_heartbeat_path is None:
            return
        out = self._stack_heartbeat_path
        out.parent.mkdir(parents=True, exist_ok=True)

        def _beat() -> None:
            import traceback

            ident = threading.main_thread().ident
            last_key: tuple | None = None
            while True:
                time.sleep(interval)
                try:
                    frame = sys._current_frames().get(ident)
                    if frame is None:
                        continue
                    key = (frame.f_code.co_filename, frame.f_lineno)
                    if key == last_key:
                        continue
                    last_key = key
                    out.write_text(
                        "".join(traceback.format_stack(frame)[-8:])
                        + f"\n# heartbeat ts={time.time():.0f} execs={self.exec_count}\n"
                    )
                except Exception:
                    pass

        threading.Thread(target=_beat, daemon=True, name="stack-heartbeat").start()

    def _calibrate_seed_baselines(self) -> None:
        """Execute every corpus seed verbatim once, before the fuzz loop.

        fuzz_one() runs ONLY mutated input: _dedup_mutate() transforms even
        seed iterations, so the pristine bytes a corpus file contains are
        never executed. The tracker's edge universe is therefore built
        exclusively from mutants that happen to stay format-valid — for
        structured targets that is a tiny sliver of what the seeds really
        reach (observed on png_read.so: 11 valid PNGs produced shm=6 edges
        after 150 execs, while one direct execution of a single seed
        records ~330; the campaign then plateaus immediately and Good-
        Turing reads the starved universe as 100% saturated).

        This pass gives every seed one unmutated execution through the
        normal pipeline (_run_target + is_new_coverage_with_edges +
        record_edges), so scheduling, corpus admission, GT estimation and
        cmplog token collection all start from the coverage the seeds
        genuinely carry. Crashes/timeouts during calibration are counted,
        not fatal — a hostile corpus must not kill startup.
        """
        # --no-calibration: skip the pass, its RSS and its per-seed executions.
        if not self._seed_calibration:
            return
        if not self.use_coverage or not self.shm_cov or self.multi_targets:
            return
        if not self.corpus:
            return
        baseline_edges = 0
        t0 = time.monotonic()
        # Widest seed seen, for the edge-id stability probe below: more edges
        # is more chances for a drifting id to show, at identical cost.
        probe_seed: bytes | None = None
        probe_width = 0
        for seed in list(self.corpus):
            returncode, stderr = self._run_target(seed)
            if self._is_crash(returncode, stderr):
                self.crash_count += 1
                continue
            if returncode == -1:  # timeout sentinel
                self.timeout_count += 1
                self._corpus_manager.save_timeout(seed)
                continue
            has_new, edge_ids = self.shm_cov.is_new_coverage_with_edges()
            has_new, edge_ids = self._confirm_new_coverage(seed, self.shm_cov, has_new, edge_ids)
            if not edge_ids:
                continue
            if len(edge_ids) > probe_width:
                probe_seed, probe_width = seed, len(edge_ids)
            hit_counts = self._only_confirmed(self.shm_cov.get_edge_counts())
            stack_depth = self.shm_cov.read_stack_depth()
            path_hash = self.shm_cov.read_path_hash()
            if path_hash == 0:
                path_hash = self.shm_cov.compute_path_hash_from_edges(edge_ids)
            self._strata_observe(seed, edge_ids)
            new = self._edge_tracker.record_edges(
                self._seed_key(seed),
                edge_ids,
                # "" skips seed_target_edges: a full per-seed duplicate that only
                # multi-target mode reads, and this pass never runs there.
                target_name="",
                hit_counts=hit_counts,
                stack_depth=stack_depth,
                path_hash=path_hash,
                hw_instructions=self._last_perf_deltas.get("instructions", 0),
                hw_branches=self._last_perf_deltas.get("branches", 0),
                hw_branch_misses=self._last_perf_deltas.get("branch_misses", 0),
            )
            baseline_edges += len(new)
        if baseline_edges or len(self.corpus):
            print(
                f"[*] Seed calibration: {len(self.corpus)} seeds -> "
                f"{baseline_edges} baseline edges "
                f"({time.monotonic() - t0:.2f}s)"
            )
        self._report_comparison_reach(len(self.corpus))
        self._report_edge_id_stability(probe_seed)

    def _report_edge_id_stability(self, seed: bytes | None, n_runs: int = 3) -> None:
        """Say whether edge ids reproduce across processes.

        Every per-edge statistic the campaign computes assumes an edge id
        means the same thing in the next process as in this one. When it
        does not, nothing downstream reports a cause: the corpus grows on
        phantom ids, each owned by exactly one seed and therefore maximally
        rare to the seed picker, and the run looks like a target with
        endless new coverage. ``--calibrate-stability`` finds those edges
        one seed at a time and masks them without ever saying why, and it is
        opt-in; this is the whole question asked once, for three executions.

        The known cause is F1: a context-sensitive build whose caller
        context is hashed from an address that moves.
        :meth:`_ensure_ctx_ids_are_exec_stable` closes that by construction
        for any target built against the current shim, which is exactly why
        this still runs -- a target built against an older one ignores the
        variable and exports no symbol that says so, so the only way to
        know is to measure. Instability also has causes the id axis knows
        nothing about: threads, time, uninitialised memory.
        Calibration is where the question is cheap: the seeds have just run,
        so the table is warm and a probe measures the steady state rather
        than the first-execution regime, which is known to differ from it
        for reasons still unresolved (F2). Warn-only, three extra
        executions, no masking: this reports, it does not decide.
        """
        if seed is None or self.shm_cov is None:
            return
        measured = self._repeat_edge_sets(seed, n_runs)
        if measured is None:
            return
        edge_sets, _hashes, dropped = measured
        if dropped:
            # A saturated table discards edges by arrival order, so set
            # divergence is not evidence of id drift -- the same argument
            # that makes _calibrate_seed_stability abstain.
            print(
                f"[*] Edge id stability: not measured, {dropped} edge(s) dropped "
                "to a full map during the probe (raise --map-size)"
            )
            return
        union = set.union(*edge_sets)
        common = set.intersection(*edge_sets)
        if not union:
            return
        jaccard = len(common) / len(union)
        # The matrix arms' preflight gate reads this (F1): under per-process ids
        # every edge is a singleton owned by one seed, i.e. maximally rare.
        substrate = getattr(self, "_matrix_substrate", None)
        if substrate is not None:
            substrate.set_stability(jaccard)
        if getattr(self, "_edge_ledger", None) is not None:
            self._strata_set_stability(jaccard)
        if jaccard == 1.0:
            print(
                f"[*] Edge id stability: {len(union)} edge ids reproduced exactly "
                f"across {n_runs} executions of one seed"
            )
            return
        msg = (
            f"Edge id stability: {n_runs} executions of one unmutated seed agreed "
            f"on {len(common)} of {len(union)} edge ids (Jaccard {jaccard:.3f}) — "
            "the ids themselves are moving, so per-edge coverage feedback is "
            "measuring the instability rather than the target"
        )
        log.warning(msg)
        print(f"[!] WARNING: {msg}")
        from fuzzer_tool.core.elf import detect_ctx_bits, detect_ctx_relative_capable

        ctx_bits = detect_ctx_bits(self.target) if self.target else None
        if ctx_bits and not self._aslr_disabled:
            # Relative mode is on by this point for any target that can
            # honour it (see _ensure_ctx_ids_are_exec_stable), so the marker
            # says which of the two remaining stories this is.
            capable = detect_ctx_relative_capable(self.target)
            requested = os.environ.get("FUZZER_KEEP_ASLR") == "1"
            if capable is False:
                print(
                    "[!]   Cause: this is a __AFL_CTX_SENSITIVE build "
                    f"(__AFL_CTX_BITS={ctx_bits}) with no "
                    "__afl_ctx_relative_capable marker, so it hashes a raw "
                    "return address, which is a different value in every "
                    "process while ASLR is on. Rebuild it against the "
                    "current adapters/afl_shim.c (tools/build_targets.sh), "
                    "or with -D__AFL_CTX_SENSITIVE=0."
                )
            elif not requested:
                # _ensure_ctx_ids_are_exec_stable should have set this before
                # anything executed. Reaching here in a real campaign means
                # that hook did not run or did not see this target.
                print(
                    "[!]   Cause: the shim can hash caller context "
                    "load-base-relative, but FUZZER_KEEP_ASLR is not set for "
                    "the target, so it is hashing raw return addresses under "
                    "live ASLR. Set it, or disable ASLR."
                )
            else:
                print(
                    "[!]   The target's shim can hash caller context "
                    "load-base-relative and FUZZER_KEEP_ASLR is set for it, "
                    "so the ids should be exec-stable and are not. This is "
                    "not the known F1 shape; look for thread scheduling, "
                    "time, or uninitialised memory in the target."
                )
        else:
            print(
                "[!]   ASLR and context hashing are not the cause here "
                "(the known one, F1, needs both). Look for thread scheduling, "
                "time, or uninitialised memory in the target; "
                "--calibrate-stability masks such edges per seed."
            )

    def _report_comparison_reach(self, n_execs: int) -> None:
        """Say whether the comparison instrumentation reached the target.

        "cmplog is on but doing nothing" is currently silent in every form
        it takes, and it takes several: an -O2 build without -fno-builtin
        (measured on cmplog_exercise.c: 20 call sites, 4 records), a preload
        that lost the symbol-lookup race to the executable's own weak sancov
        stubs, a target that never reaches its parser on the seeds it was
        given. All of them look identical from the outside -- the campaign
        runs, the token pool just stays empty -- and a user reads that as
        "this target has no interesting comparisons".

        The counters answer it directly, and calibration is where the
        question is cheap: every seed has just been executed exactly once,
        so the totals are over a known number of executions of unmutated
        input, before the fuzz loop can muddy them.

        Printed like the distance line, and warned about only in the case
        with no benign reading. Zero fires across a whole seed pass is that
        case: a target worth pointing a fuzzer at compares *something*.
        """
        if self._cmplog is None or n_execs <= 0:
            return
        self._reset_cmplog()
        self._cmplog.collect_counts()
        (l1_fired, _), (l2_fired, _) = self._cmplog.layer_totals()
        total = l1_fired + l2_fired
        if total == 0:
            print(
                "[!] Comparison instrumentation: no comparisons observed in "
                f"{n_execs} seed executions — cmplog is enabled but not "
                "reaching the target (check -fno-builtin-* on the target "
                "build, and that the shim is linked in rather than preloaded)"
            )
            return
        print(
            f"[*] Comparison instrumentation: {total} comparisons over "
            f"{n_execs} seed executions (libc {l1_fired}, trace-cmp {l2_fired})"
        )
        if l1_fired == 0:
            print(
                "[*]   Comparisons are inlined: the libc layer sees nothing, "
                "so trace-cmp records the post-expansion (0, 1) pairs rather "
                "than operands. Rebuild with -fno-builtin-memcmp "
                "(and -strcmp, -strncmp) to recover them."
            )

    def _merge_profile_dictionary(self) -> None:
        """Fold the target profile's token channels into ``self.dictionary``.

        Channel order is fixed so dictionary contents do not depend on hash
        iteration order. Strings and parser tokens may be single bytes; the
        constant channels (disassembly and literal data words) skip tokens
        shorter than 2 bytes -- single-byte values are too noisy.
        """
        profile = self._profile

        def merge(tokens, min_len=1, limit=None):
            for t in itertools.islice(tokens, limit):
                if len(t) >= min_len and t not in self.dictionary:
                    self.dictionary.append(t)

        merge((s.encode("utf-8", errors="replace") for s in profile.interesting_strings), limit=200)
        merge(profile.magic_bytes)
        merge(profile.extracted_constants, min_len=2)
        merge(profile.rodata_word_constants, min_len=2)
        merge(profile.parser_tokens)

    def _print_enabled_features(self) -> None:
        """Print every enabled feature, grouped by category."""
        groups: dict[str, list[str]] = {
            "Scheduling": [],
            "Seed selection": [],
            "Position selection": [],
            "Mutation": [],
            "Analysis": [],
            "Generation": [],
            "Execution": [],
            "Output": [],
        }

        sched_pol = getattr(self, "_power_schedule", "base")
        if sched_pol != "base":
            groups["Scheduling"].append(f"power-schedule={sched_pol}")

        ops = []
        if getattr(self, "_consolidated", False):
            ops.append("consolidated")
        if self.mc_bandit:
            ops.append("bandit")
        if self.mc_cem:
            ops.append("cem")
        if getattr(self, "_use_mopt", False):
            ops.append("mopt")
        if getattr(self, "_use_replicator", False):
            ops.append("replicator")
        if getattr(self, "_use_exp3", False):
            ops.append("exp3")
        if getattr(self, "_use_exp4", False) and self._exp4:
            ops.append("exp4")
        if getattr(self, "_eps_greedy", False):
            ops.append("eps-greedy")
        if getattr(self, "_use_hierarchical", False):
            ops.append("hierarchical")
        if getattr(self, "_gp_ucb", False):
            ops.append("gp-ucb")
        if getattr(self, "_bo_gp_ucb", False):
            ops.append("bo-gp-ucb")
        if getattr(self, "_ducb", False):
            ops.append("ducb")
        if getattr(self, "_swucb", False):
            ops.append("swucb")
        if getattr(self, "_cucb", False):
            ops.append("cucb")
        if getattr(self, "_cusum_ucb", False):
            ops.append("cusum_ucb")
        if getattr(self, "_fewa", False):
            ops.append("fewa")
        if getattr(self, "_moss", False):
            ops.append("moss")
        if getattr(self, "_fpl", False):
            ops.append("fpl")
        # corral is deliberately absent from this banner too, same reason,
        # see core/schedulers/op_corral.py.
        # gradient is deliberately absent from this banner, matching
        # op_katz/op_tang: it is Elo-only, see core/schedulers/op_gradient.py.
        # whittle is deliberately absent from this banner too, same
        # reason, see core/schedulers/op_whittle.py.
        # op_kuramoto is the exception to the Elo-only-silent pattern above:
        # shown in the banner like every other scheduler, even though it is
        # still off by default / Elo-only / absent from _FALLBACK_PRECEDENCE
        # per its own module docstring (core/schedulers/op_kuramoto.py).
        if getattr(self, "_op_kuramoto", None) is not None:
            ops.append("op-kuramoto")
        if getattr(self, "_use_successive_elim", False) and self._successive_elim:
            ops.append("successive-elim")
        if getattr(self, "_use_contextual", False):
            ops.append("contextual")
        if getattr(self, "_use_c2ucb", False):
            ops.append("c2ucb")
        if getattr(self, "_use_invasion", False):
            ops.append("invasion")
        if getattr(self, "_use_shapley", False):
            ops.append("shapley")
        if getattr(self, "_use_round_robin", False):
            ops.append("round-robin")
        if getattr(self, "_use_canary", False):
            ops.append("canary")
        if getattr(self, "_cmaes", False):
            groups["Scheduling"].append("cma-es")
        # Not an arm, so not in `ops`: it rescales the reward every arm above
        # reads. That is exactly why it is announced rather than silent -- a
        # campaign whose rewards are being divided by class size and whose log
        # does not say so is the "a suite cannot detect a reverted feature"
        # failure this repo has already paid for once.
        if getattr(self, "_shaped_reward", False):
            floor = getattr(self, "_shaped_reward_floor", 0.0)
            groups["Scheduling"].append(
                "shaped-reward" if not floor else f"shaped-reward(floor={floor:g})"
            )
        if getattr(self, "_continuum_reward", False):
            floor = getattr(self, "_continuum_reward_floor", 0.0)
            groups["Scheduling"].append(
                "continuum-reward" if not floor else f"continuum-reward(floor={floor:g})"
            )
        if ops:
            groups["Scheduling"].extend(ops)

        if getattr(self, "_use_elo", False):
            groups["Scheduling"].append("elo")

        if self.ga:
            groups["Seed selection"].append("ga")
        if self.qea:
            groups["Seed selection"].append("qea")
        if getattr(self, "_use_bayesian", False):
            groups["Seed selection"].append("bayesian")
        if getattr(self, "_use_boltzmann", False):
            groups["Seed selection"].append("boltzmann")
        if getattr(self, "_use_ecofuzz", False):
            groups["Seed selection"].append("ecofuzz")
        if self.markov_generate:
            groups["Seed selection"].append("markov-gen")
        if getattr(self, "_use_mcts", False):
            groups["Seed selection"].append("mcts")
        if getattr(self, "_use_alphabeta", False):
            groups["Seed selection"].append("alphabeta")
        if getattr(self, "_distance", None) is not None:
            groups["Seed selection"].append("aflgo")
        if getattr(self, "_katz_channel", None) is not None:
            groups["Seed selection"].append("katz")
        if getattr(self, "_tang", None) is not None:
            groups["Seed selection"].append("tang")
        if getattr(self, "_kruskal_count", None) is not None:
            groups["Seed selection"].append("kruskal-count")
        if getattr(self, "_entropy_kl", None) is not None:
            groups["Seed selection"].append("entropy-kl")
        if getattr(self, "_entropy_zscore", None) is not None:
            groups["Seed selection"].append("entropy-zscore")
        if getattr(self, "_entropy_deviation", None) is not None:
            groups["Seed selection"].append("entropy-deviation")
        if getattr(self, "_entropy_gradient", None) is not None:
            groups["Seed selection"].append("entropy-gradient")
        if getattr(self, "_entropy_loo", None) is not None:
            groups["Seed selection"].append("entropy-loo")
        if getattr(self, "_seed_residual", None) is not None:
            groups["Seed selection"].append("residual")
        if getattr(self, "_seed_strata", None) is not None:
            groups["Seed selection"].append("strata")
        if getattr(self, "_use_seed_round_robin", False) and self._seed_round_robin:
            groups["Seed selection"].append("round-robin")
        if getattr(self, "_burn_front", None) is not None:
            groups["Position selection"].append("burn-front")
        if getattr(self, "_position_arena", None) is not None:
            groups["Position selection"].append("position-arena")

        if self.markov_trained:
            groups["Mutation"].append("markov")
        if getattr(self, "_adaptive_havoc", False):
            groups["Mutation"].append("adaptive-havoc")
        if getattr(self, "_dict_picker", None) is not None:
            groups["Mutation"].append("dict-thompson")
        if self.enable_x86_mutator:
            groups["Mutation"].append("x86-mutator")
        if self.enable_arm_mutator:
            groups["Mutation"].append("arm-mutator")
        if self.enable_regex_bomb:
            groups["Mutation"].append("regex-bomb")
        if self.weizz_tags:
            groups["Mutation"].append("weizz-tags")
        if self.colorize:
            groups["Mutation"].append("colorize")

        if getattr(self, "_use_mi", False):
            groups["Analysis"].append("mi-guided")
        if getattr(self, "_pool_drift", None) is not None:
            groups["Analysis"].append("pool-drift")
        if getattr(self, "_use_renyi_weight", False):
            groups["Analysis"].append("renyi")
        if getattr(self, "_use_transfer_entropy", False):
            groups["Analysis"].append("transfer-entropy")
        if getattr(self, "_use_occupation", False):
            groups["Analysis"].append("occupation")
        if getattr(self, "_causal_sector", None) is not None:
            groups["Analysis"].append("causal-sector")
        if getattr(self, "_use_sensitivity", False):
            groups["Analysis"].append("sensitivity")
        if getattr(self, "_use_lineage", False):
            groups["Analysis"].append("lineage")
        if getattr(self, "_use_lineage_backtrack", False):
            groups["Analysis"].append("lineage-backtrack")
        if getattr(self, "_use_overlap_density", False):
            groups["Analysis"].append("overlap-density")
        if getattr(self, "_use_region_profile", False):
            groups["Analysis"].append("region-profile")
        if getattr(self, "_calibrate", 0) > 0:
            groups["Analysis"].append(f"calibrate={self._calibrate}")
        if getattr(self, "_use_bootstrap", False):
            groups["Analysis"].append("bootstrap")

        if getattr(self, "_wfc_enabled", False):
            groups["Generation"].append("wfc")
        if getattr(self, "_corpus_boost", 0) > 0:
            groups["Generation"].append(f"corpus-boost={self._corpus_boost}")
        if getattr(self, "_dirichlet_alpha", AlphaMode.FIXED) is AlphaMode.LEARNED:
            groups["Generation"].append("dirichlet-alpha=learned")

        if self.persistent:
            groups["Execution"].append("persistent")
        if self._inprocess_runner:
            groups["Execution"].append("inprocess")
        if self._adaptive_timeout:
            groups["Execution"].append("adaptive-timeout")
        if self.honggfuzz:
            groups["Execution"].append("honggfuzz")
        if self.hw_perf:
            groups["Execution"].append("hw-perf")
        if self.pt_cov:
            groups["Execution"].append("intel-pt")
        if self.branch_cov:
            groups["Execution"].append("lbr")
        if self._diff_target:
            groups["Execution"].append("differential")

        if self.debug:
            groups["Output"].append("debug")
        if self._tracer is not None:
            groups["Output"].append("trace-crashes")
        if self._format_learner is not None:
            groups["Output"].append("learn-format")

        print("[*] Enabled features:")
        for group, feats in groups.items():
            if feats:
                print(f"    {group}: {', '.join(feats)}")

    def run(self, iterations=0, max_execs=0):
        self._start_stack_heartbeat()
        if self.multi_targets:
            print(f"[*] Multi-target: {len(self.multi_targets)} targets, shared corpus")
            uninstrumented = []
            for i, t in enumerate(self.multi_targets):
                status = afl_instrumentation_status(t)
                tag = {"present": " [AFL]", "absent": " [no-AFL]", "unknown": " [AFL?]"}[status]
                if status == "absent":
                    uninstrumented.append(t)
                dist = _detect_distance(t)
                if dist:
                    tag += " [DIST]"
                print(f"  [{i}] {t}{tag}")
            if uninstrumented:
                self._warn_uninstrumented(uninstrumented)
        else:
            print(f"[*] Target: {self.target}")
            self._report_instrumentation()
            if _detect_distance(self.target):
                print("[*] Distance instrumentation: detected")
                if self._distance is None:
                    print(
                        "[*]   Directed mode idle: pass --target-functions "
                        "(function, address, or file.c:line) to engage the "
                        "distance channel (dist: stats + aflgo schedule/elo arm)"
                    )
        from fuzzer_tool.core.elf import detect_ngram_k

        print(f"[*] Ngram: k={detect_ngram_k(self.target)}")
        if self._validity.enabled:
            print(f"[*] Validity channel: reject-code {self._validity.reject_code}")
        # Static branch density: conditional branches per KB of .text
        from fuzzer_tool.core.elf import branch_density

        if self.multi_targets:
            bd_total = 0
            bd_count = 0
            for t in self.multi_targets:
                bd = branch_density(t)
                if bd is not None:
                    name = os.path.basename(t)
                    print(f"[*] Branch density: {name} {bd:.1f} cond branches/KB")
                    bd_total += bd
                    bd_count += 1
            if bd_count > 1:
                print(f"[*] Branch density: avg {bd_total / bd_count:.1f} cond branches/KB")
        else:
            bd = branch_density(self.target)
            if bd is not None:
                print(f"[*] Branch density: {bd:.1f} cond branches/KB")
        print(f"[*] Edge bitmap: {self.map_size:,} entries (auto-sized)")
        self._report_map_cache_residency()
        print(f"[*] Corpus: {self.corpus_dir} ({len(self.corpus)} seeds)")
        print(f"[*] Crashes: {self.crashes_dir}")
        print(f"[*] Max input length: {self.max_len}")
        print(f"[*] Timeout: {self.timeout}s")
        if self.honggfuzz:
            print(
                "[*] Honggfuzz power factors: enabled (novelty, freshness, fertility, density, entropy, timeout)"
            )
        if self.hw_perf:
            if self._perf_counters:
                print(f"[*] HW perf counters: {', '.join(self._perf_counters.counter_names)}")
            else:
                print(
                    "[*] HW perf counters: requested but not available (needs CAP_PERFMON or root)"
                )
        print(f"[*] Seed: {self.seed}")
        self._report_rng_health()
        # Target profile summary
        if self._profile.functions:
            profile_cache = Path(self.target).with_suffix(
                Path(self.target).suffix + ".profile_cache"
            )
            tag = " [cached]" if profile_cache.exists() and not self.refresh_profile else ""
            print(
                f"[*] Profile: {len(self._profile.functions)} functions, "
                f"{len(self._profile.hot_functions)} hot, "
                f"format={self._profile.format_signature or 'unknown'}{tag}"
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

        # Quick raw-target-speed measurement before the main loop
        try:
            _probe = b"\x00" * 64
            _n = min(100, max(10, int(len(self.corpus) * 0.1 + 1)))
            _t0 = time.perf_counter()
            for _ in range(_n):
                self._run_target(_probe)
            _t1 = time.perf_counter()
            _raw_eps = _n / (_t1 - _t0) if _t1 > _t0 else 0
            print(f"[*] Raw target speed: {_raw_eps:,.0f} eps ({_n} probes)")
        except Exception:
            pass

        i = 0
        try:
            # Run each seed as-is before mutating — catches crashes in the
            # initial corpus and gathers baseline coverage.
            for seed in list(self.corpus):
                returncode, stderr = self._run_target(seed)
                if self._diff_tracker:
                    self._check_differential(seed)
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
                # Mark seed as having been executed (even though not via
                # fuzz_one's mutate path).  This ensures loaded seeds don't
                # all show fuzz_count=0 to auto_minimize_corpus.
                meta = self.seed_meta.get(seed)
                if meta is not None:
                    meta["fuzz_count"] += 1
                    self._cached_total_fuzz += 1
                if self._is_crash(returncode, stderr):
                    self.crash_count += 1
                    self.save_crash(seed, returncode, stderr)
                    self._prune_crash_data()
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
                    rng=self._rng,
                )
                self.ga.initialize(self.corpus, self._edge_tracker)
                # Restore and announce here, not inside the differential
                # block below: nesting it there meant `--ga --resume` without
                # a differential target silently restarted GA at generation 0,
                # while `--differential-target` without `--ga` reached the
                # banner with self.ga still None and died on .pop_size.
                ga_data = self._state_store.get("ga")
                if self.resume and ga_data is not None:
                    self.ga.from_dict(ga_data)
                    print(f"[*] GA: loaded state from state store (gen={self.ga.generation})")
                print(
                    f"[*] GA: pop_size={self.ga.pop_size}, "
                    f"gen_size={self.ga.generation_size}, "
                    f"elite={self.ga.elite_fraction:.0%}, "
                    f"crossover={self.ga.crossover_rate:.0%}, "
                    f"mutation={self.ga.mutation_rate:.0%}"
                )

            # Initialize differential fuzzing if enabled
            if self._diff_target:
                from fuzzer_tool.services.differential import DifferentialTracker

                self._diff_tracker = DifferentialTracker()
                print(f"[*] Differential: comparing against {self._diff_target}")

            # Initialize QEA lifecycle if enabled
            if self._qea_enabled:
                from fuzzer_tool.core.qea import ALPHA_STRONG, QEALifecycle

                self.qea = QEALifecycle(
                    pop_size=self._ga_pop_size,
                    elite_fraction=self._ga_elite_frac,
                    generation_size=self._ga_gen_size,
                    tournament_size=self._ga_tournament_size,
                    speciation_threshold=self._ga_speciation_threshold,
                    rotation_angle=self._qea_rotation_angle,
                    strong_bias=(
                        ALPHA_STRONG if self._qea_strong_bias is None else self._qea_strong_bias
                    ),
                    elite_reset_every=self._qea_elite_reset,
                    use_correlation=self._qea_correlation,
                    correlation_delta=self._qea_correlation_delta,
                    correlation_max=self._qea_correlation_max,
                    correlation_sweeps=self._qea_correlation_sweeps,
                    use_cooling=self._qea_cooling,
                    cooling_decay=self._qea_cooling_decay,
                    cooling_min_angle=self._qea_cooling_min_angle,
                    rng=self._rng,
                )
                self.qea.initialize(self.corpus, self._edge_tracker)
                qea_data = self._state_store.get("qea")
                if self.resume and qea_data is not None:
                    self.qea.from_dict(qea_data)
                    print(f"[*] QEA: loaded state from state store (gen={self.qea.generation})")
                print(
                    f"[*] QEA: pop_size={self.qea.pop_size}, "
                    f"gen_size={self.qea.generation_size}, "
                    f"rotation_angle={self.qea.rotation_angle}, "
                    f"mutation_prob={self.qea.mutation_prob}"
                    + (
                        f", correlation=on (delta={self.qea.correlation_delta}, "
                        f"max={self.qea.correlation_max}, sweeps={self.qea.correlation_sweeps})"
                        if self.qea.use_correlation
                        else ""
                    )
                )

            if self._cmaes:
                cmaes_data = self._state_store.get("cmaes")
                if self.resume and cmaes_data is not None:
                    self._cmaes.from_dict(cmaes_data)
                    print(
                        f"[*] CMA-ES: loaded state from state store "
                        f"(gen={self._cmaes.convergence_stats()['generation']})"
                    )
                stats = self._cmaes.convergence_stats()
                print(
                    f"[*] CMA-ES: pop={self._cmaes.pop_size}, "
                    f"gen={self._cmaes.generation_size}, "
                    f"sigma={stats['sigma']:.3f}, "
                    f"top={stats['top_op']}({stats['top_prob']:.1%})"
                )

            if self._mcts is not None:
                mcts_data = self._state_store.get("mcts")
                if self.resume and mcts_data is not None:
                    self._mcts.from_dict(mcts_data)
                    print(
                        "[*] MCTS: loaded state from state store "
                        f"(nodes={self._mcts.stats()['tracked_nodes']})"
                    )
                print(f"[*] MCTS seed scheduling: exploration={self._mcts.exploration:.3f}")

            if self._alphabeta is not None:
                ab_data = self._state_store.get("alphabeta")
                if self.resume and ab_data is not None:
                    self._alphabeta.from_dict(ab_data)
                    print(
                        "[*] AlphaBeta: loaded state from state store "
                        f"(nodes={self._alphabeta.stats()['tracked_nodes']})"
                    )
                print("[*] Alpha-beta seed scheduling: Thompson-sampling descent over lineage")

            if self._kruskal_count is not None:
                self._load_kruskal_count()

            if self._edge_ledger is not None:
                self._load_strata()

            if self._entropy_kl is not None:
                self._load_entropy_kl()

            if self._entropy_zscore is not None:
                self._load_entropy_zscore()

            self._load_learned()

            # Print WFC mode status
            if self._wfc_enabled:
                print("[*] WFC: enabled — structural chunk reordering and pixel generation active")

            self._calibrate_seed_baselines()

            self._print_enabled_features()
            print("[*] Starting fuzzing...\n")

            while not _shutdown:
                if iterations and i >= iterations:
                    break
                if max_execs and self.exec_count >= max_execs:
                    break
                if self.continue_until_crash and self.crash_count > 0:
                    break
                # Cycle through targets in multi-target mode
                if self.multi_targets:
                    self._select_next_target()
                # Run any deferred minimization before picking the next seed.
                # This gives freshly-added seeds one full iteration to be selected.
                if self._minimize_pending:
                    self._flush_pending_minimize()
                seed = self._pick_seed()
                # Compute seed-level energy multiplier for mutation budget
                meta = self.seed_meta.get(seed)
                if meta is not None and self._seed_scorer:
                    # Lazy recompute aggregate cache when corpus changed
                    if not self._agg_cache_valid:
                        self._refresh_agg_cache()
                    # Denominator is the number of timed executions, not
                    # fuzz_count: the seed replay above bumps fuzz_count with
                    # no time credited, and a resumed seed used to carry a
                    # restored count against a zero numerator.
                    avg_exec_us = max(1, int(self.mean_exec_time() * 1_000_000))
                    exec_us = max(1, int(seed_exec_us(meta, avg_exec_us)))
                    bitmap_size = meta.get("coverage_edges", 0)
                    avg_bitmap_size = max(
                        1,
                        int(self._cached_total_edges / max(1, len(self.seed_meta))),
                    )
                    depth = meta.get("lineage_depth", 0)
                    fuzz_level = meta.get("fuzz_count", 0)
                    n_fuzz = fuzz_level
                    seed_key = self._seed_key(seed)

                    # Honggfuzz power factors (only when --honggfuzz enabled)
                    hf_kwargs: dict = {}
                    if self.honggfuzz:
                        new_edges = 0
                        if seed_key in self._edge_tracker.seed_edges:
                            seed_e = self._edge_tracker.seed_edges[seed_key]
                            others = set()
                            for sk, se in self._edge_tracker.seed_edges.items():
                                if sk != seed_key:
                                    others.update(se)
                            new_edges = len(seed_e - others)
                        time_added = meta.get("added_at", 0.0)
                        now = time.time()
                        child_count = meta.get("child_count", 0)
                        select_count = fuzz_level
                        timed_out = meta.get("timed_out", False)
                        rare_edges = self._edge_tracker.rare_edge_count(seed_key)

                        # Track honggfuzz factor stats
                        if new_edges > 0 and now - time_added < 600:
                            self._hf_novelty_boosts += 1
                        if now - time_added < 60:
                            self._hf_freshness_boosts += 1
                        if child_count > 0:
                            self._hf_fertility_boosts += 1
                        if bitmap_size > 0 and len(seed) > 0:
                            density = (bitmap_size * 100) / len(seed)
                            if density > 50:
                                self._hf_density_boosts += 1
                        if timed_out:
                            self._hf_timeout_penalties += 1
                        input_entropy = self._seed_entropy_pct(seed, meta)
                        if input_entropy > ENTROPY_RANDOM_PCT or input_entropy < ENTROPY_SPARSE_PCT:
                            self._hf_entropy_penalties += 1

                        stack_depth = self._edge_tracker.get_seed_stack_depth(seed_key)
                        hf_kwargs = dict(
                            new_edges=new_edges,
                            time_added=time_added,
                            now=now,
                            input_size=len(seed),
                            input_entropy=input_entropy,
                            select_count=select_count,
                            child_count=child_count,
                            rare_edge_count=rare_edges,
                            timed_out=timed_out,
                            max_cov=max(1, self._edge_tracker.get_cumulative_edge_count()),
                            hw_instructions=self._last_perf_deltas.get("instructions", 0),
                            hw_branches=self._last_perf_deltas.get("branches", 0),
                            stack_depth=stack_depth,
                        )

                    self._last_perf_score = self._seed_scorer.score(
                        exec_us=exec_us,
                        avg_exec_us=avg_exec_us,
                        bitmap_size=bitmap_size,
                        avg_bitmap_size=avg_bitmap_size,
                        handicap=0,
                        depth=depth,
                        fuzz_level=fuzz_level,
                        n_fuzz=n_fuzz,
                        total_execs=max(1, self.exec_count),
                        mean_log_n_fuzz=self._cached_mean_log_n_fuzz,
                        favored=(seed_key in self._favored),
                        avg_distance=meta.get("avg_distance", -1.0) if self._distance else -1.0,
                        max_distance=(
                            self._dist_max_observed
                            if self._distance and self._dist_max_observed is not None
                            else (self._distance.max_distance if self._distance else 0.0)
                        ),
                        anneal_progress=self._anneal_progress,
                        min_distance=self._dist_min_observed or 0.0,
                        elapsed_sec=time.time() - self.start_time,
                        t_x_minutes=self._seed_scorer.t_x_minutes,
                        katz_energy=(
                            self._katz_channel.seed_energy(seed_key)
                            if getattr(self, "_katz_channel", None) is not None
                            else 0.0
                        ),
                        **hf_kwargs,
                    )
                else:
                    # Markov-generated or synthetic seed: reset to neutral multiplier
                    self._last_perf_score = 100.0
                if self._diff_tracker:
                    self._check_differential(seed)
                # Deterministic-stage mutations are now drawn inline by
                # OperatorEngine.mutate() (see maybe_deterministic_mutation),
                # so they get the same execution/coverage/corpus-save path
                # as every other mutation instead of a separate blocking
                # loop that has to duplicate it.
                self.fuzz_one(seed)
                # Backpropagate this iteration's discovery up the MCTS path.
                # A no-op unless the mcts arm actually selected this seed —
                # update() ignores an empty path — so it stays correct when
                # Elo hands the pick to another strategy.
                if self._mcts is not None:
                    self._mcts.update(self._last_new_edge_count)
                if self._alphabeta is not None:
                    self._alphabeta.update(self._last_new_edge_count)
                i += 1
                effective_interval = self._stats_effective_interval()
                if self.exec_count - self._last_stats_exec >= effective_interval:
                    # Update favored set / top_rated cull queue every interval
                    if self._edge_tracker.seed_edges:
                        self._cull_queue()
                    # Sample Shannon entropy for rate-of-change tracking
                    if self._edge_tracker._global_edge_hits:
                        sh = self._edge_tracker.shannon_entropy_global()
                        self._record_entropy_sample(sh)
                    # Feed incremental edge count to Structure-function detector
                    current_edges = self._edge_tracker.get_cumulative_edge_count()
                    delta = current_edges - self._last_structure_edge_count
                    self._structure_fn.update(delta)
                    if self._garch is not None:
                        self._garch.update(delta)
                    self._discovery_uniformity.update(delta)
                    verdict = self._discovery_uniformity.verdict()
                    if not verdict["homogeneous"] and verdict["n"] >= 32:
                        log.info(
                            "Discovery non-Poisson-dispersed: p=%.4f n=%d",
                            verdict["p"],
                            verdict["n"],
                        )
                    # Multiple-testing correction across the tests that run
                    # on this same `delta` series: structure_function's and
                    # discovery_uniformity's dispersion tests are the same
                    # statistic under different windowing, and garch's
                    # Ljung-Box test is a different null on the same input.
                    # Display-only -- see core/multiple_testing.py and
                    # docs/handover/handover_multiple_testing_2026-09-13.md
                    # for why this does not (yet) gate any decision.
                    self._last_dispersion_corrections = collect_and_correct(self)
                    self._last_structure_edge_count = current_edges
                    # Close this tick's corpus add/prune/reject bucket --
                    # additions and evictions are recorded as they happen in
                    # corpus_manager.py; tick() just closes the window. See
                    # core/corpus_flux.py and
                    # docs/handover/handover_thermo_stochastic_concepts_2026-09-12.md
                    # (P4-T6).
                    self._corpus_flux.tick()
                    # Feed the current AFLGo distance reading (if directed
                    # mode is active and at least one execution has produced
                    # one) to the scaling-exponent detector, once per tick.
                    if self._distance is not None and self._dist_last_value is not None:
                        self._distance_trend.update(self._dist_last_value)
                    # Feed per-column edge counts to CoverageHomogeneityDetector
                    if self.shm_cov and hasattr(self, "_homogeneity"):
                        log.debug("homogeneity: shm_cov present, observing col counts")
                        hit_counts = self.shm_cov.get_edge_counts()
                        n_cols = self._homogeneity.num_columns
                        col_totals_this_tick = [0] * n_cols
                        for edge_id, _count in hit_counts.items():
                            col_totals_this_tick[edge_id % n_cols] += 1
                        col_deltas = [
                            col_totals_this_tick[i] - self._homogeneity_col_cumulative[i]
                            for i in range(n_cols)
                        ]
                        self._homogeneity_col_cumulative = col_totals_this_tick
                        log.debug("homogeneity: col_deltas=%s", col_deltas)
                        self._homogeneity.observe(col_deltas)
                        try:
                            result = self._homogeneity.is_homogeneous()
                            log.debug("homogeneity: result=%s", result)
                            if not result["homogeneous"] and result["total_edges"] > 0:
                                log.info(
                                    "Coverage clustered: χ²=%.2f, p=%.4f, V=%.3f, edges=%d",
                                    result["chi2"],
                                    result["p_value"],
                                    result["cramers_v"],
                                    result["total_edges"],
                                )
                        except Exception as ex:
                            log.debug("Coverage homogeneity check failed: %s", ex)
                        self._observe_continuum(delta, hit_counts)
                    # Capture the homogeneity result for the regime detector.
                    # `result` is only defined inside the shm_cov branch above;
                    # fall back to None when the detector wasn't initialised
                    # or shm_cov isn't available.
                    homogeneity_result = (
                        result if (self.shm_cov and hasattr(self, "_homogeneity")) else None
                    )

                    # Record coverage snapshot for temporal analysis
                    self._edge_tracker.record_coverage_snapshot(self.exec_count)
                    self._maybe_resize_on_drops()
                    if not self.quiet_stats:
                        self.print_stats()
                    self._append_coverage_log()
                    self._record_discovery_snapshot()
                    # Feed the regime detector with all observed signals.
                    # The CriticalSlowingDown detector is already fed from
                    # _print_stats_dr_str (stats.py:583); we only read its
                    # state here.  The homogeneity detector is fed above.
                    execs_since_edge = self.exec_count - self._last_new_edge_exec
                    # F0 plateau: the streaming distinct-edge estimate has
                    # stopped growing.  Opt-in via the edge tracker's
                    # enable_f0 flag; None when the estimator is not wired,
                    # so observe() leaves classification unchanged.
                    f0_plateau = self._edge_tracker.f0_plateau()
                    self._regime.observe(
                        discovery_rate=self._stats.discovery_rate(),
                        structure_delta=delta,
                        homogeneity_result=homogeneity_result,
                        execs_since_edge=execs_since_edge,
                        exec_count=self.exec_count,
                        f0_plateau=f0_plateau,
                    )
                    # Regime-driven strategy adjustment
                    if self._regime.actionable:
                        regime = self._regime.regime
                        reason = self._regime.reason
                        log.info("REGIME: %s — %s", regime.value, reason)
                        if regime is CoverageRegime.SUBCRITICAL:
                            # Subcritical: escalate mutation diversity, force stall recovery
                            if not self._stall_recovery_active:
                                self._maybe_trigger_stall_recovery(execs_since_edge)
                            # Bump havoc energy if the operator engine exposes that knob
                            ops = getattr(self, "_operators", None)
                            if ops is not None and hasattr(ops, "_havoc_energy_scale"):
                                ops._havoc_energy_scale = min(
                                    getattr(ops, "_havoc_energy_scale", 1.0) * 1.5, 5.0
                                )
                        elif regime is CoverageRegime.CRITICAL:
                            # CSD: near a coverage jump — preserve current strategy
                            log.info("REGIME: critical — preserving strategy (%s)", reason)
                        elif regime is CoverageRegime.SUPERCRITICAL:
                            # Healthy: clear any emergency state
                            if self._stall_recovery_active:
                                self._stall_recovery_exit("supercritical regime")
                                log.info("REGIME: supercritical — clearing stall recovery")
                        self._regime.acknowledge()
                    # Stall detection: no new edges in threshold execs
                    if (
                        not self._stall_recovery_active
                        and execs_since_edge >= self._stall_threshold
                        and self.exec_count > 0
                    ):
                        self._maybe_trigger_stall_recovery(execs_since_edge)
                    # Corpus-size-based pruning: minimize when corpus is
                    # significantly larger than the edge-derived target size,
                    # even if --minimize-every-execs is not set.
                    self._check_corpus_size_and_prune()
                    self._refill_format_seeds()
                    if self.job_scheduler:
                        # P3-3 step 4: memory pruning, sanitizer/crash
                        # replays, and periodic GC as one precedence-aware
                        # queue instead of three independent ad-hoc gates.
                        self._maintenance.tick(self.exec_count)
                    else:
                        # Legacy path, byte-for-byte: independent gates,
                        # crash/sanitizer replays and gc.collect keyed off
                        # the raw iteration count rather than the stats
                        # interval.
                        self._legacy_memory_prune_tick()
                        if i % 500 == 0:
                            import gc

                            gc.collect()
                    self._last_stats_exec = self.exec_count
                    if self.stats_file:
                        self._dump_stats()
                        self._save_state()
                if not self.job_scheduler:
                    if i % 500 == 0 and self.replay_n > 0:
                        self._run_crash_replays()
                    if i % 500 == 0 and (self.asan_target or self.ubsan_target):
                        self._run_sanitizer_replays()
        except (KeyboardInterrupt, SystemExit):
            pass
        except OSError as e:
            log.warning("Fuzzing interrupted by OS error: %s", e)
        except Exception:
            # Everything below this try is end-of-run persistence: _dump_stats,
            # every _state_store.set, both _save_state, and the ablation fd
            # close. None of it is in a `finally`, so before this handler any
            # exception the loop did not name — a ValueError out of a scheduler,
            # a KeyError out of a mutator — propagated straight past all of it
            # and discarded the whole campaign: Markov model, Elo ratings,
            # crash-MI counters, GA/QEA/CMA-ES/MCTS generations. Hours of work
            # for a bug in one mutation. Catch broadly so the state lands, and
            # log the traceback so the underlying defect stays loud rather than
            # being swallowed (Hard Rule 20).
            log.exception("Fuzzing aborted by unexpected error — persisting state before exit")
            self._aborted_by_error = True

        # Final coverage snapshot for temporal analysis
        self._edge_tracker.record_coverage_snapshot(self.exec_count)
        self._dump_stats()
        self._dump_coverage_report()
        if self.markov.is_trained():
            self._state_store.set("markov", self.markov.to_dict())
        if self._use_mi and self._mi:
            self._state_store.set("mi", self._mi.to_dict())
        self._state_store.set("crash_mi", self._crash_mi.save())
        self._state_store.set("length_tracker", self._length_tracker.save())
        self._flush_pending_minimize()
        if self.ga:
            self._state_store.set("ga", self.ga.to_dict())
            print(f"[*] GA: saved state (gen={self.ga.generation})")
        if self.qea:
            self._state_store.set("qea", self.qea.to_dict())
            print(f"[*] QEA: saved state (gen={self.qea.generation})")
        if self._cmaes:
            self._state_store.set("cmaes", self._cmaes.to_dict())
            print(f"[*] CMA-ES: saved state (gen={self._cmaes.generation})")
        if self._mcts is not None:
            # Drop stats for seeds minimization removed, so the persisted
            # state does not grow without bound across resumes.
            if self._lineage is not None:
                self._mcts.prune(set(self._lineage.nodes))
            self._state_store.set("mcts", self._mcts.to_dict())
            print(f"[*] MCTS: saved state ({self._mcts.stats()['tracked_nodes']} nodes)")
        if self._alphabeta is not None:
            if self._lineage is not None:
                self._alphabeta.prune(set(self._lineage.nodes))
            self._state_store.set("alphabeta", self._alphabeta.to_dict())
            print(f"[*] AlphaBeta: saved state ({self._alphabeta.stats()['tracked_nodes']} nodes)")
        if self._kruskal_count is not None:
            self._state_store.set("kruskal_count", self._kruskal_count.to_dict())
        self._save_strata()
        if self._entropy_kl is not None:
            self._state_store.set("entropy_kl", self._entropy_kl.to_dict())
        if self._entropy_zscore is not None:
            self._state_store.set("entropy_zscore", self._entropy_zscore.to_dict())
        self._save_learned()
        if self._fluctuation is not None:
            self._state_store.set("fluctuation", self._fluctuation.snapshot())
            samples = sum(len(v) for v in self._fluctuation._states.values())
            print(f"[*] Fluctuation: saved state (samples={samples})")
        if self._garch is not None:
            self._state_store.set("garch", self._garch.save())
        if self._continuum is not None:
            self._state_store.set("continuum", self._continuum.save())
        self._state_store.set("corpus_flux", self._corpus_flux.save())
        if getattr(self, "_temp_controller", None) is not None:
            # The observer's disturbance state and the PI accumulator are
            # both histories, so a resume that drops them restarts the loop
            # cold on a campaign that is anything but. Persisted together or
            # not at all -- restoring the accumulator without the observer
            # would apply an integral built against a disturbance estimate
            # that no longer exists.
            self._state_store.set("temperature_control", self._temp_controller.to_dict())
        self._save_state()
        if self._cmplog is not None:
            # Releases this run's .cmplog/.counts/.sites files. Nothing else
            # called stop(), so every run left its trio behind under a fresh
            # uuid and the cache directory grew for the life of the machine.
            with contextlib.suppress(Exception):
                self._cmplog.stop()
        if self._ablation_file:
            self._ablation_file.flush()
            self._ablation_file.close()
            self._ablation_file = None
            print(f"[*] Schedule ablation log: {self._ablation_path}")
        if not self.quiet_stats:
            self.print_stats()
        stop_word = "aborted by an unexpected error" if self._aborted_by_error else "stopped"
        print(
            f"\n\n[*] Fuzzing {stop_word}. {self.crash_count} crashes found "
            f"({len(self.crash_sigs)} unique signatures)."
        )
        if self._aborted_by_error:
            print("[!] State was persisted; see the log for the traceback.")
        if self.crash_sigs:
            print("[*] Crash signatures:")
            for sig, count in sorted(self.crash_sigs.items(), key=lambda x: -x[1]):
                print(f"    {sig} ({count}x)")
            print(f"\n[*] Crash files in: {self.crashes_dir}")
        # Show convergence stats for every active scheduler
        if self.mc and self.mc_bandit:
            print("\n[*] Bandit convergence (Thompson sampling):")
            for name, (a, b) in sorted(
                # bandit_stats() subtracts priors, so never-selected arms show
                # (0, 0) and are omitted below
                self.mc.bandit_stats().items(),
                key=lambda x: -(x[1][0] / max(x[1][0] + x[1][1], 1)),
            ):
                if a + b <= 0:
                    continue
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
        if self._cmaes:
            print("\n[*] CMA-ES convergence:")
            stats = self._cmaes.convergence_stats()
            print(
                f"    generation={stats['generation']} sigma={stats['sigma']:.4f} "
                f"top={stats['top_op']}({stats['top_prob']:.1%}) "
                f"discoveries={stats['total_discoveries']}/{stats['total_execs']}"
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
        # Seed strategy convergence (only strategies actually used this run)
        # Same table (and op_/seed_ display names) as the report's strategy
        # section; the rows functions only decide which strategies appear.
        from fuzzer_tool.services.report import strategy_table_lines

        seed_rows = self._seed_convergence_rows()
        if seed_rows:
            print("\n[*] Seed strategy convergence:")
            for line in strategy_table_lines(
                self._elo, [f"seed_{s}" for s, *_ in seed_rows], "    "
            ):
                print(line)
        pos_rows = self._position_convergence_rows()
        if pos_rows:
            print("\n[*] Position strategy convergence:")
            for line in strategy_table_lines(
                self._elo, [POS_STRATEGY_PREFIX + s for s, *_ in pos_rows], "    "
            ):
                print(line)
        # Operator strategy convergence (only schedulers actually selected)
        op_rows = self._operator_convergence_rows()
        if op_rows:
            print("\n[*] Operator strategy convergence:")
            for line in strategy_table_lines(self._elo, [s for s, *_ in op_rows], "    "):
                print(line)
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
        print()  # blank line before next epoch or shell prompt
        # Hand os.environ back the way we found it. setup_env_for_run() puts
        # the cmplog shim on the process-global LD_PRELOAD, and that shim
        # conflicts with the ASAN runtime, so anything exec'd after this run
        # -- the next epoch in a multi-target session, a replay, a caller
        # embedding Fuzzer -- would silently find no crashes.
        if self._cmplog is not None:
            self._cmplog.restore_env()
        # cmplog's restore_env() above only ever undid its own LD_PRELOAD
        # edit. __AFL_DIST_SHM_ID, __AFL_SHM_ID, AFL_MAP_SIZE, the ASAN
        # LD_PRELOAD injection and UBSAN_OPTIONS were never restored, so
        # they leaked into whatever ran next in this process -- see
        # finding #10. _restore_environ() puts back everything this Fuzzer
        # (or an earlier one in the same process) changed.
        _restore_environ()
