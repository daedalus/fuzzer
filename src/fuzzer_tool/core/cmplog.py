"""Cmplog collector: parse comparison tracing output and feed into dictionary.

The shim lives in ``adapters/afl_shim.c`` behind ``-D__AFL_CMPLOG=1`` and
provides two complementary interception layers, both writing to the same
CMP log file:

1. Symbol-based: intercepts libc comparison functions (memcmp/strcmp/strncmp/
   memchr/strcasecmp/strncasecmp/memmem/strstr/strcasestr) via LD_PRELOAD or
   build-time linking.

2. Compiler-IR-based: implements Clang's -fsanitize-coverage=trace-cmp
   callbacks (__sanitizer_cov_trace_cmp*, __sanitizer_cov_trace_switch) that
   fire after the compiler has inlined/folded comparisons into integer compares.

Both layers are compiled directly into the target (``-include afl_shim.c
-D__AFL_CMPLOG=1``), which is the only arrangement in which they reliably
fire — see the symbol-resolution note at the top of ``afl_shim.c``.

For targets that were NOT built with the shim there is still an
LD_PRELOAD fallback, compiled from the same source with
``-D__AFL_PRELOAD_ONLY``. That build deliberately contains no edge
machinery at all, so it cannot preempt an instrumented target's
``__afl_map_shm`` the way the old standalone ``cmplog_shim.so`` could.
"""

import contextlib
import hashlib
import logging
import os
import uuid

from fuzzer_tool.adapters.track_parser import conds_from_cmplog_text

log = logging.getLogger(__name__)

# ── Disk-backed directory (avoid tmpfs-full failures) ─────────────────

_CMPLOG_DIR: str | None = None


def _get_cmplog_dir() -> str:
    """Return a disk-backed (non-tmpfs) directory for cmplog artifacts.

    Falls back to tempfile.gettempdir() if XDG_CACHE_HOME and HOME
    are unavailable.
    """
    global _CMPLOG_DIR
    if _CMPLOG_DIR is not None:
        return _CMPLOG_DIR
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    _CMPLOG_DIR = os.path.join(base, "fuzzer_cmplog")
    os.makedirs(_CMPLOG_DIR, exist_ok=True)
    return _CMPLOG_DIR


def _cleanup_stale_cmplog_files():
    """Remove stale cmplog log files from previous runs at startup."""
    d = _get_cmplog_dir()
    removed = 0
    for entry in os.scandir(d):
        if entry.name.endswith((".cmplog", ".counts")) and entry.is_file():
            try:
                os.unlink(entry.path)
                removed += 1
            except OSError:
                pass
    if removed:
        log.info("Cleaned %d stale cmplog file(s) from %s", removed, d)


# Legacy fixed-name artifact from before the shim cache was keyed on the
# source digest. Never overwritten once the digest scheme landed, so it
# lingers forever unless explicitly removed. Objects built from the old
# cmplog_shim.c share the same fuzz_cmplog_shim.<digest>.so naming, so the
# digest change alone retires them through the existing prune path.
_LEGACY_SHIM_NAME = "fuzz_cmplog_shim.so"


def _prune_stale_shims(cmplog_dir: str, keep: str) -> int:
    """Delete cached shim objects other than *keep*.

    The cached ``.so`` is keyed on ``sha256(source)[:16]``, which correctly
    forces a recompile on every edit — but nothing reclaimed the superseded
    digests, so each edit permanently leaked another artifact into the cache
    directory. Drops the pre-digest fixed-name object as well.

    Args:
        cmplog_dir: Directory holding the cached shim objects.
        keep: Basename of the shim to retain (the current digest).

    Returns:
        Number of artifacts removed.
    """
    removed = 0
    try:
        entries = list(os.scandir(cmplog_dir))
    except OSError:
        return 0
    for entry in entries:
        name = entry.name
        if name == keep or not entry.is_file():
            continue
        stale = name == _LEGACY_SHIM_NAME or (
            name.startswith("fuzz_cmplog_shim.") and name.endswith(".so")
        )
        if not stale:
            continue
        try:
            os.unlink(entry.path)
            removed += 1
        except OSError:
            pass
    if removed:
        log.info("Pruned %d superseded cmplog shim object(s) from %s", removed, cmplog_dir)
    return removed


# ── Memory bounds ────────────────────────────────────────────────────
CMPLOG_TOKENS_MAX = 10_000  # max unique operand tokens
CMPLOG_PAIRS_MAX = 5_000  # max unique operand pairs
CMPLOG_FILE_MAX_BYTES = 100 * 1024 * 1024  # max cmplog file size before rotation
# The counts sidecar carries at most one short line per callback per dump,
# so it grows orders of magnitude slower than the record stream and gets a
# correspondingly smaller cap. Reaching it means the collector stopped
# draining, not that the target is chatty.
CMPLOG_COUNTS_MAX_BYTES = 4 * 1024 * 1024

# ── Wall detection ────────────────────────────────────────────────────
#
# A callback with a high fire count and a near-zero assert rate is a
# comparison the campaign reaches constantly and never passes. Below
# CMP_WALL_MIN_FIRED there is not enough evidence to call anything: a
# callback entered twice and satisfied never is an accident, not a wall.
CMP_WALL_MIN_FIRED = 1_000
# Above this assert rate the comparison is being passed often enough that
# whatever is failing is a minority of its call sites, which the callback
# granularity cannot separate anyway.
CMP_WALL_MAX_ASSERT_RATE = 0.001
# Two EWMAs of the per-execution fire count give the trend without keeping
# a history buffer. The ratio between them is what matters, not either
# value: a stall whose fire count is rising means the campaign still
# reaches the wall and keeps failing there, and a stall whose fire count is
# falling means it stopped reaching the parser depth it used to. Those want
# opposite remedies, and the edge signal is identical in both.
CMP_RATE_FAST_ALPHA = 0.1
CMP_RATE_SLOW_ALPHA = 0.01
# How far the fast rate must diverge from the slow one to call a direction.
CMP_RATE_TREND_MARGIN = 0.2

# Hash detection thresholds
_HASH_MIN_BYTES = 8  # minimum operand length to consider as hash-like
_HASH_MAX_MATCH_BYTES = 2  # max matching byte positions for a hash-like pair


class CmplogCollector:
    """Collect and process comparison tracing data from the cmplog shim.

    After each execution, reads the CMP log file, extracts operand pairs,
    and converts them into dictionary tokens for mutation.

    Args:
        max_tokens: Cap on unique operand tokens (default CMPLOG_TOKENS_MAX).
        max_pairs: Cap on unique operand pairs (default CMPLOG_PAIRS_MAX).
    """

    def __init__(self, max_tokens: int = 0, max_pairs: int = 0, workdir: str | None = None):
        self.log_path: str | None = None
        self.tokens: list[bytes] = []
        self._token_set: set[bytes] = set()
        # Operand pairs: (operand_a, operand_b) for input-to-state matching
        self.pairs: list[tuple[bytes, bytes]] = []
        self._pair_set: set[tuple[bytes, bytes]] = set()
        # PC mapping: pair -> program counter (optional, from trace-mode shim)
        self._pair_pc: dict[tuple[bytes, bytes], int | None] = {}
        # (op_a, op_b) -> (result, width). The shim already emits the
        # comparison outcome (-1 a<b / 0 a==b / 1 a>b) and operand width;
        # both are needed to negate a branch predicate rather than merely
        # replay the operands that reached it.
        self._pair_cmp: dict[tuple[bytes, bytes], tuple[int, int]] = {}
        self._shim_path: str | None = None
        self._shim_handle = None
        # Pre-mutation values of the os.environ keys setup_env_for_run owns,
        # captured once so restore_env() can put them back. None = never set up.
        self._env_saved: dict[str, str | None] | None = None
        self.workdir: str | None = workdir  # dir for runtime log files
        # Value-density signal: how often each token/pair was present
        # when a coverage gain was detected. Higher = more valuable.
        self._token_value: dict[bytes, int] = {}
        self._pair_value: dict[tuple[bytes, bytes], int] = {}
        # Hash candidates: pairs that look like checksums/CRCs and should
        # be skipped by the I2S encoding engine to avoid wasted execs.
        self.hash_candidates: set[tuple[bytes, bytes]] = set()
        # Multi-run comparison history: input_hash -> set of observed pairs.
        # Used to detect which comparisons are consistently triggered by
        # which input variants (cross-referencing colored vs uncolored runs).
        self._run_history: dict[int, set[tuple[bytes, bytes]]] = {}
        # Occurrence count: how many times each pair has been observed across runs.
        # Higher counts = more reliable comparison signals.
        self._pair_occurrence: dict[tuple[bytes, bytes], int] = {}
        # Overridable caps (0 = use module default)
        self._max_tokens = max_tokens if max_tokens > 0 else CMPLOG_TOKENS_MAX
        self._max_pairs = max_pairs if max_pairs > 0 else CMPLOG_PAIRS_MAX
        # File offset: only read new data since last collection.
        # Without this, collect_tokens re-reads the entire file (5M+ lines)
        # on every call, dominating runtime at 60%+ CPU.
        self._read_offset: int = 0
        # Per-callback comparison counters, fed by the shim's $_CMPLOG_COUNTS
        # sidecar. Keyed by callback name ("memcmp", "trace_const_cmp4", ...);
        # cumulative for the run, summed from the shim's per-dump deltas.
        self.counts_path: str | None = None
        self.cmp_fired: dict[str, int] = {}
        self.cmp_asserted: dict[str, int] = {}
        self._counts_offset: int = 0
        # What the most recent drain added, keyed the same way. Only useful
        # when the drain happens on an execution boundary -- see
        # collect_counts() -- in which case it is that execution's own
        # comparison vector rather than a slice of the run total.
        self.last_fired: dict[str, int] = {}
        self.last_asserted: dict[str, int] = {}
        # Fast and slow EWMAs of the per-execution fire count, per callback.
        # Fed only by drains that actually read something -- see
        # _update_rates.
        self.cmp_rate_fast: dict[str, float] = {}
        self.cmp_rate_slow: dict[str, float] = {}

    def start(self) -> bool:
        """Compile the LD_PRELOAD comparison-logging shim.

        Built from afl_shim.c with ``-D__AFL_PRELOAD_ONLY``: the libc and
        trace-cmp layers only, no ``__afl_map_shm`` / ``__afl_map_reset`` /
        ``__sanitizer_cov_trace_pc_guard``. Emitting those was what let the
        old cmplog_shim.so win the global lookup ahead of an instrumented
        target and leave its ``__afl_area`` NULL.

        Only needed for targets that were not built with the shim compiled
        in; when it is compiled in, ``_detect_cmplog`` finds ``__cmplog_reset``
        and no preload happens.
        """
        from fuzzer_tool.adapters.shim_factory import _find_compiler

        shim_src = os.path.join(os.path.dirname(__file__), "..", "adapters", "afl_shim.c")
        if not os.path.exists(shim_src):
            log.warning("afl_shim.c not found at %s", shim_src)
            return False

        # Use disk-backed directory (avoid tmpfs-full failures)
        cmplog_dir = _get_cmplog_dir()
        # Key the cached artifact on the source digest so any edit to
        # afl_shim.c forces a recompile instead of silently loading a
        # stale .so for the life of the machine.
        try:
            with open(shim_src, "rb") as _f:
                _digest = hashlib.sha256(_f.read()).hexdigest()[:16]
        except OSError as e:
            log.warning("Could not read cmplog shim source %s: %s", shim_src, e)
            return False
        out_path = os.path.join(cmplog_dir, f"fuzz_cmplog_shim.{_digest}.so")
        if os.path.exists(out_path):
            self._shim_path = out_path
            log.info("Cmplog shim cached: %s", out_path)
        else:
            try:
                compiler = _find_compiler()
                # Strip ASAN from subprocess env — libasan's LeakSanitizer
                # causes false-positive leak reports in the compiler itself.
                _env = os.environ.copy()
                _env.pop("ASAN_OPTIONS", None)
                _env.pop("LSAN_OPTIONS", None)
                _ld_preload = _env.get("LD_PRELOAD", "")
                if _ld_preload:
                    _parts = [p for p in _ld_preload.split(":") if "libasan" not in p]
                    _env["LD_PRELOAD"] = ":".join(_parts) if _parts else ""
                result = __import__("subprocess").run(
                    [
                        compiler,
                        "-shared",
                        "-fPIC",
                        "-O2",
                        "-D__AFL_PRELOAD_ONLY",
                        "-ldl",
                        "-o",
                        out_path,
                        shim_src,
                    ],
                    capture_output=True,
                    timeout=30,
                    env=_env,
                )
                if result.returncode == 0 and os.path.exists(out_path):
                    self._shim_path = out_path
                    log.info("Cmplog shim compiled: %s", out_path)
                else:
                    log.warning("Cmplog shim compilation failed: %s", result.stderr.decode()[:200])
            except Exception as e:
                log.warning("Cmplog shim compilation error: %s", e)

        # Reclaim superseded digests only once the current object exists, so
        # a failed compile never leaves the cache empty.
        if self._shim_path:
            _prune_stale_shims(cmplog_dir, os.path.basename(self._shim_path))

        return self._shim_path is not None

    def _counts_path_for(self, log_path: str) -> str:
        """Sidecar path for the shim's per-callback comparison counters.

        A separate file from the record stream on purpose. ``collect_tokens``
        caps its read at 10k lines and truncates whatever it did not reach,
        and ``_rotate_cmplog`` discards the stream wholesale past the size
        cap -- a counter record riding along in it would be dropped silently
        and the totals would drift low precisely on the noisiest targets.
        """
        base, dot, _ext = log_path.rpartition(".")
        return (base if dot else log_path) + ".counts"

    def setup_env(self, env: dict[str, str]) -> dict[str, str]:
        """Add cmplog env vars to the execution environment.

        Reuses (truncates) the existing log_path if one exists, or creates
        a fresh log file on first call. Adds _CMPLOG_OUT + LD_PRELOAD
        to *env*. Used for subprocess execution paths (fork+exec).

        Both the symbol-based shim and trace-cmp shim are prepended to
        LD_PRELOAD when available. They export different symbols and write
        to the same _CMPLOG_OUT file.

        Args:
            env: Current environment dict.

        Returns:
            Modified env with LD_PRELOAD and _CMPLOG_OUT set.
        """
        if not self._shim_path:
            return env

        if self.log_path is None or not os.path.exists(self.log_path):
            log_dir = self.workdir or _get_cmplog_dir()
            os.makedirs(log_dir, exist_ok=True)
            local_id = uuid.uuid4().hex[:12]
            self.log_path = os.path.join(log_dir, f"fuzz_cmplog_{local_id}.cmplog")
        else:
            # Truncate so the child writes fresh data from position 0
            with contextlib.suppress(OSError), open(self.log_path, "w") as f:
                f.truncate(0)
        self.counts_path = self._counts_path_for(self.log_path)
        env = dict(env)  # copy
        env["_CMPLOG_OUT"] = self.log_path
        env["_CMPLOG_COUNTS"] = self.counts_path

        # Prepend the unified shim to LD_PRELOAD
        if self._shim_path:
            existing = env.get("LD_PRELOAD", "")
            env["LD_PRELOAD"] = f"{self._shim_path}:{existing}" if existing else self._shim_path

        return env

    def setup_env_for_run(self):
        """Set _CMPLOG_OUT in the current process environment.

        Used by inprocess and persistent execution paths where the target
        runs inside the fuzzer process (or a long-lived child) and inherits
        os.environ rather than a per-call env dict.

        Reuses the current log_path if one exists; creates a new one on first call.
        The cmplog shim (whether LD_PRELOAD'd or compiled into the target .so)
        reads _CMPLOG_OUT at constructor time.

        The unified shim provides both libc interposition and compiler-IR
        callbacks. Placed before any ASAN library so coverage/tracecmp
        symbols resolve to the shim, not ASAN's built-in no-op stubs.

        The prior values are recorded on first call so ``restore_env()`` can
        put them back. Process-global LD_PRELOAD outlives the run that wanted
        it, and the shim conflicts with the ASAN runtime -- an unrestored
        preload turns every later subprocess exec into a run that finds
        nothing, with no error anywhere. See ``restore_env``.
        """
        if self.log_path is None or not os.path.exists(self.log_path):
            log_dir = self.workdir or _get_cmplog_dir()
            os.makedirs(log_dir, exist_ok=True)
            local_id = uuid.uuid4().hex[:12]
            self.log_path = os.path.join(log_dir, f"fuzz_cmplog_{local_id}.cmplog")

        # Snapshot once, before the first mutation. Re-snapshotting on a later
        # call would capture our own LD_PRELOAD and make restore a no-op --
        # and this is called before *every* execution, not once per run.
        if self._env_saved is None:
            self._env_saved = {
                k: os.environ.get(k) for k in ("_CMPLOG_OUT", "_CMPLOG_COUNTS", "LD_PRELOAD")
            }

        self.counts_path = self._counts_path_for(self.log_path)
        os.environ["_CMPLOG_OUT"] = self.log_path
        os.environ["_CMPLOG_COUNTS"] = self.counts_path

        if self._shim_path and self._shim_path not in os.environ.get("LD_PRELOAD", ""):
            existing = os.environ.get("LD_PRELOAD", "")
            os.environ["LD_PRELOAD"] = (
                f"{self._shim_path}:{existing}" if existing else self._shim_path
            )

    def restore_env(self) -> None:
        """Undo ``setup_env_for_run``'s mutations of ``os.environ``.

        Restores ``_CMPLOG_OUT`` and ``LD_PRELOAD`` to whatever they were
        before the first ``setup_env_for_run`` call, deleting keys that were
        absent rather than leaving an empty string behind (the linker treats
        ``LD_PRELOAD=""`` and an unset LD_PRELOAD alike, but a bare
        ``os.environ`` comparison does not, and that is what test guards and
        ``_clean_env`` callers see).

        Idempotent, and a no-op if setup was never called.
        """
        if self._env_saved is None:
            return
        for key, prior in self._env_saved.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior
        self._env_saved = None

    def preload_shims(self) -> bool:
        """Load the unified cmplog shim into the current process via ctypes.

        Used in direct_lite mode where LD_PRELOAD can't affect ctypes.CDLL
        (the dynamic linker resolves LD_PRELOAD at process start). Loads the
        shim .so with RTLD_GLOBAL so the target .so can resolve undefined
        symbols (__sanitizer_cov_trace_cmp*, etc.) at CDLL time.

        Stores the loaded shim handle for later flush/reset calls.

        Returns:
            True if the shim was loaded successfully.
        """
        import ctypes

        self._shim_handles: list[ctypes.CDLL] = []
        if self._shim_path and os.path.exists(self._shim_path):
            try:
                handle = ctypes.CDLL(self._shim_path, mode=ctypes.RTLD_GLOBAL)
                self._shim_handles = [handle]
                return True
            except OSError:
                log.debug("Failed to preload shim: %s", self._shim_path)
        return False

    def flush_shims(self):
        """Flush the cmplog buffer to disk.

        Calls __tracecmp_flush on the loaded shim handle.
        The unified shim provides __tracecmp_flush (alias for flush_buffer +
        fflush) via the same .so as __cmplog_reset.
        """
        for handle in getattr(self, "_shim_handles", []):
            try:
                fn = getattr(handle, "__tracecmp_flush", None)
                if fn is not None:
                    fn()
            except (AttributeError, OSError):
                pass

    def reset_log(self):
        """Reset the cmplog log file after a direct_lite execution.

        When cmplog is compiled into the target .so, the shim keeps the
        file open in append mode across calls. The fuzzer calls this after
        reading tokens to truncate the file so the shim writes fresh data
        on the next call.

        If the .so exposes __cmplog_reset, calls it via ctypes to truncate
        the file from inside the .so. Otherwise falls back to truncating
        the file externally (works when the .so closes/reopens on each
        constructor, e.g. LD_PRELOAD in subprocess mode — harmless no-op
        for the per-call temp-file path).

        If the file exceeds CMPLOG_FILE_MAX_BYTES, hard-resets it via
        _rotate_cmplog, which truncates in place rather than truncating
        through the normal path below. The log path never changes.
        """
        if not self.log_path:
            return

        try:
            size = os.path.getsize(self.log_path)
        except OSError:
            size = 0

        if size > CMPLOG_FILE_MAX_BYTES:
            self._rotate_cmplog()
            return

        try:
            with open(self.log_path, "w") as f:
                f.truncate(0)
        except OSError:
            pass
        self._read_offset = 0

    def _rotate_cmplog(self):
        """Hard-reset the cmplog file when it exceeds the size cap.

        Truncates **in place**, keeping the same path. An earlier version
        created a fresh file and repointed _CMPLOG_OUT at it, which was wrong
        in two ways:

        1. The superseded file was never unlinked, so disk growth stayed
           unbounded -- an unbounded *number* of 100 MB files rather than one
           unbounded file. With the workdir under corpus_dir (93ce70b) they
           piled up inside the corpus.
        2. ``os.environ`` reaches a ctypes-loaded shim (setitem calls putenv)
           and any subprocess spawned afterwards, but *not* a target already
           exec'd with the old environment. A forkserver child would call
           getenv("_CMPLOG_OUT") on its lazy reopen, get the stale path, and
           carry on filling the very file rotation meant to retire -- while
           the collector read a new file that stayed empty.

        Keeping the path fixed removes both problems: nothing accumulates and
        there is no new value to propagate. The records are transient (already
        consumed by collect_tokens), so there is nothing worth preserving.

        Truncation is safe against a live writer because the shim opens with
        O_APPEND, so its next write lands at the new end-of-file rather than
        at a stale offset -- no sparse re-growth.
        """
        if not self.log_path:
            return

        # Preferred: let the shim truncate through its own fd, which also
        # resets its internal offset. Covers the compiled-in case where the
        # fd stays open across executions.
        reset_ok = False
        for handle in getattr(self, "_shim_handles", []):
            try:
                reset = getattr(handle, "__cmplog_reset", None)
                if reset is not None:
                    reset()
                    reset_ok = True
            except (AttributeError, OSError):
                pass

        # Fallback (and belt-and-braces after the shim reset): truncate
        # externally. Correct for LD_PRELOAD/subprocess shims, which reopen
        # per exec, and a harmless no-op when the shim already zeroed it.
        try:
            with open(self.log_path, "r+b") as f:
                f.truncate(0)
        except OSError:
            if not reset_ok:
                log.warning("Cmplog: could not truncate oversized log %s", self.log_path)
                return

        self._read_offset = 0
        log.info(
            "Cmplog: truncated %s after exceeding %d bytes",
            self.log_path,
            CMPLOG_FILE_MAX_BYTES,
        )

    def collect_tokens(self) -> list[bytes]:
        """Read new cmplog data and extract operand tokens and pairs.

        Reads from the current file position up to MAX_CMPLOG_LINES_PER_READ
        lines. The token/pair sets are deduplicated, so once the pool is
        saturated, additional reads yield diminishing returns. Capping the
        read prevents 740K+ line scans from dominating CPU (was 60%+).

        Returns:
            List of unique byte sequences found in comparison operands.
            Also populates self.pairs with (operand_a, operand_b) tuples
            for input-to-state redqueen matching.
        """
        # Drained first, and outside the log-missing guard below: the
        # counters live on their own file and stay meaningful even when the
        # record stream is empty (a target whose comparisons all *pass*
        # writes no layer-1 records at all, yet is exactly the target whose
        # asserted counts you want to see).
        self.collect_counts()

        if not self.log_path or not os.path.exists(self.log_path):
            log.debug("collect_tokens: log missing or empty path=%s", self.log_path)
            return []

        try:
            if os.path.getsize(self.log_path) > CMPLOG_FILE_MAX_BYTES:
                self._rotate_cmplog()
        except OSError:
            pass

        tokens = set()
        new_pairs = []
        # Every distinct pair seen in *this* batch, first sighting or not.
        # _pair_occurrence is incremented from this, not from new_pairs.
        batch_pairs: set[tuple[bytes, bytes]] = set()
        lines_read = 0
        max_lines = 10_000  # cap per collection to bound CPU cost
        try:
            with open(self.log_path) as f:
                f.seek(self._read_offset)
                new_lines = []
                for line in f:
                    lines_read += 1
                    if lines_read > max_lines:
                        break
                    new_lines.append(line)
                self._read_offset = f.tell()
        except OSError as e:
            log.debug("Failed to read cmplog file: %s", e)
            new_lines = []

        log.debug("collect_tokens: read %d new lines from %s", len(new_lines), self.log_path)
        for c in conds_from_cmplog_text(new_lines):
            pair = (c.base.op_a, c.base.op_b)
            if pair not in self._pair_set:
                self._pair_set.add(pair)
                new_pairs.append(pair)
                if c.base.pc is not None:
                    self._pair_pc[pair] = c.base.pc
                if c.base.result is not None and c.base.width is not None:
                    self._pair_cmp[pair] = (c.base.result, c.base.width)
            batch_pairs.add(pair)
            tokens.add(c.base.op_a)
            tokens.add(c.base.op_b)

        # Clear the log for next round.
        # Truncate (not delete) so the .so's file handle stays valid
        # when cmplog is compiled into the target (direct_lite mode).
        with contextlib.suppress(OSError), open(self.log_path, "w") as f:
            f.truncate(0)
        self._read_offset = 0

        new_tokens = [t for t in tokens if t not in self._token_set]
        self._token_set.update(tokens)
        self.tokens.extend(new_tokens)
        self.pairs.extend(new_pairs)

        # Cap token/pair lists to bound memory.
        # Preserves highest-value-density entries instead of simple recency.
        if len(self.tokens) > self._max_tokens:
            excess = len(self.tokens) - self._max_tokens
            scored = [(self._token_value.get(t, 0) / max(len(t), 1), t) for t in self._token_set]
            scored.sort(key=lambda x: x[0])  # lowest value-density first
            for _, t in scored[:excess]:
                self._token_set.discard(t)
                self._token_value.pop(t, None)
            self.tokens = list(self._token_set)
        if len(self.pairs) > self._max_pairs:
            excess = len(self.pairs) - self._max_pairs
            scored = [
                (self._pair_value.get(p, 0) / max(len(p[0]) + len(p[1]), 1), p)
                for p in self._pair_set
            ]
            scored.sort(key=lambda x: x[0])  # lowest value-density first
            for _, p in scored[:excess]:
                self._pair_set.discard(p)
                self._pair_value.pop(p, None)
                # Side tables must be evicted with the pair, not just the
                # pair set: they are keyed by pair and otherwise grow without
                # bound for the life of the run, regardless of max_pairs.
                self._pair_cmp.pop(p, None)
                self._pair_pc.pop(p, None)
                # _pair_occurrence is deliberately left alone: it is
                # repopulated below for this batch, and its whole purpose is
                # to count sightings across runs, so it has to outlive the
                # pair set. It does grow unbounded — a separate concern from
                # this eviction, and one that cannot be fixed here without
                # changing what pair_confidence() means.
            self.pairs = list(self._pair_set)

        # Track pair occurrence across runs for multi-run confidence.
        # Pairs seen in many runs are reliable I2S signals; rarely-seen
        # pairs may be noise from edge-case execution paths.
        #
        # Counted over batch_pairs, NOT new_pairs. new_pairs holds only pairs
        # absent from _pair_set, and every such pair is added to _pair_set in
        # the same iteration -- so a pair could never be "new" twice, every
        # count was pinned at 1 for the life of the run, pair_confidence()
        # was a membership test wearing a counter's clothes, and
        # high_confidence_pairs(min_occurrences=2) returned [] always.
        #
        # One increment per collection batch, not per logged line: the parser
        # already dedups within a batch, and the intended unit is "runs that
        # exercised this comparison", not "times the comparison fired".
        # Raw fire counts come from the shim's own counters.
        for pair in batch_pairs:
            self._pair_occurrence[pair] = self._pair_occurrence.get(pair, 0) + 1

        if new_tokens:
            log.info(
                "Cmplog: found %d new tokens, %d new pairs (total: %d tokens, %d pairs)",
                len(new_tokens),
                len(new_pairs),
                len(self.tokens),
                len(self.pairs),
            )

        # Run hash detection on new pairs
        if new_pairs:
            n_hash = self.detect_hash_candidates(new_pairs)
            if n_hash:
                log.info("Cmplog: flagged %d hash-like pairs (skipped by encoder)", n_hash)

        return new_tokens

    def collect_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        """Fold the shim's per-callback counter deltas into the run totals.

        The shim writes ``CNT <name> <fired> <asserted>`` lines to
        ``$_CMPLOG_COUNTS`` at each sync point (per-iteration reset, flush,
        process exit, crash), zeroing its counters as it writes. Every line
        is therefore a *delta*, and summing them is correct in every
        execution mode without this side knowing anything about process
        lifetimes: a subprocess run contributes one dump per exec, a
        direct_lite run contributes many dumps from one long-lived process.

        Read from a saved offset rather than read-and-truncate: a subprocess
        target can dump between the read and the truncate, and those counts
        would vanish. Truncation happens only past the size cap, where the
        loss is bounded and explicit.

        Also records what *this* drain added, in ``last_fired`` /
        ``last_asserted``. Called on an execution boundary that is the
        executed input's own comparison vector; called on the token-
        collection schedule it is the sum over however many executions have
        happened since, which is a different and much less useful quantity.
        A drain that finds nothing clears both, so a stale vector is never
        read as the current execution's.

        Returns:
            ``(fired, asserted)`` for this drain only -- the same dicts as
            ``last_fired`` / ``last_asserted``.
        """
        self.last_fired = {}
        self.last_asserted = {}
        if not self.counts_path or not os.path.exists(self.counts_path):
            return self.last_fired, self.last_asserted

        try:
            if os.path.getsize(self.counts_path) > CMPLOG_COUNTS_MAX_BYTES:
                # Same reasoning as _rotate_cmplog: truncate in place, keep
                # the path. The shim holds the fd with O_APPEND, so its next
                # write lands at the new end rather than re-growing a hole.
                with open(self.counts_path, "r+b") as fh:
                    fh.truncate(0)
                self._counts_offset = 0
                log.debug("Cmplog: truncated oversized counts file %s", self.counts_path)
                return self.last_fired, self.last_asserted
        except OSError:
            pass

        try:
            with open(self.counts_path) as fh:
                fh.seek(self._counts_offset)
                new_lines = fh.readlines()
                self._counts_offset = fh.tell()
        except OSError as e:
            log.debug("Failed to read cmplog counts file: %s", e)
            return self.last_fired, self.last_asserted

        for line in new_lines:
            parts = line.split()
            if len(parts) != 4 or parts[0] != "CNT":
                continue
            try:
                fired = int(parts[2])
                asserted = int(parts[3])
            except ValueError:
                continue
            name = parts[1]
            self.cmp_fired[name] = self.cmp_fired.get(name, 0) + fired
            self.cmp_asserted[name] = self.cmp_asserted.get(name, 0) + asserted
            if fired:
                self.last_fired[name] = self.last_fired.get(name, 0) + fired
            if asserted:
                self.last_asserted[name] = self.last_asserted.get(name, 0) + asserted

        self._update_rates()
        return self.last_fired, self.last_asserted

    def _update_rates(self) -> None:
        """Fold this drain's fire counts into the per-callback EWMAs.

        Only non-empty drains count. A drain that read nothing is either an
        execution that compared nothing or a redundant second drain of an
        already-drained boundary, and from here those two are
        indistinguishable -- so the rates are over *observed* executions
        rather than over every execution, and a redundant drain cannot pull
        every callback's rate toward zero.

        Callbacks absent from a non-empty drain do decay, and must: a
        comparison the campaign has stopped reaching is exactly the case
        the falling-trend reading exists to name, and it presents as
        absence, not as a zero.
        """
        if not self.last_fired:
            return
        for name in self.cmp_fired:
            observed = float(self.last_fired.get(name, 0))
            fast = self.cmp_rate_fast.get(name)
            slow = self.cmp_rate_slow.get(name)
            if fast is None:
                # Seed both from the first observation instead of from zero,
                # or every callback reads as "rising" for its first few
                # hundred drains purely from the EWMAs warming up.
                self.cmp_rate_fast[name] = observed
                self.cmp_rate_slow[name] = observed
                continue
            self.cmp_rate_fast[name] = fast + CMP_RATE_FAST_ALPHA * (observed - fast)
            self.cmp_rate_slow[name] = slow + CMP_RATE_SLOW_ALPHA * (observed - slow)

    def comparison_stats(self) -> dict[str, tuple[int, int]]:
        """Per-callback ``(fired, asserted)`` counts, cumulative for the run.

        ``fired`` is how many times the comparison callback was entered;
        ``asserted`` is how many of those had their predicate hold --
        operands equal for the cmp family, needle found for the search
        family, a non-empty span for strspn/strcspn, ``a == b`` for the
        trace-cmp callbacks, and a matched case for switch dispatch.

        Neither number is derivable from the record stream: the records
        carry no callback identity, the collector dedups away multiplicity,
        and satisfied layer-1 comparisons are never written at all (the
        record writer drops ``result == 0`` to keep solved comparisons out
        of the I2S pair pool).
        """
        return {
            name: (fired, self.cmp_asserted.get(name, 0))
            for name, fired in sorted(self.cmp_fired.items())
        }

    def total_comparisons(self) -> tuple[int, int]:
        """Run totals as ``(fired, asserted)`` across every callback."""
        return sum(self.cmp_fired.values()), sum(self.cmp_asserted.values())

    def layer_totals(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Totals split by instrumentation layer, as ``(layer1, layer2)``.

        Layer 1 is the libc interposition (``memcmp``, ``strstr``, ...) and
        sees real operands; layer 2 is the compiler's trace-cmp callbacks
        and sees whatever survived to IR. The ratio says what kind of target
        this is, and it is the only thing that does:

        * layer 2 alone means the comparisons were inlined, which is the
          regime where trace-cmp yields the degenerate ``(0, 1)`` pairs --
          the record stream is nearly worthless there even though it is
          large;
        * layer 1 alone means the parser goes through libc, so token
          extraction and the pair pool are worth the budget;
        * neither means the instrumentation is not reaching the target at
          all, which is otherwise entirely silent.

        Classified on the ``trace_`` prefix rather than a second copy of the
        callback list, so a new interceptor lands on the right side without
        anyone having to remember this function exists.
        """
        l1 = l2 = (0, 0)
        for name, fired in self.cmp_fired.items():
            asserted = self.cmp_asserted.get(name, 0)
            if name.startswith("trace_"):
                l2 = (l2[0] + fired, l2[1] + asserted)
            else:
                l1 = (l1[0] + fired, l1[1] + asserted)
        return l1, l2

    def fire_trend(self, name: str) -> str:
        """``"rising"``, ``"falling"`` or ``"flat"`` for one callback.

        ``"unknown"`` until both EWMAs exist. Compares the fast rate against
        the slow one rather than against an absolute figure: what the fire
        count is worth knowing about is its direction, and the absolute
        value varies by orders of magnitude between callbacks on the same
        target.
        """
        fast = self.cmp_rate_fast.get(name)
        slow = self.cmp_rate_slow.get(name)
        if fast is None or slow is None:
            return "unknown"
        if slow <= 0:
            return "rising" if fast > 0 else "flat"
        ratio = fast / slow
        if ratio > 1.0 + CMP_RATE_TREND_MARGIN:
            return "rising"
        if ratio < 1.0 - CMP_RATE_TREND_MARGIN:
            return "falling"
        return "flat"

    def comparison_walls(
        self,
        min_fired: int = CMP_WALL_MIN_FIRED,
        max_assert_rate: float = CMP_WALL_MAX_ASSERT_RATE,
    ) -> dict[str, tuple[int, int, str]]:
        """Callbacks the campaign reaches constantly and never passes.

        Args:
            min_fired: Evidence floor. A callback entered twice and
                satisfied never is an accident, not a wall.
            max_assert_rate: Above this the comparison is being passed often
                enough that whatever is failing is a minority of its call
                sites -- which callback granularity cannot separate anyway.

        Returns:
            ``{callback: (fired, asserted, trend)}`` for every wall, ordered
            by fire count descending.

        This is the question edge coverage cannot answer. A wall does not
        show up as a plateau, because the campaign is not stuck -- it is
        reaching the comparison over and over and failing it, which looks
        identical from the map to not reaching it at all.

        The honest limit is granularity: one bucket per callback, not per
        call site, so this says *which family* is the wall and never which
        comparison. Two memcmp sites are one bucket, and a target with one
        satisfied memcmp per execution and a million unsatisfied ones reads
        as an assert rate of 10^-6 rather than as two separate facts.
        """
        walls = {}
        for name, fired in self.cmp_fired.items():
            if fired < min_fired:
                continue
            asserted = self.cmp_asserted.get(name, 0)
            if asserted > fired * max_assert_rate:
                continue
            walls[name] = (fired, asserted, self.fire_trend(name))
        return dict(sorted(walls.items(), key=lambda kv: -kv[1][0]))

    def wall_summary(self) -> str:
        """One line naming the walls, or empty when there are none.

        Written for the stall reason string, where a stall with a rising
        fire count and a stall with a falling one are different animals and
        currently read identically.
        """
        walls = self.comparison_walls()
        if not walls:
            return ""
        parts = [
            f"{name} {fired:,}x {trend}" if trend != "unknown" else f"{name} {fired:,}x"
            for name, (fired, _asserted, trend) in list(walls.items())[:3]
        ]
        return "walls: " + ", ".join(parts)

    def detect_hash_candidates(self, pairs: list[tuple[bytes, bytes]]) -> int:
        """Identify pairs that look like checksum/CRC comparisons.

        Hash-like comparisons have long operands that share very few byte
        positions — they can't be cracked by I2S substitution and would
        waste execution time if fed to the encoding engine.

        Criteria (from Redqueen's ``cmp.py::could_be_hash()``):
        - Both operands >= ``_HASH_MIN_BYTES`` bytes.
        - Operands share <= ``_HASH_MAX_MATCH_BYTES`` byte positions
          (i.e. the values are fundamentally different, not an encoding
          transform of each other).

        Args:
            pairs: Newly collected operand pairs to screen.

        Returns:
            Number of pairs flagged as hash-like.
        """
        n = 0
        for op_a, op_b in pairs:
            if len(op_a) < _HASH_MIN_BYTES or len(op_b) < _HASH_MIN_BYTES:
                continue
            if len(op_a) != len(op_b):
                continue
            # Count matching byte positions
            matches = sum(1 for a, b in zip(op_a, op_b, strict=False) if a == b)
            if matches <= _HASH_MAX_MATCH_BYTES:
                self.hash_candidates.add((op_a, op_b))
                n += 1
        return n

    def is_hash_candidate(self, op_a: bytes, op_b: bytes) -> bool:
        """Check if a pair has been flagged as hash-like."""
        return (op_a, op_b) in self.hash_candidates

    def high_confidence_pairs(self, min_occurrences: int = 2) -> list[tuple[bytes, bytes]]:
        """Return pairs observed in at least *min_occurrences* runs.

        High-confidence pairs are more likely to be genuine I2S candidates
        rather than one-off noise from edge-case execution paths.
        """
        return [p for p, count in self._pair_occurrence.items() if count >= min_occurrences]

    def pair_confidence(self, op_a: bytes, op_b: bytes) -> int:
        """Return how many times a pair has been observed."""
        return self._pair_occurrence.get((op_a, op_b), 0)

    def pair_cmp(self, op_a: bytes, op_b: bytes) -> tuple[int, int] | None:
        """Observed ``(result, width)`` for a pair, or None if unrecorded.

        ``result`` is the shim's comparison outcome: -1 for a<b, 0 for a==b,
        1 for a>b. Negating it is what turns a recorded comparison into a
        solvable constraint for reaching the opposite branch.
        """
        return self._pair_cmp.get((op_a, op_b))

    def branch_records(self) -> list[tuple[bytes, bytes, int, int, int | None]]:
        """All pairs with a recorded outcome, as (a, b, result, width, pc)."""
        out = []
        for pair, (result, width) in self._pair_cmp.items():
            out.append((pair[0], pair[1], result, width, self._pair_pc.get(pair)))
        return out

    def pair_pc(self, op_a: bytes, op_b: bytes) -> int | None:
        """Return the program counter for a pair, if known (trace mode)."""
        return self._pair_pc.get((op_a, op_b))

    def mark_coverage_gain(self) -> None:
        """Bump value signal for all currently tracked tokens and pairs.

        Called by the fuzzer when a coverage gain is detected during a
        fuzz iteration where cmplog data was used.  Tokens/pairs that
        are frequently present during gains are preferentially retained
        during eviction.
        """
        for t in self._token_set:
            self._token_value[t] = self._token_value.get(t, 0) + 1
        for p in self._pair_set:
            self._pair_value[p] = self._pair_value.get(p, 0) + 1

    def get_tokens(self) -> list[bytes]:
        """Get all collected tokens."""
        return self.tokens

    def stop(self):
        """Release run-scoped resources: the log file and the env mutations.

        The shim itself is cached in tempdir for reuse and is not removed.
        """
        self.restore_env()
        if self.counts_path and os.path.exists(self.counts_path):
            with contextlib.suppress(OSError):
                os.unlink(self.counts_path)
        self.counts_path = None
        if self.log_path and os.path.exists(self.log_path):
            with contextlib.suppress(OSError):
                os.unlink(self.log_path)
            self.log_path = None
