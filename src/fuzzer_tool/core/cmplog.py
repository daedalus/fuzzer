"""Cmplog collector: parse comparison tracing output and feed into dictionary.

A single unified shim (cmplog_shim.c) provides two complementary
interception layers, both writing to the same CMP log file:

1. Symbol-based: intercepts libc comparison functions (memcmp/strcmp/strncmp/
   memchr/strcasecmp/strncasecmp/memmem/strstr/strcasestr) via LD_PRELOAD or
   build-time linking.

2. Compiler-IR-based: implements Clang's -fsanitize-coverage=trace-cmp
   callbacks (__sanitizer_cov_trace_cmp*, __sanitizer_cov_trace_switch) that
   fire after the compiler has inlined/folded comparisons into integer compares.

Both layers are compiled into a single .so — no need for separate shims.
"""

import contextlib
import logging
import os
import tempfile

log = logging.getLogger(__name__)

# ── Memory bounds ────────────────────────────────────────────────────
CMPLOG_TOKENS_MAX = 10_000  # max unique operand tokens
CMPLOG_PAIRS_MAX = 5_000  # max unique operand pairs

# Hash detection thresholds
_HASH_MIN_BYTES = 8  # minimum operand length to consider as hash-like
_HASH_MAX_MATCH_BYTES = 2  # max matching byte positions for a hash-like pair


class CmplogCollector:
    """Collect and process comparison tracing data from the cmplog shim.

    After each execution, reads the CMP log file, extracts operand pairs,
    and converts them into dictionary tokens for mutation.
    """

    def __init__(self):
        self.log_path: str | None = None
        self.tokens: list[bytes] = []
        self._token_set: set[bytes] = set()
        # Operand pairs: (operand_a, operand_b) for input-to-state matching
        self.pairs: list[tuple[bytes, bytes]] = []
        self._pair_set: set[tuple[bytes, bytes]] = set()
        self._shim_path: str | None = None
        self._shim_handle = None
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

    def start(self) -> bool:
        """Compile and prepare the unified cmplog shim."""
        from fuzzer_tool.adapters.shim_factory import _find_compiler

        shim_src = os.path.join(os.path.dirname(__file__), "..", "adapters", "cmplog_shim.c")
        if not os.path.exists(shim_src):
            log.warning("cmplog_shim.c not found at %s", shim_src)
            return False

        # Cache shim in tempdir — don't recompile if already exists
        out_path = os.path.join(tempfile.gettempdir(), "fuzz_cmplog_shim.so")
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
                    [compiler, "-shared", "-fPIC", "-O2", "-ldl", "-o", out_path, shim_src],
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

        return self._shim_path is not None

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
            fd, self.log_path = tempfile.mkstemp(suffix=".cmplog", prefix="fuzz_cmplog_")
            os.close(fd)
        else:
            # Truncate so the child writes fresh data from position 0
            with contextlib.suppress(OSError):
                with open(self.log_path, "w") as f:
                    f.truncate(0)
        env = dict(env)  # copy
        env["_CMPLOG_OUT"] = self.log_path

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
        """
        if self.log_path is None or not os.path.exists(self.log_path):
            fd, self.log_path = tempfile.mkstemp(suffix=".cmplog", prefix="fuzz_cmplog_")
            os.close(fd)
        os.environ["_CMPLOG_OUT"] = self.log_path

        if self._shim_path and self._shim_path not in os.environ.get("LD_PRELOAD", ""):
            existing = os.environ.get("LD_PRELOAD", "")
            os.environ["LD_PRELOAD"] = (
                f"{self._shim_path}:{existing}" if existing else self._shim_path
            )

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
        """
        if not self.log_path:
            return
        try:
            with open(self.log_path, "w") as f:
                f.truncate(0)
        except OSError:
            pass

    def collect_tokens(self) -> list[bytes]:
        """Read the cmplog file and extract operand tokens and pairs.

        Returns:
            List of unique byte sequences found in comparison operands.
            Also populates self.pairs with (operand_a, operand_b) tuples
            for input-to-state redqueen matching.
        """
        if not self.log_path or not os.path.exists(self.log_path):
            return []

        tokens = set()
        new_pairs = []
        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("CMP "):
                        continue
                    parts = line[4:].split()
                    if len(parts) < 2:
                        continue
                    hex_a, hex_b = parts[0], parts[1]
                    try:
                        operand_a = bytes.fromhex(hex_a)
                        operand_b = bytes.fromhex(hex_b)
                        tokens.add(operand_a)
                        tokens.add(operand_b)
                        # Track pairs for input-to-state matching
                        pair = (operand_a, operand_b)
                        if pair not in self._pair_set:
                            self._pair_set.add(pair)
                            new_pairs.append(pair)
                    except ValueError:
                        continue
        except OSError as e:
            log.debug("Failed to read cmplog file: %s", e)

        # Clear the log for next round.
        # Truncate (not delete) so the .so's file handle stays valid
        # when cmplog is compiled into the target (direct_lite mode).
        with contextlib.suppress(OSError):
            with open(self.log_path, "w") as f:
                f.truncate(0)

        new_tokens = [t for t in tokens if t not in self._token_set]
        self._token_set.update(tokens)
        self.tokens.extend(new_tokens)
        self.pairs.extend(new_pairs)

        # Cap token/pair lists to bound memory.
        # Preserves highest-value-density entries instead of simple recency.
        if len(self.tokens) > CMPLOG_TOKENS_MAX:
            excess = len(self.tokens) - CMPLOG_TOKENS_MAX
            scored = [
                (self._token_value.get(t, 0) / max(len(t), 1), t)
                for t in self._token_set
            ]
            scored.sort(key=lambda x: x[0])  # lowest value-density first
            for _, t in scored[:excess]:
                self._token_set.discard(t)
                self._token_value.pop(t, None)
            self.tokens = list(self._token_set)
        if len(self.pairs) > CMPLOG_PAIRS_MAX:
            excess = len(self.pairs) - CMPLOG_PAIRS_MAX
            scored = [
                (self._pair_value.get(p, 0) / max(len(p[0]) + len(p[1]), 1), p)
                for p in self._pair_set
            ]
            scored.sort(key=lambda x: x[0])  # lowest value-density first
            for _, p in scored[:excess]:
                self._pair_set.discard(p)
                self._pair_value.pop(p, None)
            self.pairs = list(self._pair_set)

            # Track pair occurrence across runs for multi-run confidence.
            # Pairs seen in many runs are reliable I2S signals; rarely-seen
            # pairs may be noise from edge-case execution paths.
            for pair in new_pairs:
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
        return [
            p for p, count in self._pair_occurrence.items()
            if count >= min_occurrences
        ]

    def pair_confidence(self, op_a: bytes, op_b: bytes) -> int:
        """Return how many times a pair has been observed."""
        return self._pair_occurrence.get((op_a, op_b), 0)

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
        """Clean up log file only (shim is cached in tempdir for reuse)."""
        if self.log_path and os.path.exists(self.log_path):
            with contextlib.suppress(OSError):
                os.unlink(self.log_path)
            self.log_path = None
