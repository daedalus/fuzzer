"""Cmplog collector: parse comparison tracing output and feed into dictionary.

Two complementary interception layers, both writing to the same CMP log file:

1. Symbol-based (cmplog_shim.c): intercepts libc comparison functions
   (memcmp/strcmp/strncmp/memchr/strcasecmp/strncasecmp/memmem/strstr/
   strcasestr) via LD_PRELOAD or build-time linking.

2. Compiler-IR-based (tracecmp_shim.c): intercepts Clang's
   -fsanitize-coverage=trace-cmp callbacks (__sanitizer_cov_trace_cmp*,
   __sanitizer_cov_trace_switch) that fire after the compiler has inlined/
   folded comparisons into integer compares.

Both shims coexist — they export different symbols, write to the same
_CMPLOG_OUT file, and the collector parses all CMP lines transparently.
"""

import contextlib
import logging
import os
import tempfile

log = logging.getLogger(__name__)

# ── Memory bounds ────────────────────────────────────────────────────
CMPLOG_TOKENS_MAX = 10_000  # max unique operand tokens
CMPLOG_PAIRS_MAX = 5_000  # max unique operand pairs


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
        self._tracecmp_shim_path: str | None = None
        self._shim_handle = None

    def start(self) -> bool:
        """Compile and prepare the cmplog shim and trace-cmp shim."""
        from fuzzer_tool.adapters.shim_factory import _find_compiler, build_tracecmp_shim

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
                result = __import__("subprocess").run(
                    [compiler, "-shared", "-fPIC", "-O2", "-ldl", "-o", out_path, shim_src],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode == 0 and os.path.exists(out_path):
                    self._shim_path = out_path
                    log.info("Cmplog shim compiled: %s", out_path)
                else:
                    log.warning("Cmplog shim compilation failed: %s", result.stderr.decode()[:200])
            except Exception as e:
                log.warning("Cmplog shim compilation error: %s", e)

        # Also build the trace-cmp shim (compiler-IR-based comparison tracing)
        try:
            self._tracecmp_shim_path = build_tracecmp_shim()
            if self._tracecmp_shim_path:
                log.info("Trace-cmp shim compiled: %s", self._tracecmp_shim_path)
        except Exception as e:
            log.debug("Trace-cmp shim build skipped: %s", e)

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
        if not self._shim_path and not self._tracecmp_shim_path:
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

        # Prepend both shims to LD_PRELOAD
        shims = []
        if self._shim_path:
            shims.append(self._shim_path)
        if self._tracecmp_shim_path:
            shims.append(self._tracecmp_shim_path)
        if shims:
            existing = env.get("LD_PRELOAD", "")
            combined = ":".join(shims)
            env["LD_PRELOAD"] = f"{combined}:{existing}" if existing else combined

        return env

    def setup_env_for_run(self):
        """Set _CMPLOG_OUT in the current process environment.

        Used by inprocess and persistent execution paths where the target
        runs inside the fuzzer process (or a long-lived child) and inherits
        os.environ rather than a per-call env dict.

        Reuses the current log_path if one exists; creates a new one on first call.
        The cmplog shim (whether LD_PRELOAD'd or compiled into the target .so)
        reads _CMPLOG_OUT at constructor time.

        Both shims (symbol-based + trace-cmp) are prepended to LD_PRELOAD
        when available.
        """
        if self.log_path is None or not os.path.exists(self.log_path):
            fd, self.log_path = tempfile.mkstemp(suffix=".cmplog", prefix="fuzz_cmplog_")
            os.close(fd)
        os.environ["_CMPLOG_OUT"] = self.log_path

        # Collect all shim paths to prepend to LD_PRELOAD
        shims = []
        if self._shim_path and self._shim_path not in os.environ.get("LD_PRELOAD", ""):
            shims.append(self._shim_path)
        if self._tracecmp_shim_path and self._tracecmp_shim_path not in os.environ.get(
            "LD_PRELOAD", ""
        ):
            shims.append(self._tracecmp_shim_path)

        if shims:
            existing = os.environ.get("LD_PRELOAD", "")
            combined = ":".join(shims)
            os.environ["LD_PRELOAD"] = f"{combined}:{existing}" if existing else combined

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

        # Cap token/pair lists to bound memory
        if len(self.tokens) > CMPLOG_TOKENS_MAX:
            half = CMPLOG_TOKENS_MAX // 2
            self.tokens = self.tokens[-half:]
            self._token_set = set(self.tokens)
        if len(self.pairs) > CMPLOG_PAIRS_MAX:
            half = CMPLOG_PAIRS_MAX // 2
            self.pairs = self.pairs[-half:]
            self._pair_set = set(self.pairs)

        if new_tokens:
            log.info(
                "Cmplog: found %d new tokens, %d new pairs (total: %d tokens, %d pairs)",
                len(new_tokens),
                len(new_pairs),
                len(self.tokens),
                len(self.pairs),
            )

        return new_tokens

    def get_tokens(self) -> list[bytes]:
        """Get all collected tokens."""
        return self.tokens

    def stop(self):
        """Clean up log file only (shim is cached in tempdir for reuse)."""
        if self.log_path and os.path.exists(self.log_path):
            with contextlib.suppress(OSError):
                os.unlink(self.log_path)
            self.log_path = None
