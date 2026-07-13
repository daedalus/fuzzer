"""Cmplog collector: parse comparison tracing output and feed into dictionary.

Intercepts memcmp/strcmp/strncmp/memchr via LD_PRELOAD and collects
operand pairs that differ. These operands are fed into the dictionary
and mutation engine, enabling the fuzzer to discover magic bytes and
protocol constants that blind mutation cannot find.
"""

import contextlib
import logging
import os
import tempfile

log = logging.getLogger(__name__)


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

    def start(self) -> bool:
        """Compile and prepare the cmplog shim."""
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
            return True

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
                return True
            log.warning("Cmplog shim compilation failed: %s", result.stderr.decode()[:200])
        except Exception as e:
            log.warning("Cmplog shim compilation error: %s", e)
        return False

    def setup_env(self, env: dict[str, str]) -> dict[str, str]:
        """Add cmplog env vars to the execution environment.

        Args:
            env: Current environment dict.

        Returns:
            Modified env with LD_PRELOAD and _CMPLOG_OUT set.
        """
        if not self._shim_path:
            return env

        fd, self.log_path = tempfile.mkstemp(suffix=".cmplog", prefix="fuzz_cmplog_")
        os.close(fd)
        env = dict(env)  # copy
        env["_CMPLOG_OUT"] = self.log_path

        # Prepend to LD_PRELOAD
        existing = env.get("LD_PRELOAD", "")
        if existing:
            env["LD_PRELOAD"] = f"{self._shim_path}:{existing}"
        else:
            env["LD_PRELOAD"] = self._shim_path

        return env

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

        # Clear the log for next round
        with contextlib.suppress(OSError):
            os.unlink(self.log_path)

        new_tokens = [t for t in tokens if t not in self._token_set]
        self._token_set.update(tokens)
        self.tokens.extend(new_tokens)
        self.pairs.extend(new_pairs)

        if new_tokens:
            log.info(
                "Cmplog: found %d new tokens, %d new pairs (total: %d tokens, %d pairs)",
                len(new_tokens), len(new_pairs), len(self.tokens), len(self.pairs),
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
