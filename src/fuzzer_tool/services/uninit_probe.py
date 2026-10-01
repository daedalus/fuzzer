"""Uninitialized-memory probe: does the output depend on the malloc fill?

lcamtuf's browser image bugs (IJG jpeg, GIF/BMP/TIFF decoders) leaked heap
bytes the decoder never wrote. Same input, different fill byte for fresh
allocations, different output: that is the leak, found without MSAN::

    run 1  fill 0xAA ─┐
    run 2  fill 0xAA ─┴─ differ?  NOISY (target is nondeterministic)
    run 3  fill 0x55 ──── differs from run 1?  LEAK  else  STABLE

The fill reaches both allocators: ASAN (``malloc_fill_byte``) and glibc
(``MALLOC_PERTURB_``, which fills with ``value ^ 0xff``). Stack memory is
not covered; that still needs MSAN.
"""

import logging
import os
import tempfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.adapters.process import run_target_digest

log = logging.getLogger(__name__)

FILL_A = 0xAA
FILL_B = 0x55

# ASAN fills only the first max_malloc_fill_size bytes (default 4 KiB).
_ASAN_FILL_MAX = 1 << 28

TIMEOUT_RC = -1

# Distinct leaking inputs saved per run; later ones are counted, not written.
MAX_SAVED = 256

SAVE_PREFIX = "uninit_"

# In-process libraries cannot be exec'd.
_SO_SUFFIXES = (".so", ".dylib", ".dll")

Run = Callable[[bytes, dict[str, str]], tuple[int, bytes]]


class Verdict(Enum):
    STABLE = "stable"
    LEAK = "leak"
    NOISY = "noisy"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


def fill_env(base: dict[str, str], fill: int) -> dict[str, str]:
    """Copy of *base* that makes fresh heap memory read as *fill*."""
    env = dict(base)
    asan = f"malloc_fill_byte={fill}:max_malloc_fill_size={_ASAN_FILL_MAX}"
    prior = env.get("ASAN_OPTIONS")
    env["ASAN_OPTIONS"] = f"{prior}:{asan}" if prior else asan
    env["MALLOC_PERTURB_"] = str(fill ^ 0xFF)
    return env


def probe(run: Run, data: bytes, base_env: dict[str, str] | None = None) -> Verdict:
    """Classify *data* with three runs: control pair, then the other fill."""
    base = {} if base_env is None else base_env
    env_a = fill_env(base, FILL_A)

    a1 = run(data, env_a)
    a2 = run(data, env_a)
    if TIMEOUT_RC in (a1[0], a2[0]):
        return Verdict.TIMEOUT
    if a1 != a2:
        return Verdict.NOISY

    b = run(data, fill_env(base, FILL_B))
    if b[0] == TIMEOUT_RC:
        return Verdict.TIMEOUT
    return Verdict.LEAK if b != a1 else Verdict.STABLE


class UninitProbe:
    """Probe inputs against one executable; save each leaking input once.

    Args:
        target: Executable path.
        timeout: Per-run timeout in seconds.
        out_dir: Where leaking inputs are written (``uninit_<hash>``).
        target_args: File-mode argv; ``{file}`` becomes the input path.
            None feeds the input on stdin only.
    """

    def __init__(
        self,
        target: str,
        timeout: float,
        out_dir: str | Path,
        target_args: list[str] | None = None,
    ):
        self._target = target
        self._timeout = timeout
        self._out_dir = Path(out_dir)
        self._target_args = target_args
        self._saved: set[str] = set()
        self.leaks = 0
        self.noisy = 0

    def check(self, data: bytes) -> Verdict:
        """Probe *data*; on LEAK, save it."""
        if self._target.lower().endswith(_SO_SUFFIXES):
            return Verdict.SKIPPED

        verdict = self._probe_with_file(data)
        if verdict is Verdict.NOISY:
            self.noisy += 1
        if verdict is Verdict.LEAK:
            self._save(data)
        return verdict

    def _probe_with_file(self, data: bytes) -> Verdict:
        # File-mode targets read argv; stdin carries the input too, the
        # same way fuzz_loader feeds both.
        if self._target_args is None:
            return probe(self._runner([self._target]), data, dict(os.environ))

        fd, path = tempfile.mkstemp(prefix="uninit_probe_")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            argv = [a.replace("{file}", path) for a in self._target_args]
            return probe(self._runner([self._target, *argv]), data, dict(os.environ))
        finally:
            os.unlink(path)

    def _runner(self, cmd: list[str]) -> Run:
        return lambda data, env: run_target_digest(cmd, data, self._timeout, env)

    def _save(self, data: bytes) -> None:
        key = hash_data(data)
        if key in self._saved:
            return
        self.leaks += 1
        if len(self._saved) >= MAX_SAVED:
            return
        self._saved.add(key)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        (self._out_dir / f"{SAVE_PREFIX}{key}").write_bytes(data)
        log.warning("uninit probe: output depends on heap fill; saved %s%s", SAVE_PREFIX, key)
