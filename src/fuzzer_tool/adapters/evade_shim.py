"""Build and inject the AntiFuzz-evasion LD_PRELOAD shim (``antifuzz_evade.c``).

Thin driver over the C source next to this module: it compiles the preload
once per process and hands callers the ``LD_PRELOAD`` value to prepend for a
target hardened with AntiFuzz's self-ptrace and delay-on-malformed tricks
(USENIX Sec '19, §4.2/§4.3). The C file is the mechanism; this layer only
compiles it and composes the env var, so callers work in terms of "evade" not
compiler flags.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import tempfile

log = logging.getLogger(__name__)

_SRC = os.path.join(os.path.dirname(__file__), "antifuzz_evade.c")

# Built once per process; the .so is pure code with no per-target state.
_cached_so: str | None = None


def _compile(cc: str = "clang") -> str | None:
    fd, so = tempfile.mkstemp(suffix=".so", prefix=f"antifuzz_evade_{os.getpid()}_")
    os.close(fd)

    env = os.environ.copy()
    # The compiler must not run under the target's sanitizer runtime.
    env.pop("ASAN_OPTIONS", None)
    env.pop("LSAN_OPTIONS", None)
    env.pop("LD_PRELOAD", None)

    cmd = [cc, "-shared", "-fPIC", "-O2", "-o", so, _SRC, "-ldl"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30, env=env)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log.warning("antifuzz_evade: compile failed (%s)", exc)
        with contextlib.suppress(OSError):
            os.unlink(so)
        return None

    if result.returncode != 0:
        log.warning("antifuzz_evade: compile failed: %s", result.stderr.decode()[:400])
        with contextlib.suppress(OSError):
            os.unlink(so)
        return None

    return so


def build_evade_shim(cc: str = "clang") -> str | None:
    """Path to the compiled preload ``.so``, or None when it cannot build."""
    global _cached_so
    if _cached_so and os.path.exists(_cached_so):
        return _cached_so

    _cached_so = _compile(cc)
    return _cached_so


def evade_ld_preload(existing: str | None = None, cc: str = "clang") -> str | None:
    """``LD_PRELOAD`` value that prepends the evasion shim to *existing*.

    Returns None (caller leaves LD_PRELOAD untouched) when the shim cannot
    be built, so evasion is best-effort and never breaks a run.
    """
    so = build_evade_shim(cc)
    if so is None:
        return None

    if not existing:
        return so

    return f"{so}:{existing}" if so not in existing.split(":") else existing
