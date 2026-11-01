"""libFuzzer-style custom mutator / crossover plugins (``--custom-mutator``).

A plugin is a shared object exporting either or both of::

    size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                   size_t MaxSize, unsigned int Seed);

    size_t LLVMFuzzerCustomCrossOver(const uint8_t *Data1, size_t Size1,
                                     const uint8_t *Data2, size_t Size2,
                                     uint8_t *Out, size_t MaxOutSize,
                                     unsigned int Seed);

so a harness written for libFuzzer / OSS-Fuzz (or for atheris, whose
``Setup(custom_mutator=..., custom_crossover=...)`` mirrors these) can be
reused without change.  Each exported symbol becomes one operator
(``custom_mutator``, ``custom_crossover``) in ``REGISTRY``, so the operator
schedulers weigh it against the built-in mutators like any other.

Two callbacks, two directions
-----------------------------

* fuzzer -> plugin: the two functions above.  The wrapper copies the seed
  into a ``MaxSize`` buffer, calls the plugin, and returns the first
  ``ret`` bytes as a new ``bytes``.  A return of 0 or above ``MaxSize`` is
  a decline (``None``).
* plugin -> fuzzer: ``size_t LLVMFuzzerMutate(uint8_t *Data, size_t Size,
  size_t MaxSize)``.  A typical plugin parses the input, mutates the
  parsed structure, and calls this to apply ordinary byte-level havoc to
  one field.  The plugin leaves the symbol undefined and expects the host
  to provide it, but this host is Python, so a few lines of C (built with
  the system ``cc`` on first use, loaded ``RTLD_GLOBAL`` before the
  plugin) export the symbol and forward to a ``ctypes`` callback that runs
  :func:`havoc`.

The havoc behind ``LLVMFuzzerMutate`` is self-contained (bit/byte flips,
interesting values, arithmetic, block delete/duplicate/insert/copy/swap),
not a reach into ``OperatorEngine``: ``core/mutator_interface.py`` exists to
stop mutators depending on the whole fuzzer, and a callback invoked from
foreign C has no engine to hand anyway.  It draws from the fuzzer's own rng,
so runs stay reproducible under ``--seed``.

Limits
------

* Not crash-isolated.  The plugin runs inside the fuzzer process; a
  segfault in it ends the fuzzer.  Test a plugin standalone first.
* One plugin per process: the ``LLVMFuzzerMutate`` forwarder is a single
  global symbol.  A second ``register_custom_mutator`` call rebinds the
  existing operators to the new plugin.
* Needs a C compiler (``$CC`` or ``cc``) the first time.
* AFL++'s ``afl_custom_*`` API is not implemented.
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import logging
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase

log = logging.getLogger(__name__)

#: Buffer capacity handed to the plugin when the fuzzer has no length cap
#: (``max_len == 0``).  libFuzzer's own default ``-max_len`` is 4096; the
#: capacity grows with the input so a long seed is never truncated first.
DEFAULT_CAP = 4096

MUTATOR_SYMBOL = "LLVMFuzzerCustomMutator"
CROSSOVER_SYMBOL = "LLVMFuzzerCustomCrossOver"

_HOST_SRC = r"""
#include <stddef.h>
#include <stdint.h>
typedef size_t (*fz_mutate_cb)(uint8_t *, size_t, size_t);
static fz_mutate_cb g_cb;
void fz_set_mutate_cb(fz_mutate_cb cb) { g_cb = cb; }
size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t max_size) {
    return g_cb ? g_cb(data, size, max_size) : size;
}
"""

# ---------------------------------------------------------------------------
# Byte-level havoc behind LLVMFuzzerMutate
# ---------------------------------------------------------------------------

_INTERESTING_8 = (0, 1, 16, 32, 64, 100, 127, 128, 255)
_INTERESTING_16 = (0, 128, 255, 256, 512, 1000, 1024, 4096, 32767, 32768, 65535)
_INTERESTING_32 = (0, 32768, 65535, 65536, 100663045, 2147483647, 2147483648, 4294967295)
_MAX_BLOCK = 64
_N_OPS = 10


def _write_int(buf: bytearray, pos: int, value: int, width: int, rng) -> None:
    order = "little" if rng.randint(0, 1) else "big"
    buf[pos : pos + width] = (value & ((1 << (8 * width)) - 1)).to_bytes(width, order)


def _havoc_once(buf: bytearray, max_size: int, rng) -> None:
    n = len(buf)
    op = rng.randrange(_N_OPS)
    if n == 0:
        if max_size > 0:
            buf.extend(rng.randbytes(rng.randint(1, min(max_size, 8))))
        return
    if op == 0:  # bit flip
        buf[rng.randrange(n)] ^= 1 << rng.randrange(8)
    elif op == 1:  # random byte
        buf[rng.randrange(n)] = rng.randrange(256)
    elif op == 2:  # interesting byte
        buf[rng.randrange(n)] = rng.choice(_INTERESTING_8)
    elif op == 3:  # byte arithmetic
        pos = rng.randrange(n)
        delta = rng.randint(1, 35)
        buf[pos] = (buf[pos] + (delta if rng.randint(0, 1) else -delta)) & 0xFF
    elif op == 4:  # interesting 16/32-bit value
        width = 2 if n < 4 or rng.randint(0, 1) else 4
        if n >= width:
            vals = _INTERESTING_16 if width == 2 else _INTERESTING_32
            _write_int(buf, rng.randrange(n - width + 1), rng.choice(vals), width, rng)
    elif op == 5:  # delete a block
        if n >= 2:
            ln = rng.randint(1, min(_MAX_BLOCK, n - 1))
            start = rng.randrange(n - ln + 1)
            del buf[start : start + ln]
    elif op == 6:  # duplicate a block
        ln = rng.randint(1, min(_MAX_BLOCK, n, max_size - n)) if max_size > n else 0
        if ln:
            start = rng.randrange(n - ln + 1)
            at = rng.randrange(n + 1)
            buf[at:at] = buf[start : start + ln]
    elif op == 7:  # insert random bytes
        if max_size > n:
            at = rng.randrange(n + 1)
            buf[at:at] = rng.randbytes(rng.randint(1, min(_MAX_BLOCK, max_size - n)))
    elif op == 8:  # overwrite a block with another block of the buffer
        if n >= 2:
            ln = rng.randint(1, min(_MAX_BLOCK, n - 1))
            src = rng.randrange(n - ln + 1)
            dst = rng.randrange(n - ln + 1)
            buf[dst : dst + ln] = bytes(buf[src : src + ln])
    else:  # swap two bytes
        i, j = rng.randrange(n), rng.randrange(n)
        buf[i], buf[j] = buf[j], buf[i]


def havoc(data: bytes, max_size: int, rng=None) -> bytes:
    """One libFuzzer-``Mutate``-style byte mutation of *data*, at most *max_size* long.

    Retries a few times so a no-op draw (e.g. a bit flip undone by nothing,
    a swap of a byte with itself) does not come back unchanged; when the
    input is already longer than *max_size* it is truncated first, as
    ``LLVMFuzzerMutate`` requires ``Size <= MaxSize``.
    """
    rng = rng if rng is not None else random.Random()
    buf = bytearray(data[:max_size] if max_size >= 0 else data)
    original = bytes(buf)
    for _ in range(8):
        _havoc_once(buf, max_size, rng)
        if len(buf) > max_size:
            del buf[max_size:]
        if bytes(buf) != original:
            break
    return bytes(buf)


# ---------------------------------------------------------------------------
# Host shim: exports LLVMFuzzerMutate and forwards it into Python
# ---------------------------------------------------------------------------

_MUTATE_CB = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)

#: rng for the plugin call in flight; read by the LLVMFuzzerMutate callback.
_current_rng: Any = None
_fallback_rng = random.Random()


def _on_mutate(ptr, size, max_size) -> int:
    """Body of ``LLVMFuzzerMutate``: mutate ``Data[:Size]`` in place, return the new size."""
    if not ptr:
        return 0
    try:
        out = havoc(ctypes.string_at(ptr, size), max_size, _current_rng or _fallback_rng)
        ctypes.memmove(ptr, out, len(out))
        return len(out)
    except Exception:  # noqa: BLE001 - must not unwind into foreign C
        log.warning("LLVMFuzzerMutate callback failed", exc_info=True)
        return size


_host: ctypes.CDLL | None = None
_host_cb: Any = None  # keeps the CFUNCTYPE object alive for the process
_host_dir: str | None = None


def _cleanup_host_dir() -> None:
    if _host_dir:
        shutil.rmtree(_host_dir, ignore_errors=True)


def _load_host() -> ctypes.CDLL:
    """Build (once) and load the ``LLVMFuzzerMutate`` forwarder, ``RTLD_GLOBAL``."""
    global _host, _host_cb, _host_dir
    if _host is not None:
        return _host
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("--custom-mutator needs a C compiler (set $CC or install cc/gcc/clang)")
    tmp = tempfile.mkdtemp(prefix="fz_cm_host_")
    src = Path(tmp) / "host.c"
    so = Path(tmp) / "libfzhost.so"
    src.write_text(_HOST_SRC)
    proc = subprocess.run(
        [cc, "-shared", "-fPIC", "-O2", "-o", str(so), str(src)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"building the LLVMFuzzerMutate host shim failed: {proc.stderr.strip()}")
    _host_dir = tmp
    atexit.register(_cleanup_host_dir)
    host = ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
    host.fz_set_mutate_cb.argtypes = [ctypes.c_void_p]
    host.fz_set_mutate_cb.restype = None
    _host_cb = _MUTATE_CB(_on_mutate)
    host.fz_set_mutate_cb(ctypes.cast(_host_cb, ctypes.c_void_p))
    _host = host
    return host


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class CustomMutatorPlugin:
    """A loaded plugin ``.so`` with whichever of the two entry points it exports."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = str(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"--custom-mutator: no such file: {self.path}")
        _load_host()  # first: the plugin's undefined LLVMFuzzerMutate must resolve at dlopen
        try:
            self._lib = ctypes.CDLL(os.path.abspath(self.path))
        except OSError as exc:
            raise RuntimeError(f"--custom-mutator: cannot load {self.path}: {exc}") from exc

        self._mutate = self._bind(
            MUTATOR_SYMBOL,
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint],
        )
        self._cross = self._bind(
            CROSSOVER_SYMBOL,
            [
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint,
            ],
        )
        if self._mutate is None and self._cross is None:
            raise RuntimeError(
                f"--custom-mutator: {self.path} exports neither {MUTATOR_SYMBOL} nor {CROSSOVER_SYMBOL}"
            )

    def _bind(self, symbol: str, argtypes: list) -> Any:
        try:
            fn = getattr(self._lib, symbol)
        except AttributeError:
            return None
        fn.argtypes = argtypes
        fn.restype = ctypes.c_size_t
        return fn

    @property
    def has_mutator(self) -> bool:
        return self._mutate is not None

    @property
    def has_crossover(self) -> bool:
        return self._cross is not None

    @staticmethod
    def _capacity(max_len: int, *lengths: int) -> int:
        return max_len or max(DEFAULT_CAP, 2 * max(lengths))

    def mutate(self, data: bytes, max_len: int, rng) -> bytes | None:
        """Call ``LLVMFuzzerCustomMutator``; ``None`` for a size of 0 or above ``MaxSize``."""
        global _current_rng
        if self._mutate is None:
            return None
        cap = self._capacity(max_len, len(data))
        data = data[:cap]
        buf = (ctypes.c_uint8 * cap)()
        ctypes.memmove(buf, data, len(data))
        _current_rng = rng
        try:
            n = self._mutate(buf, len(data), cap, rng.randint(0, 0xFFFFFFFF))
        finally:
            _current_rng = None
        if n == 0 or n > cap:
            return None
        return ctypes.string_at(buf, n)

    def crossover(self, a: bytes, b: bytes, max_len: int, rng) -> bytes | None:
        """Call ``LLVMFuzzerCustomCrossOver``; same decline rule as :meth:`mutate`."""
        global _current_rng
        if self._cross is None:
            return None
        cap = self._capacity(max_len, len(a), len(b))
        out = (ctypes.c_uint8 * cap)()
        _current_rng = rng
        try:
            n = self._cross(a, len(a), b, len(b), out, cap, rng.randint(0, 0xFFFFFFFF))
        finally:
            _current_rng = None
        if n == 0 or n > cap:
            return None
        return ctypes.string_at(out, n)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class _PluginOp(MutatorBase):
    category = "adaptive"

    def __init__(self, plugin: CustomMutatorPlugin) -> None:
        self.plugin = plugin

    def rebind(self, plugin: CustomMutatorPlugin) -> None:
        self.plugin = plugin


class CustomMutatorOp(_PluginOp):
    """``custom_mutator``: the plugin's ``LLVMFuzzerCustomMutator``."""

    name = "custom_mutator"

    def mutate(
        self,
        data: bytes,
        rng: Any,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx: Any,
    ) -> bytes | None:
        return self.plugin.mutate(data, max_len, rng)


class CustomCrossoverOp(_PluginOp):
    """``custom_crossover``: the plugin's ``LLVMFuzzerCustomCrossOver`` against a corpus entry."""

    name = "custom_crossover"

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        return len(context.corpus) > 1

    def mutate(
        self,
        data: bytes,
        rng: Any,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx: Any,
    ) -> bytes | None:
        corpus = context.corpus if context is not None else ()
        if len(corpus) < 2:
            return None
        other = corpus[rng.randrange(len(corpus))]
        if not isinstance(other, bytes | bytearray) or other == data:
            return None
        return self.plugin.crossover(data, bytes(other), max_len, rng)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_OP_NAMES = (CustomMutatorOp.name, CustomCrossoverOp.name)


def register_custom_mutator(path: str | os.PathLike[str]) -> list[MutatorBase]:
    """Load *path* and register one operator per exported entry point.

    Safe to call again: an operator that already exists is rebound to the
    new plugin instead of raising a duplicate-registration error.  Raises
    ``FileNotFoundError`` / ``RuntimeError`` on a missing, unloadable or
    symbol-less plugin; the caller asked for it by name, so it fails loudly.
    """
    from fuzzer_tool.core import operator_categories
    from fuzzer_tool.core.operator_registry import REGISTRY

    plugin = CustomMutatorPlugin(path)
    existing = {getattr(m, "name", None): m for m in REGISTRY.mutators()}
    wanted: list[tuple[type[_PluginOp], bool]] = [
        (CustomMutatorOp, plugin.has_mutator),
        (CustomCrossoverOp, plugin.has_crossover),
    ]
    out: list[MutatorBase] = []
    for cls, present in wanted:
        if not present:
            continue
        mutator = existing.get(cls.name)
        if mutator is not None:
            mutator.rebind(plugin)
        else:
            mutator = cls(plugin)
            REGISTRY.register_mutator(mutator)
        out.append(mutator)
    operator_categories.refresh()
    log.info(
        "Custom mutator plugin %s: %s",
        plugin.path,
        ", ".join(m.name for m in out),
    )
    return out


def unregister_custom_mutators() -> None:
    """Remove the plugin operators from ``REGISTRY`` (test teardown)."""
    from fuzzer_tool.core import operator_categories
    from fuzzer_tool.core.operator_registry import REGISTRY

    removed = False
    for name in _OP_NAMES:
        if REGISTRY._ops.pop(name, None) is not None:  # noqa: SLF001
            removed = True
    if removed:
        REGISTRY._categories_cache = None  # noqa: SLF001
        REGISTRY._version += 1  # noqa: SLF001
        with contextlib.suppress(Exception):
            operator_categories.refresh()


__all__ = [
    "CROSSOVER_SYMBOL",
    "MUTATOR_SYMBOL",
    "CustomCrossoverOp",
    "CustomMutatorOp",
    "CustomMutatorPlugin",
    "havoc",
    "register_custom_mutator",
    "unregister_custom_mutators",
]
