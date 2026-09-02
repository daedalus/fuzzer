"""On-disk cache for de Bruijn sequence construction.

Implements handover 10f
(``docs/handover/handover_combinatorics_permutations_2026-09-02.md``):
``core/mutations/structured.py``'s ``de_bruijn_bytes`` and
``de_bruijn_bits`` are pure functions of ``(k, n)`` (or ``n`` alone for
the bit variant) already memoized per-process via ``@lru_cache``, but a
parallel fuzzing campaign starts N worker processes that each rebuild the
same handful of sequences from scratch. Since the construction is a pure
function of its inputs, a disk cache keyed by those inputs deduplicates
the work across every process on the machine, following the same
XDG-aware ``~/.cache/`` convention as ``cfg_cache.py``.

Much simpler than ``cfg_cache.py``: the payload is raw ``bytes`` (no
pickling, no class allowlist needed), and invalidation is a single
fingerprint over ``_de_bruijn_symbols``' source folded into the
filename — a source edit changes the fingerprint and every artifact
under the old fingerprint is simply never looked up again (no need to
open and validate stale files, they just accumulate as orphans until
manually cleared).
"""

import hashlib
import inspect
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

_CACHE_SUBDIR = "fuzzer_debruijn_cache"
_cache_dir_memo: str | None = None


def env_enabled() -> bool:
    """False when FUZZER_DISABLE_DEBRUIJN_CACHE opts out (cfg_cache.py style)."""
    return os.environ.get("FUZZER_DISABLE_DEBRUIJN_CACHE", "") not in ("1", "true", "yes")


def _cache_dir() -> str:
    global _cache_dir_memo
    if _cache_dir_memo is None:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        _cache_dir_memo = os.path.join(base, _CACHE_SUBDIR)
        os.makedirs(_cache_dir_memo, exist_ok=True)
    return _cache_dir_memo


def fingerprint(fn: Callable) -> str:
    """sha256 over *fn*'s source, so an algorithm edit invalidates automatically."""
    try:
        src = inspect.getsource(fn)
    except OSError:
        # Source unavailable (e.g. compiled/frozen build) -- fall back to a
        # fixed tag rather than raising; the cache degrades to "always miss"
        # for that process instead of crashing the mutator.
        return "nosource"
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _artifact_path(kind: str, key: str, fp: str) -> Path:
    return Path(_cache_dir()) / f"{kind}_{key}_{fp}.bin"


def load(kind: str, key: str, fp: str) -> bytes | None:
    """Cached sequence bytes for *(kind, key, fp)*, or None on miss/disable."""
    if not env_enabled():
        return None
    path = _artifact_path(kind, key, fp)
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None  # plain miss -- not worth a warning
    except OSError as e:
        log.warning("de Bruijn cache %s unreadable, recomputing: %s", path.name, e)
        return None


def store(kind: str, key: str, fp: str, data: bytes) -> None:
    """Write *data* for *(kind, key, fp)* atomically. Best-effort."""
    if not env_enabled() or not data:
        return
    final = _artifact_path(kind, key, fp)
    try:
        fd, tmp = tempfile.mkstemp(dir=str(final.parent), suffix=".tmp")
        try:
            with open(fd, "wb") as f:
                f.write(data)
            Path(tmp).replace(final)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as exc:
        log.warning("could not write de Bruijn cache %s: %s", final.name, exc)
