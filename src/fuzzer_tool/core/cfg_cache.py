"""On-disk cache for decoded function CFGs.

Amortizes the pure-Python x86-64 decode cost across runs: today
``TargetDistance._build_cfgs``, tomorrow the whole-program ICFG
(docs/kscheduler_centrality_port.md W1). Artifacts live under
``~/.cache/fuzzer_cfgcache/`` (XDG_CACHE_HOME aware), one gzip pickle per
(binary, decoder) pair.

Invalidation is three-way:

1. **Binary identity** — NT_GNU_BUILD_ID when the ELF carries one,
   else sha256 of the file bytes.
2. **Decoder fingerprint** — sha256 over the ``cfg.py`` + ``elf.py``
   source bytes, so any change to decode behavior invalidates without a
   manual version bump.
3. **SCHEMA_VERSION** — explicit payload-format version; bump when the
   pickled layout changes in a way the fingerprint cannot see.

All components are folded into the artifact filename AND verified inside
the payload (defense-in-depth for renamed/copied files).

Loading goes through an unpickler whose global allowlist extends the
audited ``state_store._ALLOWED_GLOBALS`` with exactly two first-party
pure-data classes; anything else is refused, never executed.
"""

import gzip
import hashlib
import logging
import os
import pickle
import tempfile
from pathlib import Path

from fuzzer_tool.core import cfg as _cfg_mod
from fuzzer_tool.core import elf as _elf_mod
from fuzzer_tool.core.cfg import FunctionCFG
from fuzzer_tool.core.state_store import _ALLOWED_GLOBALS, UnsafeStateError

log = logging.getLogger(__name__)

# Payload format version. Bump on any layout change the decoder
# fingerprint cannot see (e.g. renaming payload keys).
SCHEMA_VERSION = 1

# Parallel-decode gates: below BOTH thresholds the pool startup cost
# (~tens of ms) exceeds the decode itself and the serial path runs.
PARALLEL_MIN_BYTES = 512 * 1024
PARALLEL_MIN_FUNCS = 16
MAX_WORKERS = min(os.cpu_count() or 1, 8)

# state_store's allowlist covers builtins/collections/numpy only; CFG
# dataclasses pickle by reference and need their own two entries.
_CFG_GLOBALS = _ALLOWED_GLOBALS | {
    ("fuzzer_tool.core.cfg", "FunctionCFG"),
    ("fuzzer_tool.core.cfg", "BasicBlock"),
}

_CACHE_SUBDIR = "fuzzer_cfgcache"
_cache_dir_memo: str | None = None


class _CfgUnpickler(pickle.Unpickler):
    """Unpickler restricted to :data:`_CFG_GLOBALS`."""

    def find_class(self, module: str, name: str):  # noqa: D102
        if (module, name) in _CFG_GLOBALS:
            return super().find_class(module, name)
        raise UnsafeStateError(f"refusing to load disallowed object {module}.{name} from cfg cache")


def env_enabled() -> bool:
    """False when FUZZER_DISABLE_CFG_CACHE opts out (fuzzer.py:92 style)."""
    return os.environ.get("FUZZER_DISABLE_CFG_CACHE", "") not in ("1", "true", "yes")


def _cache_dir() -> str:
    global _cache_dir_memo
    if _cache_dir_memo is None:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        _cache_dir_memo = os.path.join(base, _CACHE_SUBDIR)
        os.makedirs(_cache_dir_memo, exist_ok=True)
    return _cache_dir_memo


def _artifact_path(identity: str) -> Path:
    return Path(_cache_dir()) / f"{identity}.pkl.gz"


def decoder_fingerprint() -> str:
    """sha256 over the decoder sources, so edits invalidate automatically."""
    h = hashlib.sha256()
    for mod in (_cfg_mod, _elf_mod):
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()[:16]


def identity(path: str) -> str | None:
    """Cache key for *path*, or None when the file cannot be identified."""
    try:
        bid = _elf_mod.build_id(path)
        st_size = os.path.getsize(path)
        if bid is not None:
            binary_part = bid.hex()
        else:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            binary_part = h.hexdigest()
    except OSError as e:
        log.debug("cfg-cache identity unavailable for %s: %s", path, e)
        return None
    combined = "|".join((binary_part, decoder_fingerprint(), str(SCHEMA_VERSION), str(st_size)))
    return hashlib.sha256(combined.encode()).hexdigest()


def load(identity_key: str) -> dict[str, FunctionCFG] | None:
    """Cached CFGs for *identity_key*, or None on miss/corruption/disable."""
    if not env_enabled():
        return None
    path = _artifact_path(identity_key)
    try:
        with open(path, "rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as fh:
            payload = _CfgUnpickler(fh).load()
    except FileNotFoundError:
        return None  # plain miss — not worth a warning
    except (
        OSError,
        EOFError,
        ValueError,
        pickle.UnpicklingError,
        UnsafeStateError,
    ) as e:
        log.warning("cfg cache %s unreadable, recomputing: %s", path.name, e)
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SCHEMA_VERSION
        or payload.get("decoder_fp") != decoder_fingerprint()
    ):
        return None
    cfgs = payload.get("cfgs")
    if not isinstance(cfgs, dict):
        return None
    return cfgs


def store(identity_key: str, cfgs: dict[str, FunctionCFG]) -> None:
    """Merge *cfgs* into the cached artifact atomically. Best-effort."""
    if not env_enabled() or not identity_key or not cfgs:
        return
    merged: dict[str, FunctionCFG] = {**(load(identity_key) or {}), **cfgs}
    payload = {
        "schema": SCHEMA_VERSION,
        "decoder_fp": decoder_fingerprint(),
        "cfgs": merged,
    }
    final = _artifact_path(identity_key)
    try:
        fd, tmp = tempfile.mkstemp(dir=str(final.parent), suffix=".tmp")
        try:
            with open(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            Path(tmp).replace(final)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except (OSError, pickle.PicklingError) as exc:
        log.warning("could not write cfg cache %s: %s", final.name, exc)


def should_parallelize(total_decode_bytes: int, n_functions: int) -> bool:
    """Pool gate: spawn workers only above the amortization thresholds."""
    return (
        total_decode_bytes >= PARALLEL_MIN_BYTES or n_functions >= PARALLEL_MIN_FUNCS
    ) and n_functions > 1
