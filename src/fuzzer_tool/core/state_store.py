"""Single-file persisted fuzzer state.

Replaces the eleven per-component JSON files that used to litter the
corpus directory (markov.json, mi.json, elo.json, ga.json, qea.json,
state.json, sensitivity.json, crash_mi.json, length_tracker.json,
seed_quality.json, edge_tracker.json) with one compressed pickle,
``state.pkl.gz``.

Why pickle rather than JSON: the component ``save()`` payloads are
numeric dicts with non-string keys (edge ids, byte values, positions).
JSON coerces every key to a string, so each component had to re-cast keys
back to ``int`` on load — an easy place to drift, and the reason several
loaders carry defensive ``int(k)`` conversions. Pickle round-trips the
structures exactly.

**Sanitized loading.** Pickle executes arbitrary code by design, and these
files live in the corpus directory — somewhere a fuzzing campaign writes
constantly and a user might reasonably share or copy between machines.
Loading one unrestricted would be a straightforward code-execution
vector. :class:`_SafeUnpickler` therefore refuses every global except a
small allowlist of container/scalar types, so a tampered state file can
at worst produce wrong numbers, not execute code.

Use ``--no-save-state`` to skip writing the file entirely.
"""

from __future__ import annotations

import gzip
import io
import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STATE_FILENAME = "state.pkl.gz"

# Legacy per-component JSON files, kept so an existing corpus directory can
# be migrated on first load and then cleaned up.
LEGACY_JSON_FILES = {
    "markov": "markov.json",
    "mi": "mi.json",
    "elo": "elo.json",
    "ga": "ga.json",
    "qea": "qea.json",
    "corpus": "state.json",
    "sensitivity": "sensitivity.json",
    "crash_mi": "crash_mi.json",
    "length_tracker": "length_tracker.json",
    "seed_quality": "seed_quality.json",
    "edge_tracker": "edge_tracker.json",
}

# Types a state payload is allowed to contain. Everything the components'
# save() methods produce is built from these.
_ALLOWED_GLOBALS = {
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "str"),
    ("builtins", "bytes"),
    ("builtins", "bytearray"),
    ("builtins", "bool"),
    ("builtins", "complex"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("collections", "deque"),
    ("collections", "Counter"),
    # numpy scalars/arrays appear in a few saved payloads
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class UnsafeStateError(Exception):
    """Raised when a state file references a disallowed global."""


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that refuses anything outside :data:`_ALLOWED_GLOBALS`.

    Without this, loading a state file from an untrusted corpus directory
    would execute whatever the file's author chose.
    """

    def find_class(self, module: str, name: str):  # noqa: D102
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise UnsafeStateError(
            f"refusing to load disallowed object {module}.{name} from state file"
        )


def _safe_loads(raw: bytes) -> Any:
    return _SafeUnpickler(io.BytesIO(raw)).load()


class StateStore:
    """Load/save the whole fuzzer state as one compressed pickle.

    Sections are addressed by logical name (``"markov"``, ``"elo"``, ...)
    so callers keep using their existing ``save()``/``load()`` dicts.
    """

    def __init__(self, corpus_dir: str | Path, enabled: bool = True):
        self.corpus_dir = Path(corpus_dir)
        self.path = self.corpus_dir / STATE_FILENAME
        self.enabled = enabled
        self._data: dict[str, Any] = {}
        self._loaded = False

    # ── reading ──────────────────────────────────────────────────────

    def load(self) -> dict[str, Any]:
        """Read the state file, falling back to legacy JSON if absent."""
        self._loaded = True
        if self.path.exists():
            try:
                with gzip.open(self.path, "rb") as fh:
                    data = _safe_loads(fh.read())
                if isinstance(data, dict):
                    self._data = data
                    return self._data
                log.warning("state file %s is not a dict — ignoring", self.path)
            except UnsafeStateError:
                # Do not silently continue: a rejected global means the file
                # was tampered with or written by something else entirely.
                log.error("refusing to load untrusted state file %s", self.path)
            except (OSError, EOFError, pickle.UnpicklingError, ValueError) as exc:
                log.warning("could not read state file %s: %s", self.path, exc)
            self._data = {}
            return self._data

        self._data = self._load_legacy_json()
        return self._data

    def _load_legacy_json(self) -> dict[str, Any]:
        """Migrate a pre-existing set of per-component JSON files."""
        import json

        migrated: dict[str, Any] = {}
        for section, filename in LEGACY_JSON_FILES.items():
            p = self.corpus_dir / filename
            if not p.exists():
                continue
            try:
                migrated[section] = json.loads(p.read_text())
            except (OSError, ValueError) as exc:
                log.warning("skipping unreadable legacy state %s: %s", p, exc)
        if migrated:
            log.info(
                "migrated %d legacy JSON state files into %s",
                len(migrated),
                STATE_FILENAME,
            )
        return migrated

    def get(self, section: str, default: Any = None) -> Any:
        if not self._loaded:
            self.load()
        return self._data.get(section, default)

    # ── writing ──────────────────────────────────────────────────────

    def set(self, section: str, value: Any) -> None:
        self._data[section] = value
        self._loaded = True

    def save(self) -> bool:
        """Write the state file atomically. No-op when disabled."""
        if not self.enabled:
            return False
        if not self._data:
            return False
        try:
            self.corpus_dir.mkdir(parents=True, exist_ok=True)
            # Write to a temp file in the same directory, then rename, so a
            # crash mid-write cannot leave a truncated state file behind.
            fd, tmp = tempfile.mkstemp(dir=str(self.corpus_dir), suffix=".tmp")
            try:
                with open(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
                    pickle.dump(self._data, fh, protocol=pickle.HIGHEST_PROTOCOL)
                Path(tmp).replace(self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except (OSError, pickle.PicklingError) as exc:
            log.warning("could not write state file %s: %s", self.path, exc)
            return False
        return True

    def cleanup_legacy(self) -> int:
        """Remove per-component JSON files superseded by the pickle."""
        removed = 0
        for filename in LEGACY_JSON_FILES.values():
            p = self.corpus_dir / filename
            try:
                if p.exists():
                    p.unlink()
                    removed += 1
            except OSError:
                pass
        return removed

    def __contains__(self, section: str) -> bool:
        if not self._loaded:
            self.load()
        return section in self._data

    def __len__(self) -> int:
        return len(self._data)
