"""Fuzz targets: fuzzer-tool entry points that must survive any input.

Every target consumes raw bytes, so one crash-file format and one replay
path serve all of them. Object-shaped targets decode the bytes through the
allowlisted state unpickler first; a decode failure is not a finding.

    bytes ──► Target.run ──► ok / _Skip (not applicable) / exception (finding)
                                   │
                         slow (> SLOW_SECONDS) is a finding too
"""

from __future__ import annotations

import functools
import gzip
import pickle
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.core import elf
from fuzzer_tool.core.analyzers.analyzer_format_learner import FormatLearner
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.markov import MarkovEnsemble
from fuzzer_tool.core.mutations.generic import load_dictionary, parse_dict_line
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.state_store import STATE_FILENAME, StateStore, _safe_loads

from . import corpus
from .mutators import crash_key

if TYPE_CHECKING:
    from collections.abc import Callable

OP_PREFIX = "op."
MOCK_MAX_LEN = 4096
# Op output may grow, but not unboundedly (Hard Rule 54).
OP_GROWTH_CAP = 16
ELF_PROBE_VADDR = 0x400000
LEARNER_MAX_TRANSITIONS = 2000
TRACKER_MAP_SIZE = 1024
_HASH_SEED_HEX = 8
_TRANSITION_TYPES = (bytes, str, (int, type(None)), int, int, int, set, set)


class _Skip(Exception):
    """Input does not reach the target (undecodable, gated op, mock gap)."""


class OracleError(AssertionError):
    """A target returned, but its output broke the contract."""


@dataclass(frozen=True)
class Target:
    """A named entry point and the crash-file extension for its inputs."""

    name: str
    ext: str
    run: Callable[[bytes], None]


# ── decoding ─────────────────────────────────────────────────────────────────


def _decode(data: bytes) -> Any:
    """Allowlisted unpickle; anything it rejects never reaches the target."""
    try:
        return _safe_loads(data)
    except Exception as exc:
        raise _Skip(corpus.err(exc)) from exc


def _typed(row: Any) -> bool:
    """Whether *row* matches the ``record_transition`` argument types."""
    if not isinstance(row, tuple) or len(row) != len(_TRANSITION_TYPES):
        return False
    return all(isinstance(v, t) for v, t in zip(row, _TRANSITION_TYPES, strict=True))


# ── ELF / dictionary ─────────────────────────────────────────────────────────


def _run_elf_bytes(data: bytes) -> None:
    """Bytes-level ELF parsers: contract is None/[] on malformed input."""
    elf.func_table(data)
    elf.data_ranges(data)
    elf.word_section_spans(data)
    elf.find_load_segment(data, ELF_PROBE_VADDR)


def _with_file(data: bytes, fn: Callable[[str], Any]) -> None:
    """Run a path-taking parser on *data* written to a throwaway file."""
    with tempfile.TemporaryDirectory(prefix="selffuzz_") as tmp:
        path = Path(tmp) / "input"
        path.write_bytes(data)
        fn(str(path))


def _elf_path_probe(path: str) -> None:
    elf.is_elf_executable(path)
    elf.detect_elf_type(path)
    elf.build_id(path)


def _run_elf_path(data: bytes) -> None:
    """Path-level ELF probes the CLI runs on every target binary."""
    _with_file(data, _elf_path_probe)


def _run_dict(data: bytes) -> None:
    """AFL dictionary: per-line parser and the file loader."""
    for line in data.decode("latin-1").splitlines():
        parse_dict_line(line)
    _with_file(data, load_dictionary)


# ── persisted state ──────────────────────────────────────────────────────────


def _run_state(data: bytes) -> None:
    """``--resume`` load of a corrupt state file: must return, never raise."""
    with tempfile.TemporaryDirectory(prefix="selffuzz_") as tmp:
        (Path(tmp) / STATE_FILENAME).write_bytes(gzip.compress(data))
        StateStore(tmp).load()


def _restore_contract(restore: Callable, snapshot: Callable, data: bytes) -> None:
    """Restore applies fully, or raises ValueError leaving the object unchanged."""
    obj = _decode(data)
    before = pickle.dumps(snapshot())
    try:
        restore(obj)
    except ValueError as exc:
        if pickle.dumps(snapshot()) != before:
            raise OracleError(f"rejected restore modified the object: {exc}") from exc


def _run_markov(data: bytes) -> None:
    m = MarkovEnsemble(rng=RandPool(1))
    _restore_contract(m.from_dict, m.to_dict, data)


def _run_edge_tracker(data: bytes) -> None:
    et = EdgeTracker(map_size=TRACKER_MAP_SIZE)
    _restore_contract(et.from_dict, et.to_dict, data)


def _run_learner_state(data: bytes) -> None:
    fl = FormatLearner(rng=RandPool(1))
    _restore_contract(fl.load_state, fl.get_state, data)


def _run_learner(data: bytes) -> None:
    """Replay typed transitions, then every read path of the learned model."""
    rows = _decode(data)
    if not isinstance(rows, list) or not all(_typed(r) for r in rows):
        raise _Skip("not a transition list")

    fl = FormatLearner(rng=RandPool(1))
    for row in rows[:LEARNER_MAX_TRANSITIONS]:
        fl.record_transition(*row)

    for cluster in list(fl.clusters.values()):
        cluster.backtest()

    probe = rows[0][0] if rows else b""
    fl.get_format_summary()
    fl.get_learned_value(0, 4, probe)
    fl.weighted_position(probe, len(probe))
    fl.load_state(fl.get_state())


# ── operators ────────────────────────────────────────────────────────────────


class MockFuzzer:
    """Minimal fuzzer surface ``OperatorEngine`` reads; ungated state only.

    Mirrors ``tests/test_regression_operator_registry.py::_MockFuzzer``.
    """

    def __init__(self) -> None:
        self.dictionary = [b"IHDR", b"\xff\xd8\xff"]
        self.markov_trained = False
        self.mc = None
        self.mc_cem = False
        self.grammar = None
        self._cmplog = None
        self.enable_regex_bomb = False
        self.enable_x86_mutator = False
        self.enable_arm_mutator = False
        self.max_len = MOCK_MAX_LEN
        self.reset(0)

    def reset(self, seed: int) -> None:
        """Fresh per-call state so a replay sees what the campaign saw."""
        self._rng = RandPool(seed)
        self.seed_meta: dict = {}
        self.corpus: list = []
        self._op_declines: dict = {}
        self._dict_scratch: list = []
        self._dict_scratch_idx = 0


@functools.cache
def _mock() -> MockFuzzer:
    return MockFuzzer()


@functools.cache
def _op_table() -> dict[str, Callable]:
    """Dispatch table over the mock; cached so tests can patch entries."""
    from fuzzer_tool.services.operators import OperatorEngine

    return OperatorEngine(_mock()).build_dispatch()


@functools.cache
def all_ops() -> list[str]:
    """Every dispatchable op, sorted; gated ones skip at run time."""
    return sorted(_op_table())


def op_names(data: bytes = b"abc") -> list[str]:
    """Ops the registry arms for *data* on the mock (gated ops drop out)."""
    table = _op_table()
    return sorted(n for n in REGISTRY.available(_mock(), data) if n in table)


def _is_mock_gap(exc: BaseException) -> bool:
    """AttributeError on the mock itself: harness gap, not a target bug."""
    return isinstance(exc, AttributeError) and f"'{MockFuzzer.__name__}' object" in str(exc)


def _check_op_output(name: str, data: bytes, buf: bytearray, out: Any) -> None:
    """Contract: None (in-place) or bytes-like, bounded growth."""
    if out is None:
        out = buf
    if not isinstance(out, bytes | bytearray):
        raise OracleError(f"{name} returned {type(out).__name__}")

    cap = OP_GROWTH_CAP * max(len(data), MOCK_MAX_LEN)
    if len(out) > cap:
        raise OracleError(f"{name} grew {len(data)} -> {len(out)} bytes")


def _run_op(name: str, data: bytes) -> None:
    """One op on *data*, seeded from the input so replay is exact."""
    mock = _mock()
    mock.reset(int(hash_data(data)[:_HASH_SEED_HEX], 16))
    if name not in REGISTRY.available(mock, data):
        raise _Skip("op gated off")

    buf = bytearray(data)
    idx = mock._rng.randrange(len(data)) if data else 0
    try:
        out = _op_table()[name](buf, idx, data)
    except AttributeError as exc:
        if _is_mock_gap(exc):
            raise _Skip(str(exc)) from exc
        raise

    _check_op_output(name, data, buf, out)


def _op_target(name: str) -> Target | None:
    """Resolve ``op.<name>`` lazily: the registry grows, crash files don't."""
    if not name.startswith(OP_PREFIX):
        return None

    op = name.removeprefix(OP_PREFIX)
    if op not in _op_table():
        return None
    return Target(name, "bin", functools.partial(_run_op, op))


TARGETS: dict[str, Target] = {
    "elf_bytes": Target("elf_bytes", "elf", _run_elf_bytes),
    "elf_path": Target("elf_path", "elf", _run_elf_path),
    "dict": Target("dict", "dict", _run_dict),
    "state": Target("state", "pkl", _run_state),
    "markov": Target("markov", "pkl", _run_markov),
    "edge_tracker": Target("edge_tracker", "pkl", _run_edge_tracker),
    "learner_state": Target("learner_state", "pkl", _run_learner_state),
    "learner": Target("learner", "pkl", _run_learner),
}


# ── execution ────────────────────────────────────────────────────────────────


def resolve(name: str) -> Target | None:
    """Static target or a lazily built ``op.<name>``."""
    return TARGETS.get(name) or _op_target(name)


def execute(name: str, data: bytes) -> tuple[str | None, str | None]:
    """Run one case: ``(crash_key, detail)``, both None when clean."""
    target = resolve(name)
    if target is None:
        raise KeyError(f"unknown target {name!r}")

    started = time.monotonic()
    try:
        target.run(data)
    except _Skip:
        return None, None
    except Exception as exc:
        return crash_key(exc), corpus.err(exc)

    elapsed = time.monotonic() - started
    if elapsed > corpus.SLOW_SECONDS:
        return f"slow@{name}", f"slow: {elapsed:.2f}s"
    return None, None


def run_target(name: str, data: bytes) -> str | None:
    """Failure detail for one case, or None."""
    return execute(name, data)[1]
