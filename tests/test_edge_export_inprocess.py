"""`edge-dump` on a real shim-linked `.so`: same backends and flags as `minimize -c`.

--inprocess                     loader
--inprocess-direct              direct_lite
--inprocess --inprocess-direct  direct
-c / --coverage                 accepted; coverage is always on
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.cli import commands
from tests import test_minimize_inprocess as _mi

# Guard sets from TARGET_SRC: a={1,10,11}, aaaa=a, b={1,20,21}, c=a+{30}.
SEEDS = {"a": b"a", "a_dup": b"aaaa", "b": b"b", "c": b"c"}

# Shared fixture: fresh shim-linked .so per call.
build_so = _mi.build_so

MODE_FLAGS = {
    "loader": ["--inprocess"],
    "direct_lite": ["--inprocess-direct"],
    "direct": ["--inprocess", "--inprocess-direct"],
}


def _dump(tmp_path: Path, so: str, flags: list[str], tag: str, monkeypatch) -> dict:
    """Run `edge-dump` through main(); return the loaded pickle."""
    corpus = tmp_path / f"corpus_{tag}" / "seeds"
    corpus.mkdir(parents=True)
    for name, data in SEEDS.items():
        (corpus / name).write_bytes(data)

    out = tmp_path / f"{tag}.pkl"
    argv = ["fuzzer-tool", "edge-dump", so, "-d", str(corpus.parent), "-o", str(out)]
    argv += [*flags, "--inprocess-func", "fuzz_test"]
    monkeypatch.setattr(sys, "argv", argv)
    assert commands.main() == 0

    with open(out, "rb") as f:
        return pickle.load(f)


def _sets(d: dict) -> dict[bytes, set[int]]:
    return {data: set(d[hash_data(data)].tolist()) for data in SEEDS.values()}


@pytest.mark.parametrize("mode", list(MODE_FLAGS))
def test_inprocess_modes_export_edges(mode, build_so, tmp_path, monkeypatch):
    """Each backend sees the guard structure of TARGET_SRC."""
    e = _sets(_dump(tmp_path, build_so(), MODE_FLAGS[mode], mode, monkeypatch))

    assert e[b"a"] == e[b"aaaa"]
    assert e[b"a"] < e[b"c"]
    assert len(e[b"c"] - e[b"a"]) == 1
    assert e[b"a"] != e[b"b"]


def test_coverage_flag_accepted(build_so, tmp_path, monkeypatch):
    """-c (minimize parity) yields the same export as without it."""
    with_c = _dump(tmp_path, build_so(), ["--inprocess-direct", "-c"], "c", monkeypatch)
    without = _dump(tmp_path, build_so(), ["--inprocess-direct"], "noc", monkeypatch)
    assert _sets(with_c) == _sets(without)


def test_backends_agree_with_control(build_so, tmp_path, monkeypatch):
    """Adversarial: loader vs direct_lite must match; control: loader vs loader."""
    ref = _sets(_dump(tmp_path, build_so(), MODE_FLAGS["loader"], "ref", monkeypatch))
    ctl = _sets(_dump(tmp_path, build_so(), MODE_FLAGS["loader"], "ctl", monkeypatch))
    assert ref == ctl

    lite = _sets(_dump(tmp_path, build_so(), MODE_FLAGS["direct_lite"], "lite", monkeypatch))
    assert lite == ref
