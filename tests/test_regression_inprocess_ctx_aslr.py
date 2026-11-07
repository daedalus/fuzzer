"""In-process `.so` targets: caller-context edge ids must not depend on the load base.

`disable_aslr()` only fixes the layout of *children*; a `.so` loaded into the
fuzzer's own process sits at a randomized base. The shim hashed the raw return
address into ``__afl_guard_run``, so two `edge-dump --inprocess-direct` runs of
one input reported disjoint edge ids (measured: {131..153} vs {225..255}), and
loader vs direct_lite disagreed. A resumed direct_lite campaign read every edge
as new. Fixed: ``InProcessRunner`` switches the shim to base-relative context.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from tests import test_minimize_inprocess as _mi
from tests.test_regression_shim_audit import _clean_env

build_so = _mi.build_so
SEEDS = {"a": b"a", "b": b"b", "c": b"c"}


def _dump(tmp: Path, so: str, flag: str, tag: str) -> set[int]:
    """One `edge-dump` in a fresh process; the union of all edge ids."""
    corpus = tmp / f"c_{tag}" / "seeds"
    corpus.mkdir(parents=True)
    for name, data in SEEDS.items():
        (corpus / name).write_bytes(data)

    out = tmp / f"{tag}.pkl"
    r = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "edge-dump", so, "-d", str(corpus.parent),
         "-o", str(out), flag, "--inprocess-func", "fuzz_test"],
        capture_output=True, text=True, timeout=120, cwd=tmp, env=_clean_env(),
    )  # fmt: skip
    assert r.returncode == 0, r.stdout[-1000:] + r.stderr[-1000:]

    with open(out, "rb") as f:
        return set().union(*(set(v.tolist()) for v in pickle.load(f).values()))


@pytest.fixture
def so(build_so) -> str:
    return build_so()


def test_regression_direct_ids_stable_across_processes(so, tmp_path):
    """Falsification: two direct_lite processes report the same edge ids."""
    first = _dump(tmp_path, so, "--inprocess-direct", "d1")
    second = _dump(tmp_path, so, "--inprocess-direct", "d2")

    assert first
    assert first == second


def test_regression_direct_ids_match_loader(so, tmp_path):
    """Adversarial: the loader child (exec'd without ASLR) agrees with direct_lite."""
    loader = _dump(tmp_path, so, "--inprocess", "l1")
    direct = _dump(tmp_path, so, "--inprocess-direct", "d1")

    assert loader == direct
