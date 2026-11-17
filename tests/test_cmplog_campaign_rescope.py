"""Campaign dedup scope must not hide pairs the pool has evicted.

Under ``--cmplog-dedup-scope campaign`` the shim drops every record it
already wrote this campaign, but the collector evicts pairs past
``max_pairs``. An evicted pair then never comes back. Fix: once the pool
has evicted ``max_pairs`` pairs since the last rescope (a full turnover),
the fuzzer calls ``__cmplog_rescope()``.

The end-to-end case runs in a child process: the shim's tables live in
the loaded ``.so`` and must not leak into the pytest process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.cmplog import CmplogCollector, DedupScope, Preload
from fuzzer_tool.services.fuzz_round import FuzzRound
from tests.conftest import requires_clang

AFL_SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
REPO_SRC = Path(__file__).parent.parent / "src"

_MAX_PAIRS = 8
_CHURN = 40  # distinct pairs run between the two probes: several turnovers
_PROBE = 1

# One compare per call: run(v) records the pair (MAGIC, v); clang passes
# the constant first.
_LIB_C = r"""
#include <stdint.h>
volatile int sink;
void run(uint32_t v) { sink += (v == 0xA5A5A5A5u); }
"""

# Child: drive the collector + shim like the fuzz loop, report what the
# pool held. argv: lib, workdir, scope, policy (on/off).
_CHILD = r"""
import ctypes, json, sys
from fuzzer_tool.core.cmplog import CmplogCollector, DedupScope, Preload

lib_path, workdir, scope, policy = sys.argv[1:5]
probe, churn, max_pairs = (int(a) for a in sys.argv[5:8])
c = CmplogCollector(workdir=workdir, preload=Preload.NONE, max_pairs=max_pairs,
                    dedup_scope=DedupScope(scope))
c._shim_path = workdir + "/unused_shim.so"
c.setup_env_for_run()
lib = ctypes.CDLL(lib_path)
rescopes = 0

def step(v):
    global rescopes
    lib.run(ctypes.c_uint32(v))
    lib.__tracecmp_flush()
    c.collect_tokens()
    lib.__cmplog_reset()
    if policy == "on" and c.take_stale_scope():
        lib.__cmplog_rescope()
        rescopes += 1

def held(v):
    return any(int.from_bytes(b, "little") == v for _, b in c.pairs)

step(probe)
first = held(probe)
for v in range(1000, 1000 + churn):
    step(v)
evicted = not held(probe)
step(probe)
print(json.dumps({"first": first, "evicted": evicted, "back": held(probe),
                  "rescopes": rescopes, "evictions": c.evicted_pair_count}))
c.stop()
"""


@pytest.fixture(scope="module")
def lib(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cmplog_rescope")
    src = d / "lib.c"
    src.write_text(_LIB_C)
    so = d / "lib.so"
    proc = subprocess.run(
        ["clang", "-O2", "-shared", "-fPIC", "-fsanitize-coverage=trace-pc-guard,trace-cmp"]
        + ["-D__AFL_CMPLOG=1", "-include", str(AFL_SHIM), "-o", str(so), str(src), "-ldl"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"lib did not build: {proc.stderr[-400:]}")
    return so


def _scenario(lib: Path, tmp_path: Path, scope: DedupScope, policy: str) -> dict:
    work = tmp_path / f"{scope.value}_{policy}"
    work.mkdir()
    env = dict(os.environ, PYTHONPATH=str(REPO_SRC))
    for k in ("_CMPLOG_OUT", "__AFL_CMPLOG_DEDUP_SCOPE", "__AFL_SHM_ID"):
        env.pop(k, None)
    args = [str(lib), str(work), scope.value, policy, str(_PROBE), str(_CHURN), str(_MAX_PAIRS)]
    run = subprocess.run(
        [sys.executable, "-c", _CHILD, *args], env=env, capture_output=True, text=True, timeout=120
    )
    assert run.returncode == 0, run.stderr[-2000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


@requires_clang
class TestEndToEnd:
    def test_control_drain_scope_relearns(self, lib, tmp_path):
        """Rule 46: drain scope re-sends every exec, so the probe comes back."""
        r = _scenario(lib, tmp_path, DedupScope.DRAIN, "off")
        assert r["first"] and r["evicted"] and r["back"]

    def test_falsify_campaign_without_rescope_loses_pair(self, lib, tmp_path):
        """The bug: the shim remembers what the pool forgot."""
        r = _scenario(lib, tmp_path, DedupScope.CAMPAIGN, "off")
        assert r["first"] and r["evicted"]
        assert not r["back"]

    def test_regression_campaign_scope_relearns_evicted_pairs(self, lib, tmp_path):
        r = _scenario(lib, tmp_path, DedupScope.CAMPAIGN, "on")
        assert r["first"] and r["evicted"] and r["back"]
        assert 1 <= r["rescopes"] <= r["evictions"] // _MAX_PAIRS


# ── take_stale_scope: one rescope per full pool turnover ────────────────


def _collector(tmp_path: Path, scope: DedupScope) -> CmplogCollector:
    return CmplogCollector(
        workdir=str(tmp_path), preload=Preload.NONE, max_pairs=_MAX_PAIRS, dedup_scope=scope
    )


def test_stale_after_one_turnover(tmp_path):
    c = _collector(tmp_path, DedupScope.CAMPAIGN)
    c.evicted_pair_count = _MAX_PAIRS - 1
    assert not c.take_stale_scope()
    c.evicted_pair_count = _MAX_PAIRS
    assert c.take_stale_scope()


def test_stale_marks_and_rearms(tmp_path):
    """True once per turnover: the mark moves to the eviction count."""
    c = _collector(tmp_path, DedupScope.CAMPAIGN)
    c.evicted_pair_count = _MAX_PAIRS + 3
    assert c.take_stale_scope()
    assert not c.take_stale_scope()
    c.evicted_pair_count = 2 * _MAX_PAIRS + 3
    assert c.take_stale_scope()


def test_adversarial_drain_scope_never_stale(tmp_path):
    c = _collector(tmp_path, DedupScope.DRAIN)
    c.evicted_pair_count = 100 * _MAX_PAIRS
    assert not c.take_stale_scope()


# ── fuzz round wiring ───────────────────────────────────────────────────


# _collect_due: every round under 500 pairs, every 20th from 2000 on.
_EAGER_POOL = 0
_SAMPLED_POOL = 2000


def _round(stale: bool, collect_now: bool = True) -> tuple[FuzzRound, list[str]]:
    calls: list[str] = []
    cmplog = SimpleNamespace(
        pairs=[None] * (_EAGER_POOL if collect_now else _SAMPLED_POOL),
        collect_tokens=lambda *a: [],
        take_stale_scope=lambda: calls.append("stale?") or stale,
    )
    f = SimpleNamespace(
        _cmplog=cmplog,
        _cmplog_skip_counter=0,
        _rewind_cmplog_shim=lambda: calls.append("rewind"),
        _rescope_cmplog_shim=lambda: calls.append("rescope"),
        _add_dict_tokens=lambda t: None,
    )
    r = FuzzRound.__new__(FuzzRound)
    r._f = f
    return r, calls


def test_round_rescopes_when_stale():
    r, calls = _round(stale=True)
    r._collect_tokens()
    assert calls == ["rewind", "stale?", "rescope"]


def test_falsify_round_keeps_scope_when_fresh():
    r, calls = _round(stale=False)
    r._collect_tokens()
    assert "rescope" not in calls


def test_adversarial_no_check_on_skipped_round():
    """Skipped rounds drained nothing: no eviction could have happened."""
    r, calls = _round(stale=True, collect_now=False)
    r._collect_tokens()
    assert calls == []
