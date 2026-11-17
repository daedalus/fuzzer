"""core/deadness: sound static dead-function proofs and the verdict cache (--prove-dead)."""

import shutil
import subprocess

import pytest

from fuzzer_tool.core.deadness import MAX_VERDICTS, DeadProver, Verdict, VerdictCache
from fuzzer_tool.core.elf import func_table

BASE = r"""
#include <stdlib.h>
static volatile int sink;
__attribute__((noinline)) int live_fn(int a) { sink = a; return a + 1; }
__attribute__((noinline)) int dead_fn(int a) { sink = a * 7; return a - 1; }
__attribute__((noinline)) int dead_caller(int a) { return dead_fn(a) + 2; }
__attribute__((noinline)) int ind_fn(int a) { sink = a ^ 5; return a; }
int (*volatile fp)(int) = ind_fn;
__attribute__((noinline)) int cmp_fn(const void *x, const void *y) {
    return *(const int *)x - *(const int *)y;
}
__attribute__((constructor, noinline)) void ctor_fn(void) { sink = 3; }
int main(int argc, char **argv) {
    int v[2] = {argc, 1};
    qsort(v, 2, sizeof(int), cmp_fn);
    CALL_DEAD
    return live_fn(argc) + fp(argc);
}
"""


def _cc(tmp_path, name: str, extra: list[str], defines: str = "") -> str:
    if not shutil.which("clang"):
        pytest.skip("clang not available")
    src = tmp_path / f"{name}.c"
    src.write_text(BASE.replace("CALL_DEAD", defines))
    out = tmp_path / name
    r = subprocess.run(
        ["clang", "-O0", "-fno-inline", *extra, "-o", str(out), str(src)], capture_output=True
    )
    assert r.returncode == 0, r.stderr.decode()
    return str(out)


def _addr(path: str, fname: str) -> int:
    with open(path, "rb") as fh:
        ft = func_table(fh.read())
    return int(ft.starts[ft.names.index(fname)])


def _verdicts(path: str) -> dict[str, Verdict]:
    with open(path, "rb") as fh:
        prover = DeadProver(fh.read())
    names = ("main", "live_fn", "dead_fn", "dead_caller", "ind_fn", "cmp_fn", "ctor_fn")
    return {n: prover.verdict(_addr(path, n)) for n in names}


@pytest.fixture(
    scope="module", params=[["-fPIE", "-pie"], ["-fno-pie", "-no-pie"]], ids=["pie", "exec"]
)
def binary(request, tmp_path_factory):
    return _cc(tmp_path_factory.mktemp("dead"), "prog", request.param)


def test_transitively_dead_functions_proven(binary):
    v = _verdicts(binary)
    assert v["dead_caller"] is Verdict.DEAD  # never called, never referenced
    assert v["dead_fn"] is Verdict.DEAD  # called only from dead code
    assert v["main"] is Verdict.ALIVE
    assert v["live_fn"] is Verdict.ALIVE


def test_adversarial_indirect_call_target_not_dead(binary):
    # ind_fn is only reached through a function pointer: a direct-call walk
    # from main never sees it. The naive check below says dead; the prover
    # must not.
    with open(binary, "rb") as fh:
        prover = DeadProver(fh.read())
    naive = prover._direct_reach_from_main()
    assert not naive[prover._func_index(_addr(binary, "ind_fn"))]
    assert prover.verdict(_addr(binary, "ind_fn")) is Verdict.ALIVE


def test_adversarial_callback_and_constructor_not_dead(binary):
    # qsort calls cmp_fn from libc; the loader calls ctor_fn via .init_array.
    v = _verdicts(binary)
    assert v["cmp_fn"] is Verdict.ALIVE
    assert v["ctor_fn"] is Verdict.ALIVE


def test_falsification_called_function_is_alive(tmp_path):
    path = _cc(tmp_path, "calls", ["-fPIE", "-pie"], "sink = dead_caller(argc);")
    v = _verdicts(path)
    assert v["dead_caller"] is Verdict.ALIVE
    assert v["dead_fn"] is Verdict.ALIVE


def test_shared_object_exports_are_roots(tmp_path):
    if not shutil.which("clang"):
        pytest.skip("clang not available")
    src = tmp_path / "lib.c"
    src.write_text(
        "static volatile int s;\n"
        "__attribute__((used, noinline)) static int hidden(int a) { s = a; return a; }\n"
        "int exported(int a) { s = a + 1; return a; }\n"
    )
    out = tmp_path / "lib.so"
    r = subprocess.run(
        ["clang", "-O0", "-shared", "-fPIC", "-o", str(out), str(src)], capture_output=True
    )
    assert r.returncode == 0, r.stderr.decode()
    with open(out, "rb") as fh:
        prover = DeadProver(fh.read())
    # No main and no internal caller: the naive walk calls it dead.
    exported = _addr(str(out), "exported")
    assert not prover._direct_reach_from_main()[prover._func_index(exported)]
    assert prover.verdict(exported) is Verdict.ALIVE  # dlsym-able


def test_not_elf_is_unknown():
    prover = DeadProver(b"not an elf at all" * 8)
    assert prover.verdict(0x1000) is Verdict.UNKNOWN


def test_address_outside_code_is_unknown(binary):
    with open(binary, "rb") as fh:
        prover = DeadProver(fh.read())
    assert prover.verdict(0) is Verdict.UNKNOWN


# --- verdict cache -----------------------------------------------------------


def test_cache_never_stores_unknown():
    cache = VerdictCache(b"id")
    with pytest.raises(ValueError):
        cache.put(0x10, Verdict.UNKNOWN)
    assert cache.get(0x10) is None


def test_cache_bounded_lru():
    cache = VerdictCache(b"id", cap=2)
    cache.put(1, Verdict.DEAD)
    cache.put(2, Verdict.ALIVE)
    cache.get(1)  # refresh 1
    cache.put(3, Verdict.DEAD)
    assert cache.get(2) is None
    assert cache.get(1) is Verdict.DEAD and cache.get(3) is Verdict.DEAD
    assert VerdictCache(b"id").cap == MAX_VERDICTS


def test_cache_state_round_trip_and_identity_mismatch():
    cache = VerdictCache(b"build-1")
    cache.put(0x40, Verdict.DEAD)
    cache.put(0x80, Verdict.ALIVE)
    same = VerdictCache(b"build-1")
    same.load_state_dict(cache.state_dict())
    assert same.get(0x40) is Verdict.DEAD and same.get(0x80) is Verdict.ALIVE

    # A rebuilt binary must not inherit proofs about the old one.
    other = VerdictCache(b"build-2")
    other.load_state_dict(cache.state_dict())
    assert other.get(0x40) is None


def test_cache_rejects_unknown_in_loaded_state():
    cache = VerdictCache(b"x")
    cache.load_state_dict({"identity": b"x", "verdicts": {1: "unknown", 2: "dead"}})
    assert cache.get(1) is None
    assert cache.get(2) is Verdict.DEAD


# --- directed consumer -------------------------------------------------------


def test_target_distance_drops_proven_dead_targets(binary):
    from fuzzer_tool.core.analyzers.analyzer_distance import TargetDistance
    from fuzzer_tool.core.run_mode import Switch

    td = TargetDistance(binary, targets=["dead_fn", "live_fn"], prove_dead=Switch.ON)
    assert td.load()
    assert td.target_addrs == {_addr(binary, "live_fn")}
    assert td.dead_targets == {_addr(binary, "dead_fn")}

    plain = TargetDistance(binary, targets=["dead_fn", "live_fn"])
    assert plain.load()
    assert plain.target_addrs == {_addr(binary, "live_fn"), _addr(binary, "dead_fn")}


def test_target_distance_keeps_indirect_target(binary):
    from fuzzer_tool.core.analyzers.analyzer_distance import TargetDistance
    from fuzzer_tool.core.run_mode import Switch

    td = TargetDistance(binary, targets=["ind_fn"], prove_dead=Switch.ON)
    assert td.load()
    assert td.target_addrs == {_addr(binary, "ind_fn")}
    assert not td.dead_targets


def test_activation_drops_dead_target_and_caches_proof(binary, monkeypatch, capsys):
    from types import SimpleNamespace

    from fuzzer_tool.core.analyzer_registry import _activate_distance

    monkeypatch.setenv("__AFL_DIST_SHM_ID", "")  # restored after the test
    cache = VerdictCache(b"id")
    f = SimpleNamespace(
        target=binary,
        _distance_targets=["dead_fn", "live_fn"],
        _use_cfg_cache=False,
        debug=False,
        _dead_verdicts=cache,
    )
    _activate_distance(f)
    try:
        assert f._distance.target_addrs == {_addr(binary, "live_fn")}
        assert cache.get(_addr(binary, "dead_fn")) is Verdict.DEAD
        assert cache.get(_addr(binary, "live_fn")) is Verdict.ALIVE
        assert "statically unreachable" in capsys.readouterr().out
    finally:
        if f._dist_table_shm is not None:
            f._dist_table_shm.cleanup()


def test_cached_verdict_skips_prover(binary, monkeypatch):
    from fuzzer_tool.core import deadness
    from fuzzer_tool.core.analyzers.analyzer_distance import TargetDistance
    from fuzzer_tool.core.run_mode import Switch

    cache = VerdictCache(b"id")
    cache.put(_addr(binary, "live_fn"), Verdict.ALIVE)
    cache.put(_addr(binary, "dead_fn"), Verdict.DEAD)

    def boom(*_a, **_k):
        raise AssertionError("prover ran despite cached verdicts")

    monkeypatch.setattr(deadness.DeadProver, "__init__", boom)
    td = TargetDistance(
        binary, targets=["dead_fn", "live_fn"], prove_dead=Switch.ON, verdicts=cache
    )
    assert td.load()
    assert td.dead_targets == {_addr(binary, "dead_fn")}


def test_fuzzer_persists_verdicts():
    import inspect

    from fuzzer_tool.services import fuzzer

    src = inspect.getsource(fuzzer)
    assert 'self._state_store.set("dead_verdicts", self._dead_verdicts.state_dict())' in src
    assert 'self._dead_verdicts.load_state_dict(self._state_store.get("dead_verdicts", {}))' in src
