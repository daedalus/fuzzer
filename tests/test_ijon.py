"""IJON annotations: target-declared state feedback (Aschermann et al., S&P 2020).

    target: IJON_MAX(x) ─> ijon_max(slot, x)       ─> IjonMaxShm slot (u64, max)
            IJON_SET(x) ─> ijon_map_set(site ^ x)  ─> synthetic edge (once per run)
            IJON_INC(x) ─> ijon_map_inc(site ^ x)  ─> synthetic edge (hit-count bucketed)
    fuzzer: IjonChannel.observe() ─> IjonMax: any slot above campaign max -> admit

Segment: [u32 head = slots + 1][u32 dirty][u64 slots[head - 1]][4 pad]
"""

from __future__ import annotations

import ctypes
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import IjonMaxShm, ShmCoverage
from fuzzer_tool.core.elf import detect_ijon_capable
from fuzzer_tool.core.ijon import IJON_MAP_SLOTS, IjonMax
from fuzzer_tool.services.ijon_channel import Exec, IjonChannel
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import SHIM, _build, _clean_env

MAP_ENTRIES = 8192
IJON_ENV = "__AFL_IJON_SHM_ID"
U64_MAX = (1 << 64) - 1


def _vals(**at: int) -> np.ndarray:
    """Slot vector with *at* = {"s<i>": value}; every other slot 0."""
    v = np.zeros(IJON_MAP_SLOTS, dtype=np.uint64)
    for k, x in at.items():
        v[int(k[1:])] = x
    return v


# ── IjonMax (core) ───────────────────────────────────────────────────


def test_first_nonzero_value_is_new():
    t = IjonMax()
    assert t.observe(_vals(s3=1))
    assert t.hits == 1


def test_strictly_higher_is_new():
    t = IjonMax()
    t.observe(_vals(s3=10))
    assert t.observe(_vals(s3=11))


def test_falsify_equal_or_lower_is_not_new():
    """A tracker that admits every nonzero run would pass the two tests above."""
    t = IjonMax()
    t.observe(_vals(s3=10, s7=4))
    assert not t.observe(_vals(s3=10))
    assert not t.observe(_vals(s3=9, s7=4))
    assert not t.observe(_vals())
    assert t.hits == 1


def test_max_is_per_slot():
    """Raising slot 7 is new even when slot 3 fell."""
    t = IjonMax()
    t.observe(_vals(s3=10))
    assert t.observe(_vals(s3=1, s7=1))
    assert not t.observe(_vals(s3=10, s7=1))


def test_adversarial_u64_max_no_overflow():
    t = IjonMax()
    assert t.observe(_vals(s0=U64_MAX))
    assert not t.observe(_vals(s0=U64_MAX))
    assert int(t.best()[0]) == U64_MAX


def test_state_roundtrip():
    t = IjonMax()
    t.observe(_vals(s5=42))
    u = IjonMax()
    u.from_dict(t.to_dict(), {}.get)
    assert not u.observe(_vals(s5=42))
    assert u.observe(_vals(s5=43))


@pytest.mark.parametrize("bad", [{}, {"best": b"\x01"}, {"best": "x"}, None])
def test_adversarial_bad_state_starts_fresh(bad):
    """Corrupt or size-mismatched state must not crash --resume."""
    u = IjonMax()
    u.from_dict(bad, {}.get)
    assert u.observe(_vals(s0=1))


# ── IjonMaxShm (adapter) ─────────────────────────────────────────────


def test_layout_head_counts_words_with_header():
    s = IjonMaxShm(slots=IJON_MAP_SLOTS)
    try:
        assert ctypes.c_uint32.from_address(s._ptr).value == IJON_MAP_SLOTS + 1
        assert not s.dirty()
    finally:
        s.cleanup()


def test_read_into_clears_slots_and_dirty():
    s = IjonMaxShm(slots=8)
    try:
        ctypes.c_uint64.from_address(s._ptr + 8 + 3 * 8).value = 99
        ctypes.c_uint32.from_address(s._ptr + 4).value = 1
        out = np.zeros(8, dtype=np.uint64)
        s.read_into(out)
        assert int(out[3]) == 99
        assert not s.dirty()
        s.read_into(out)
        assert not out.any()
    finally:
        s.cleanup()


@pytest.mark.parametrize("slots", [0, 3, 100])
def test_adversarial_non_power_of_two_refused(slots):
    with pytest.raises(ValueError):
        IjonMaxShm(slots=slots)


# ── IjonChannel (service) ────────────────────────────────────────────


def _channel(slots: int = 8) -> IjonChannel:
    return IjonChannel(IjonMaxShm(slots=slots), IjonMax(slots=slots))


def _poke(ch: IjonChannel, slot: int, value: int) -> None:
    ptr = ch._shm._ptr
    ctypes.c_uint64.from_address(ptr + 8 + slot * 8).value = value
    ctypes.c_uint32.from_address(ptr + 4).value = 1


def test_channel_reports_new_max():
    ch = _channel()
    try:
        _poke(ch, 2, 5)
        assert ch.observe(Exec.CLEAN, b"in")
        _poke(ch, 2, 5)
        assert not ch.observe(Exec.CLEAN, b"in")
    finally:
        ch.close()


def test_adversarial_truncated_run_clears_but_never_reports():
    """A crash's partial values neither admit nor leak into the next exec."""
    ch = _channel()
    try:
        _poke(ch, 2, 5)
        assert not ch.observe(Exec.TRUNCATED, b"in")
        assert not ch._shm.dirty()
        assert not ch.observe(Exec.CLEAN, b"in")
        _poke(ch, 2, 5)
        assert ch.observe(Exec.CLEAN, b"in"), "truncated value must not have raised the max"
    finally:
        ch.close()


def test_clean_untouched_segment_is_not_new():
    ch = _channel()
    try:
        assert not ch.observe(Exec.CLEAN, b"in")
    finally:
        ch.close()


# ── FuzzRound admission ──────────────────────────────────────────────


def _round(ijon, timeout=False, crash=False):
    from fuzzer_tool.services.fuzz_round import FuzzRound

    f = SimpleNamespace(
        _exec_perplexity=SimpleNamespace(due=lambda: False),
        _doppler=None,
        _perf_novelty=False,
        _perf_novelty_hits=0,
        _hw_max=None,
        _last_perf_deltas={},
        _ijon=ijon,
    )
    r = FuzzRound(f, b"seed")
    r._is_timeout = timeout
    r._is_crash = crash
    r._scanned_shm = None
    r._observe_counts()
    return r


def test_ijon_max_admits_and_credits():
    ch = _channel()
    try:
        _poke(ch, 1, 7)
        r = _round(ch)
        assert r._is_ijon_max
        assert r._admits()
        r._judge()
        assert r._success
    finally:
        ch.close()


def test_ijon_ignores_crash():
    ch = _channel()
    try:
        _poke(ch, 1, 7)
        assert not _round(ch, crash=True)._is_ijon_max
    finally:
        ch.close()


def test_no_ijon_channel_is_inert():
    assert not _round(None)._is_ijon_max


# ── live shim ────────────────────────────────────────────────────────

_SRC = """
#include "ijon.h"
#include <stdlib.h>
int main(int argc, char **argv) {
    unsigned long long v = argc > 1 ? strtoull(argv[1], 0, 10) : 0;
    ijon_max(3, v);
    ijon_max(3, v / 2);            /* lower: must not overwrite */
    ijon_min(5, v);
    ijon_max(3 + 512, v + 1);      /* wraps to slot 3 */
    int reps = argc > 2 ? atoi(argv[2]) : 1;
    for (int i = 0; i < reps; i++) IJON_SET(v & 3);
    return 0;
}
"""


@pytest.fixture(scope="module")
def ijon_exe(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("ijon"),
        _SRC,
        "-fsanitize-coverage=trace-pc-guard",
        f"-I{SHIM.parent}",
    )


def _run(exe, ijon: IjonMaxShm | None, *argv: str) -> tuple[int, ShmCoverage]:
    shm = ShmCoverage(size=MAP_ENTRIES)
    # Edge ids are exec-stable only without ASLR or base-relative; the fuzzer
    # disables ASLR, pytest does not. IJON_STATE runs on the ctx-hashing slow path.
    env = _clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES), FUZZER_KEEP_ASLR="1")
    if ijon is not None:
        env[IJON_ENV] = ijon.env_id
    rc = subprocess.run([str(exe), *argv], env=env, timeout=30).returncode
    return rc, shm


def _slots(s: IjonMaxShm) -> np.ndarray:
    out = np.zeros(s.slots, dtype=np.uint64)
    s.read_into(out)
    return out


@requires_clang
def test_capability_symbol(ijon_exe):
    assert detect_ijon_capable(str(ijon_exe)) is True


@requires_clang
def test_live_max_min_and_wrap(ijon_exe):
    s = IjonMaxShm(slots=IJON_MAP_SLOTS)
    try:
        rc, shm = _run(ijon_exe, s, "100")
        shm.cleanup()
        assert rc == 0
        assert s.dirty()
        v = _slots(s)
        assert int(v[3]) == 101, "wrapped slot raised past 100; lower value ignored"
        assert int(v[5]) == U64_MAX - 100
        assert int(np.count_nonzero(v)) == 2
    finally:
        s.cleanup()


@requires_clang
def test_live_set_values_are_distinct_synthetic_edges(ijon_exe):
    _, a = _run(ijon_exe, None, "1")
    _, b = _run(ijon_exe, None, "2")
    try:
        ea, eb = a.get_edge_ids(), b.get_edge_ids()
        synth = 0x80000000
        sa = {e for e in ea - eb if e & synth}
        sb = {e for e in eb - ea if e & synth}
        assert sa and sb, "IJON_SET(1) and IJON_SET(2) must map to different ids"
    finally:
        a.cleanup()
        b.cleanup()


@requires_clang
def test_falsify_set_is_once_per_run(ijon_exe):
    """IJON_SET 50x in one run must leave the same count as 1x."""
    _, once = _run(ijon_exe, None, "1", "1")
    _, many = _run(ijon_exe, None, "1", "50")
    try:
        co, cm = once.get_edge_counts(), many.get_edge_counts()
        set_ids = {e for e in co if e & 0x80000000}
        assert set_ids
        assert {e: co[e] for e in set_ids} == {e: cm[e] for e in set_ids}
    finally:
        once.cleanup()
        many.cleanup()


@requires_clang
def test_adversarial_lying_header_is_refused(ijon_exe):
    """A head larger than the segment must not become a write bound."""
    s = IjonMaxShm(slots=8)
    try:
        ctypes.c_uint32.from_address(s._ptr).value = 1 << 20
        rc, shm = _run(ijon_exe, s, "100")
        shm.cleanup()
        assert rc == 0
        assert not s.dirty()
    finally:
        s.cleanup()


@requires_clang
def test_adversarial_no_segment_is_inert(ijon_exe):
    rc, shm = _run(ijon_exe, None, "100")
    shm.cleanup()
    assert rc == 0


# ── Lightweight round ────────────────────────────────────────────────


def _lite_round(poke: bool) -> bool:
    import tempfile

    from fuzzer_tool.core.run_mode import RunMode
    from tests.test_lightweight_mode import _CLEAN, _MUTANT, _make
    from tests.test_lightweight_mode import _round as lite_round

    with tempfile.TemporaryDirectory(prefix="ijon_lite_") as d:
        f = _make(d, run_mode=RunMode.LITE)
        f._ijon = _channel()
        try:
            if poke:
                _poke(f._ijon, 0, 3)
            lite_round(f, _CLEAN, crash=False, new_cov=False)
            return _MUTANT in f.corpus
        finally:
            f._ijon.close()


def test_regression_ijon_admits_in_lightweight():
    """--lightweight skips _observe; a new IJON max must still admit."""
    assert _lite_round(poke=True)


def test_lite_without_new_max_admits_nothing():
    """Control: the same LITE round with an untouched max-map is boring."""
    assert not _lite_round(poke=False)


# ── Upstream API parity (RUB-SysSec/ijon afl-llvm-rt.o.c) ────────────

_M64 = (1 << 64) - 1


def _ref_simple_hash(x: int) -> int:
    """splitmix64 finalizer, transcribed from the upstream runtime."""
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def _ref_hashint(old: int, val: int) -> int:
    return _ref_simple_hash(((old & 0xFFFFFFFF) << 32) | (val & 0xFFFFFFFF)) & 0xFFFFFFFF


def _ref_hashmem(old: int, data: bytes) -> int:
    old = _ref_hashint(old, len(data))
    for b in data:
        # upstream passes a (signed) char: bytes >= 0x80 sign-extend
        old = _ref_hashint(old, b - 256 if b >= 0x80 else b)
    return old


_API_SRC = r"""
#include "ijon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__attribute__((noinline)) static uint32_t inner(void) { return ijon_hashstack(); }
__attribute__((noinline)) static uint32_t outer_a(void) { return inner(); }
__attribute__((noinline)) static uint32_t outer_b(void) { return inner(); }
__attribute__((noinline)) static void mark(void) { IJON_SET(1); }   /* one site */

int main(int argc, char **argv) {
    const char *mode = argv[1];
    if (!strcmp(mode, "hash")) {
        printf("%llu %u %u %u\n", (unsigned long long)ijon_simple_hash(12345ULL),
               ijon_hashint(7, 9), ijon_hashstr(3, (char *)"maze"),
               ijon_hashmem(3, (char *)"\x01\xff", 2));
    } else if (!strcmp(mode, "dist")) {
        printf("%u %u %u %u\n", ijon_strdist((char *)"abcd", (char *)"abXd"),
               ijon_strdist((char *)"ab", (char *)"ab"),
               ijon_memdist((char *)"ab\0c", (char *)"ab\0d", 4),
               ijon_memdist((char *)"xyz", (char *)"xyz", 2));
    } else if (!strcmp(mode, "bits")) {
        printf("%d %d %d\n", IJON_BITS(0u), IJON_BITS(1u), IJON_BITS(0x80000000u));
    } else if (!strcmp(mode, "stack")) {
        uint32_t a[2];
        for (int i = 0; i < 2; i++) a[i] = outer_a();   /* one call site */
        printf("%u %u %u\n", a[0], a[1], outer_b());
    } else if (!strcmp(mode, "set")) {          /* plain IJON_SET(1) */
        mark();
    } else if (!strcmp(mode, "xor")) {          /* same site, other state */
        ijon_xor_state(0x1234);
        mark();
    } else if (!strcmp(mode, "ctx")) {          /* CTX must restore state */
        int v = IJON_CTX(41 + 1);
        if (v != 42) return 3;
        mark();
    } else if (!strcmp(mode, "push")) {
        ijon_push_state(atoi(argv[2]));
        ijon_push_state(atoi(argv[3]));
        mark();
    } else if (!strcmp(mode, "off")) {          /* feedback disabled */
        ijon_disable_feedback();
        IJON_SET(1);
        ijon_max(1, 77);
        ijon_enable_feedback();
        ijon_max(2, 5);
    } else if (!strcmp(mode, "strdist")) {
        IJON_STRDIST((char *)argv[2], (char *)"MAGIC");
    }
    return 0;
}
"""


@pytest.fixture(scope="module")
def api_exe(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("ijon_api"),
        _API_SRC,
        "-fsanitize-coverage=trace-pc-guard",
        f"-I{SHIM.parent}",
    )


def _out(exe, *argv: str) -> list[int]:
    # Base-relative return addresses: hashstack must not move with ASLR.
    env = _clean_env(FUZZER_KEEP_ASLR="1")
    r = subprocess.run([str(exe), *argv], capture_output=True, text=True, timeout=30, env=env)
    assert r.returncode == 0, r.stderr
    return [int(x) for x in r.stdout.split()]


def _synth(exe, *argv: str) -> set[int]:
    _, shm = _run(exe, None, *argv)
    try:
        return {e for e in shm.get_edge_ids() if e & 0x80000000}
    finally:
        shm.cleanup()


@requires_clang
def test_hash_helpers_match_upstream(api_exe):
    got = _out(api_exe, "hash")
    want = [
        _ref_simple_hash(12345),
        _ref_hashint(7, 9),
        _ref_hashmem(3, b"maze"),
        _ref_hashmem(3, b"\x01\xff"),
    ]
    assert got == want


@requires_clang
def test_dist_helpers_match_upstream(api_exe):
    # strdist stops at the first mismatch or NUL; memdist ignores NUL.
    assert _out(api_exe, "dist") == [2, 2, 3, 2]


@requires_clang
def test_bits(api_exe):
    assert _out(api_exe, "bits") == [0, 31, 0]


@requires_clang
def test_hashstack_deterministic_and_stack_sensitive(api_exe):
    a1, a2, b = _out(api_exe, "stack")
    assert a1 == a2
    assert a1 != b
    assert _out(api_exe, "stack")[0] == a1, "must be stable across processes"


@requires_clang
def test_xor_state_moves_set_id(api_exe):
    assert _synth(api_exe, "set") != _synth(api_exe, "xor")


@requires_clang
def test_ctx_restores_state(api_exe):
    assert _synth(api_exe, "ctx") == _synth(api_exe, "set")


@requires_clang
def test_push_state_is_history_sensitive(api_exe):
    assert _synth(api_exe, "push", "1", "2") != _synth(api_exe, "push", "2", "1")
    assert _synth(api_exe, "push", "1", "2") == _synth(api_exe, "push", "1", "2")


@requires_clang
def test_disable_feedback_suppresses_ijon(api_exe):
    s = IjonMaxShm(slots=IJON_MAP_SLOTS)
    try:
        rc, shm = _run(api_exe, s, "off")
        synth = {e for e in shm.get_edge_ids() if e & 0x80000000}
        shm.cleanup()
        assert rc == 0
        v = _slots(s)
        assert int(v[1]) == 0, "disabled ijon_max must not write"
        assert int(v[2]) == 5, "re-enabled ijon_max must write"
        assert synth == set(), "disabled IJON_SET must not map"
    finally:
        s.cleanup()


@requires_clang
def test_strdist_progress_is_coverage(api_exe):
    assert _synth(api_exe, "strdist", "MAxxx") != _synth(api_exe, "strdist", "MAGxx")


# ── Paper parity (Aschermann et al., S&P 2020, §III, §IV-A) ──────────


def test_tracker_keeps_one_input_per_slot():
    """§IV-A4: the best input per slot; a better one replaces it."""
    t = IjonMax()
    t.observe(_vals(s3=5), b"five")
    t.observe(_vals(s3=5, s4=1), b"tie")
    assert t.slot_inputs() == {3: b"five", 4: b"tie"}
    t.observe(_vals(s3=9), b"nine")
    assert t.slot_inputs() == {3: b"nine", 4: b"tie"}


def test_falsify_lower_value_never_replaces_slot_input():
    t = IjonMax()
    t.observe(_vals(s3=9), b"nine")
    t.observe(_vals(s3=8), b"eight")
    assert t.slot_inputs() == {3: b"nine"}


def test_slot_inputs_roundtrip_by_hash():
    t = IjonMax()
    t.observe(_vals(s2=4), b"seed-a")
    u = IjonMax()
    from fuzzer_tool.core.hashing import key16

    u.from_dict(t.to_dict(), {key16(b"seed-a"): b"seed-a"}.get)
    assert u.slot_inputs() == {2: b"seed-a"}


def test_adversarial_missing_slot_input_dropped_on_resume():
    """A slot input pruned from the corpus must not resurrect."""
    t = IjonMax()
    t.observe(_vals(s2=4), b"gone")
    u = IjonMax()
    u.from_dict(t.to_dict(), {}.get)
    assert u.slot_inputs() == {}
    assert not u.observe(_vals(s2=4), b"x"), "the max itself is kept"


def _sched(pct: int, randints, choice_idxs=()):
    from tests.support.scripted_rng import ScriptedRng

    ch = _channel(slots=IJON_MAP_SLOTS)
    ch.schedule_maxmap = pct
    ch.tracker.observe(_vals(s1=1, s5=2), b"a")
    ch.tracker.observe(_vals(s5=3), b"b")
    rng = ScriptedRng(randints=randints, choice_idxs=choice_idxs)
    try:
        return ch.pick(rng)
    finally:
        ch.close()


def test_schedule_draw_above_threshold_picks_max_queue():
    """§IV-A4: draw 0..99; >= IJON_SCHEDULE_MAXMAP picks a slot input."""
    assert _sched(50, randints=[50], choice_idxs=[1]) == b"b"


def test_schedule_draw_below_threshold_defers_to_queue():
    assert _sched(50, randints=[49]) is None


def test_schedule_100_never_uses_max_queue():
    assert _sched(100, randints=[99]) is None


def test_schedule_0_always_uses_max_queue():
    assert _sched(0, randints=[0], choice_idxs=[0]) == b"a"


def test_adversarial_schedule_empty_max_queue():
    from tests.support.scripted_rng import ScriptedRng

    ch = _channel()
    try:
        assert ch.pick(ScriptedRng(randints=[99])) is None
    finally:
        ch.close()


def test_seed_picker_takes_ijon_pick():
    from types import SimpleNamespace

    from fuzzer_tool.services.seed_picker import IJON_MAXMAP, SeedPicker

    f = SimpleNamespace(
        _rng=None,
        _stall_recovery_active=False,
        _target_arena=None,
        _ijon=SimpleNamespace(pick=lambda rng: b"slot-input"),
        _seed_strategy=None,
    )
    sp = SeedPicker.__new__(SeedPicker)
    sp.f = f
    sp._update_temperature = lambda: None
    assert sp.pick_seed() == b"slot-input"
    assert f._seed_strategy == IJON_MAXMAP


_PAPER_SRC = r"""
#include "ijon.h"
#include <stdlib.h>
#include <string.h>

static volatile int sink;
__attribute__((noinline)) static void work(int a) {
    if (a > 3) sink = a * 2;
    else sink = a;
}

int main(int argc, char **argv) {
    const char *mode = argv[1];
    int n = argc > 2 ? atoi(argv[2]) : 0;
    if (!strcmp(mode, "state")) {           /* IJON_STATE(n) then the same code */
        IJON_STATE(n);
        work(7);
    } else if (!strcmp(mode, "state2")) {   /* two calls cancel */
        IJON_STATE(n);
        IJON_STATE(n);
        work(7);
    } else if (!strcmp(mode, "off")) {      /* IJON_DISABLE masks edges */
        IJON_DISABLE();
        work(7);
        work(1);
    } else if (!strcmp(mode, "max2")) {     /* IJON_MAX(slot, x) */
        IJON_MAX(3, 40);
        IJON_MIN(4, 2);
        IJON_MAX(41);
    } else if (!strcmp(mode, "hash")) {
        unsigned a = IJON_HASH_INT(1, 2), b = IJON_HASH_STR(1, (char *)"x"),
                 c = IJON_HASH_MEM(1, (char *)"x", 1), d = IJON_HASH_STACK(0);
        (void)d;   /* stack hash depends on the call site; compile check only */
        return (a == ijon_hashint(1, 2) && b == ijon_hashstr(1, (char *)"x") &&
                c == ijon_hashmem(1, (char *)"x", 1))
                   ? 0 : 5;
    }
    return 0;
}
"""


@pytest.fixture(scope="module")
def paper_exe(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("ijon_paper"),
        _PAPER_SRC,
        "-fsanitize-coverage=trace-pc-guard",
        f"-I{SHIM.parent}",
    )


def _real(exe, *argv: str) -> set[int]:
    _, shm = _run(exe, None, *argv)
    try:
        return {e for e in shm.get_edge_ids() if not e & 0x80000000}
    finally:
        shm.cleanup()


@requires_clang
def test_state_moves_real_edges(paper_exe):
    """§IV-A3: same code under another state is different coverage."""
    base, five = _real(paper_exe, "state", "0"), _real(paper_exe, "state", "5")
    assert base != five
    assert five == _real(paper_exe, "state", "5"), "deterministic"


@requires_clang
def test_state_twice_cancels(paper_exe):
    """Same branch of main, so only the state differs: 5^5 == 0^0."""
    assert _real(paper_exe, "state2", "5") == _real(paper_exe, "state2", "0")


@requires_clang
def test_disable_masks_real_edges(paper_exe):
    """§IV-A1: after IJON_DISABLE, work() adds no edges."""
    off = _real(paper_exe, "off")
    on = _real(paper_exe, "state", "0")
    assert len(off) < len(on)


@requires_clang
def test_max_two_arg_slot(paper_exe):
    s = IjonMaxShm(slots=IJON_MAP_SLOTS)
    try:
        rc, shm = _run(paper_exe, s, "max2")
        shm.cleanup()
        assert rc == 0
        v = _slots(s)
        assert int(v[3]) == 40
        assert int(v[4]) == U64_MAX - 2
        assert int(np.count_nonzero(v)) == 3, "1-arg IJON_MAX uses a site slot"
    finally:
        s.cleanup()


@requires_clang
def test_paper_hash_macros(paper_exe):
    rc, shm = _run(paper_exe, None, "hash")
    shm.cleanup()
    assert rc == 0


# ── CLI ──────────────────────────────────────────────────────────────


def test_cli_schedule_maxmap_parses(monkeypatch):
    from tests.test_lightweight_mode import _cli

    args, _ = _cli(monkeypatch, "--ijon", "--ijon-schedule-maxmap", "30")
    assert args.ijon is True
    assert args.ijon_schedule_maxmap == 30


@pytest.mark.parametrize("bad", ["-1", "101", "x"])
def test_adversarial_cli_schedule_maxmap_out_of_range(monkeypatch, bad):
    from tests.test_lightweight_mode import _cli

    with pytest.raises(SystemExit):
        _cli(monkeypatch, "--ijon-schedule-maxmap", bad)


def test_observe_without_data_keeps_max_queue():
    """Adversarial: a caller passing no input must not enqueue b''."""
    t = IjonMax()
    assert t.observe(_vals(s1=1))
    assert t.slot_inputs() == {}


def test_from_dict_without_resolver_restores_maxima_only():
    t = IjonMax()
    t.observe(_vals(s1=4), b"in")
    u = IjonMax()
    u.from_dict(t.to_dict())
    assert u.slot_inputs() == {}
    assert not u.observe(_vals(s1=4))
