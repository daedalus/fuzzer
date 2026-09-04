"""End-to-end: does an ordering mutation reach the right region's estimator?

`OperatorEngine.record_coverage_diff` folds a coverage-edge diff into
`LiveBitMaskEstimator` for whichever region one byte offset falls in. That
contract is only honoured if the offset describes what the mutation touched.
`targets/order_sensitivity.c` makes the question answerable: four regions with
known order-sensitivity, so every liveness verdict has a ground truth.

    region 0 [0,4096)      LIVE  adjacent records are compared
    region 1 [4096,8192)   DEAD  commutative sum
    region 2 [8192,12288)  DEAD  commutative xor
    region 3 [12288,16384) LIVE  but COLD: unreachable in one adjacent swap

Region 3 is the case tools/gen_synthetic_target.py cannot produce and which
docs/sweeps/synthetic_liveness_calibration_2026-08-29.md names as the
unvalidated one: live, but quiet enough to look dead under a weak mutator.

Requires gcc; skipped without it. ASLR is disabled per child, as
tools/sweep_liveness_thresholds.py learned to do -- production disables it
before running any target, and a harness that does not is characterising a
program the fuzzer never executes.
"""

import ctypes
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fuzzer_tool.adapters.shm import ShmCoverage  # noqa: E402
from fuzzer_tool.core.mutations import chunk_shuffle  # noqa: E402
from fuzzer_tool.core.rand_pool import RandPool  # noqa: E402
from fuzzer_tool.services.operators import OperatorEngine  # noqa: E402
from support.operator_env import make_minimal_fuzzer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM = os.path.join(ROOT, "src", "fuzzer_tool", "adapters", "afl_shim.c")
SRC = os.path.join(ROOT, "targets", "order_sensitivity.c")

REC, NREC = 512, 32
REGIONS = [(0, 8), (8, 16), (16, 24), (24, 32)]  # record slots per 4096-byte region
TRUTH = ["LIVE", "DEAD", "DEAD", "LIVE"]
NOBS = 260  # per region; _LIVENESS_SWITCH_AFTER is 200

pytestmark = pytest.mark.skipif(
    shutil.which("gcc") is None or not os.path.exists(SHIM),
    reason="needs gcc and afl_shim.c",
)

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_ADDR_NO_RANDOMIZE = 0x0040000


def _no_aslr():
    _libc.personality(_ADDR_NO_RANDOMIZE)


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("order_attr")
    exe = str(tmp / "order_sensitivity")
    r = subprocess.run(
        ["gcc", "-O1", "-D__AFL_CTX_SENSITIVE=0", f"-include{SHIM}", "-o", exe, SRC],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-800:]
    inp = str(tmp / "in.bin")

    def run(data):
        # Fresh SHM per execution: the map accumulates, so a reused segment
        # makes every comparison read as "coverage moved".
        cov = ShmCoverage(size=8192)
        try:
            with open(inp, "wb") as f:
                f.write(data)
            env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE="8192")
            subprocess.run([exe, inp], env=env, capture_output=True, preexec_fn=_no_aslr)
            return frozenset(cov.get_edge_ids())
        finally:
            cov.cleanup()

    return run


def _base():
    rng = __import__("random").Random(0)
    b = bytearray(rng.randrange(1, 200) for _ in range(REC * NREC))
    for i in range(24, 32):
        b[i * REC] = 50 + (i - 24) * 3
    b[27 * REC] = 250  # region 3's max sits at local slot 3
    return bytes(b)


BASE = _base()
IDENT = list(range(NREC))


def _build(order):
    return b"".join(BASE[r * REC : (r + 1) * REC] for r in order)


class TestGroundTruth:
    """The premise. If these drift the verdict tests mean nothing."""

    def test_target_is_deterministic(self, bench):
        parent = _build(IDENT)
        assert len({bench(parent) for _ in range(5)}) == 1

    @pytest.mark.parametrize(
        "idx,by_adjacent_swap,by_free_permutation",
        [(0, True, True), (1, False, False), (2, False, False), (3, False, True)],
    )
    def test_region_order_sensitivity(self, bench, idx, by_adjacent_swap, by_free_permutation):
        import random

        rng = random.Random(1)
        lo, hi = REGIONS[idx]
        e0 = bench(_build(IDENT))

        moved_swap = False
        for _ in range(60):
            o = list(IDENT)
            i = rng.randrange(lo, hi - 1)
            o[i], o[i + 1] = o[i + 1], o[i]
            if bench(_build(o)) != e0:
                moved_swap = True
                break
        assert moved_swap is by_adjacent_swap

        moved_free = False
        for _ in range(60):
            o = list(IDENT)
            seg = o[lo:hi]
            rng.shuffle(seg)
            o[lo:hi] = seg
            if bench(_build(o)) != e0:
                moved_free = True
                break
        assert moved_free is by_free_permutation


def _verdicts(bench, arm, seed):
    import random

    rng = random.Random(seed)
    eng = OperatorEngine(make_minimal_fuzzer(seed=seed))
    eng._ctx_cache = eng.ctx
    parent = _build(IDENT)
    pe = bench(parent)
    sound = total = 0
    for k in range(NOBS * 4):
        r = k % 4
        lo, hi = REGIONS[r]
        if arm == "fabricated":
            mut = bytes(chunk_shuffle(parent, rng=RandPool(rng.randrange(1 << 30)), stride=REC))
            off = rng.randrange(len(parent))
        else:
            # The shipped operator, not a re-implementation of it: this is
            # what makes the file a regression test for `region_shuffle`
            # rather than for an idea about it.
            off = (lo + rng.randrange(hi - lo)) * REC
            buf = bytearray(parent)
            eng._op_region_shuffle(buf, off, parent)
            mut = bytes(buf)
        touched = {
            i
            for i in range(4)
            if mut[REGIONS[i][0] * REC : REGIONS[i][1] * REC]
            != parent[REGIONS[i][0] * REC : REGIONS[i][1] * REC]
        }
        eng.record_coverage_diff(parent, off, set(pe), set(bench(mut)))
        total += 1
        if touched <= {off // 4096}:
            sound += 1
    verdicts = [
        "DEAD" if eng._region_liveness_factor(parent, i) < 1.0 else "LIVE" for i in range(4)
    ]
    return sound / total, verdicts


class TestAttribution:
    def test_whole_buffer_shuffle_misreads_the_dead_regions(self, bench):
        """The status quo. A fabricated offset cannot resolve a dead region.

        Every observation carries a nonzero diff (the live half moved) into a
        uniformly drawn region, so no region's mask ever stays empty and both
        known-dead regions are left unresolved -- reported as not-dead.
        """
        m1, verdicts = _verdicts(bench, "fabricated", 11)
        assert m1 < 0.05, m1
        assert verdicts[1] == "LIVE" and verdicts[2] == "LIVE", verdicts

    def test_region_confined_shuffle_classifies_every_region(self, bench):
        """Confined mutation plus a true offset gets all four right.

        Including region 3: a free permutation inside the region reaches its
        firing condition, which a single adjacent swap from the parent never
        does. That is why the operator draws a permutation rather than
        walking a Gray code -- the estimator needs independent samples of the
        ordering space, not a connected walk through it.
        """
        m1, verdicts = _verdicts(bench, "confined", 11)
        assert m1 == 1.0, m1
        assert verdicts == TRUTH, verdicts
