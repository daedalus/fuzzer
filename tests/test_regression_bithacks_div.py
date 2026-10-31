"""Lemire's signed 32-bit division (2026-10-31) in ``targets/bithacks_diff.c``.

``div32`` divides by a divisor > 2 that is not a power of two, with a 33-bit multiplier
and no add-back. ``fastdiv_s32`` (fastmod 0.2.0) divides by a runtime divisor
without a branch. Both must equal C's ``/`` (truncation toward zero).

A driver ``#include``s the target and compares both against ``/`` on edge and
hashed inputs. Hard Rule 46: the reference compared to itself must pass, and a
deliberately wrong divisor must fail, or the comparison proves nothing.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "targets" / "bithacks_diff.c"

# UB traps without the runtime library (not installed in every container).
_UBSAN = ("-fsanitize=undefined", "-fsanitize-trap=undefined")

# Driver modes. Each prints "bad=<mismatches>".
_DRIVER = r"""
#define main bithacks_main
#include "@SRC@"
#undef main

static long bad;
static void cmp(long got, long ref) { bad += got != ref; }

// Dividends: edges, multiples of d +-1 (rounding boundaries), hashed sweep.
#define FOR_X(d, body) do { \
    static const int32_t E[] = {INT_MIN, INT_MIN + 1, -1, 0, 1, INT_MAX - 1, INT_MAX}; \
    for (unsigned i = 0; i < sizeof E / sizeof *E; i++) { int32_t x = E[i]; body; } \
    for (int64_t k = -3; k <= 3; k++) for (int64_t e = -1; e <= 1; e++) { \
        int64_t xx = (int64_t)(d) * k + e; if (xx < INT_MIN || xx > INT_MAX) continue; \
        int32_t x = (int32_t)xx; body; } \
    int64_t lim = INT_MAX / ((d) < 0 ? -(int64_t)(d) : (d)); \
    for (int64_t e = -1; e <= 1; e++) for (int sg = -1; sg <= 1; sg += 2) { \
        int64_t xx = sg * (lim * (d) + e); if (xx < INT_MIN || xx > INT_MAX) continue; \
        int32_t x = (int32_t)xx; body; } \
    for (uint32_t i = 0; i < 4096; i++) { int32_t x = (int32_t)h32(i); body; } \
} while (0)

static bool skip32(int32_t d) { return d < 3 || (d & (d - 1)) == 0; }
static bool skipfd(int32_t d) { return d == 0 || d == 1 || d == -1 || d == INT_MIN; }

static const int32_t BIG[] = {INT_MAX, INT_MAX - 2, (1 << 30) + 1, (1 << 30) - 1, 1000003, 65537};

static void run_div32(int32_t off) {
    // Constant divisors from the article: folded at compile time.
    FOR_X(7, cmp(div32(x, 7), x / 7)); FOR_X(60, cmp(div32(x, 60), x / 60));
    FOR_X(365, cmp(div32(x, 365), x / 365)); FOR_X(1000, cmp(div32(x, 1000), x / 1000));
    for (int32_t d = 3; d <= 1100; d++) if (!skip32(d)) FOR_X(d, cmp(div32(x, d + off), x / d));
    for (unsigned i = 0; i < sizeof BIG / sizeof *BIG; i++) { int32_t d = BIG[i];
        FOR_X(d, cmp(div32(x, d - off), x / d)); }
}

static void one_fd(int32_t d, int32_t off) {
    uint64_t M = fastdiv_m(d + off); FOR_X(d, cmp(fastdiv_s32(x, M, d), x / d));
}

static void run_fastdiv(int32_t off) {
    for (int32_t d = 2; d <= 1100; d++) { one_fd(d, off); one_fd(-d, -off); }
    for (int k = 1; k <= 30; k++) { one_fd(1 << k, off); one_fd(-(1 << k), -off); }
    for (unsigned i = 0; i < sizeof BIG / sizeof *BIG; i++) { one_fd(BIG[i], -off); one_fd(-BIG[i], off); }
}

static void run_self(void) {
    for (int32_t d = 2; d <= 1100; d++) { volatile int32_t dd = d; FOR_X(d, cmp(x / dd, x / d)); }
}

// The 3-op sign extension through fastdiv, exhaustive (v16, b) for b < 32.
static void run_signext(void) {
    for (unsigned b = 1; b < 32; b++) for (unsigned x = 0; x < 65536; x++) {
        int ref = (int)((uint32_t)x << (32 - b)) >> (32 - b);
        int32_t prod = (int32_t)((unsigned)x * (unsigned)multipliers[b]);
        cmp(fastdiv_s32(prod, divM[b], divisors[b]), ref);
    }
}

// The new CHKs are wired into check32 and pass on hashed inputs.
static void run_wired(void) {
    for (uint32_t i = 0; i < 1u << 16; i++) { check32(h32(i)); check32(i); check32(~i); }
    int seen = 0;
    for (int i = 0; i < nids; i++) {
        if (!names[i] || (!strstr(names[i], "div32") && !strstr(names[i], "fastdiv_s32"))) continue;
        seen++; bad += fails[i] != 0;
    }
    if (seen < 3) bad += 1000000;
}

int main(int argc, char **argv) {
    init_tables(); init_R();
    const char *m = argv[1];
    if (!strcmp(m, "div32")) run_div32(0);
    else if (!strcmp(m, "div32_wrong")) run_div32(1);
    else if (!strcmp(m, "fastdiv")) run_fastdiv(0);
    else if (!strcmp(m, "fastdiv_wrong")) run_fastdiv(1);
    else if (!strcmp(m, "self")) run_self();
    else if (!strcmp(m, "signext")) run_signext();
    else if (!strcmp(m, "wired")) run_wired();
    else return 2;
    printf("bad=%ld\n", bad);
    return 0;
}
"""


@pytest.fixture(scope="module")
def driver(tmp_path_factory) -> Path:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    d = tmp_path_factory.mktemp("bithacks_div")
    src = d / "drv.c"
    src.write_text(_DRIVER.replace("@SRC@", str(_SRC)))
    out = d / "drv"
    cmd = [
        "clang", "-O2", "-w", *_UBSAN,
        str(src), "-o", str(out), "-lm",
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:2000]
    yield out
    shutil.rmtree(d, ignore_errors=True)


def _bad(driver: Path, mode: str) -> int:
    r = subprocess.run([str(driver), mode], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[:2000]
    return int(r.stdout.strip().split("=")[1])


def test_control_reference_matches_itself(driver):
    assert _bad(driver, "self") == 0


def test_control_wrong_divisor_is_detected(driver):
    """Sensitivity: off-by-one divisor must mismatch, else the oracle is blind."""
    assert _bad(driver, "div32_wrong") > 0
    assert _bad(driver, "fastdiv_wrong") > 0


def test_div32_matches_c_division(driver):
    assert _bad(driver, "div32") == 0


def test_fastdiv_s32_matches_c_division(driver):
    """Adversarial: +-2 (add-back path), powers of two, INT_MIN dividend, INT_MAX."""
    assert _bad(driver, "fastdiv") == 0


def test_signext_3op_via_fastdiv_exhaustive(driver):
    assert _bad(driver, "signext") == 0


def test_checks_wired_into_check32(driver):
    assert _bad(driver, "wired") == 0


def test_regression_fuzz_mode_accepts_edge_inputs(tmp_path):
    """Fuzz mode aborts on any MISMATCH; INT_MIN / -1 / 0 words must not trip the new checks."""
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    out = tmp_path / "bh"
    cmd = ["clang", "-O1", "-w", *_UBSAN, str(_SRC), "-o", str(out), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:2000]
    # Bytes 8..11 become check32's v; the rest stays zero.
    for w in (0x80000000, 0xFFFFFFFF, 0, 1, 2, 7, 0x7FFFFFFF, 0xFFFFFFFE):
        buf = bytes(8) + w.to_bytes(8, "little") + bytes(8)
        r = subprocess.run([str(out)], input=buf, capture_output=True, timeout=10)
        assert r.returncode == 0, (hex(w), r.stderr[:500])
