"""Tests for configurable n-gram history depth (__AFL_NGRAM_K).

The shim is compiled into the driver TU via ``-include``, so the tests can
call ``__afl_map_edge`` directly and recover each edge id from the rolling
path hash (``e = acc_after ^ (acc_before * 31)``) -- no SHM-set ambiguity,
no dependence on XOR-space collisions. All behavioral builds run with
``__AFL_CTX_SENSITIVE=0`` so the n-gram variable is isolated from caller
context.

k=2 must stay byte-identical to the pre-ngram shim (exported
``__afl_prev_loc``, XOR ids): corpora and resume state depend on it. Only
k>2 may change ids.
"""

import os
import shutil
import subprocess

import pytest

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)

# The differing block sits TWO positions before the final block: its value
# lands inside the k=3 trigram window but outside the k=2 pair window, so
# the FINAL edge id separates the two grammars exactly.
_DRIVER = """
#include <stdlib.h>
#include <stdio.h>
static uint32_t emit(uint32_t cur) {
    uint32_t before = __afl_path_hash_acc;
    __afl_map_edge(cur);
    return __afl_path_hash_acc ^ (before * 31u);
}
int main(int argc, char **argv) {
    (void)argc;
    switch (argv[1][0]) {
    case 'p': {
        unsigned long s = strtoul(argv[2], NULL, 10);
        printf("%u\\n", emit(41));
        printf("%u\\n", emit((uint32_t)s));
        printf("%u\\n", emit(29));
        printf("%u\\n", emit(15));
        break;
    }
    case 'r':
        for (int i = 0; i < 3; i++) printf("%u\\n", emit(41 + (uint32_t)i));
        __afl_map_reset();
        for (int i = 0; i < 3; i++) printf("%u\\n", emit(41 + (uint32_t)i));
        break;
    case 'f':
        for (uint32_t i = 1; i <= 4000; i++) (void)emit(i);
        break;
    default:
        return 2;
    }
    return 0;
}
"""

CTX_OFF = ["-D__AFL_CTX_SENSITIVE=0"]


def _cc():
    return shutil.which("clang") or shutil.which("gcc")


def _compile(tmp_path, name, extra_flags, cc):
    src = tmp_path / f"{name}.c"
    src.write_text(_DRIVER)
    exe = tmp_path / name
    r = subprocess.run(
        [cc, "-O1", "-g", *extra_flags, "-include", SHIM, "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    return r, str(exe)


def _run_edges(exe, args, shm_id=None, map_size="1024"):
    """Run one driver mode, return stdout edge ids as int list."""
    env = {**os.environ, "AFL_MAP_SIZE": map_size}
    if shm_id:
        env["__AFL_SHM_ID"] = shm_id
    else:
        env.pop("__AFL_SHM_ID", None)
    r = subprocess.run([exe, *args], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [int(line) for line in r.stdout.split()]


def _nm_symbols(path):
    """Exact symbol-name set — substring matching would confuse
    __afl_prev_loc with __afl_prev_locs."""
    out = subprocess.run(["nm", path], capture_output=True, text=True).stdout
    return {line.split()[-1] for line in out.splitlines() if line.strip()}


@pytest.fixture(scope="module")
def cc():
    compiler = _cc()
    if compiler is None:
        pytest.skip("no C compiler")
    return compiler


class TestShimBuildMatrix:
    def test_marker_symbol_encodes_k(self, cc, tmp_path):
        for define, expected in [(None, 2), ("2", 2), ("3", 3)]:
            flags = list(CTX_OFF)
            if define is not None:
                flags.append(f"-D__AFL_NGRAM_K={define}")
            r, exe = _compile(tmp_path, f"k{define or 'default'}", flags, cc)
            assert r.returncode == 0, r.stderr
            assert f"__afl_ngram_k_{expected}" in _nm_symbols(exe)

    def test_default_and_explicit_k2_are_identical(self, cc, tmp_path):
        outs = []
        for name, flags in [
            ("dflt", list(CTX_OFF)),
            ("expl", [*CTX_OFF, "-D__AFL_NGRAM_K=2"]),
        ]:
            r, exe = _compile(tmp_path, name, flags, cc)
            assert r.returncode == 0, r.stderr
            outs.append(_run_edges(exe, ["p", "5"]))
        assert outs[0] == outs[1]

    def test_k2_exports_prev_loc_k3_does_not(self, cc, tmp_path):
        _, k2 = _compile(tmp_path, "abi2", [*CTX_OFF, "-D__AFL_NGRAM_K=2"], cc)
        _, k3 = _compile(tmp_path, "abi3", [*CTX_OFF, "-D__AFL_NGRAM_K=3"], cc)
        assert "__afl_prev_loc" in _nm_symbols(k2)
        # Ring statics are file-scope; the exported legacy word must not
        # survive a k>2 build or old tooling would read a stale layout.
        assert "__afl_prev_loc" not in _nm_symbols(k3)


class TestNgramDiscrimination:
    def test_k3_distinguishes_history_k2_is_blind(self, cc, tmp_path):
        """Two paths sharing their final hop: k=2 must give the same final
        edge id, k=3 must split it (falsifies 'same final edge' blindness).
        Needs an attached SHM or __afl_map_edge early-returns on the null
        area check and every recovered id collapses to 0."""
        from fuzzer_tool.adapters.shm import ShmCoverage

        finals = {}
        shm = ShmCoverage(size=1024)
        try:
            for k in ("2", "3"):
                _, exe = _compile(tmp_path, f"disc{k}", [*CTX_OFF, f"-D__AFL_NGRAM_K={k}"], cc)
                a = _run_edges(exe, ["p", "5"], shm_id=shm.env_id)
                b = _run_edges(exe, ["p", "7"], shm_id=shm.env_id)
                assert len(a) == len(b) == 4
                assert any(a), "ids must be nonzero once attached"
                finals[k] = (a[-1], b[-1])
        finally:
            shm.cleanup()
        assert finals["2"][0] == finals["2"][1], "k=2 must stay blind"
        assert finals["3"][0] != finals["3"][1], "k=3 must split the history"

    def test_reset_clears_the_ring(self, cc, tmp_path):
        """Same sequence around __afl_map_reset: both halves must produce
        identical edge lists or the ring leaked history across iterations."""
        from fuzzer_tool.adapters.shm import ShmCoverage

        _, exe = _compile(tmp_path, "resetshm", [*CTX_OFF, "-D__AFL_NGRAM_K=3"], cc)
        shm = ShmCoverage(size=1024)
        try:
            out = _run_edges(exe, ["r"], shm_id=shm.env_id)
        finally:
            shm.cleanup()
        first, second = out[:3], out[3:]
        assert first == second


class TestConfigGuards:
    def test_below_min_k_is_a_build_error(self, cc, tmp_path):
        r, _ = _compile(tmp_path, "bad_low", [*CTX_OFF, "-D__AFL_NGRAM_K=1"], cc)
        assert r.returncode != 0
        assert "__AFL_NGRAM_K must be >= 2" in r.stderr

    def test_oversized_k_is_a_build_error(self, cc, tmp_path):
        r, _ = _compile(tmp_path, "bad_high", [*CTX_OFF, "-D__AFL_NGRAM_K=5000"], cc)
        assert r.returncode != 0
        assert "__AFL_NGRAM_K too large" in r.stderr


class TestDropCounterAtK3:
    def test_scattered_fnv_ids_still_report_drops(self, cc, tmp_path):
        """Adversarial: FNV scatters ids across the u32 space, unlike the
        dense guard-era range. The bounded probe must still count losses."""
        from fuzzer_tool.adapters.shm import ShmCoverage

        _, exe = _compile(tmp_path, "fillk3", [*CTX_OFF, "-D__AFL_NGRAM_K=3"], cc)
        shm = ShmCoverage(size=1024)
        try:
            env = {**os.environ, "__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": "1024"}
            proc = subprocess.run([exe, "f"], env=env, capture_output=True)
            state = (
                f"rc={proc.returncode} "
                f"edge_count={shm.read_edge_count()} "
                f"dropped={shm.read_dropped_edges()} "
                f"stderr={proc.stderr[-200:]!r}"
            )
            assert shm.read_dropped_edges() > 0, state
        finally:
            shm.cleanup()
