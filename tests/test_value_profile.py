"""--value-profile: comparison operand distance as synthetic edges.

shim   per site: ids for popcount(a^b) and bitlen(|a-b|), once per run
shm    ValueProfileShm: [u32 head][u32 on]; Python flips `on`
mode   off | on | stall (switched on at the first stall, kept on)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage, ValueProfileShm
from fuzzer_tool.core.elf import detect_vp_capable
from fuzzer_tool.services.value_profile import VP_ENV, ValueProfile, ValueProfileChannel
from tests.conftest import requires_clang

SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
MAP_ENTRIES = 8192
MAGIC = 0x12345678
LOOP = 100

# argv[1] is a hex u32 compared against MAGIC; argv[2] repeats the compare.
_SRC = r"""
#include <stdint.h>
#include <stdlib.h>
volatile int sink;
__attribute__((noinline)) static void check(uint32_t x) {
    if (x == 0x12345678u) sink = 1;
}
int main(int argc, char **argv) {
    uint32_t x = (uint32_t)strtoul(argc > 1 ? argv[1] : "0", 0, 16);
    int reps = argc > 2 ? atoi(argv[2]) : 1;
    for (int i = 0; i < reps; i++) check(x);
    return 0;
}
"""


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("vp")
    src = d / "vp.c"
    src.write_text(_SRC)
    obj, out = d / "vp.o", d / "vp"
    cc = [
        "clang", "-O1", "-fno-omit-frame-pointer", "-D__AFL_CTX_SENSITIVE=0", "-D__AFL_CMPLOG=1",
        "-fsanitize-coverage=trace-pc-guard,trace-cmp", "-include", str(SHIM),
        "-c", str(src), "-o", str(obj),
    ]  # fmt: skip
    r = subprocess.run(cc, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    r = subprocess.run(["clang", str(obj), "-o", str(out), "-ldl"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return str(out)


@pytest.fixture
def vp():
    seg = ValueProfileShm()
    yield seg
    seg.cleanup()


def _run(exe, vp, x, reps=1):
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(MAP_ENTRIES))
        env.pop("LD_PRELOAD", None)
        if vp is not None:
            env[VP_ENV] = vp.env_id
        r = subprocess.run([exe, f"{x:x}", str(reps)], env=env, capture_output=True, timeout=30)
        assert r.returncode == 0, r.stderr[-800:]
        return cov.get_edge_counts()
    finally:
        cov.cleanup()


def _vp_ids(exe, vp, x, reps=1):
    """Ids this run adds over the same input run without the segment."""
    base = _run(exe, None, x, reps)
    return {e: c for e, c in _run(exe, vp, x, reps).items() if e not in base}


@requires_clang
class TestShim:
    def test_closer_operand_is_new_coverage(self, exe, vp):
        vp.enable()
        far = _vp_ids(exe, vp, 0)
        near = _vp_ids(exe, vp, MAGIC ^ 1)  # one bit away

        assert far and near
        assert set(near) - set(far), "a closer operand reached no new distance id"

    def test_off_word_adds_nothing(self, exe, vp):
        """Falsification: with `on` clear the two inputs look identical."""
        assert _vp_ids(exe, vp, 0) == {}
        assert _vp_ids(exe, vp, MAGIC ^ 1) == {}

    def test_loop_does_not_move_hit_counts(self, exe, vp):
        """Adversarial: a compare in a hot loop marks each distance id once."""
        vp.enable()
        counts = _vp_ids(exe, vp, MAGIC ^ 1, reps=LOOP)
        assert counts and set(counts.values()) == {1}

    def test_enable_mid_campaign(self, exe, vp):
        """stall mode: flipping `on` takes effect on the next exec."""
        assert _vp_ids(exe, vp, MAGIC ^ 1) == {}
        vp.enable()
        assert _vp_ids(exe, vp, MAGIC ^ 1)

    def test_capability_marker(self, exe):
        assert detect_vp_capable(exe) is True


class TestChannel:
    def test_off_opens_nothing(self, monkeypatch):
        monkeypatch.delenv(VP_ENV, raising=False)
        assert ValueProfileChannel.open(ValueProfile.OFF) is None
        assert VP_ENV not in os.environ

    def test_on_enables_at_open(self, monkeypatch):
        monkeypatch.delenv(VP_ENV, raising=False)
        ch = ValueProfileChannel.open(ValueProfile.ON)
        try:
            assert os.environ[VP_ENV] == ch.shm.env_id
            assert ch.shm.is_on()
        finally:
            ch.cleanup()

    def test_stall_waits_then_stays_on(self, monkeypatch):
        monkeypatch.delenv(VP_ENV, raising=False)
        ch = ValueProfileChannel.open(ValueProfile.STALL)
        try:
            assert not ch.shm.is_on()
            ch.on_stall()
            assert ch.shm.is_on()
            ch.on_stall()  # a second stall changes nothing
            assert ch.shm.is_on()
        finally:
            ch.cleanup()


class TestWiring:
    def test_cli_choices(self, monkeypatch):
        from tests.test_lightweight_mode import _cli

        args, _ = _cli(monkeypatch, "--value-profile", "stall")
        assert args.value_profile == "stall"
        args, _ = _cli(monkeypatch)
        assert args.value_profile == "off"

    def test_cli_rejects_unknown(self, monkeypatch):
        from tests.test_lightweight_mode import _cli

        with pytest.raises(SystemExit):
            _cli(monkeypatch, "--value-profile", "sometimes")

    def test_stall_enter_switches_on(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        calls = []
        f = SimpleNamespace(_vp=SimpleNamespace(on_stall=lambda: calls.append(1)))
        Fuzzer._vp_on_stall(f)
        assert calls == [1]
        Fuzzer._vp_on_stall(SimpleNamespace(_vp=None))  # off: no-op
