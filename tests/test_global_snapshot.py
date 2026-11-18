"""In-process global snapshot restore (``--snapshot-restore globals``).

Ported from wtf's snapshot model, scoped to the target ``.so``'s writable
data: ``.data``/``.bss`` are copied after load and written back before every
direct-mode exec, so one input's global state cannot leak into the next.

The shim's own globals live in ``afl_shim_data``/``afl_shim_bss`` (afl_shim.c
``#pragma clang section``) and are never restored.
"""

from __future__ import annotations

import ast
import re
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# One .bss counter and one .data flag: rc = hits * 10 + (flag intact).
PROBE_SRC = r"""
#include <stddef.h>
static int hits;
int magic = 7;
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    hits++;
    if (len && buf[0] == 'S')
        magic = 0;
    return hits * 10 + (magic == 7);
}
"""

PRAGMA_RE = re.compile(r"^#pragma clang section.*$", re.MULTILINE)


def _build(tmp: Path, name: str, shim: Path, extra: tuple[str, ...] = ()) -> Path:
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    src = tmp / f"{name}.c"
    src.write_text(PROBE_SRC)
    so = tmp / f"{name}.so"
    proc = subprocess.run(
        ["clang", "-O1", "-fPIC", "-shared", *extra, "-o", str(so), str(src), str(shim)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"probe build failed: {proc.stderr[-400:]}")
    return so


@pytest.fixture(scope="module")
def probe_so(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("snap"), "probe", SHIM)


@pytest.fixture(scope="module")
def legacy_so(tmp_path_factory) -> Path:
    """Shim built without the section pragma: its globals sit in .data/.bss."""
    tmp = tmp_path_factory.mktemp("snap_legacy")
    shim = tmp / "afl_shim.c"
    shim.write_text(PRAGMA_RE.sub("", SHIM.read_text()))
    return _build(tmp, "legacy", shim)


# Child process: each run gets its own dlopen handle and signal state.
_PROBE = textwrap.dedent(
    """
    import sys
    from fuzzer_tool.adapters.inprocess import InProcessRunner
    from fuzzer_tool.core.run_mode import StateRestore

    so, restore, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    r = InProcessRunner(
        so, "fuzz_test", timeout=5.0,
        direct=(mode == "direct"), direct_lite=(mode == "lite"),
        restore=StateRestore(restore),
    )
    rcs = [r.run_one(b"S")[0] for _ in range(3)]
    r.stop()
    print(repr(rcs))
    """
)


def _probe(
    so: Path, restore: str, mode: str = "lite", env: dict | None = None
) -> tuple[list[int], str]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(so), restore, mode],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return ast.literal_eval(proc.stdout.strip().splitlines()[-1]), proc.stderr


def _sections(so: Path) -> dict[str, tuple[int, int]]:
    """name -> (addr, size), parsed by readelf: an oracle independent of core.elf."""
    if shutil.which("readelf") is None:
        pytest.skip("readelf not available")
    out = subprocess.run(["readelf", "-SW", str(so)], capture_output=True, text=True).stdout
    secs = {}
    for m in re.finditer(r"\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)", out):
        secs[m.group(1)] = (int(m.group(2), 16), int(m.group(3), 16))
    return secs


def _overlaps(span: tuple[int, int], sec: tuple[int, int]) -> bool:
    return span[0] < sec[0] + sec[1] and sec[0] < span[0] + span[1]


class TestDataRanges:
    def test_covers_target_globals(self, probe_so):
        from fuzzer_tool.core.elf import data_ranges

        dr = data_ranges(probe_so.read_bytes())
        secs = _sections(probe_so)

        assert dr is not None and dr.shim_isolated
        for name in (".data", ".bss"):
            addr, size = secs[name]
            assert any(s <= addr and addr + size <= s + n for s, n in dr.spans), name

    def test_excludes_shim_and_linker_state(self, probe_so):
        """Adversarial: restoring any of these clobbers the shim or the loader."""
        from fuzzer_tool.core.elf import data_ranges

        dr = data_ranges(probe_so.read_bytes())
        secs = _sections(probe_so)
        banned = ("afl_shim_data", "afl_shim_bss", ".got", ".got.plt", ".dynamic", ".init_array")

        for name in banned:
            if name not in secs:
                continue
            assert not any(_overlaps(s, secs[name]) for s in dr.spans), name

    def test_legacy_shim_not_isolated(self, legacy_so):
        from fuzzer_tool.core.elf import data_ranges

        dr = data_ranges(legacy_so.read_bytes())

        assert dr is not None and not dr.shim_isolated

    def test_spans_sorted_disjoint(self, probe_so):
        from fuzzer_tool.core.elf import data_ranges

        spans = data_ranges(probe_so.read_bytes()).spans

        for (a, n), (b, _) in zip(spans, spans[1:], strict=False):
            assert a + n < b

    @pytest.mark.parametrize(
        "blob",
        [
            b"",
            b"\x7fELF",
            b"MZ" + b"\0" * 100,
            # ELF64 LE header whose section table points past the file.
            b"\x7fELF\x02\x01" + b"\0" * 34 + struct.pack("<Q", 2**64 - 1) + b"\0" * 16,
        ],
    )
    def test_malformed_returns_none(self, blob):
        """Adversarial: garbage headers must not raise or yield spans."""
        from fuzzer_tool.core.elf import data_ranges

        assert data_ranges(blob) is None

    def test_truncated_real_header(self, probe_so):
        """Adversarial: a real image cut short keeps its header but loses sections."""
        from fuzzer_tool.core.elf import data_ranges

        assert data_ranges(probe_so.read_bytes()[:4096]) is None


class TestRunnerRestore:
    def test_control_off_leaks(self, probe_so):
        """Control: without restore the probe must see its own leak, else it proves nothing."""
        rcs, _ = _probe(probe_so, "off")

        assert len(set(rcs)) == len(rcs)

    @pytest.mark.parametrize("mode", ["lite", "direct"])
    def test_globals_restored_each_exec(self, probe_so, mode):
        """Falsification: every exec must see the state the first one saw."""
        rcs, _ = _probe(probe_so, "globals", mode)
        fresh = 1 * 10 + 0  # first exec: hits=1, magic cleared by 'S'

        assert rcs == [fresh] * len(rcs)

    def test_legacy_shim_refused(self, legacy_so):
        """Adversarial: an old shim would be clobbered, so restore stays off and says so."""
        rcs, err = _probe(legacy_so, "globals")

        assert len(set(rcs)) == len(rcs)
        assert "snapshot-restore" in err


class TestGlobalSnapshot:
    def test_restore_reverts_and_bounds(self):
        """Driver level, on a plain buffer: bytes written after capture are reverted."""
        import ctypes

        from fuzzer_tool.adapters.global_snapshot import GlobalSnapshot

        buf = ctypes.create_string_buffer(b"abcdefgh")
        base = ctypes.addressof(buf)
        snap = GlobalSnapshot(base, ((0, 4), (6, 2)))
        ctypes.memmove(base, b"XXXXXXXX", 8)

        snap.restore()

        assert buf.raw[:8] == b"abcdXXgh"
        assert snap.size == 6


_ASAN_FLAGS = ("-fsanitize=address", "-fsanitize-recover=address", "-lasan")
_ASAN_REPORT = "ERROR: AddressSanitizer"


@pytest.fixture(scope="module")
def asan_run(tmp_path_factory) -> tuple[Path, dict]:
    """ASAN probe + the env the fuzzer runs it under (libasan preloaded, recover on)."""
    import os

    so = _build(tmp_path_factory.mktemp("snap_asan"), "asan", SHIM, _ASAN_FLAGS)
    ldd = subprocess.run(["ldd", str(so)], capture_output=True, text=True).stdout
    m = re.search(r"(\S*libasan\.so\S*) => (\S+)", ldd)
    if m is None:
        pytest.skip("libasan runtime not found")
    env = dict(os.environ)
    env["LD_PRELOAD"] = m.group(2)
    env["ASAN_OPTIONS"] = "halt_on_error=0:detect_leaks=0:verify_asan_link_order=0"
    return so, env


class TestAsanRedzones:
    def test_control_off_no_reports(self, asan_run):
        """Control: the probe itself is clean, so a report can only come from restore."""
        so, env = asan_run
        _, err = _probe(so, "off", env=env)

        assert _ASAN_REPORT not in err

    def test_regression_asan_restore_skips_redzones(self, asan_run):
        """Restore wrote over ASAN global redzones: 3 reports per exec, 800 -> 130 eps on libbase64."""
        so, env = asan_run
        rcs, err = _probe(so, "globals", env=env)
        fresh = 1 * 10 + 0

        assert _ASAN_REPORT not in err
        assert rcs == [fresh] * len(rcs)


class _FakeShadow:
    """Per-granule ASAN shadow model: 0 clean, 1..7 partial, 0xf9 redzone."""

    def __init__(self, shadow: list[int]):
        self.bad = [not (s == 0 or (s < 8 and i % 8 < s)) for s in shadow for i in range(8)]

    def region(self, beg: int, size: int) -> int:
        return next((a for a in range(beg, beg + size) if self.bad[a]), 0)

    def addr(self, a: int) -> bool:
        return self.bad[a]


def _covered(spans) -> set[int]:
    return {a for s, n in spans for a in range(s, s + n)}


class TestAsanSplit:
    @pytest.mark.parametrize(
        "shadow",
        [
            [0, 0, 0, 0],
            [0, 4, 0xF9, 0xF9, 0, 0],
            [0xF9, 0xF9, 0, 0],
            [0, 0, 0xF9, 0xF9],
            [0xF9, 0xF9, 0xF9],
            [1, 0xF9, 7, 0xF9, 0, 3],
        ],
    )
    def test_keeps_exactly_addressable(self, shadow):
        """Adversarial: partial granules, leading/trailing/all-redzone spans."""
        from fuzzer_tool.adapters.global_snapshot import _AsanShadow

        fake = _FakeShadow(shadow)
        # Base 8 offsets relative spans; byte 0 of the model sits at absolute 8.
        model = _FakeShadow([0] + shadow)
        split = _AsanShadow(model.region, model.addr).addressable(8, ((0, len(fake.bad)),))
        want = {a for a, bad in enumerate(fake.bad) if not bad}

        assert _covered(split) == want
        assert all(n > 0 for _, n in split)
        for (a, n), (b, _) in zip(split, split[1:], strict=False):
            assert a + n < b

    def test_clean_spans_unchanged(self):
        """Falsification: nothing poisoned -> the spans pass through as given."""
        from fuzzer_tool.adapters.global_snapshot import _AsanShadow

        model = _FakeShadow([0] * 8)
        spans = ((0, 16), (24, 40))

        assert _AsanShadow(model.region, model.addr).addressable(0, spans) == spans


def _fuzz_kwargs(monkeypatch, tmp_path, extra: list[str]) -> dict:
    from fuzzer_tool.cli import commands

    exe = tmp_path / "t"
    exe.write_bytes(b"\x7fELF")
    exe.chmod(0o755)
    captured: dict = {}
    monkeypatch.setattr(commands, "Fuzzer", lambda **kw: captured.update(kw) or MagicMock())
    argv = ["fuzzer-tool", "fuzz", str(exe), "-d", str(tmp_path / "c"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    commands.main()
    return captured


class TestCli:
    def test_default_globals(self, monkeypatch, tmp_path):
        from fuzzer_tool.core.run_mode import StateRestore

        kw = _fuzz_kwargs(monkeypatch, tmp_path, [])

        assert kw["state_restore"] is StateRestore.GLOBALS

    def test_gate_disables(self, monkeypatch, tmp_path):
        """Falsification: the disable gate must reach the Fuzzer as OFF."""
        from fuzzer_tool.core.run_mode import StateRestore

        kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--no-snapshot-restore"])

        assert kw["state_restore"] is StateRestore.OFF

    @pytest.mark.parametrize(
        ("argv", "want"),
        [
            (["--no-snapshot-restore", "--snapshot-restore", "globals"], "globals"),
            (["--snapshot-restore", "globals", "--no-snapshot-restore"], "off"),
        ],
    )
    def test_gate_last_wins(self, monkeypatch, tmp_path, argv, want):
        """Adversarial: conflicting flags resolve to the last one, never silently mixed."""
        from fuzzer_tool.core.run_mode import StateRestore

        kw = _fuzz_kwargs(monkeypatch, tmp_path, argv)

        assert kw["state_restore"] is StateRestore(want)

    def test_fuzzer_default_globals(self):
        """API callers (not just the CLI) get restore on by default."""
        import inspect

        from fuzzer_tool.core.run_mode import StateRestore
        from fuzzer_tool.services.fuzzer import Fuzzer

        default = inspect.signature(Fuzzer.__init__).parameters["state_restore"].default

        assert default is StateRestore.GLOBALS

    def test_globals_reaches_fuzzer(self, monkeypatch, tmp_path):
        from fuzzer_tool.core.run_mode import StateRestore

        kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--snapshot-restore", "globals"])

        assert kw["state_restore"] is StateRestore.GLOBALS

    def test_rejects_unknown(self, monkeypatch, tmp_path):
        """Adversarial: a typo fails loudly instead of fuzzing with leaking state."""
        with pytest.raises(SystemExit):
            _fuzz_kwargs(monkeypatch, tmp_path, ["--snapshot-restore", "heap"])

    def test_fuzzer_passes_to_runner(self):
        """Wiring: both InProcessRunner constructions forward the mode."""
        import inspect

        from fuzzer_tool.services import fuzzer

        src = inspect.getsource(fuzzer)

        assert src.count("restore=self.state_restore") >= 2
