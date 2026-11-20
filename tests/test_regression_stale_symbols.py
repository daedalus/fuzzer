"""Regression: in-process crash stacks were symbolized against a rebuilt target.

UBSAN/ASAN symbolize in the fuzzer process by opening the target's *path*.
A rebuild that replaced the .so mid-run left the old image mapped but put a
new file at that path, so every frame resolved against the wrong binary.
Measured on ffmpeg_read_9.0.2_ubsan.so: a matroskadec.c:3496 float-cast
overflow was reported as avg_c / vc1_decode_frame, and func_hash (dedup)
was computed from those names.

``LoadedImage`` holds an fd to the file that was loaded and re-symbolizes
frames inside its range when the path no longer names that file.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.adapters import loaded_image
from fuzzer_tool.adapters.loaded_image import LoadedImage
from fuzzer_tool.core.sanitizer import crash_frames
from tests.conftest import requires_clang

_ROOT = Path(__file__).resolve().parent.parent
SHIM = _ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
SYMBOLIZER = shutil.which("llvm-symbolizer")

needs_symbolizer = pytest.mark.skipif(SYMBOLIZER is None, reason="llvm-symbolizer required")

# v2 puts a large function in front, so v1's target_fn offset lands in pad_fn.
PAD = "volatile int sink;\nint pad_fn(int x) {" + " sink += x;" * 400 + " return sink; }\n"
FUNCS = "int target_fn(int x) { return x * 3 + 1; }\nint other_fn(int x) { return x - 7; }\n"


def _so(tmp_path: Path, name: str, src: str, *flags: str) -> Path:
    c, so = tmp_path / f"{name}.c", tmp_path / f"{name}.so"
    c.write_text(src)
    r = subprocess.run(
        ["clang", "-O0", "-g", "-shared", "-fPIC", *flags, "-o", str(so), str(c)],
        capture_output=True,
    )
    if r.returncode != 0:
        pytest.skip(f"clang cannot build .so: {r.stderr[-200:]!r}")
    return so


def _symbolize_path(path: Path, offset: int) -> str:
    """Path-based symbolization: what the sanitizer runtime does."""
    out = subprocess.run(
        [SYMBOLIZER, f"--obj={path}", hex(offset)], capture_output=True, text=True
    ).stdout
    return out.splitlines()[0]


class _Loaded:
    """v1 loaded in this process, v2 built beside it, a stale report in hand."""

    def __init__(self, tmp_path: Path):
        self.path = _so(tmp_path, "lib", FUNCS)
        self.v2 = _so(tmp_path, "v2", PAD + FUNCS)
        self.image = LoadedImage.open(str(self.path))
        lib = ctypes.CDLL(str(self.path))
        self.image.locate()
        self.pc = ctypes.cast(lib.target_fn, ctypes.c_void_p).value + 1
        libc = ctypes.CDLL(None)
        self.outside = ctypes.cast(libc.printf, ctypes.c_void_p).value + 1
        self.text = (
            "libx.c:1:1: runtime error: boom\n"
            f"    #0 {self.pc:#x} in stale_name /old/x.c:9:9\n"
            f"    #1 {self.pc:#x} in stale_inline /old/x.c:8:8\n"
            f"    #2 {self.outside:#x} in keep_me /lib/y.c:1:1\n"
            "SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior libx.c:1:1\n"
        )

    def replace(self) -> None:
        os.replace(self.v2, self.path)


def _indices(text: str) -> list[int]:
    return [int(i) for i in re.findall(r"^\s*#(\d+) ", text, re.M)]


# ── unit: LoadedImage.fix ─────────────────────────────────────────────


@requires_clang
@needs_symbolizer
def test_regression_replaced_target_resymbolized(tmp_path):
    t = _Loaded(tmp_path)
    t.replace()

    # Control: by path, the old offset now names another function.
    assert _symbolize_path(t.path, t.pc - t.image.base) != "target_fn"

    fixed = t.image.fix(t.text)
    frames = crash_frames(fixed)
    assert frames[0].func == "target_fn"
    assert frames[0].path.endswith("lib.c")
    assert "stale_name" not in fixed and "stale_inline" not in fixed
    assert frames[-1].func == "keep_me"  # outside the image: untouched
    assert _indices(fixed) == list(range(len(frames)))
    assert fixed.splitlines()[-1].startswith("SUMMARY:")


@requires_clang
def test_untouched_target_text_unchanged(tmp_path):
    """Falsification: same file on disk -> the runtime's names stand."""
    t = _Loaded(tmp_path)
    assert t.image.fix(t.text) == t.text


@requires_clang
@needs_symbolizer
def test_deleted_target_resymbolized(tmp_path):
    """Adversarial: the path is gone, the held fd still has the old bytes."""
    t = _Loaded(tmp_path)
    t.path.unlink()
    assert crash_frames(t.image.fix(t.text))[0].func == "target_fn"


@requires_clang
def test_rewritten_in_place_strips_names(tmp_path):
    """Adversarial: same inode, new mtime -> old bytes unrecoverable; frames go
    unsymbolized instead of keeping a guess."""
    t = _Loaded(tmp_path)
    st = os.stat(t.path)
    os.utime(t.path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))

    fixed = t.image.fix(t.text)
    assert "stale_name" not in fixed and "stale_inline" not in fixed
    first = crash_frames(fixed)[0]
    assert first.func == "" and first.module == f"lib.so+{t.pc - t.image.base:#x}"
    assert crash_frames(fixed)[-1].func == "keep_me"


@requires_clang
def test_no_symbolizer_strips_names(tmp_path, monkeypatch):
    """Adversarial: no llvm-symbolizer -> unsymbolized, never stale names."""
    monkeypatch.setattr(loaded_image.shutil, "which", lambda _name: None)
    t = _Loaded(tmp_path)
    t.replace()

    fixed = t.image.fix(t.text)
    assert "stale_name" not in fixed
    assert crash_frames(fixed)[0].module.startswith("lib.so+0x")


def test_unmapped_path_is_inert(tmp_path):
    """Adversarial: never loaded (no mapping) -> fix is a no-op."""
    f = tmp_path / "never.so"
    f.write_bytes(b"\x7fELF")
    image = LoadedImage.open(str(f))
    image.locate()
    f.unlink()
    text = "    #0 0x1234 in a /x.c:1:1\n"
    assert image.fix(text) == text


# ── end to end: direct_lite runner, real UBSAN report ───────────────────

UB_V1 = """
#include <limits.h>
#include <stddef.h>
int fuzz_shm_run(const unsigned char *d, size_t n) {
    if (n && d[0] == 'O') { int x = INT_MAX; return x + (int)n; }
    return 0;
}
"""

_PROBE = r"""
import os, re, subprocess, sys
from fuzzer_tool.adapters.inprocess import InProcessRunner
path, v2 = sys.argv[1], sys.argv[2]
r = InProcessRunner(path, function_name="fuzz_shm_run", direct_lite=True, capture_stderr=True)
ino = os.stat(path).st_ino
os.replace(v2, path)                      # the rebuild lands mid-run
rc, err = r.run_one(b"O1")
m = re.search(r"#0 (0x[0-9a-f]+) in (\S+)", err)
pc = int(m.group(1), 16)
base = min(int(l.split("-")[0], 16) for l in open("/proc/self/maps")
           if l.split()[4] == str(ino) and int(l.split()[2], 16) == 0)
by_path = subprocess.run(["llvm-symbolizer", "--obj=" + path, hex(pc - base)],
                         capture_output=True, text=True).stdout.splitlines()[0]
print(m.group(2), by_path)
r.stop()
"""


@requires_clang
@needs_symbolizer
def test_regression_runner_resymbolizes_after_rebuild(tmp_path):
    from fuzzer_tool.cli import ldpreload_wrapper as wrapper

    runtime = wrapper._resolve_ubsan()
    if runtime is None:
        pytest.skip("clang UBSAN runtime not installed")
    flags = ("-O1", "-fsanitize=undefined", "-include", str(SHIM))
    path = _so(tmp_path, "ub", UB_V1, *flags)
    pad = "volatile int sink;\nint pad_fn(int x) {" + " sink += x;" * 400 + " return sink; }\n"
    v2 = _so(tmp_path, "ub_v2", pad + UB_V1, *flags)

    env = dict(os.environ, LD_PRELOAD=runtime, UBSAN_OPTIONS=":".join(wrapper._UBSAN_DEFAULTS))
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(path), str(v2)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-600:]
    reported, by_path = proc.stdout.split()

    assert by_path != "fuzz_shm_run"  # control: the bug reproduces by path
    assert reported == "fuzz_shm_run"
