"""Regression: trace-cmp produced nothing usable on optimized targets.

Three separate defects made the compiler-IR layer inert, each silent:

1. **The callbacks were swallowed.** ``-fsanitize-coverage`` links
   compiler-rt's sancov runtime, which ships *weak no-op definitions* of
   ``__sanitizer_cov_trace_{,const_}cmp{1,2,4,8}``. The executable is
   searched before LD_PRELOAD libraries in the global symbol lookup order,
   so those stubs win and the preloaded shim is never reached. Measured on
   an -O2 trace-cmp build of ``cmplog_exercise.c``: 20 call sites in the
   binary, 4 CMP lines at runtime (all from the libc memchr interceptor).
   The shim must be *linked in* -- a strong definition beats the weak stub.

2. **trace-cmp cannot recover memcmp constants at all.** SanitizerCoverage
   instruments IR ``icmp``; clang's ExpandMemCmp is a CodeGen pass that runs
   after it. So the comparison trace-cmp sees is ``memcmp_result == 0``, and
   it logs the literal pair ``(0, 1)``; only later does the memcmp become
   ``cmpl $0x6C504D43,(%rbx)``. 11 of 20 logged pairs on that build were
   that degenerate pair -- the same "looks unsolved but is already solved"
   pool pollution the log_cmp result filter exists to prevent.

3. **The fix is ``-fno-builtin-<fn>``,** which keeps the call at the PLT so
   the libc layer sees the real operands, with every other -O2 optimization
   intact. It is complementary to trace-cmp, not a replacement: trace-cmp
   still catches genuine inline integer compares and switch dispatch that
   the libc layer cannot see.

The shim is now compiled into the target with ``-D__AFL_CMPLOG=1`` rather
than linked as a separate ``cmplog_shim.o``. "Linked in" in the numbers
below means that gate is on.

Constants from ``cmplog_exercise.c`` reaching the pair pool, seed
``AAAAAAAAAAAAAAAA``:

    -O2                                    0/10   (5 operands)
    -O2 + trace-cmp, preloaded             0/10   (5 operands)
    -O2 + trace-cmp, linked                0/10  (12 operands)
    -O0                                    9/10  (21 operands)
    -O2 + no-builtin                      10/10  (24 operands)
    -O2 + no-builtin + trace-cmp, linked  10/10  (36 operands)
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.conftest import requires_clang

REPO = Path(__file__).parent.parent
SCRIPT = REPO / "tools" / "build_targets.sh"
AFL_SHIM = REPO / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
EXERCISE = REPO / "targets" / "cmplog_exercise.c"

# Every magic constant cmplog_exercise.c compares against.
CONSTANTS = [
    b"CMPl",
    b"OG!",
    b"fuzz",
    b"CMPLOGfuzz",
    b"CMPLOG_ACTIVE",
    b"TEST_",
    b"COMPARE_A",
    b"COMPARE_B",
    b"FUZZ_",
    b"BENCH_",
]

NOBUILTIN = [
    "-fno-builtin-memcmp",
    "-fno-builtin-bcmp",
    "-fno-builtin-strcmp",
    "-fno-builtin-strncmp",
    "-fno-builtin-strcasecmp",
    "-fno-builtin-strncasecmp",
    "-fno-builtin-memchr",
    "-fno-builtin-strstr",
    "-fno-builtin-memmem",
]


# ── Script-level wiring ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


def test_tracecmp_is_on_by_default(script_text: str):
    """It was WITH_TRACECMP=0, so the whole layer never built."""
    assert re.search(r"^WITH_TRACECMP=1", script_text, re.MULTILINE)
    assert "--no-tracecmp" in script_text, "no opt-out for the new default"


def test_nobuiltin_flags_are_defined_and_used(script_text: str):
    assert "NOBUILTIN_CMP=" in script_text
    body = script_text[script_text.index("build_tracecmp_targets()") :]
    assert "$NOBUILTIN_CMP" in body, (
        "trace-cmp targets built without -fno-builtin: memcmp folds at -O2 "
        "and the constants never reach the pool"
    )


def test_tracecmp_targets_link_the_shim(script_text: str):
    """LD_PRELOAD loses to the runtime's weak no-op stubs."""
    body = script_text[script_text.index("build_tracecmp_targets()") :]
    body = body[: body.index("\n}\n")]
    assert "$CMPLOG_CFLAGS" in body and "$CMPLOG_LIBS" in body, (
        "the cmplog layer must be compiled into trace-cmp targets, not preloaded"
    )
    assert "cmplog_obj" not in body, (
        "the separate cmplog object is gone; it now lives in afl_shim.c"
    )


def test_cmplog_exercise_gets_a_tracecmp_build(script_text: str):
    """The target the search operators are measured on was never covered."""
    body = script_text[script_text.index("build_tracecmp_targets()") :]
    assert "cmplog_exercise" in body[: body.index("\n}\n")]


# ── Compiled behaviour ───────────────────────────────────────────────


def _compile(tmpdir, name, flags, link_shim):
    out = os.path.join(tmpdir, name)
    cmd = ["clang", "-O2", "-g", *flags]
    if link_shim:
        cmd += ["-D__AFL_CMPLOG=1"]
    cmd += ["-include", str(AFL_SHIM), "-o", out, str(EXERCISE)]
    if link_shim:
        cmd += ["-ldl"]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr.decode()[:400]
    return out


def _pool(tmpdir, exe, preload=None):
    """Run the target and return the operands that reached the pair pool."""
    log = os.path.join(tmpdir, os.path.basename(exe) + ".cmplog")
    if os.path.exists(log):
        os.unlink(log)
    env = dict(os.environ, _CMPLOG_OUT=log)
    env.pop("LD_PRELOAD", None)
    env.pop("ASAN_OPTIONS", None)
    if preload:
        env["LD_PRELOAD"] = preload
    subprocess.run([exe], input=b"A" * 16, capture_output=True, env=env, timeout=60)

    operands: set[bytes] = set()
    if os.path.exists(log):
        with open(log) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "CMP":
                    for hexstr in (parts[1], parts[2]):
                        with contextlib.suppress(ValueError):
                            operands.add(bytes.fromhex(hexstr))
    return operands


def _found(operands):
    return [c for c in CONSTANTS if any(c in op for op in operands)]


@requires_clang
def test_o2_alone_starves_the_pool():
    """Baseline: the defect. -O2 folds every comparison away."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = _compile(tmp, "plain", [], link_shim=True)
        assert _found(_pool(tmp, exe)) == [], (
            "unexpected: -O2 exposed constants without -fno-builtin"
        )


@requires_clang
def test_tracecmp_alone_does_not_recover_memcmp_constants():
    """Recorded negative result -- trace-cmp is not the fix for these.

    Pinned so nobody re-derives it and adds trace-cmp expecting an uplift.
    """
    with tempfile.TemporaryDirectory() as tmp:
        exe = _compile(tmp, "tc", ["-fsanitize-coverage=trace-cmp,trace-pc-guard"], link_shim=True)
        assert _found(_pool(tmp, exe)) == [], (
            "trace-cmp recovered memcmp constants -- if clang moved "
            "ExpandMemCmp before SanitizerCoverage, revisit $NOBUILTIN_CMP"
        )


@requires_clang
def test_nobuiltin_recovers_every_constant():
    """The fix, in isolation."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = _compile(tmp, "nb", NOBUILTIN, link_shim=True)
        missing = set(CONSTANTS) - set(_found(_pool(tmp, exe)))
        assert not missing, f"constants still absent from the pool: {sorted(missing)}"


@requires_clang
def test_combined_build_beats_either_layer_alone():
    """The two layers are complementary: more operands than either alone."""
    with tempfile.TemporaryDirectory() as tmp:
        nb = _pool(tmp, _compile(tmp, "nb2", NOBUILTIN, link_shim=True))
        both = _pool(
            tmp,
            _compile(
                tmp,
                "both",
                [*NOBUILTIN, "-fsanitize-coverage=trace-cmp,trace-pc-guard"],
                link_shim=True,
            ),
        )
        assert len(_found(both)) == len(CONSTANTS)
        assert len(both) > len(nb), (
            f"trace-cmp added nothing on top of no-builtin "
            f"({len(both)} vs {len(nb)} operands) -- the callbacks are "
            "probably resolving to the runtime's weak no-op stubs again"
        )


@requires_clang
def test_linked_shim_beats_preloaded_shim():
    """The weak-stub shadowing bug, measured directly."""
    with tempfile.TemporaryDirectory() as tmp:
        shim_so = os.path.join(tmp, "shim.so")
        r = subprocess.run(
            [
                "clang",
                "-shared",
                "-fPIC",
                "-O2",
                "-D__AFL_PRELOAD_ONLY",
                "-ldl",
                "-o",
                shim_so,
                str(AFL_SHIM),
            ],
            capture_output=True,
            timeout=120,
        )
        assert r.returncode == 0, r.stderr.decode()[:400]

        flags = [*NOBUILTIN, "-fsanitize-coverage=trace-cmp,trace-pc-guard"]
        preloaded = _pool(tmp, _compile(tmp, "pre", flags, link_shim=False), preload=shim_so)
        linked = _pool(tmp, _compile(tmp, "lnk", flags, link_shim=True))

        assert len(linked) > len(preloaded), (
            f"compiling the shim in gained nothing ({len(linked)} vs "
            f"{len(preloaded)} operands) -- expected the preloaded build to "
            "lose its trace-cmp callbacks to the runtime's weak stubs"
        )


@requires_clang
def test_nobuiltin_keeps_memcmp_a_real_call():
    """The mechanism behind the fix, not just its effect."""
    with tempfile.TemporaryDirectory() as tmp:
        folded = _compile(tmp, "folded", [], link_shim=False)
        kept = _compile(tmp, "kept", NOBUILTIN, link_shim=False)

        def calls_memcmp(path):
            out = subprocess.run(
                ["objdump", "-d", path], capture_output=True, text=True, timeout=120
            ).stdout
            return bool(re.search(r"call.*<memcmp(@plt)?>", out))

        assert not calls_memcmp(folded), "-O2 unexpectedly kept the memcmp call"
        assert calls_memcmp(kept), "-fno-builtin-memcmp did not keep the call"
