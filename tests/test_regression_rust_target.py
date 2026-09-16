"""End-to-end check for rust/buggy_target + targets/rust_target.c.

This is the fuzzer's first non-C/C++ target. The risk it's guarding
against isn't "does Rust code compile" -- it's whether a crash originating
in a linked-in Rust static lib is still correctly detected through the
*same* mechanisms the fuzzer already relies on for C targets:

  - __afl_guarded_call's sigsetjmp/siglongjmp (in-process/direct_lite mode,
    driven via ctypes exactly as adapters/inprocess.py does it)
  - a plain subprocess exit-by-signal (persistent/subprocess mode)

Both are exercised here rather than assumed, because the two are not
equivalent: running the executable directly (no sigsetjmp ever
established) exposed a real pre-existing quirk in afl_shim.c's crash
handler -- __afl_crash_handler always siglongjmp's to __afl_jmp_buf, and
if nothing has set that buffer up yet (true for main()'s SIGILL/SIGSEGV
path when nothing calls __afl_guarded_call first) the jump target is
whatever garbage was last on the stack. Confirmed to be pre-existing and
not specific to this target: targets/test_target.c's own 'S' trigger
(a NULL function-pointer call), built and run the same standalone way,
crashes the same way for the same reason. Documented here rather than
"fixed" because afl_shim.c is explicitly out of scope for this pass (see
docs/handover/handover_rust_target_2026-09-15.md) and because it does not
affect either real execution mode: a subprocess's exit status is a signal
either way, and __afl_guarded_call is unaffected since it always runs with
its jmp buf already set up before the target executes.

This suite also had a real bug of its own, now fixed: an earlier version
built targets/rust_target.c with gcc. It linked, ran, and every
crash-detection assertion here passed -- but gcc's -fsanitize-coverage=
has no trace-pc-guard variant, so the wrapper carried zero instrumented
call sites, silently. Crash detection doesn't depend on coverage
instrumentation, which is exactly why nothing here caught it. Fixtures
now mirror build_targets.sh's _pick_cc (prefer clang), and
test_wrapper_actually_has_instrumented_call_sites checks the built binary
directly rather than trusting a clean build.

Optional nightly-rustc coverage: tests below tagged with the
"nightly" reason skip unless a nightly toolchain is available (env
RUSTC_NIGHTLY, or the default path tools/fetch_rust_nightly.sh installs
to) -- verify real per-basic-block instrumentation *inside* the crate
itself, not just the wrapper. Skipped in CI unless someone has
deliberately opted into that unofficial third-party toolchain (see
tools/fetch_rust_nightly.sh's header comment before doing so); nothing
else in this file depends on it.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATE_DIR = os.path.join(ROOT, "rust", "buggy_target")
WRAPPER_SRC = os.path.join(ROOT, "targets", "rust_target.c")
SHIM = os.path.join(ROOT, "src", "fuzzer_tool", "adapters", "afl_shim.c")

pytestmark = pytest.mark.skipif(
    shutil.which("cargo") is None or not os.path.exists(WRAPPER_SRC),
    reason="cargo not installed, or targets/rust_target.c not present",
)


def _pick_cc():
    """Mirror tools/build_targets.sh's _pick_cc: prefer clang, since gcc's
    -fsanitize-coverage= has no trace-pc-guard variant (confirmed via
    objdump: gcc -> 0 calls to __sanitizer_cov_trace_pc_guard in the built
    wrapper, clang -> 84). An earlier version of this test suite built
    with gcc unconditionally, which silently gave the wrapper zero
    instrumented call sites -- crash detection still passed (it doesn't
    depend on coverage instrumentation), which is exactly how that went
    unnoticed. Returns (cc, extra_flags_list).
    """
    if shutil.which("clang"):
        return "clang", ["-fsanitize-coverage=trace-pc-guard"]
    return "gcc", []


def _find_nightly_rustc():
    """Same auto-detection tools/build_rust_target.sh uses: $RUSTC_NIGHTLY
    first, then the default path tools/fetch_rust_nightly.sh installs to.
    Returns a path or None -- callers skip rather than fail when None.
    """
    candidate = os.environ.get("RUSTC_NIGHTLY") or os.path.expanduser(
        "~/.cache/rust-nightly-jolt/bin/rustc"
    )
    return candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else None


@pytest.fixture(scope="module")
def rust_target_so(tmp_path_factory):
    subprocess.run(
        ["cargo", "build", "--release"], cwd=CRATE_DIR, check=True,
        capture_output=True,
    )
    rlib = os.path.join(CRATE_DIR, "target", "release", "libbuggy_rust_target.a")
    assert os.path.exists(rlib), "cargo build did not produce the expected staticlib"

    out = tmp_path_factory.mktemp("rust_target") / "rust_target.so"
    cc, cov_flag = _pick_cc()
    cmd = [
        cc, "-O2", "-g", "-fno-omit-frame-pointer", *cov_flag,
        "-shared", "-fPIC",
        "-include", SHIM, "-o", str(out), WRAPPER_SRC, rlib,
        "-lpthread", "-ldl", "-lm", "-lrt", "-lutil", "-lgcc_s",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(out)


@pytest.fixture(scope="module")
def rust_target_exe(tmp_path_factory):
    rlib = os.path.join(CRATE_DIR, "target", "release", "libbuggy_rust_target.a")
    out = tmp_path_factory.mktemp("rust_target_exe") / "rust_target"
    cc, cov_flag = _pick_cc()
    cmd = [
        cc, "-O2", "-g", "-fno-omit-frame-pointer", *cov_flag,
        "-include", SHIM, "-o", str(out), WRAPPER_SRC, rlib,
        "-lpthread", "-ldl", "-lm", "-lrt", "-lutil", "-lgcc_s",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(out)


def _run_guarded(so_path: str, data: bytes) -> int:
    """Call fuzz_shm_run via __afl_guarded_call, in a forked child.

    Mirrors adapters/inprocess.py's direct-call convention exactly (see
    that file's ``lib = ctypes.CDLL(target)`` / ``__afl_guarded_call``
    block). Forked per case: __afl_guarded_call's sigsetjmp/siglongjmp is
    only meant to survive one signal per call in this harness -- the real
    fuzzer isolates each in-process execution the same way at a higher
    level (fresh child per campaign restart / bounded consecutive crashes),
    which a shared-process test loop should not skip.
    """
    pid = os.fork()
    if pid == 0:
        lib = ctypes.CDLL(so_path)
        fn = lib.fuzz_shm_run
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        buf = (ctypes.c_uint8 * len(data))(*data)
        guarded = lib.__afl_guarded_call
        guarded.restype = ctypes.c_int
        guarded.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        rc = guarded(ctypes.cast(fn, ctypes.c_void_p), buf, len(data))
        os._exit(0 if rc == 0 else 128 + (-rc))
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status), f"child for {data!r} did not exit cleanly: {status}"
    return os.WEXITSTATUS(status)


@pytest.mark.parametrize(
    "data,expect_crash",
    [
        (b"not a trigger", False),
        (b"RUST\x00", False),  # unrecognized selector byte
        (b"RUSTS", True),      # wild out-of-bounds read -- always faults
        (b"RUSTO\x0a", False), # small OOB read -- silent without ASAN (that's the point)
        (b"RUSTW\x0a", False), # small OOB write -- silent without ASAN (that's the point)
        (b"RUSTP", True),      # safe-Rust panic (checked index)
    ],
)
def test_guarded_call_crash_detection(rust_target_so, data, expect_crash):
    """S and P are the two bugs a plain (non-ASAN) build can catch on its
    own. O and W are deliberately NOT expected to crash here — see
    test_asan_catches_small_overflows below for what they're actually
    for. A plain build silently reading/writing a few bytes past a small
    allocation without crashing is realistic, not a bug in this test: it's
    the exact gap ASAN's redzones exist to close, verified in
    docs/handover/handover_rust_target_2026-09-15.md.
    """
    exit_code = _run_guarded(rust_target_so, data)
    if expect_crash:
        assert exit_code > 128, f"expected a signal-crash exit code, got {exit_code}"
    else:
        assert exit_code == 0


def test_panic_reports_as_abort_not_segv(rust_target_so):
    """SIGABRT (6) and SIGSEGV (11) must stay distinguishable.

    A target that always converts everything to the same signal removes a
    real triage signal from the fuzzer's crash bucketing. 'P' panics via
    checked indexing (no unsafe code on that path at all) and must come
    back as SIGABRT under panic=abort; 'S' is a genuine wild pointer read
    and must come back as SIGSEGV.
    """
    assert _run_guarded(rust_target_so, b"RUSTP") == 128 + 6
    assert _run_guarded(rust_target_so, b"RUSTS") == 128 + 11


def test_subprocess_mode_detects_crash_via_exit_status(rust_target_exe):
    """The other real execution mode: plain fork+exec, crash read from wait status.

    No __afl_guarded_call involved here -- this is exactly how
    persistent_subprocess.py's non-persistent fallback and any one-shot
    execution path observe a crash: os.waitpid() on a child that received
    a fatal signal. Confirms the crash still surfaces correctly even
    though (per the module docstring) the *particular* signal number seen
    by this path can differ from __afl_guarded_call's for the same input,
    because __afl_crash_handler's siglongjmp has nothing to jump to before
    main() ever calls __afl_guarded_call itself.
    """
    proc = subprocess.run(
        [rust_target_exe], input=b"RUSTS", stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert proc.returncode < 0, "expected the child to die by signal"

    proc = subprocess.run(
        [rust_target_exe], input=b"benign, no trigger here",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert proc.returncode == 0


@pytest.mark.skipif(shutil.which("clang") is None, reason="only meaningful with clang available")
def test_wrapper_actually_has_instrumented_call_sites(rust_target_so):
    """Guards against the exact regression this suite already had once.

    Building targets/rust_target.c with gcc links and runs fine and every
    crash-detection test above still passes with it -- gcc silently
    produces a wrapper with zero __sanitizer_cov_trace_pc_guard call
    sites, and nothing about crash detection notices. This test is the
    thing that would have caught it: when clang is available (which the
    fixtures now prefer, matching build_targets.sh's _pick_cc), the built
    .so must actually contain real instrumented call sites, not just
    build without error.
    """
    result = subprocess.run(
        ["objdump", "-d", rust_target_so], check=True, capture_output=True, text=True,
    )
    call_sites = result.stdout.count("call")  # cheap check, refined below
    guard_calls = sum(
        1 for line in result.stdout.splitlines()
        if "call" in line and "__sanitizer_cov_trace_pc_guard" in line
    )
    assert guard_calls > 0, (
        "expected real trace-pc-guard call sites in the clang-built wrapper; "
        f"found 0 (raw 'call' mentions: {call_sites}) -- coverage instrumentation regressed"
    )


@pytest.fixture(scope="module")
def rust_target_so_nightly(tmp_path_factory):
    nightly_rustc = _find_nightly_rustc()
    if nightly_rustc is None:
        pytest.skip(
            "no nightly rustc found (checked $RUSTC_NIGHTLY and "
            "~/.cache/rust-nightly-jolt/bin/rustc) -- run "
            "tools/fetch_rust_nightly.sh to opt in (reads its header "
            "comment first: unofficial third-party toolchain)"
        )

    env = dict(os.environ)
    env["RUSTC"] = nightly_rustc
    env["RUSTFLAGS"] = (
        "-Cpasses=sancov-module "
        "-Cllvm-args=-sanitizer-coverage-level=3 "
        "-Cllvm-args=-sanitizer-coverage-trace-pc-guard"
    )
    build_dir = tmp_path_factory.mktemp("nightly_cargo_target")
    subprocess.run(
        ["cargo", "build", "--release", "--target-dir", str(build_dir)],
        cwd=CRATE_DIR, check=True, capture_output=True, env=env,
    )
    rlib = build_dir / "release" / "libbuggy_rust_target.a"
    assert rlib.exists(), "nightly cargo build did not produce the expected staticlib"

    out = tmp_path_factory.mktemp("rust_target_nightly") / "rust_target.so"
    cc, cov_flag = _pick_cc()  # still clang for the wrapper side
    cmd = [
        cc, "-O2", "-g", "-fno-omit-frame-pointer", *cov_flag,
        "-shared", "-fPIC",
        "-include", SHIM, "-o", str(out), WRAPPER_SRC, str(rlib),
        "-lpthread", "-ldl", "-lm", "-lrt", "-lutil", "-lgcc_s",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(out)


def test_nightly_build_instruments_inside_the_crate(rust_target_so_nightly):
    """The thing stable rustc genuinely cannot do (see the handover doc):
    real __sanitizer_cov_trace_pc_guard calls from *inside*
    rust_fuzz_entry, not just from targets/rust_target.c's few lines
    around it. Verified empirically while wiring this up -- 12 guard
    calls landed inside rust_fuzz_entry itself once release-mode inlining
    folded all four bug functions into it -- so this checks the same
    per-function property rather than just a higher total call count
    (which the wrapper's own instrumentation would also produce and so
    wouldn't actually distinguish "did the crate get instrumented" from
    "did the wrapper").
    """
    result = subprocess.run(
        ["objdump", "-d", "-C", rust_target_so_nightly],
        check=True, capture_output=True, text=True,
    )
    lines = result.stdout.splitlines()
    in_fn = False
    guard_calls_in_rust_fuzz_entry = 0
    for line in lines:
        if line.rstrip().endswith("<rust_fuzz_entry>:"):
            in_fn = True
            continue
        if in_fn and line.strip() == "":
            break
        if in_fn and "call" in line and "__sanitizer_cov_trace_pc_guard>" in line:
            guard_calls_in_rust_fuzz_entry += 1
    assert guard_calls_in_rust_fuzz_entry > 0, (
        "expected real trace-pc-guard calls inside rust_fuzz_entry itself "
        "with nightly rustc + -Z sanitizer-coverage-trace-pc-guard; found 0"
    )


def test_nightly_build_still_detects_all_crashes(rust_target_so_nightly):
    """Coverage instrumentation is additive; it must not change what
    counts as a crash. This build has coverage instrumentation only (no
    ASAN — see test_asan_catches_small_overflows below for that), so S
    and P are the ones expected to crash on their own, same as the
    stable/plain build; O and W are deliberately quiet here too (small
    overflows into live, mapped memory), which is the correct behavior to
    verify, not a gap in this test.
    """
    assert _run_guarded(rust_target_so_nightly, b"benign") == 0
    assert _run_guarded(rust_target_so_nightly, b"RUSTS") == 128 + 11
    assert _run_guarded(rust_target_so_nightly, b"RUSTO\x0a") == 0
    assert _run_guarded(rust_target_so_nightly, b"RUSTW\x0a") == 0
    assert _run_guarded(rust_target_so_nightly, b"RUSTP") == 128 + 6


def _find_matching_asan_clang(nightly_rustc: str):
    """Mirror tools/build_rust_target.sh's clang-version-matching logic:
    a mismatched major LLVM version between rustc's bundled ASAN
    expectations and the C compiler's own runtime fails at process start
    with "incompatible ASan runtimes", not a build error — confirmed
    empirically (system clang's default LLVM 18 against this nightly's
    LLVM 20). Returns a clang binary name or None.
    """
    result = subprocess.run(
        [nightly_rustc, "--version", "--verbose"], capture_output=True, text=True,
    )
    match = re.search(r"^LLVM version: (\d+)", result.stdout, re.MULTILINE)
    if not match:
        return None
    candidate = f"clang-{match.group(1)}"
    return candidate if shutil.which(candidate) else None


@pytest.fixture(scope="module")
def rust_target_exe_asan_nightly(tmp_path_factory):
    """The build that took the most work to get right this session --
    real ASAN instrumentation of the Rust crate itself, linked with a C
    compiler whose runtime actually matches. See
    docs/handover/handover_rust_target_2026-09-15.md for the two separate
    bugs this uncovered along the way (an ASAN-runtime version mismatch,
    and a write silently optimized away because it was UB independent of
    the bounds check being missing — deriving a `*mut u8` from a `&[u8]`
    and writing through it).
    """
    nightly_rustc = _find_nightly_rustc()
    if nightly_rustc is None:
        pytest.skip("no nightly rustc found — see tools/fetch_rust_nightly.sh")
    asan_cc = _find_matching_asan_clang(nightly_rustc)
    if asan_cc is None:
        pytest.skip(
            "no clang matching this nightly rustc's LLVM version found — "
            "install it (e.g. apt install clang-20) to run this test"
        )

    env = dict(os.environ)
    env["RUSTC"] = nightly_rustc
    env["RUSTFLAGS"] = (
        "-Z sanitizer=address "
        "-Cpasses=sancov-module "
        "-Cllvm-args=-sanitizer-coverage-level=3 "
        "-Cllvm-args=-sanitizer-coverage-trace-pc-guard"
    )
    build_dir = tmp_path_factory.mktemp("asan_nightly_cargo_target")
    subprocess.run(
        ["cargo", "build", "--release", "--target-dir", str(build_dir)],
        cwd=CRATE_DIR, check=True, capture_output=True, env=env,
    )
    rlib = build_dir / "release" / "libbuggy_rust_target.a"
    assert rlib.exists(), "ASAN cargo build did not produce the expected staticlib"

    out = tmp_path_factory.mktemp("rust_target_asan_nightly") / "rust_target"
    cmd = [
        asan_cc, "-O2", "-g", "-fno-omit-frame-pointer", "-fsanitize=address",
        "-include", SHIM, "-o", str(out), WRAPPER_SRC, str(rlib),
        # Deliberately NOT passing -lasan: forcing one on top of what the
        # matched clang auto-selects is exactly what reproduces the
        # "incompatible ASan runtimes" failure.
        "-lpthread", "-ldl", "-lm", "-lrt", "-lutil", "-lgcc_s",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(out)


def test_asan_catches_small_overflows(rust_target_exe_asan_nightly):
    """The actual fix this session's work was for: with real Rust-side
    ASAN instrumentation (and an exact-size heap allocation on the C
    side — see targets/rust_target.c's main(), which used to read into a
    fixed 256-byte scratch buffer that silently absorbed any overrun
    shorter than its own leftover capacity), a SMALL out-of-bounds
    read/write that a plain build can't catch at all now produces a real,
    fully symbolicated "AddressSanitizer: heap-buffer-overflow" report
    pointing at the actual Rust source line — not just a raw crash.
    """
    for trigger, direction in [(b"RUSTO\x0a", "READ"), (b"RUSTW\x0a", "WRITE")]:
        proc = subprocess.run(
            [rust_target_exe_asan_nightly], input=trigger, capture_output=True,
        )
        stderr = proc.stderr.decode(errors="replace")
        assert "AddressSanitizer: heap-buffer-overflow" in stderr, (
            f"expected a real ASAN report for {trigger!r}, got: {stderr[:500]}"
        )
        assert "rust/buggy_target/src/lib.rs" in stderr, (
            f"expected the report to symbolicate into the Rust source for {trigger!r}"
        )

    benign = subprocess.run(
        [rust_target_exe_asan_nightly], input=b"benign, nothing here", capture_output=True,
    )
    assert benign.returncode == 0
    assert b"AddressSanitizer" not in benign.stderr
