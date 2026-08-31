"""Regression test for finding #13 (docs/bugreport_2026-08-21_merged.md).

``InProcessRunner._run_c_direct`` used to install a *Python-level*
``signal.signal(SIGSEGV, ...)`` handler to "catch" target crashes. That
cannot work: SIGSEGV/SIGABRT are synchronous faults delivered at the
faulting instruction, and returning normally from a plain Python handler
resumes execution at that same instruction. For a genuine wild-pointer
fault this produces an infinite fault-loop that hangs the fuzzer process
permanently — verified below against the actual pre-fix source.

The fix routes ``_run_c_direct`` through ``__afl_guarded_call``
(afl_shim.c), the same C-level sigsetjmp/siglongjmp escape
``_run_c_direct_lite`` already used correctly, which can actually unwind
the stack out of the faulting call.

Because a *reproduction* of the original bug really does hang forever,
every risky call here runs in a subprocess with a hard wall-clock
``timeout=`` (SIGKILL on expiry) rather than in-process — a hang shows up
as ``subprocess.TimeoutExpired``, not as a frozen test run.
"""

import base64
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TARGETS_DIR = Path(__file__).parent.parent / "targets"
SHIM_PATH = TARGETS_DIR.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# Small standalone driver: build an InProcessRunner in direct mode against
# a target .so and run one input through it, printing "RC=<n>" so the
# parent process (with its hard timeout) can read the outcome without
# ever risking its own process on the crash.
_PROBE = textwrap.dedent(
    """\
    import base64, sys
    sys.path.insert(0, {src_path!r})
    from fuzzer_tool.adapters.inprocess import InProcessRunner

    runner = InProcessRunner(
        target={target!r},
        function_name="fuzz_shm_run",
        timeout=2.0,
        shm_size=4096,
        direct=True,
        coverage_env_id=None,
        cov=False,
        debug=False,
    )
    rc, stderr = runner.run_one(base64.b64decode({data_b64!r}))
    print(f"RC={{rc}}")
    """
)

# Hard wall-clock ceiling for the subprocess. The fixed code returns in
# well under a second; this only needs to be comfortably above that so a
# real hang (which would otherwise run forever) is reliably caught.
_WATCHDOG_SECONDS = 10


def _run_probe(src_path: str, target: str, data: bytes) -> subprocess.CompletedProcess:
    script = _PROBE.format(
        src_path=src_path,
        target=target,
        data_b64=base64.b64encode(data).decode(),
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=_WATCHDOG_SECONDS,
    )


@pytest.fixture(scope="module")
def target_so(tmp_path_factory):
    """Compile test_target.c with afl_shim.c linked directly in (-include),
    so __afl_guarded_call is a real, defined symbol in the target's own
    shared object — exactly the direct-mode build shape the fix relies on.
    """
    tmpdir = tmp_path_factory.mktemp("direct_crash")
    out = tmpdir / "target_shim.so"
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            "-o",
            str(out),
            str(TARGETS_DIR / "test_target.c"),
            "-include",
            str(SHIM_PATH),
        ],
        check=True,
        capture_output=True,
    )
    return out


class TestDirectModeCrashRecovery:
    """Fixed code: a real SIGSEGV in --inprocess-direct is survived, not hung."""

    def test_safe_input_returns_zero(self, target_so):
        result = _run_probe(
            str(Path(__file__).parent.parent / "src"), str(target_so), b"safe input"
        )
        assert result.returncode == 0, result.stderr
        assert "RC=0" in result.stdout

    def test_segv_trigger_recovers_promptly(self, target_so):
        """CRASHS dereferences a NULL function pointer (real SIGSEGV).

        Must return well within the watchdog window with the 128+SIGSEGV
        convention (139), not time out.
        """
        result = _run_probe(
            str(Path(__file__).parent.parent / "src"), str(target_so), b"CRASHS"
        )
        assert result.returncode == 0, (
            f"probe did not exit cleanly (stdout={result.stdout!r} "
            f"stderr={result.stderr!r})"
        )
        assert "RC=139" in result.stdout, result.stdout

    def test_abort_trigger_recovers(self, target_so):
        """CRASHA hits abort(), intercepted by afl_shim.c's override — rc=0."""
        result = _run_probe(
            str(Path(__file__).parent.parent / "src"), str(target_so), b"CRASHA"
        )
        assert result.returncode == 0, result.stderr
        assert "RC=0" in result.stdout


class TestPreFixSourceActuallyHung:
    """Characterization test: prove the ORIGINAL code really did hang forever.

    This pins the bug report's claim against the actual pre-fix source
    (extracted from git history at build time), so the regression this
    suite guards against is not hypothetical.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def prefix_src_dir(tmp_path_factory):
        import subprocess as sp

        repo_root = Path(__file__).parent.parent
        # Commit that introduced the fix; ^ is the pre-fix parent.
        fix_commit = sp.run(
            ["git", "log", "--format=%H", "-1", "--follow", "--diff-filter=M",
             "--grep=finding #13", "--", "src/fuzzer_tool/adapters/inprocess.py"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        if not fix_commit:
            pytest.skip("fix commit for finding #13 not found in history")
        fix_sha = fix_commit[0]
        old_src = sp.run(
            ["git", "show", f"{fix_sha}^:src/fuzzer_tool/adapters/inprocess.py"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout
        assert "signal.signal(signal.SIGSEGV, _crash_handler)" in old_src, (
            "expected the pre-fix source to contain the broken Python-level "
            "SIGSEGV handler — history layout may have changed"
        )

        tmpdir = tmp_path_factory.mktemp("prefix_src")
        pkg_dir = tmpdir / "fuzzer_tool"
        import shutil

        shutil.copytree(repo_root / "src" / "fuzzer_tool", pkg_dir)
        (pkg_dir / "adapters" / "inprocess.py").write_text(old_src)
        return tmpdir

    def test_old_code_hangs_on_segv(self, target_so, prefix_src_dir):
        with pytest.raises(subprocess.TimeoutExpired):
            _run_probe(str(prefix_src_dir), str(target_so), b"CRASHS")
