"""Regression: ldpreload_wrapper must not preload runtimes the target already links.

`_detect_ubsan` matched *any* `__ubsan_handle_*` occurrence, so targets built
with `-fsanitize=undefined` that already link the runtime (defined `T`
symbols, the common clang case) triggered a UBSAN standalone preload on top
of libasan.  Preloading libasan and the UBSAN standalone together into the
fuzzer process trips an ASAN init CHECK (`sanitizer_signal_interceptors.inc`)
and hangs startup.  Only strong *undefined* imports (`U`) must trigger a
preload; defined (`T`) and weak-undefined (`w`) references must not.
"""

from unittest.mock import MagicMock, patch

from fuzzer_tool.cli.ldpreload_wrapper import _detect_asan, _detect_ubsan


def _nm_run(stdout: bytes) -> MagicMock:
    r = MagicMock()
    r.returncode = 0
    r.stdout = stdout
    return r


def test_ubsan_not_detected_when_handlers_defined():
    """Targets that link the UBSAN runtime (T symbols) need no preload."""
    nm_out = (
        b"0000000000109650 T __ubsan_handle_add_overflow\n"
        b"0000000000109960 T __ubsan_handle_builtin_unreachable\n"
    )
    with patch("fuzzer_tool.cli.ldpreload_wrapper.subprocess.run", return_value=_nm_run(nm_out)):
        assert _detect_ubsan("fake_target") is False


def test_ubsan_detected_when_handlers_unresolved():
    """Targets with unresolved __ubsan_handle_* imports need the preload."""
    nm_out = (
        b"                 U __ubsan_handle_add_overflow\n"
        b"                 U __ubsan_handle_builtin_unreachable\n"
    )
    with patch("fuzzer_tool.cli.ldpreload_wrapper.subprocess.run", return_value=_nm_run(nm_out)):
        assert _detect_ubsan("fake_target") is True


def test_ubsan_not_detected_for_weak_undefined():
    """Weak undefined (w) handlers are resolvable-or-NULL — no preload.

    `__ubsan_handle_cfi_bad_type` appears as a weak undefined reference in
    targets that otherwise link the UBSAN runtime; preloading the standalone
    runtime for it hangs the fuzzer process.
    """
    nm_out = (
        b"000000000010d420 w __ubsan_handle_cfi_bad_type\n"
        b"0000000000109650 T __ubsan_handle_add_overflow\n"
    )
    with patch("fuzzer_tool.cli.ldpreload_wrapper.subprocess.run", return_value=_nm_run(nm_out)):
        assert _detect_ubsan("fake_target") is False


def test_ubsan_detected_with_strong_undefined_alongside_weak():
    """A single strong undefined import is enough to trigger the preload."""
    nm_out = (
        b"                 U __ubsan_handle_add_overflow\n"
        b"000000000010d420 w __ubsan_handle_cfi_bad_type\n"
    )
    with patch("fuzzer_tool.cli.ldpreload_wrapper.subprocess.run", return_value=_nm_run(nm_out)):
        assert _detect_ubsan("fake_target") is True


def test_ubsan_not_detected_on_nm_failure():
    """nm failure must not cause a (hang-inducing) preload."""
    r = MagicMock()
    r.returncode = 1
    r.stdout = b""
    with patch("fuzzer_tool.cli.ldpreload_wrapper.subprocess.run", return_value=r):
        assert _detect_ubsan("fake_target") is False


def test_asan_not_detected_when_runtime_linked():
    """Targets that link their own ASAN runtime must not get a preload.

    LD_PRELOADing a second libasan on top of a statically-linked runtime
    breaks the child's AFL SHM coverage (verified on png_read: shm stayed 0).
    """
    nm_out = b"00000000000d7100 T __asan_init\n"
    with patch("fuzzer_tool.cli.ldpreload_wrapper.subprocess.run", return_value=_nm_run(nm_out)):
        assert _detect_asan("fake_target") is False


def test_asan_detected_when_init_unresolved():
    """A target with unresolved __asan_init still needs the preload."""
    nm_out = b"                 U __asan_init\n"
    with patch("fuzzer_tool.cli.ldpreload_wrapper.subprocess.run", return_value=_nm_run(nm_out)):
        assert _detect_asan("fake_target") is True
