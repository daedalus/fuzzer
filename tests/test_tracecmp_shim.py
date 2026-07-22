"""Tests for the unified cmplog_shim.c — both libc and compiler-IR tracing."""

import ctypes
import os
import subprocess
import tempfile

SHIM_REL = os.path.join(
    os.path.dirname(__file__), "..", "src", "fuzzer_tool", "adapters", "cmplog_shim.c"
)


def _build_shim() -> str:
    """Compile the unified cmplog shim into a temp .so."""
    assert os.path.exists(SHIM_REL), f"cmplog_shim.c not found at {SHIM_REL}"
    fd, out_path = tempfile.mkstemp(suffix=".so", prefix="test_cmplog_")
    os.close(fd)
    result = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-ldl", "-o", out_path, SHIM_REL],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Shim compilation failed: {result.stderr.decode()[:200]}"
    return out_path


class TestUnifiedShimCompilation:
    """Test that the unified shim compiles and exports all expected symbols."""

    def test_compiles(self):
        assert os.path.exists(SHIM_REL)
        path = _build_shim()
        os.unlink(path)

    def test_exports_all_symbols(self):
        path = _build_shim()
        try:
            result = subprocess.run(["nm", "-D", path], capture_output=True, text=True)
            symbols = result.stdout

            # libc interposition symbols
            for sym in [
                "memcmp",
                "strcmp",
                "strncmp",
                "memchr",
                "strcasecmp",
                "strncasecmp",
                "memmem",
                "strstr",
                "strcasestr",
            ]:
                assert sym in symbols, f"Missing exported symbol: {sym}"

            # Compiler-IR callback symbols
            for sym in [
                "__sanitizer_cov_trace_cmp1",
                "__sanitizer_cov_trace_cmp2",
                "__sanitizer_cov_trace_cmp4",
                "__sanitizer_cov_trace_cmp8",
                "__sanitizer_cov_trace_const_cmp1",
                "__sanitizer_cov_trace_const_cmp2",
                "__sanitizer_cov_trace_const_cmp4",
                "__sanitizer_cov_trace_const_cmp8",
                "__sanitizer_cov_trace_switch",
            ]:
                assert sym in symbols, f"Missing exported symbol: {sym}"

            # Public API symbols
            for sym in ["__cmplog_reset", "__cmplog_get_path", "__tracecmp_flush", "__tracecmp_reset",
                         "__tracecmp_get_path"]:
                assert sym in symbols, f"Missing exported symbol: {sym}"
        finally:
            os.unlink(path)


class TestTracecmpCallbacks:
    """Test the unified shim's compiler-IR callback logging."""

    def test_cmp4_callback(self):
        shim_path = _build_shim()
        with tempfile.NamedTemporaryFile(suffix=".cmplog", delete=False, mode="w") as f:
            log_path = f.name
        try:
            os.environ["_CMPLOG_OUT"] = log_path
            lib = ctypes.CDLL(shim_path)
            getattr(lib, "__sanitizer_cov_trace_cmp4")(0x41424344, 0x45464748)
            getattr(lib, "__tracecmp_flush")()

            with open(log_path) as f:
                content = f.read()
            assert "CMP " in content, f"Expected CMP line, got: {content!r}"
            lines = [ln for ln in content.strip().split("\n") if ln.startswith("CMP ")]
            assert len(lines) >= 1
            parts = lines[0].split()
            assert len(parts) >= 4
            # 0x41424344 little-endian → 44434241
            assert parts[1] == "44434241", f"Expected a=44434241, got {parts[1]}"
            # 0x45464748 little-endian → 48474645
            assert parts[2] == "48474645", f"Expected b=48474645, got {parts[2]}"
        finally:
            os.unlink(log_path)

    def test_cmp1_callback(self):
        shim_path = _build_shim()
        with tempfile.NamedTemporaryFile(suffix=".cmplog", delete=False, mode="w") as f:
            log_path = f.name
        try:
            os.environ["_CMPLOG_OUT"] = log_path
            lib = ctypes.CDLL(shim_path)
            getattr(lib, "__sanitizer_cov_trace_cmp1")(0x89, 0x50)
            getattr(lib, "__tracecmp_flush")()

            with open(log_path) as f:
                content = f.read()
            lines = [ln for ln in content.strip().split("\n") if ln.startswith("CMP ")]
            assert len(lines) >= 1
            parts = lines[0].split()
            assert parts[1] == "89", f"Expected a=89, got {parts[1]}"
            assert parts[2] == "50", f"Expected b=50, got {parts[2]}"
        finally:
            os.unlink(log_path)

    def test_switch_callback(self):
        shim_path = _build_shim()
        with tempfile.NamedTemporaryFile(suffix=".cmplog", delete=False, mode="w") as f:
            log_path = f.name
        try:
            os.environ["_CMPLOG_OUT"] = log_path
            lib = ctypes.CDLL(shim_path)
            # Clang layout: ref[0]=count, ref[1]=bit-width, ref[2..]=case values
            case_count = 4
            ArrayType = ctypes.c_uint64 * (case_count + 2)
            arr = ArrayType()
            arr[0] = case_count
            arr[1] = 64
            arr[2] = 0x00
            arr[3] = 0x49
            arr[4] = 0x50
            arr[5] = 0x53
            ref_ptr = ctypes.cast(ctypes.addressof(arr), ctypes.POINTER(ctypes.c_uint64))
            getattr(lib, "__sanitizer_cov_trace_switch")(0x41, ref_ptr)
            getattr(lib, "__tracecmp_flush")()

            with open(log_path) as f:
                content = f.read()
            lines = [ln for ln in content.strip().split("\n") if ln.startswith("CMP ")]
            assert len(lines) == 4, f"Expected 4 CMP lines, got {len(lines)}"
        finally:
            os.unlink(log_path)

    def test_const_cmp4_callback(self):
        shim_path = _build_shim()
        with tempfile.NamedTemporaryFile(suffix=".cmplog", delete=False, mode="w") as f:
            log_path = f.name
        try:
            os.environ["_CMPLOG_OUT"] = log_path
            lib = ctypes.CDLL(shim_path)
            getattr(lib, "__sanitizer_cov_trace_const_cmp4")(0xDEADBEEF, 0xCAFEBABE)
            getattr(lib, "__tracecmp_flush")()

            with open(log_path) as f:
                content = f.read()
            lines = [ln for ln in content.strip().split("\n") if ln.startswith("CMP ")]
            assert len(lines) >= 1
            parts = lines[0].split()
            assert parts[1] == "efbeadde"
            assert parts[2] == "bebafeca"
        finally:
            os.unlink(log_path)


class TestUnifiedShimCollector:
    """Test that CmplogCollector parses unified shim output."""

    def test_collect_combined_output(self, tmp_path):
        from fuzzer_tool.core.cmplog import CmplogCollector

        c = CmplogCollector()
        log_file = tmp_path / "test.cmplog"
        log_file.write_text(
            "CMP 48656c6c6f 576f726c64 1 5\n"      # libc: memcmp("Hello","World")
            "CMP 89 50 -23 1\n"                     # IR: 1-byte cmp
            "CMP 41424344 45464748 -4 4\n"          # IR: 4-byte cmp
        )
        c.log_path = str(log_file)
        tokens = c.collect_tokens()
        assert b"Hello" in tokens
        assert b"World" in tokens
        assert bytes.fromhex("89") in tokens
        assert bytes.fromhex("50") in tokens
        assert bytes.fromhex("41424344") in tokens
        assert bytes.fromhex("45464748") in tokens
