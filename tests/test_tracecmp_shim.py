"""Tests for tracecmp_shim.c — compiler-IR comparison tracing callbacks."""

import ctypes
import os
import subprocess
import tempfile

from fuzzer_tool.core.cmplog import CmplogCollector


class TestTracecmpShimCompilation:
    """Test that the trace-cmp shim compiles and exports expected symbols."""

    def test_compiles(self):
        shim_src = os.path.join(
            os.path.dirname(__file__), "..", "src", "fuzzer_tool", "adapters", "tracecmp_shim.c"
        )
        assert os.path.exists(shim_src), f"tracecmp_shim.c not found at {shim_src}"

        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            out_path = f.name
        try:
            result = subprocess.run(
                ["gcc", "-shared", "-fPIC", "-O2", "-o", out_path, shim_src],
                capture_output=True,
                timeout=30,
            )
            assert result.returncode == 0, f"Compilation failed: {result.stderr.decode()[:200]}"
            assert os.path.exists(out_path)
        finally:
            os.unlink(out_path)

    def test_exports_trace_cmp_symbols(self):
        shim_src = os.path.join(
            os.path.dirname(__file__), "..", "src", "fuzzer_tool", "adapters", "tracecmp_shim.c"
        )
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            out_path = f.name
        try:
            subprocess.run(
                ["gcc", "-shared", "-fPIC", "-O2", "-o", out_path, shim_src],
                capture_output=True,
                timeout=30,
            )
            result = subprocess.run(["nm", "-D", out_path], capture_output=True, text=True)
            symbols = result.stdout
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
                "__tracecmp_reset",
                "__tracecmp_flush",
            ]:
                assert sym in symbols, f"Missing exported symbol: {sym}"
        finally:
            os.unlink(out_path)


class TestTracecmpShimFunctionality:
    """Test the trace-cmp shim's logging behavior."""

    def _build_shim(self) -> str:
        shim_src = os.path.join(
            os.path.dirname(__file__), "..", "src", "fuzzer_tool", "adapters", "tracecmp_shim.c"
        )
        # Unique path per call — Python caches .so loads, so each test
        # needs its own copy to trigger the constructor independently.
        fd, out_path = tempfile.mkstemp(suffix=".so", prefix="test_tracecmp_")
        os.close(fd)
        result = subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O2", "-o", out_path, shim_src],
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Shim compilation failed: {result.stderr.decode()[:200]}"
        return out_path

    def test_cmp4_callback(self):
        """Test that trace_cmp4 logs a CMP line with correct format."""
        shim_path = self._build_shim()
        with tempfile.NamedTemporaryFile(suffix=".cmplog", delete=False, mode="w") as f:
            log_path = f.name
        try:
            os.environ["_CMPLOG_OUT"] = log_path
            lib = ctypes.CDLL(shim_path)
            getattr(lib, "__sanitizer_cov_trace_cmp4")(0x41424344, 0x45464748)
            # Flush buffer to disk (without truncating)
            getattr(lib, "__tracecmp_flush")()

            with open(log_path) as f:
                content = f.read()
            assert "CMP " in content, f"Expected CMP line in log, got: {content!r}"
            lines = [ln for ln in content.strip().split("\n") if ln.startswith("CMP ")]
            assert len(lines) >= 1
            parts = lines[0].split()
            assert len(parts) >= 4
            hex_a, hex_b = parts[1], parts[2]
            # 0x41424344 in LE = 44434241
            assert hex_a == "44434241", f"Expected operand a=44434241, got {hex_a}"
            # 0x45464748 in LE = 48474645
            assert hex_b == "48474645", f"Expected operand b=48474645, got {hex_b}"
        finally:
            os.unlink(log_path)

    def test_cmp1_callback(self):
        """Test 1-byte comparison logging."""
        shim_path = self._build_shim()
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
            assert parts[1] == "89", f"Expected operand a=89, got {parts[1]}"
            assert parts[2] == "50", f"Expected operand b=50, got {parts[2]}"
        finally:
            os.unlink(log_path)

    def test_switch_callback(self):
        """Test switch statement logging — one CMP line per case."""
        shim_path = self._build_shim()
        with tempfile.NamedTemporaryFile(suffix=".cmplog", delete=False, mode="w") as f:
            log_path = f.name
        try:
            os.environ["_CMPLOG_OUT"] = log_path
            lib = ctypes.CDLL(shim_path)
            # Clang layout: ref[0]=count, ref[1]=bit-width, ref[2..]=case values
            case_count = 4
            ArrayType = ctypes.c_uint64 * (case_count + 2)
            arr = ArrayType()
            arr[0] = case_count  # ref[0] = count
            arr[1] = 64          # ref[1] = bit-width (uint64_t)
            arr[2] = 0x00        # ref[2] = case value 0
            arr[3] = 0x49        # ref[3] = case value 1
            arr[4] = 0x50        # ref[4] = case value 2
            arr[5] = 0x53        # ref[5] = case value 3
            ref_ptr = ctypes.cast(
                ctypes.addressof(arr),
                ctypes.POINTER(ctypes.c_uint64),
            )
            getattr(lib, "__sanitizer_cov_trace_switch")(0x41, ref_ptr)
            getattr(lib, "__tracecmp_flush")()

            with open(log_path) as f:
                content = f.read()
            lines = [ln for ln in content.strip().split("\n") if ln.startswith("CMP ")]
            assert len(lines) == 4, f"Expected 4 CMP lines for switch, got {len(lines)}"
        finally:
            os.unlink(log_path)

    def test_const_cmp4_callback(self):
        """Test const variant — same format as regular trace_cmp4."""
        shim_path = self._build_shim()
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
            # 0xDEADBEEF in LE = efbeadde
            assert parts[1] == "efbeadde"
            # 0xCAFEBABE in LE = beadface
            assert parts[2] == "bebafeca"
        finally:
            os.unlink(log_path)


class TestCmplogCollectorDualShim:
    """Test that CmplogCollector handles interleaved trace-cmp output."""

    def test_collect_tokens_tracecmp_format(self, tmp_path):
        """Verify CmplogCollector parses trace-cmp CMP lines correctly."""
        c = CmplogCollector()
        log_file = tmp_path / "test.cmplog"
        log_file.write_text(
            "CMP 48656c6c6f 576f726c64 1 5\n"  # symbol-based
            "CMP 89 50 -23 1\n"  # trace-cmp 1-byte
            "CMP 41424344 45464748 -4 4\n"  # trace-cmp 4-byte
            "CMP 0000000000000000 000000000000000a -10 8\n"  # trace-cmp switch
        )
        c.log_path = str(log_file)
        tokens = c.collect_tokens()
        assert b"Hello" in tokens
        assert b"World" in tokens
        assert bytes.fromhex("89") in tokens
        assert bytes.fromhex("50") in tokens
        assert bytes.fromhex("41424344") in tokens
        assert bytes.fromhex("45464748") in tokens

    def test_build_tracecmp_shim(self):
        """Test that shim_factory.build_tracecmp_shim works."""
        from fuzzer_tool.adapters.shim_factory import build_tracecmp_shim

        path = build_tracecmp_shim()
        try:
            assert path is not None
            assert os.path.exists(path)
            assert path.endswith(".so")
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
