"""Regression: save_crash() must chmod crash-sidecar files read-only.

Crash reproducer artifacts are authoritative records of sanitizer hits — a
corpus-minimize sweep or dedup pass that mutates an existing sidecar silently
destroys the reproducer. All files written by save_crash() except the shell
script must be locked to read-only (0o444) so stray writes fail loudly.
"""

import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.adapters.filesystem import save_crash


@pytest.fixture
def tmp_crash_dir():
    with tempfile.TemporaryDirectory(prefix="fuzz_crash_protect_") as d:
        yield Path(d)


class TestCrashFileProtection:
    """Crash sidecar artifacts are read-only on disk."""

    def test_bin_is_readonly(self, tmp_crash_dir: Path):
        hashes, sigs = set(), {}
        name = save_crash(b"crash_seed", -11, "SIGSEGV", tmp_crash_dir, hashes, sigs)
        assert name is not False
        crash_bin = tmp_crash_dir / f"{name}.bin"
        mode = crash_bin.stat().st_mode & 0o777
        assert mode == 0o444, f".bin is {oct(mode)}, want 0o444"

    def test_txt_sidecar_is_readonly(self, tmp_crash_dir: Path):
        hashes, sigs = set(), {}
        name = save_crash(b"crash_seed", -11, "SIGSEGV", tmp_crash_dir, hashes, sigs)
        assert name is not False
        sidecar = tmp_crash_dir / f"{name}.txt"
        mode = sidecar.stat().st_mode & 0o777
        assert mode == 0o444, f".txt is {oct(mode)}, want 0o444"

    def test_json_sidecar_is_readonly(self, tmp_crash_dir: Path):
        hashes, sigs = set(), {}
        name = save_crash(b"crash_seed", -11, "SIGSEGV", tmp_crash_dir, hashes, sigs)
        assert name is not False
        json_sidecar = tmp_crash_dir / f"{name}.json"
        mode = json_sidecar.stat().st_mode & 0o777
        assert mode == 0o444, f".json is {oct(mode)}, want 0o444"

    def test_hexdump_is_readonly(self, tmp_crash_dir: Path):
        hashes, sigs = set(), {}
        name = save_crash(b"crash_seed", -11, "SIGSEGV", tmp_crash_dir, hashes, sigs)
        assert name is not False
        hexdump = tmp_crash_dir / f"{name}.hex"
        mode = hexdump.stat().st_mode & 0o777
        assert mode == 0o444, f".hex is {oct(mode)}, want 0o444"

    def test_reproducer_script_is_executable(self, tmp_crash_dir: Path):
        hashes, sigs = set(), {}
        name = save_crash(b"crash_seed", -11, "SIGSEGV", tmp_crash_dir, hashes, sigs)
        assert name is not False
        script = tmp_crash_dir / f"{name}.sh"
        mode = script.stat().st_mode & 0o777
        assert mode == 0o755, f".sh is {oct(mode)}, want 0o755"


class TestCrashFileProtectionAdversarial:
    """A write to a protected crash file must fail."""

    def test_write_to_bin_fails(self, tmp_crash_dir: Path):
        hashes, sigs = set(), {}
        name = save_crash(b"crash_seed", -11, "SIGSEGV", tmp_crash_dir, hashes, sigs)
        assert name is not False
        crash_bin = tmp_crash_dir / f"{name}.bin"
        with pytest.raises(PermissionError):
            crash_bin.write_bytes(b"mutated")

    def test_write_to_txt_fails(self, tmp_crash_dir: Path):
        hashes, sigs = set(), {}
        name = save_crash(b"crash_seed", -11, "SIGSEGV", tmp_crash_dir, hashes, sigs)
        assert name is not False
        sidecar = tmp_crash_dir / f"{name}.txt"
        with pytest.raises(PermissionError):
            sidecar.write_text("tampered")

    def test_write_to_json_fails(self, tmp_crash_dir: Path):
        hashes, sigs = set(), {}
        name = save_crash(b"crash_seed", -11, "SIGSEGV", tmp_crash_dir, hashes, sigs)
        assert name is not False
        json_sidecar = tmp_crash_dir / f"{name}.json"
        with pytest.raises(PermissionError):
            json_sidecar.write_text('{"tampered": true}')
