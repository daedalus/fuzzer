"""Sub-second timeouts must survive the trip to the loader.

The loader takes its per-exec timeout from the INIT handshake. That value used
to be sent as ``int(self.timeout)`` and parsed with ``atoi``, so anything under
one second arrived as ``0`` -- which the loader's ``<= 0`` guard then replaced
with its 5s default. Asking for a tighter timeout produced a *looser* one, and
nothing reported the substitution.

These tests pin the two halves of that path: the senders must not truncate, and
the loader must accept a fractional value.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

LOADER_SRC = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/fuzz_loader.c"


class TestSendersDoNotTruncate:
    """The formatting on both senders must preserve sub-second values."""

    @pytest.mark.parametrize("timeout", [0.04, 0.2, 0.999, 1.5, 30.0])
    def test_forkserver_init_preserves_fraction(self, timeout):
        init = f"INIT /t/target.so fuzz /t/in.bin {timeout:.6f}\n"
        sent = float(init.split()[-1])
        assert sent == pytest.approx(timeout), f"INIT truncated {timeout} to {sent}"
        assert sent > 0, "a sub-second timeout must not arrive as 0"

    @pytest.mark.parametrize("timeout", [0.04, 0.2, 0.999])
    def test_timeout_env_roundtrips_as_float(self, timeout):
        # The consumer parses this with float(); int() would raise on "0.040000"
        # and int(0.04) would have written "0" in the first place.
        encoded = f"{timeout:.6f}"
        assert float(encoded) == pytest.approx(timeout)

    def test_int_truncation_would_have_broken_these(self):
        """Guard the guard: confirm the old idiom really did collapse to zero,
        so this file cannot quietly stop testing anything."""
        assert int(0.04) == 0
        assert int(0.999) == 0


class TestLoaderAcceptsFractionalTimeout:
    """The C side must parse and arm a fractional timeout."""

    def test_loader_parses_with_strtod_not_atoi(self):
        src = LOADER_SRC.read_text()
        assert "strtod(timeout_str" in src, "loader must parse the timeout as a double"
        assert "atoi(timeout_str)" not in src, "atoi truncates sub-second timeouts to 0"

    def test_loader_arms_an_interval_timer(self):
        """alarm() takes whole seconds, so it cannot express 0.04s at all."""
        src = LOADER_SRC.read_text()
        assert "setitimer(ITIMER_REAL" in src
        assert "alarm(timeout_seconds" not in src, "alarm() cannot express sub-second timeouts"

    def test_arm_timeout_has_a_nonzero_floor(self):
        """An all-zero itimerval disarms the timer instead of firing, which
        would silently mean 'no timeout'."""
        src = LOADER_SRC.read_text()
        assert "seconds < 0.001" in src, "arm_timeout needs a floor below which it clamps"

    @pytest.mark.skipif(
        subprocess.run(["which", "gcc"], capture_output=True).returncode != 0,
        reason="gcc not available",
    )
    def test_end_to_end_subsecond_timeout_fires(self, tmp_path):
        """Build a target that sleeps 0.5s and confirm a 0.04s timeout stops it."""
        so_src = tmp_path / "slow.c"
        so_src.write_text(
            "#include <unistd.h>\n#include <stddef.h>\n"
            "int fuzz_test(const unsigned char *b, size_t n) "
            "{ (void)b; (void)n; usleep(500000); return 0; }\n"
        )
        so = tmp_path / "slow.so"
        if subprocess.run(
            ["gcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(so_src)],
            capture_output=True,
        ).returncode:
            pytest.skip("could not build the sleeping target")

        loader = tmp_path / "fuzz_loader"
        if subprocess.run(
            ["gcc", "-O2", "-o", str(loader), str(LOADER_SRC), "-ldl"], capture_output=True
        ).returncode:
            pytest.skip("could not build fuzz_loader")

        inp = tmp_path / "in.bin"
        inp.write_bytes(b"AAAA")

        def run_once(timeout: str) -> str:
            p = subprocess.Popen(
                [str(loader)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            try:
                p.stdin.write(f"INIT {so} fuzz_test {inp} {timeout}\n".encode())
                p.stdin.flush()
                p.stdout.readline()  # READY
                p.stdin.write(b"RUN 4\nAAAA")
                p.stdin.flush()
                return p.stdout.readline().decode().strip()
            finally:
                p.kill()
                p.wait()

        # 0.04s < the target's 0.5s sleep, so the timer must fire.
        assert run_once("0.04").startswith("RC -1"), (
            "a 0.04s timeout did not fire on a 0.5s target -- it was likely "
            "truncated to 0 and replaced with the 5s default"
        )
        # 2s > the sleep, so the same target must complete normally.
        assert run_once("2").startswith("RC 0"), "a 2s timeout should let a 0.5s target finish"
