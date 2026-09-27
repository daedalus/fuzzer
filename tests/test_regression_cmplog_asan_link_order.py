"""Regression: the cmplog shim preload silenced dynamic-runtime ASAN targets.

A target linked against a shared ASAN runtime (gcc default, clang
``-shared-libasan``) aborts before ``main`` when anything is preloaded ahead
of that runtime ("ASan runtime does not come first in initial library
list"). The cmplog shim is preloaded on every exec, so every exec died at
startup: 0 crashes, 0 edges, no error.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core.cmplog import CmplogCollector

ASAN_SRC = Path(__file__).parent.parent / "targets" / "asan_target.c"
LINK_ORDER_OFF = "verify_asan_link_order=0"


def _collector() -> CmplogCollector:
    c = CmplogCollector()
    c._shim_path = "/tmp/fake_shim.so"
    return c


class TestSetupEnv:
    def test_regression_link_order_check_disabled(self):
        env = _collector().setup_env({"PATH": "/usr/bin"})
        assert LINK_ORDER_OFF in env["ASAN_OPTIONS"].split(":")

    def test_existing_options_kept(self):
        env = _collector().setup_env({"ASAN_OPTIONS": "detect_leaks=0"})
        assert env["ASAN_OPTIONS"].split(":") == ["detect_leaks=0", LINK_ORDER_OFF]

    def test_adversarial_user_choice_not_overridden(self):
        """An explicit verify_asan_link_order is the user's call."""
        env = _collector().setup_env({"ASAN_OPTIONS": "verify_asan_link_order=1"})
        assert env["ASAN_OPTIONS"] == "verify_asan_link_order=1"

    def test_falsification_no_shim_no_change(self):
        """Without a shim nothing is preloaded, so nothing to relax."""
        env = CmplogCollector().setup_env({"PATH": "/usr/bin"})
        assert "ASAN_OPTIONS" not in env


class TestSetupEnvForRun:
    def test_set_then_restored(self, monkeypatch):
        monkeypatch.setenv("ASAN_OPTIONS", "detect_leaks=0")
        c = _collector()
        c.setup_env_for_run()
        assert LINK_ORDER_OFF in os.environ["ASAN_OPTIONS"].split(":")

        c.restore_env()
        assert os.environ["ASAN_OPTIONS"] == "detect_leaks=0"
        if c.log_path:
            Path(c.log_path).unlink(missing_ok=True)

    def test_absent_key_restored_as_absent(self, monkeypatch):
        monkeypatch.delenv("ASAN_OPTIONS", raising=False)
        c = _collector()
        c.setup_env_for_run()
        c.restore_env()
        assert "ASAN_OPTIONS" not in os.environ
        if c.log_path:
            Path(c.log_path).unlink(missing_ok=True)


def _shared_asan_target(out: Path) -> Path:
    """clang ``-shared-libasan`` build of asan_target.c, or skip."""
    if not shutil.which("clang"):
        pytest.skip("clang not available")
    rt = subprocess.run(
        ["clang", "-print-file-name=libclang_rt.asan-x86_64.so"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not os.path.isfile(rt):
        pytest.skip("shared ASAN runtime not installed")
    r = subprocess.run(
        [
            "clang",
            "-g",
            "-w",
            "-fsanitize=address",
            "-shared-libasan",
            f"-Wl,-rpath,{os.path.dirname(rt)}",
            "-o",
            str(out),
            str(ASAN_SRC),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return out


def test_regression_shared_asan_target_crashes_found(tmp_path):
    target = _shared_asan_target(tmp_path / "asan_target")
    seeds = tmp_path / "corpus" / "seeds"
    seeds.mkdir(parents=True)
    crashes = tmp_path / "crashes"
    crashes.mkdir()
    (seeds / "seed").write_bytes(b"BUG!S")  # stack-buffer-overflow on exec

    r = subprocess.run(
        ["python3", "-m", "fuzzer_tool", "fuzz", str(target)]
        + ["-d", str(tmp_path / "corpus"), "-o", str(crashes)]
        + ["-n", "20", "-t", "2", "-s", "42"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert r.returncode == 0, r.stderr
    assert list(crashes.glob("crash_*")), r.stdout[-2000:]
