"""Network (TCP/UDP) mode was removed; nothing may still expose it."""

import importlib.util
import subprocess
import sys

NET_FLAGS = ("--net-host", "--net-port", "--net-proto", "--net-keepalive", "--net-settle-ms")


def _fuzz_help() -> str:
    r = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", "--help"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    return r.stdout


def test_regression_network_mode_removed():
    """Falsification: no --net-* flag in help, no adapter module."""
    out = _fuzz_help()
    assert "--persistent" in out  # control: help lists sibling backends
    assert not [f for f in NET_FLAGS if f in out]
    assert importlib.util.find_spec("fuzzer_tool.adapters.network") is None


def test_regression_net_flag_rejected():
    """Adversarial: passing a removed flag is an argparse error, not ignored."""
    r = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", "/bin/true", "--net-host", "127.0.0.1"],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "--net-host" in r.stderr
