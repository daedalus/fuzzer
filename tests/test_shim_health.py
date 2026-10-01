"""Shim health checks: static marker, runtime counters, and their wiring.

Static: ``__afl_scoped_crash_handler`` tells a current shim from one whose
crash handler jumps outside ``__afl_guarded_call``.
Runtime: ``__afl_shim_health()`` exports counters for failures the shim
survives silently (refused segments, dropped cmplog records, stray signals).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from fuzzer_tool.core.elf import detect_scoped_crash_handler
from fuzzer_tool.core.shim_health import Attach, ShimField, health_issues, stale_shim_issues
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

ENTRY_SRC = """
#include <stdint.h>
#include <stddef.h>
int ok_entry(const uint8_t *d, size_t n) { (void)d; return (int)n; }
"""


def _healthy() -> list[int]:
    counters = [0] * len(ShimField)
    counters[ShimField.ATTACHED] = 1
    counters[ShimField.MAP_ENTRIES] = 8192
    return counters


# ── Static marker ──────────────────────────────────────────────────────


@requires_clang
def test_scoped_marker_present_in_current_shim(tmp_path):
    exe = _build(tmp_path, "int main(void) { return 0; }", "-fsanitize-coverage=trace-pc-guard")

    assert detect_scoped_crash_handler(str(exe)) is True
    assert stale_shim_issues(str(exe)) == []


@requires_clang
def test_scoped_marker_absent_in_plain_binary(tmp_path):
    src = tmp_path / "plain.c"
    src.write_text("int main(void) { return 0; }\n")
    exe = tmp_path / "plain"
    subprocess.run(["clang", str(src), "-o", str(exe)], check=True)

    assert detect_scoped_crash_handler(str(exe)) is False
    issues = stale_shim_issues(str(exe))
    assert len(issues) == 1 and "__afl_scoped_crash_handler" in issues[0]


def test_unreadable_target_is_unknown_not_stale(tmp_path):
    """Adversarial: a non-ELF path must not raise or produce a false alarm."""
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\x7fELF" + b"\xff" * 12)

    assert detect_scoped_crash_handler(str(junk)) is None
    assert stale_shim_issues(str(junk)) == []
    assert stale_shim_issues(str(tmp_path / "missing")) == []


# ── Counter interpretation ─────────────────────────────────────────────


def test_healthy_counters_raise_nothing():
    assert health_issues(_healthy(), Attach.EXPECTED) == []


def test_each_anomaly_is_reported():
    counters = _healthy()
    counters[ShimField.ATTACHED] = 0
    counters[ShimField.SEG_REJECTED] = 2
    counters[ShimField.CMPLOG_DROPPED] = 5
    counters[ShimField.STRAY_SIGNALS] = 1

    text = "\n".join(health_issues(counters, Attach.EXPECTED))

    assert "not attached" in text
    assert "2 SHM segment" in text
    assert "5 cmplog record" in text
    assert "1 crash signal" in text


def test_detached_is_fine_when_not_expected():
    counters = _healthy()
    counters[ShimField.ATTACHED] = 0

    assert health_issues(counters, Attach.NOT_EXPECTED) == []


def test_short_or_long_counter_vectors():
    """Adversarial: an older shim exports fewer fields, a newer one more."""
    assert health_issues([1, 8192], Attach.EXPECTED) == []
    longer = [*_healthy(), 99, 99]
    assert health_issues(longer, Attach.EXPECTED) == []
    assert health_issues([], Attach.NOT_EXPECTED) == []


# ── Runtime export ─────────────────────────────────────────────────────


@requires_clang
def test_partial_read_respects_caller_bound(tmp_path):
    """Adversarial: n smaller than the field count writes only n slots."""
    exe = _build(
        tmp_path,
        """
        #include <stdio.h>
        int main(void) {
            uint64_t h[4] = {7, 7, 7, 7};
            uint32_t total = __afl_shim_health(h, 2);
            uint32_t again = __afl_shim_health(NULL, 0);
            printf("%u %u %llu %llu\\n", total, again,
                   (unsigned long long)h[2], (unsigned long long)h[3]);
            return 0;
        }
        """,
        "-fsanitize-coverage=trace-pc-guard",
    )
    out = subprocess.run([str(exe)], capture_output=True, text=True, env=_clean_env(), check=True)
    total, again, h2, h3 = (int(x) for x in out.stdout.split())

    assert total == again >= len(ShimField)
    assert (h2, h3) == (7, 7)


@requires_clang
def test_read_shim_health_via_ctypes(tmp_path):
    so = _build(tmp_path, ENTRY_SRC, "-fsanitize-coverage=trace-pc-guard", shared=True)
    plain = tmp_path / "plain.so"
    (tmp_path / "plain.c").write_text("int f(void) { return 0; }\n")
    subprocess.run(
        ["clang", "-shared", "-fPIC", str(tmp_path / "plain.c"), "-o", str(plain)], check=True
    )
    script = """
        import ctypes, os, signal, sys
        from fuzzer_tool.adapters.inprocess import read_shim_health
        signal.signal(signal.SIGSEGV, lambda *a: None)
        lib = ctypes.CDLL(sys.argv[1])
        os.kill(os.getpid(), signal.SIGSEGV)   # stray: counted, then handed back
        print(*read_shim_health(lib))
        print(read_shim_health(ctypes.CDLL(sys.argv[2])))
    """
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(so), str(plain)],
        capture_output=True, text=True, timeout=60, env=_clean_env(),
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    first, second = r.stdout.splitlines()
    counters = [int(x) for x in first.split()]

    assert counters[ShimField.ATTACHED] == 0
    assert counters[ShimField.MAP_ENTRIES] == 8192
    assert counters[ShimField.STRAY_SIGNALS] == 1
    assert second == "None"


# ── Wiring ─────────────────────────────────────────────────────────────


def test_report_section_lists_runtime_issues():
    from fuzzer_tool.services.report import _shim_health

    counters = _healthy()
    counters[ShimField.SEG_REJECTED] = 1
    runner = SimpleNamespace(shim_health=lambda: tuple(counters))
    f = SimpleNamespace(_inprocess_runner=runner, use_coverage=True)

    text = _shim_health(f)

    assert "Shim Health" in text and "1 SHM segment" in text


def test_report_section_silent_when_healthy_or_unavailable():
    from fuzzer_tool.services.report import _shim_health

    healthy = SimpleNamespace(shim_health=lambda: tuple(_healthy()))
    unavailable = SimpleNamespace(shim_health=lambda: None)

    assert _shim_health(SimpleNamespace(_inprocess_runner=healthy, use_coverage=True)) == ""
    assert _shim_health(SimpleNamespace(_inprocess_runner=unavailable, use_coverage=True)) == ""
    assert _shim_health(SimpleNamespace(_inprocess_runner=None, use_coverage=True)) == ""


@requires_clang
def test_preflight_warns_on_stale_shim(tmp_path, capsys):
    from fuzzer_tool.services.fuzzer import Fuzzer

    src = tmp_path / "old.c"
    src.write_text("int main(void) { return 0; }\n")
    exe = tmp_path / "old"
    subprocess.run(["clang", str(src), "-o", str(exe)], check=True)

    Fuzzer._warn_stale_shim(SimpleNamespace(), str(exe))

    assert "__afl_scoped_crash_handler" in capsys.readouterr().out


def test_shim_field_order_matches_c_enum():
    """The index order is the ABI: parse it out of the C source."""
    shim = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"
    text = shim.read_text()
    body = text[text.index("__AFL_HEALTH_ATTACHED = 0") : text.index("__AFL_HEALTH_FIELDS\n")]
    c_names = [
        tok.strip().strip(",").removeprefix("__AFL_HEALTH_").split(" ")[0]
        for tok in body.split("\n")
    ]
    c_names = [n for n in c_names if n]

    assert c_names == [f.name for f in ShimField]
