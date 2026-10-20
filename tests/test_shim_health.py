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


# ── cmplog drop breakdown ──────────────────────────────────────────────


def test_drop_warning_names_the_cause():
    counters = _healthy()
    counters[ShimField.CMPLOG_DROPPED] = 1_143_458_223 + 4
    counters[ShimField.CMPLOG_CONTENDED] = 1_143_458_223
    counters[ShimField.CMPLOG_WRITE_FAILED] = 4

    text = "\n".join(health_issues(counters, Attach.EXPECTED))

    assert "1143458227 cmplog record(s) dropped" in text
    assert "1143458223 another thread held the lock" in text
    assert "4 failed flush(es)" in text
    assert "same-thread" not in text  # zero parts are left out


def test_drop_warning_without_breakdown_is_unchanged():
    counters = _healthy()
    counters[ShimField.CMPLOG_DROPPED] = 5

    text = "\n".join(health_issues(counters, Attach.EXPECTED))

    assert text.endswith(
        "5 cmplog record(s) dropped (writer contention, failed write or lost sink)"
    )


def test_drop_warning_names_lost_sink():
    counters = _healthy()
    counters[ShimField.CMPLOG_DROPPED] = 2
    counters[ShimField.CMPLOG_NO_SINK] = 2

    text = "\n".join(health_issues(counters, Attach.EXPECTED))

    assert "2 batch(es) lost when the sink could not be reopened" in text


def test_watch_one_shot_warning_carries_the_breakdown():
    from fuzzer_tool.core.shim_health import ShimWatch

    counters = _healthy()
    counters[ShimField.CMPLOG_DROPPED] = 3
    counters[ShimField.CMPLOG_REENTRANT] = 3

    fresh = ShimWatch().counters(counters, Attach.EXPECTED)

    assert any("3 same-thread re-entry" in line for line in fresh)


THREADED_SRC = r"""
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
volatile uint64_t g1 = 7, g2 = 9, sink;
#define S(i) if ((a ^ (i*2654435761u)) == (b + i)) sink++;
__attribute__((noinline)) static void run(uint64_t n) {
    for (uint64_t k = 0; k < n; k++) {
        uint64_t a = g1 + (k & 63), b = g2 + ((k >> 3) & 63);
        S(1) S(2) S(3) S(4)
    }
}
static void *th(void *x) { (void)x; run(3000000); return 0; }
int main(void) {
    pthread_t t; pthread_create(&t, 0, th, 0); run(3000000); pthread_join(t, 0);
    printf("%llu %llu %llu %llu\n",
        (unsigned long long)__afl_health[__AFL_HEALTH_CMPLOG_DROPPED],
        (unsigned long long)__afl_health[__AFL_HEALTH_CMPLOG_REENTRANT],
        (unsigned long long)__afl_health[__AFL_HEALTH_CMPLOG_CONTENDED],
        (unsigned long long)__afl_health[__AFL_HEALTH_CMPLOG_WRITE_FAILED]);
    return 0;
}
"""


def test_second_thread_drops_are_counted_as_contended(tmp_path):
    """Measured: a second thread running instrumented code loses most records
    to the try-lock; the shim must file them under CONTENDED, not REENTRANT."""
    import shutil

    cc = shutil.which("gcc")
    if cc is None:
        import pytest

        pytest.skip("gcc not available")
    shim = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"
    src = tmp_path / "t.c"
    src.write_text(THREADED_SRC)
    exe = tmp_path / "t"
    subprocess.run(
        [cc, "-O2", "-w", "-D__AFL_CMPLOG=1", "-fsanitize-coverage=trace-cmp",
         "-include", str(shim), str(src), "-o", str(exe), "-ldl", "-lpthread"],
        check=True,
    )
    env = {**_clean_env(), "_CMPLOG_OUT": "/dev/null"}

    out = subprocess.run([str(exe)], env=env, capture_output=True, text=True, check=True).stdout
    dropped, reentrant, contended, write_failed = (int(x) for x in out.split())

    assert contended > 0
    assert reentrant == 0
    assert write_failed == 0
    # The counters are non-atomic, so two threads may lose an update; the sum
    # is only exact up to that noise.
    assert abs(dropped - (reentrant + contended + write_failed)) <= max(8, dropped // 1000)


STOLEN_FD_SRC = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
volatile uint64_t g1 = 7, g2 = 9, sink;
#define S(i) if ((a ^ (i*2654435761u)) == (b + i)) sink++;
__attribute__((noinline)) static void run(uint64_t n) {
    for (uint64_t k = 0; k < n; k++) {
        uint64_t a = g1 + (k & 63), b = g2 + ((k >> 3) & 63);
        S(1) S(2) S(3) S(4)
    }
}
int main(void) {
    run(10);                                  /* records pending in the buffer */
    int other = open("/dev/null", O_WRONLY);  /* replace the sink under the shim */
    dup2(other, __afl_cmplog_fd);
    setenv("_CMPLOG_OUT", "/nonexistent-dir/cmplog.bin", 1);
    __afl_cmplog_flush();
    printf("%llu %llu %llu %llu %llu\n",
        (unsigned long long)__afl_health[__AFL_HEALTH_CMPLOG_DROPPED],
        (unsigned long long)__afl_health[__AFL_HEALTH_CMPLOG_REENTRANT],
        (unsigned long long)__afl_health[__AFL_HEALTH_CMPLOG_CONTENDED],
        (unsigned long long)__afl_health[__AFL_HEALTH_CMPLOG_WRITE_FAILED],
        (unsigned long long)__afl_health[__AFL_HEALTH_CMPLOG_NO_SINK]);
    return 0;
}
"""


def test_lost_sink_batch_is_counted_as_no_sink(tmp_path):
    """The sink fd is replaced under the shim and the reopen fails: the pending
    batch used to vanish with no counter."""
    import shutil

    import pytest

    cc = shutil.which("gcc")
    if cc is None:
        pytest.skip("gcc not available")
    shim = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"
    src = tmp_path / "s.c"
    src.write_text(STOLEN_FD_SRC)
    exe = tmp_path / "s"
    subprocess.run(
        [cc, "-O2", "-w", "-D__AFL_CMPLOG=1", "-fsanitize-coverage=trace-cmp",
         "-include", str(shim), str(src), "-o", str(exe), "-ldl", "-lpthread"],
        check=True,
    )
    env = {**_clean_env(), "_CMPLOG_OUT": str(tmp_path / "cmplog.bin")}

    out = subprocess.run([str(exe)], env=env, capture_output=True, text=True, check=True).stdout
    dropped, reentrant, contended, write_failed, no_sink = (int(x) for x in out.split())

    assert no_sink == 1
    assert (reentrant, contended, write_failed) == (0, 0, 0)
    assert dropped == 1
