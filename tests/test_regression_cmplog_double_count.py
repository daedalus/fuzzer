"""Regression: a compiled-in cmplog target counted each comparison twice.

The fuzzer also LD_PRELOADed its shim. The target's own ``memcmp``
interceptor forwards through ``dlsym(RTLD_NEXT)``, which lands on the
preloaded copy, and both copies dump: one ``memcmp`` call read as
``memcmp: 2``. Per-exec vectors and cmp-progress saw doubled counts.

Oracle: the bare binary, run with only ``_CMPLOG_COUNTS`` set.
"""

import os
import shutil
import subprocess

import pytest

from tests.conftest import requires_clang
from tests.test_regression_cmplog_forkserver import (
    INPUTS,
    Exec,
    Sink,
    _fuzzer,
    _teardown,
    target,  # noqa: F401  (module fixture)
)


def _bare_counts(exe: str, data: bytes, tmp_path) -> dict[str, int]:
    """Fired counts from the binary alone: no fuzzer, no preload."""
    counts = tmp_path / "bare.counts"
    counts.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    env["_CMPLOG_COUNTS"] = str(counts)
    subprocess.run([exe], input=data, env=env, capture_output=True, timeout=10)

    fired: dict[str, int] = {}
    for line in counts.read_text().splitlines():
        _tag, name, n_fired, _n_asserted = line.split()
        fired[name] = fired.get(name, 0) + int(n_fired)
    return fired


def _fuzzer_counts(exe: str, tmp_path, backend: Exec) -> list[dict[str, int]]:
    """Per-exec fired counts over INPUTS through the fuzzer."""
    f = _fuzzer(exe, tmp_path, Sink.FIFO, backend)
    try:
        f._cmplog.collect_counts()
        out = []
        for data in INPUTS:
            f._runner.run_target(data)
            out.append(f._cmplog.collect_counts()[0])
        return out
    finally:
        _teardown(f)


@requires_clang
@pytest.mark.parametrize("backend", list(Exec), ids=lambda b: b.value)
def test_regression_cmplog_compiled_in_counts_once(target, tmp_path, backend):  # noqa: F811
    """Falsification: fuzzer vectors equal the bare binary's, per input."""
    bare = [_bare_counts(target, d, tmp_path) for d in INPUTS]
    again = [_bare_counts(target, d, tmp_path) for d in INPUTS]
    assert bare == again, "control: bare oracle not reproducible"

    assert _fuzzer_counts(target, tmp_path / backend.value, backend) == bare


_PLAIN = """
#include <stdio.h>
#include <string.h>

int main(void) {
    char buf[64];
    size_t n = fread(buf, 1, sizeof(buf), stdin);
    return n >= 4 && memcmp(buf, "QXZT", 4) == 0;
}
"""


@pytest.fixture(scope="module")
def plain_target(tmp_path_factory):
    """No shim compiled in: interception can only come from the preload."""
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    d = tmp_path_factory.mktemp("cmplog_plain")
    src = d / "p.c"
    src.write_text(_PLAIN)
    exe = d / "p"
    r = subprocess.run(
        ["clang", "-O0", "-fno-builtin", "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")
    return str(exe)


@requires_clang
def test_uninstrumented_target_keeps_preload(plain_target, tmp_path):
    """Adversarial: dropping the preload must not blind a shim-less target."""
    for fired in _fuzzer_counts(plain_target, tmp_path, Exec.SPAWN):
        assert fired.get("memcmp") == 1, fired
