"""trace-cmp callbacks return at once when nothing consumes them.

With records paused (the collector parses one round in twenty) or cmplog
off, every callback still tested four or five settings: 33 Ir per
``const_cmp4``, 82 per ``switch``, 22% of ffmpeg_read's instructions. One
flag, ``__afl_cmp_on``, now gates them: set at init from the configured
consumers (counts, sites, records, COMPCOV level), and moved after that
only by ``__cmplog_pause``. A closed or stolen fd leaves it on, which is
slower but never drops a record a consumer would have taken.
"""

from __future__ import annotations

import subprocess

import pytest

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

# Distinct operands per phase: the record dedup keys on (a, b, pc).
RUN, PAUSED, RESUMED = 0x11223344, 0x55667788, 0x0A0B0C0D
#: Little-endian hex of each operand, as a CMP record prints it.
HEX = {v: v.to_bytes(4, "little").hex() for v in (RUN, PAUSED, RESUMED)}

_DRIVER = f"""
#include <stdio.h>
static void phase(uint32_t a) {{
    __sanitizer_cov_trace_const_cmp4(a, a + 1);
    __tracecmp_flush();
}}
int main(void) {{
    printf("on %d\\n", __afl_cmp_on);
    phase({RUN}u);
    __cmplog_pause(1);
    printf("paused %d\\n", __afl_cmp_on);
    phase({PAUSED}u);
    __cmplog_pause(0);
    printf("resumed %d\\n", __afl_cmp_on);
    phase({RESUMED}u);
    return 0;
}}
"""


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("cmp"), _DRIVER, "-D__AFL_CMPLOG=1")


def _flags(exe, **env: str) -> dict[str, int]:
    """Run the driver; return the printed flag per phase."""
    r = subprocess.run(
        [str(exe)], env=_clean_env(**env), capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0, r.stderr
    return {k: int(v) for k, v in (line.split() for line in r.stdout.splitlines())}


@requires_clang
def test_regression_cmp_off_without_consumers(exe):
    """Falsification: nothing configured, so the callbacks have nothing to do."""
    assert _flags(exe) == {"on": 0, "paused": 0, "resumed": 0}


@requires_clang
def test_records_follow_pause(exe, tmp_path):
    """Records configured: written running, dropped paused, back on resume."""
    log = tmp_path / "cmp.log"
    flags = _flags(exe, _CMPLOG_OUT=str(log))
    text = log.read_text()

    assert flags == {"on": 1, "paused": 0, "resumed": 1}
    assert HEX[RUN] in text
    assert HEX[PAUSED] not in text
    assert HEX[RESUMED] in text


@requires_clang
@pytest.mark.parametrize(
    "env",
    [
        {"_CMPLOG_COUNTS": "counts.bin"},
        {"_CMPLOG_SITE_COUNTS": "sites.bin"},
        {"__AFL_COMPCOV_LEVEL": "1"},
    ],
)
def test_other_consumers_ignore_pause(exe, tmp_path, env):
    """Adversarial: pause stops records only; counts, sites and COMPCOV keep the flag on."""
    env = {k: str(tmp_path / v) if k.startswith("_CMPLOG") else v for k, v in env.items()}
    assert _flags(exe, **env) == {"on": 1, "paused": 1, "resumed": 1}
