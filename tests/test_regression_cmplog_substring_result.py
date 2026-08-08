"""Regression: memmem/strstr/strcasestr hardcoded their comparison result.

``log_cmp`` drops entries whose ``result == 0``. That is the filter keeping
already-solved comparisons out of the pair pool -- there is nothing for
input-to-state to solve once the operands already match, and a solved pair
displaces an unsolved one under ``CMPLOG_PAIRS_MAX``.

``memcmp``/``strcmp`` pass the real libc return value, so the filter works
for them. The three substring interceptors passed a literal ``-1``, so a
*successful* match was logged as if it were still unsolved. On any target
that searches for a token it will usually find (a magic string in a valid
seed, a delimiter scan in a parser), that is a steady stream of pool
entries the search operators can never make progress on.

The fix passes ``result ? 0 : -1`` -- ``result`` being the returned pointer,
non-NULL exactly when the substring was found.
"""

import contextlib
import os
import subprocess
import tempfile

from tests.conftest import requires_clang

SHIM_REL = os.path.join(
    os.path.dirname(__file__), "..", "src", "fuzzer_tool", "adapters", "cmplog_shim.c"
)

# Distinctive operands so libc's own internal string work cannot be mistaken
# for the harness's calls.
_FOUND = b"ZQFOUNDZQ"
_MISSING = b"ZQABSENTZQ"

_HARNESS = r"""
#define _GNU_SOURCE
#include <stddef.h>
#include <string.h>

/* The results must be consumed: memmem/strstr are pure, so clang deletes
 * calls whose return value is unused and the interceptors never fire. */
volatile size_t sink = 0;

int main(void) {
    static volatile char hay[] = "prefix ZQFOUNDZQ suffix";
    const char *h = (const char *)hay;
    size_t hl = strlen(h);

    /* hits -- must NOT reach the pool */
    sink += (size_t)memmem(h, hl, "ZQFOUNDZQ", 9);
    sink += (size_t)strstr(h, "ZQFOUNDZQ");
    sink += (size_t)strcasestr(h, "zqfoundzq");

    /* misses -- must reach the pool */
    sink += (size_t)memmem(h, hl, "ZQABSENTZQ", 10);
    sink += (size_t)strstr(h, "ZQABSENTZQ");
    sink += (size_t)strcasestr(h, "zqabsentzq");
    return 0;
}
"""


def _build(tmpdir, source, out_name, extra):
    path = os.path.join(tmpdir, out_name)
    src = os.path.join(tmpdir, out_name + ".c")
    with open(src, "w") as fh:
        fh.write(source)
    r = subprocess.run(["clang", "-O1", "-o", path, src, *extra], capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr.decode()[:400]
    return path


def _logged_operands(tmpdir):
    """Run the harness under the shim; return the set of logged operands."""
    with open(SHIM_REL) as fh:
        shim_source = fh.read()
    shim = _build(tmpdir, shim_source, "shim.so", ["-shared", "-fPIC", "-ldl"])
    harness = _build(tmpdir, _HARNESS, "harness", [])
    log_path = os.path.join(tmpdir, "out.cmplog")

    env = dict(os.environ, LD_PRELOAD=shim, _CMPLOG_OUT=log_path)
    env.pop("ASAN_OPTIONS", None)
    r = subprocess.run([harness], env=env, capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr.decode()[:400]

    operands = set()
    with open(log_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "CMP":
                for hexstr in (parts[1], parts[2]):
                    with contextlib.suppress(ValueError):
                        operands.add(bytes.fromhex(hexstr))
    return operands


@requires_clang
def test_successful_substring_match_is_not_pooled():
    with tempfile.TemporaryDirectory() as tmpdir:
        operands = _logged_operands(tmpdir)
    solved = [op for op in operands if _FOUND in op]
    assert not solved, (
        f"a substring match that succeeded reached the pool: {solved!r} -- "
        "log_cmp's result==0 filter was bypassed by a hardcoded -1"
    )


@requires_clang
def test_failed_substring_match_is_still_pooled():
    """The filter must not swallow the entries that are actually useful."""
    with tempfile.TemporaryDirectory() as tmpdir:
        operands = _logged_operands(tmpdir)
    assert any(_MISSING in op for op in operands), (
        "an unsolved substring comparison never reached the pool; the "
        "interceptors log nothing at all"
    )
