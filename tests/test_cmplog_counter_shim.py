"""End-to-end test of the shim's comparison counters against a real binary.

The Python-side tests parse a hand-written sidecar; these compile the shim,
run a target through it, and check the numbers the interceptors actually
produce. Three properties matter and none is visible from the record stream:

1. **Satisfied layer-1 comparisons are counted.** ``__afl_cmplog_bytes``
   drops ``result == 0`` on purpose -- a solved comparison is pollution for
   the input-to-state pair pool -- so the record proving a memcmp matched is
   the one never written. The counter sits ahead of that filter.
2. **A switch dispatch counts once, not once per case.** The record writer
   emits one CMP line per case, so counting records would report a ten-case
   jump table as ten comparisons with nine unsatisfied -- the arm actually
   taken drowned in the arms that never could be.
3. **Dumps are deltas.** The shim zeroes as it writes, which is what lets
   the collector sum blindly across subprocess runs (one dump each) and
   direct_lite runs (many dumps, one process) without knowing which it has.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_gcc

REPO = Path(__file__).parent.parent
AFL_SHIM = REPO / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

pytestmark = requires_gcc


def _parse_counts(path: Path) -> dict[str, tuple[int, int]]:
    """Sum the shim's per-dump deltas, the way collect_counts does."""
    out: dict[str, tuple[int, int]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] != "CNT":
            continue
        fired, asserted = out.get(parts[1], (0, 0))
        out[parts[1]] = (fired + int(parts[2]), asserted + int(parts[3]))
    return out


@pytest.fixture(scope="module")
def shim_obj(tmp_path_factory) -> Path:
    """The preload-only shim, compiled once for the module."""
    obj = tmp_path_factory.mktemp("shim") / "afl_shim.o"
    proc = subprocess.run(
        ["gcc", "-O2", "-fPIC", "-D__AFL_PRELOAD_ONLY", "-c", str(AFL_SHIM), "-o", str(obj)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"shim did not compile: {proc.stderr[-400:]}")
    return obj


def _build_and_run(tmp_path: Path, shim_obj: Path, source: str, extra_cflags=()) -> dict:
    src = tmp_path / "target.c"
    src.write_text(source)
    exe = tmp_path / "target"
    proc = subprocess.run(
        ["gcc", "-O1", *extra_cflags, str(src), str(shim_obj), "-o", str(exe), "-ldl"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"target did not build: {proc.stderr[-400:]}")

    counts = tmp_path / "counts.txt"
    env = dict(os.environ)
    env["_CMPLOG_COUNTS"] = str(counts)
    env["_CMPLOG_OUT"] = str(tmp_path / "records.cmplog")
    run = subprocess.run([str(exe)], capture_output=True, text=True, env=env, timeout=60)
    assert run.returncode == 0, run.stderr
    assert counts.exists(), "shim wrote no counts sidecar"
    return _parse_counts(counts)


NOBUILTIN = ("-fno-builtin-memcmp", "-fno-builtin-strcmp", "-fno-builtin-strstr")


def test_satisfied_layer1_comparisons_are_counted(tmp_path, shim_obj):
    """The half of the picture the record stream discards entirely."""
    source = """
    #include <string.h>
    #include <stdio.h>
    int main(void) {
        volatile int r = 0;
        r += memcmp("HELLO_WORLD", "HELLO_WORLD", 11) == 0;   /* satisfied */
        r += memcmp("HELLO_WORLD", "HELLO_WORLD", 11) == 0;   /* satisfied */
        r += memcmp("HELLO_WORLD", "GOODBYE_ABC", 11) == 0;   /* not */
        r += strcmp("HELLO_WORLD", "HELLO_WORLD") == 0;       /* satisfied */
        r += strcmp("HELLO_WORLD", "nope") == 0;              /* not */
        printf("%d\\n", r);
        return 0;
    }
    """
    counts = _build_and_run(tmp_path, shim_obj, source, NOBUILTIN)
    assert counts["memcmp"] == (3, 2)
    assert counts["strcmp"] == (2, 1)


def test_search_family_asserts_on_found(tmp_path, shim_obj):
    """For strstr/memmem/memchr, "asserted" means the needle was located.

    These are the interceptors that pass ``result ? 0 : -1`` to the record
    writer, so every successful search is dropped from the stream.
    """
    source = """
    #include <string.h>
    #include <stdio.h>
    int main(void) {
        const char *h = "HELLO_WORLD";
        volatile int r = 0;
        r += strstr(h, "WORLD") != NULL;
        r += strstr(h, "HELLO") != NULL;
        r += strstr(h, "LO_WO") != NULL;
        r += strstr(h, "ZZZZ")  != NULL;
        printf("%d\\n", r);
        return 0;
    }
    """
    counts = _build_and_run(tmp_path, shim_obj, source, NOBUILTIN)
    assert counts["strstr"] == (4, 3)


def test_switch_counts_dispatches_not_cases(tmp_path, shim_obj):
    """Calls the callback directly against the sancov switch ABI.

    Driving it through a real ``switch`` would make the test hostage to
    whether the compiler chose a jump table on that optimization level; the
    property under test is the shim's, not the compiler's.
    """
    source = """
    #include <stdint.h>
    #include <stdio.h>
    void __sanitizer_cov_trace_switch(uint64_t val, void *cases);
    int main(void) {
        /* ABI: cases[0] = count, cases[1] = bit width, cases[2..] = values */
        uint64_t cases[12] = {10, 32, 0,1,2,3,4,5,6,7,8,9};
        __sanitizer_cov_trace_switch(1,  cases);   /* matches */
        __sanitizer_cov_trace_switch(5,  cases);   /* matches */
        __sanitizer_cov_trace_switch(9,  cases);   /* matches */
        __sanitizer_cov_trace_switch(77, cases);   /* default */
        __sanitizer_cov_trace_switch(88, cases);   /* default */
        printf("done\\n");
        return 0;
    }
    """
    counts = _build_and_run(tmp_path, shim_obj, source)
    # Five dispatches, three matched -- NOT the 50 CMP records the same run
    # writes to the record stream (5 dispatches x 10 cases).
    assert counts["trace_switch"] == (5, 3)


def test_dumps_are_deltas(tmp_path, shim_obj):
    """Three sync points, three deltas, summing to the true totals."""
    source = """
    #include <string.h>
    #include <stdio.h>
    void __cmplog_reset(void);
    int main(void) {
        volatile int r = 0;
        r += memcmp("aaaa", "aaaa", 4) == 0;
        __cmplog_reset();
        r += memcmp("aaaa", "bbbb", 4) == 0;
        r += memcmp("cccc", "dddd", 4) == 0;
        __cmplog_reset();
        r += memcmp("eeee", "eeee", 4) == 0;
        printf("%d\\n", r);   /* the destructor dumps the tail */
        return 0;
    }
    """
    counts_file = tmp_path / "counts.txt"
    src = tmp_path / "target.c"
    src.write_text(source)
    exe = tmp_path / "target"
    proc = subprocess.run(
        ["gcc", "-O1", "-fno-builtin-memcmp", str(src), str(shim_obj), "-o", str(exe), "-ldl"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"target did not build: {proc.stderr[-400:]}")

    env = dict(os.environ)
    env["_CMPLOG_COUNTS"] = str(counts_file)
    env["_CMPLOG_OUT"] = str(tmp_path / "records.cmplog")
    subprocess.run([str(exe)], capture_output=True, env=env, timeout=60, check=True)

    lines = [line for line in counts_file.read_text().splitlines() if line.startswith("CNT")]
    # Three separate dumps, not one cumulative snapshot repeated.
    assert len(lines) == 3, lines
    assert _parse_counts(counts_file)["memcmp"] == (4, 2)


def test_counting_is_off_without_the_env_var(tmp_path, shim_obj):
    """memcmp is hot in most targets; an unread counter is pure overhead."""
    source = """
    #include <string.h>
    #include <stdio.h>
    int main(void) {
        volatile int r = memcmp("aaaa", "bbbb", 4) == 0;
        printf("%d\\n", r);
        return 0;
    }
    """
    src = tmp_path / "target.c"
    src.write_text(source)
    exe = tmp_path / "target"
    proc = subprocess.run(
        ["gcc", "-O1", "-fno-builtin-memcmp", str(src), str(shim_obj), "-o", str(exe), "-ldl"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"target did not build: {proc.stderr[-400:]}")

    counts_file = tmp_path / "counts.txt"
    env = dict(os.environ)
    env.pop("_CMPLOG_COUNTS", None)
    env["_CMPLOG_OUT"] = str(tmp_path / "records.cmplog")
    subprocess.run([str(exe)], capture_output=True, env=env, timeout=60, check=True)
    assert not counts_file.exists()
