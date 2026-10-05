"""``__fuzz_event`` / ``__fuzz_event_at`` in afl_shim.c.

The target calls the hook; the shim appends 12-byte records
(event, input offset, state-var hash) to ``$__LTL_EVENTS_OUT``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core.ltl import NO_OFFSET, read_events
from tests.conftest import requires_clang

AFL_SHIM = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"

_C = """
int main(int argc, char **argv) {
    (void)argv;
    __fuzz_event(1);
    __fuzz_event_at(2, 40);
    if (argc > 1) __sfuzz_state(7, 3);
    __fuzz_event_at(3, 41);
    if (argc > 2) __sfuzz_state(7, 4);
    __fuzz_event(4);
    return 0;
}
"""


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("ltl_shim")
    src = d / "t.c"
    src.write_text(_C)
    out = d / "t"
    proc = subprocess.run(
        [
            "clang",
            "-O1",
            "-fsanitize-coverage=trace-pc-guard",
            "-include",
            str(AFL_SHIM),
            "-o",
            str(out),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"build failed: {proc.stderr[-800:]}")
    return out


def _run(exe, tmp_path, argc=1, env_path=True):
    out = tmp_path / "ev"
    env = {"__LTL_EVENTS_OUT": str(out)} if env_path else {}
    argv = [str(exe)] + ["x"] * (argc - 1)
    rc = subprocess.run(argv, env=env, capture_output=True, timeout=60).returncode
    return rc, out


@requires_clang
def test_events_recorded_in_order(exe, tmp_path):
    rc, out = _run(exe, tmp_path)
    assert rc == 0
    recs = read_events(str(out))
    assert [r.event for r in recs] == [1, 2, 3, 4]


@requires_clang
def test_offsets(exe, tmp_path):
    _, out = _run(exe, tmp_path)
    recs = read_events(str(out))
    assert recs[0].offset == NO_OFFSET
    assert [r.offset for r in recs[1:3]] == [40, 41]


@requires_clang
def test_hash_zero_until_a_state_var_moves(exe, tmp_path):
    _, out = _run(exe, tmp_path, argc=2)
    recs = read_events(str(out))
    assert recs[0].state_hash == 0
    assert recs[1].state_hash == 0
    assert recs[2].state_hash != 0
    assert recs[3].state_hash == recs[2].state_hash


@requires_clang
def test_hash_tracks_state_var_value(exe, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, out_a = _run(exe, tmp_path / "a", argc=2)
    one = read_events(str(out_a))[3].state_hash
    _, out_b = _run(exe, tmp_path / "b", argc=3)
    two = read_events(str(out_b))[3].state_hash
    assert one != two


@requires_clang
def test_hash_is_deterministic_across_runs(exe, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, a = _run(exe, tmp_path / "a", argc=3)
    _, b = _run(exe, tmp_path / "b", argc=3)
    assert read_events(str(a)) == read_events(str(b))


@requires_clang
def test_inert_without_the_env_var(exe, tmp_path):
    rc, out = _run(exe, tmp_path, env_path=False)
    assert rc == 0
    assert not out.exists()


@requires_clang
def test_appends_across_runs_like_a_persistent_loop(exe, tmp_path):
    _run(exe, tmp_path)
    _, out = _run(exe, tmp_path)
    assert len(read_events(str(out))) == 8


@requires_clang
def test_unwritable_path_does_not_crash_the_target(exe, tmp_path):
    env = {"__LTL_EVENTS_OUT": str(tmp_path / "no" / "such" / "dir" / "ev")}
    rc = subprocess.run([str(exe)], env=env, capture_output=True, timeout=60).returncode
    assert rc == 0
