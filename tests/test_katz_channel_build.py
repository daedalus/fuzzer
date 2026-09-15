"""Regression: ``KatzChannel.build()`` must succeed for undirected campaigns.

``KatzChannel.build`` never passes ``targets=`` to the ``TargetDistance``
it constructs internally -- K-Scheduler is specifically the *undirected*
channel (see ``services/fuzzer.py``'s ``if not targets:`` call site,
mutually exclusive with directed-mode distance over the shared
``__AFL_DIST_SHM_ID`` slot). A prior version of ``build()`` nonetheless
rejected construction whenever ``td.target_addrs`` was empty -- which,
given that this method never supplies targets in the first place, means
the check was unconditionally true. K-Scheduler could never activate at
all, for any binary, in any campaign, from the commit that introduced the
check (``fe8fd42``, itself a "perf" commit) onward. Every other Katz test
in this suite (``test_katz_channel.py``, ``test_katz_beta.py``)
constructs ``KatzChannel`` directly via its plain ``__init__`` with a
hand-built ICFG, bypassing ``.build()`` entirely, which is exactly why
this was never caught.

Builds a real trace-pc-marked ELF with gcc (no clang dependency) and
drives the actual classmethod end to end.
"""

import shutil
import subprocess

import pytest

from fuzzer_tool.services.katz_channel import KatzChannel, _target_has_trace_pc

SRC = """\
__attribute__((noinline)) void __sanitizer_cov_trace_pc(void) {}
__attribute__((noinline)) int leaf(int x) {
    __sanitizer_cov_trace_pc();
    return x + 1;
}
__attribute__((noinline)) int target_fn(int a) {
    __sanitizer_cov_trace_pc();
    int r = 0;
    if (a > 10) { __sanitizer_cov_trace_pc(); r = leaf(a); }
    else { __sanitizer_cov_trace_pc(); r = a - 1; }
    return r;
}
int main(void) {
    __sanitizer_cov_trace_pc();
    return target_fn(42);
}
"""


@pytest.fixture(scope="module")
def trace_pc_binary(tmp_path_factory):
    if not shutil.which("gcc"):
        pytest.skip("gcc not available")
    d = tmp_path_factory.mktemp("katz_build")
    src = d / "t.c"
    src.write_text(SRC)
    exe = d / "t"
    r = subprocess.run(
        ["gcc", "-O0", "-g", "-no-pie", "-o", str(exe), str(src)], capture_output=True
    )
    assert r.returncode == 0, r.stderr.decode()
    return str(exe)


class TestKatzChannelBuildSucceedsWithoutTargets:
    def test_trace_pc_detected(self, trace_pc_binary):
        assert _target_has_trace_pc(trace_pc_binary)

    def test_build_returns_a_real_channel(self, trace_pc_binary):
        ch = KatzChannel.build(trace_pc_binary)
        assert ch is not None
        assert ch.n_nodes > 0
        assert ch.icfg.n_edges > 0

    def test_channel_has_no_target_dependent_state(self, trace_pc_binary):
        # The whole point: nothing about the built channel depends on any
        # target having been resolved -- td (kept only as _td for the
        # distance-table upload) was constructed with no targets at all.
        ch = KatzChannel.build(trace_pc_binary)
        assert ch._td.target_addrs == set()
