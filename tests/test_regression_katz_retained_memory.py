"""Regression: the Katz channel retained ~1.85 GB of build-only state.

Measured on ffmpeg_read_asan.so (2.74M ICFG nodes): RSS 2102 MB after
``KatzChannel.build``, 245 MB once ``_cfgs``, ``_td``, ``node_index`` and
the ``node_addrs`` int list were gone. Only the build and ``upload()`` read
them; the per-exec path needs src/dst and node indices.
"""

from array import array

import numpy as np

from fuzzer_tool.core.icfg import InterproceduralCFG
from fuzzer_tool.services.katz_channel import KatzChannel
from tests.test_katz_channel_build import trace_pc_binary  # noqa: F401  (fixture)

_STEP = 0x10


def _icfg(n: int) -> InterproceduralCFG:
    addrs = [0x1000 + _STEP * i for i in range(n)]
    src = np.arange(n - 1, dtype=np.int64)
    return InterproceduralCFG(addrs, ["f"] * n, src, src + 1, cfgs={})


def test_node_addrs_are_packed():
    """Falsification: node starts are 8-byte words, not an int-object list."""
    icfg = _icfg(4)

    assert isinstance(icfg.node_addrs, array)
    assert icfg.node_addrs.typecode == "Q"
    assert not hasattr(icfg, "node_index")


def test_node_lookup_edges():
    """Adversarial: below, between, above and on every node start."""
    n = 5
    icfg = _icfg(n)
    first = icfg.node_addrs[0]
    last = icfg.node_addrs[-1]

    for i, a in enumerate(icfg.node_addrs):
        assert icfg._node_at(a) == i

    assert icfg._node_at(first - 1) is None
    assert icfg._node_at(first + 1) is None
    assert icfg._node_at(last + _STEP) is None
    assert icfg._node_at(0) is None


def test_bottleneck_ignores_unmapped():
    """Adversarial: an address with no node is dropped, as with the old dict."""
    icfg = _icfg(3)
    a = list(icfg.node_addrs)
    unmapped = a[-1] + _STEP

    cut = icfg.bottleneck_edges({a[0], unmapped}, {a[2]})

    assert cut
    assert all(u in a and v in a for u, v in cut)


def test_channel_releases_build_state(trace_pc_binary):  # noqa: F811
    """Falsification: CFGs go after build, TargetDistance after upload."""
    ch = KatzChannel.build(trace_pc_binary)
    assert ch is not None
    assert ch.icfg._cfgs is None
    assert ch._td is not None  # upload() still needs it

    try:
        assert ch.upload()
        assert ch._td is None
    finally:
        ch.cleanup()
