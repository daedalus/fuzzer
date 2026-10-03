"""Regression: the whole-program ICFG bypassed the on-disk CFG cache.

``cfg_cache``'s docstring names ``core/icfg.py`` as a client, but
``_decode_all_cfgs`` decoded every function serially on every start:
163 s of ffmpeg campaign startup (14M instructions through the pure-Python
decoder), outside ``--profile-hotpath``'s window. It now goes through
``TargetDistance``'s cached, parallel decode.
"""

import pytest

from fuzzer_tool.core import cfg_cache
from fuzzer_tool.core.analyzers import analyzer_distance as distance_mod
from fuzzer_tool.core.analyzers.analyzer_distance import TargetDistance
from fuzzer_tool.core.icfg import build_interprocedural_cfg
from tests.test_icfg import tp_target  # noqa: F401  (fixture)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.delenv("FUZZER_DISABLE_CFG_CACHE", raising=False)
    monkeypatch.setattr(cfg_cache, "_cache_dir_memo", None)


def _count_decodes(monkeypatch) -> dict:
    calls = {"n": 0}
    orig = distance_mod.build_function_cfg

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(distance_mod, "build_function_cfg", counting)
    return calls


def _icfg(path):
    td = TargetDistance(path, targets=["target_fn"])
    assert td.load()
    return build_interprocedural_cfg(td)


def _same(a, b):
    assert a.node_addrs == b.node_addrs
    assert a.n_edges == b.n_edges
    assert (a.src == b.src).all() and (a.dst == b.dst).all()


def test_regression_icfg_second_build_hits_cache(tp_target, monkeypatch):  # noqa: F811
    calls = _count_decodes(monkeypatch)
    first = _icfg(tp_target)
    assert calls["n"] > 0, "fixture decoded nothing; cannot prove caching"

    before = calls["n"]
    second = _icfg(tp_target)
    assert calls["n"] == before
    _same(first, second)


def test_cached_equals_uncached(tp_target, monkeypatch):  # noqa: F811
    """Falsification: the cache must not change the graph."""
    _icfg(tp_target)  # warm
    cached = _icfg(tp_target)
    monkeypatch.setenv("FUZZER_DISABLE_CFG_CACHE", "1")
    _same(cached, _icfg(tp_target))


def test_control_uncached_against_itself(tp_target, monkeypatch):  # noqa: F811
    """Hard Rule 46."""
    monkeypatch.setenv("FUZZER_DISABLE_CFG_CACHE", "1")
    _same(_icfg(tp_target), _icfg(tp_target))


def test_corrupt_cache_rebuilds_same_graph(tp_target, monkeypatch):  # noqa: F811
    """Adversarial: a garbage artifact is refused, not trusted."""
    good = _icfg(tp_target)
    for f in cfg_cache.Path(cfg_cache._cache_dir()).glob("*"):
        f.write_bytes(b"\x1f\x8bgarbage")
    _same(good, _icfg(tp_target))


def test_regression_no_targets_skips_cache_load(tp_target, monkeypatch):  # noqa: F811
    """Katz builds TargetDistance without targets: load() wants no CFGs, so
    it must not unpickle the whole-program artifact (15 s on ffmpeg) only for
    the ICFG to unpickle it again."""
    _icfg(tp_target)  # warm
    loads = {"n": 0}
    orig = cfg_cache.load

    def counting(ident):
        loads["n"] += 1
        return orig(ident)

    monkeypatch.setattr(cfg_cache, "load", counting)
    td = TargetDistance(tp_target)
    assert td.load()
    assert loads["n"] == 0
    build_interprocedural_cfg(td)
    assert loads["n"] == 1


def test_regression_icfg_decode_never_forks_a_pool(tp_target, monkeypatch):  # noqa: F811
    """Forking the decode pool from a threaded process (xdist worker, fuzzer
    with the cmplog FIFO drain running) deadlocked the children. The ICFG
    decodes serially; the pool bought nothing there (160 s vs 163 s)."""

    class _NoPool:
        def __init__(self, *a, **k):
            raise AssertionError("ICFG decode forked a process pool")

    monkeypatch.setattr(distance_mod, "ProcessPoolExecutor", _NoPool)
    monkeypatch.setattr(cfg_cache, "should_parallelize", lambda *a: True)
    td = TargetDistance(tp_target)  # Katz builds it without targets
    assert td.load()
    assert build_interprocedural_cfg(td).n_nodes > 0
