"""Regression: ``edge_diagnostic.py op-caches`` read caches that moved.

6b3b7c3b moved ``_boundary_cache``/``_root_hash_cache``/``_plan_cache`` onto
``FractalVoronoiMutator`` and ``_noise_cache`` onto ``PerlinNoiseMutator``;
the mode still read them as module globals and died with AttributeError on
its first row. The fuzz run is stubbed out: the row is what broke.
"""

import importlib.util
from pathlib import Path

from fuzzer_tool.core.mutations import fractal_voronoi, perlin_noise
from fuzzer_tool.core.operator_registry import REGISTRY

TOOL = Path(__file__).resolve().parent.parent / "tools" / "edge_diagnostic.py"


def _load():
    spec = importlib.util.spec_from_file_location("edge_diagnostic_caches", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_fuzzer = lambda: None
    mod.quiet_run = lambda f, n: None
    return mod


def _rows(capsys):
    _load().op_caches_mode(None)
    lines = capsys.readouterr().out.splitlines()
    return [line.split() for line in lines[1:]]


def _registered(name):
    return next(m for m in REGISTRY.mutators() if m.name == name)


def test_regression_op_caches_mode_prints_rows(capsys):
    rows = _rows(capsys)
    assert len(rows) == 6
    assert all(len(r) == 10 for r in rows)


def test_adversarial_reports_the_registered_instances(capsys):
    """Counts come from the instances the fuzzer dispatches to, not fresh ones."""
    pn = _registered(perlin_noise.PerlinNoiseMutator.name)
    fv = _registered(fractal_voronoi.FractalVoronoiMutator.name)
    pn_before = dict(pn._noise_cache)
    try:
        pn._noise_cache[-1] = object()
        row = _rows(capsys)[0]
        assert int(row[1]) == len(pn._noise_cache)
        assert int(row[6]) == len(fv._boundary_cache)
        assert int(row[8]) == len(fv._plan_cache)
    finally:
        pn._noise_cache.clear()
        pn._noise_cache.update(pn_before)
