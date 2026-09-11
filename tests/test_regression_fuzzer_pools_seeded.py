"""Regression: every RandPool-taking object Fuzzer builds gets its pool.

A class that takes ``rng: RandPool | None = None`` falls back to
``RandPool()`` -- OS entropy, which ``--seed`` cannot reach. Built without
``rng=`` in the services layer, it makes a seeded campaign irreproducible
in whatever it drives. Found this way: the Markov model (``markov_bytes``,
see test_regression_markov_seeded), ``GALifecycle`` (GA breeding),
``BayesianSeedQuality`` (Thompson draws over seeds) and ``StatsReporter``.

Read from source rather than listed, so a class that gains an ``rng``
parameter later, or a new construction site, is covered without editing
this file.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "fuzzer_tool"


def _rng_slot_by_class() -> dict[str, int]:
    """Class name -> positional index of ``rng`` in __init__ (self excluded)."""
    slots = {}
    for path in _SRC.rglob("*.py"):
        for cls in ast.walk(ast.parse(path.read_text())):
            if not isinstance(cls, ast.ClassDef):
                continue
            for fn in cls.body:
                if not (isinstance(fn, ast.FunctionDef) and fn.name == "__init__"):
                    continue
                pos = [a.arg for a in fn.args.args[1:]]
                kwonly = [a.arg for a in fn.args.kwonlyargs]
                if "rng" in pos:
                    slots[cls.name] = pos.index("rng")
                elif "rng" in kwonly:
                    slots[cls.name] = -1
    return slots


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _unseeded_constructions(slots):
    found, bad = 0, []
    for path in (_SRC / "services").rglob("*.py"):
        for call in ast.walk(ast.parse(path.read_text())):
            if not isinstance(call, ast.Call):
                continue
            name = _call_name(call)
            if name not in slots:
                continue
            found += 1
            passed = any(k.arg in ("rng", None) for k in call.keywords)
            slot = slots[name]
            passed = passed or (slot >= 0 and len(call.args) > slot)
            if not passed:
                bad.append(f"{path.relative_to(_SRC)}:{call.lineno} {name}")
    return found, bad


def test_discovery_found_constructions():
    """Guard: an empty scan would make the next test vacuously true."""
    slots = _rng_slot_by_class()
    assert {"MarkovChain", "GALifecycle", "MOSSScheduler"} <= set(slots)
    found, _ = _unseeded_constructions(slots)
    assert found >= 20


def test_services_pass_the_pool_to_every_rng_taking_class():
    found, bad = _unseeded_constructions(_rng_slot_by_class())
    assert not bad, f"built without rng=, so --seed does not reach them: {bad}"
