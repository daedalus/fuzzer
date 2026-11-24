"""The two ``_op_*`` engine helpers the 2026-11-24 review called dead are live.

The review's P1 finding read ``_op_declined`` and ``_op_cost_feature``
(services/operators.py:6232, 6558 at that commit) as dead operator
handlers because no registry spec dispatches to them. They are not
registry handlers at all — they are engine-internal helpers, and both are
load-bearing:

* ``_op_declined`` is called from ~30 format-aware handlers to record a
  decline (buffer returned untouched, so the effectiveness hash does not
  credit the operator for havoc's work).
* ``_op_cost_feature`` supplies the per-arm entry of ``_context_vector``,
  which the ``contextual`` (LinUCB) and ``c2ucb`` strategies call on every
  ``select_op``/``record``.

Deleting either would break the build, so this file pins the design: they
must stay out of the registry dispatch (they are helpers, not handlers)
and keep live call sites in the engine.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services.operators import CONTEXT_DIM, OperatorEngine

REPO = Path(__file__).resolve().parents[1]
OPERATORS_SRC = REPO / "src" / "fuzzer_tool" / "services" / "operators.py"


def test_helpers_are_not_registry_handlers_by_design():
    """No spec dispatches to them — that is the design, not dead code."""
    handlers = {spec.handler_name for spec in REGISTRY.specs() if spec.handler_name}
    assert "_op_declined" not in handlers
    assert "_op_cost_feature" not in handlers


def test_declined_records_and_returns_buffer_untouched():
    eng = SimpleNamespace(f=SimpleNamespace(_op_declines={}))

    out = OperatorEngine._op_declined(eng, "utf8_seq_mutate", b"abc")

    assert out == b"abc", "a decline must not mutate the buffer (havoc credit bug)"
    assert eng.f._op_declines["utf8_seq_mutate"] == 1


def test_cost_feature_feeds_the_context_vector():
    """_context_vector -> _op_cost_feature is the contextual/c2ucb path."""
    f = SimpleNamespace(
        _current_context_shared=[0.25] * (CONTEXT_DIM - 1),
        _op_time_ema={"bit_flip": 1e-3},  # 1 ms
    )
    eng = SimpleNamespace(f=f)
    eng._op_cost_feature = lambda op: OperatorEngine._op_cost_feature(eng, op)

    vec = OperatorEngine._context_vector(eng, "bit_flip")

    assert len(vec) == CONTEXT_DIM
    assert vec[:-1] == [0.25] * (CONTEXT_DIM - 1)
    # 1 ms -> log10(1e-3 + 1e-6) ≈ -3, squashed by (x + 6) / 7.
    assert vec[-1] == pytest.approx((math.log10(1e-3 + 1e-6) + 6.0) / 7.0)


def test_cost_feature_neutral_without_timing():
    eng = SimpleNamespace(f=SimpleNamespace(_op_time_ema={}))

    assert OperatorEngine._op_cost_feature(eng, "never_timed") == 0.5


def test_both_helpers_keep_live_call_sites():
    """A refactor that drops the last caller must update this file too."""
    tree = ast.parse(OPERATORS_SRC.read_text())
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "_op_declined" in calls, "_op_declined lost its last call site"
    assert "_op_cost_feature" in calls, "_op_cost_feature lost its last call site"
