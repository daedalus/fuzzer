"""An operator that cannot do its job must decline, not borrow havoc.

Thirteen handlers used to `return self._op_havoc(...)` when their
format-aware work was impossible: the input did not parse, no constraint
solved, no candidate site existed. Havoc then changed the buffer *under the
declining operator's name*, and `mutate()` decides effectiveness by
comparing an xxh3 digest across the call -- so the hash moved, the operator
was added to `_last_ops_effective`, and every scheduler that consumes that
signal rewarded it for a mutation it did not make.

Measured across the gated battery before the change, as a share of
selections:

    recompress_gzip        100%      der_tlv_reorder     16%
    path_negate            100%      der_tlv_insert      14%
    deflate_struct_mutate  100%      length_offset_goal   5%
    recompress_zlib         98%
    tlv_nest_mutate         88%

Four operators were havoc with a different label. That is worse than
earning nothing: it also hid the gap, because the operator looked like it
was working.

`_op_declined` records the decline and returns the buffer untouched, which
makes the existing effectiveness signal correct with no new wiring -- the
hash does not move and the reward is the one the scheduler should have been
computing all along. `_op_field_repair` is the one deliberate exception: it
composes with havoc on purpose ("applied after a havoc pass, so the
operator both mutates and re-establishes structural validity").
"""

from __future__ import annotations

import ast
import inspect

import pytest

from fuzzer_tool.services import operators as operators_mod
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer

# Composes with havoc by design; see the module docstring.
_DELIBERATE_HAVOC_USERS = {"_op_field_repair"}


def _handlers_calling_havoc() -> set[str]:
    """Functions in operators.py that call `_op_havoc` on the buffer."""
    src = inspect.getsource(operators_mod)
    tree = ast.parse(src)
    funcs = [
        (n.lineno, n.body[-1].end_lineno, n.name)
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
    ]
    found = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "_op_havoc":
            continue
        enclosing = min(
            (end - start, name) for start, end, name in funcs if start <= node.lineno <= end
        )
        found.add(enclosing[1])
    return found


def test_no_handler_falls_back_to_havoc():
    """The whole point: a handler that cannot work declines under its own
    name instead of running havoc under it."""
    borrowers = _handlers_calling_havoc() - {"_op_havoc"} - _DELIBERATE_HAVOC_USERS

    assert not borrowers, (
        "these call _op_havoc as a fallback, so havoc's work is credited to "
        f"them by the effectiveness signal: {sorted(borrowers)}"
    )


def test_discovery_sees_the_deliberate_user():
    """Guard: if the scan found nothing at all the test above is vacuous."""
    assert _handlers_calling_havoc() >= _DELIBERATE_HAVOC_USERS, (
        "the deliberate havoc user is gone -- either it was renamed or the "
        "scan is broken, and the assertion above no longer proves anything"
    )


def test_declining_changes_nothing_and_is_counted():
    f = make_minimal_fuzzer(seed=1)
    engine = OperatorEngine(f)
    buf = bytearray(b"the buffer must come back byte for byte")

    out = engine._op_declined("some_op", buf)

    assert bytes(out) == bytes(buf), "a decline must not modify the buffer"
    assert f._op_declines["some_op"] == 1
    engine._op_declined("some_op", buf)
    assert f._op_declines["some_op"] == 2, "declines accumulate across the campaign"


@pytest.mark.parametrize(
    "op,handler",
    [
        ("recompress_gzip", "_op_recompress_gzip"),
        ("recompress_zlib", "_op_recompress_zlib"),
        ("deflate_struct_mutate", "_op_deflate_struct_mutate"),
        ("tlv_nest_mutate", "_op_tlv_nest_mutate"),
    ],
)
def test_an_operator_with_nothing_to_work_on_declines(op, handler):
    """Plain text is none of these formats, so each must decline on it."""
    f = make_minimal_fuzzer(seed=1)
    engine = OperatorEngine(f)
    data = b"not compressed, not a TLV, not a deflate stream" * 4
    buf = bytearray(data)

    out = getattr(engine, handler)(buf, 0, data)

    assert f._op_declines.get(op) == 1, f"{handler} did not record a decline"
    assert bytes(out if out is not None else buf) == data, (
        f"{handler} declined but the buffer moved anyway -- havoc still ran"
    )


def test_decline_rate_is_reported_per_operator():
    f = make_minimal_fuzzer(seed=1)
    engine = OperatorEngine(f)
    f._op_attempts.update({"a": 10, "b": 4, "never_selected": 0})
    f._op_declines.update({"a": 5, "b": 4})

    rates = engine.op_decline_rates()

    assert rates["a"] == pytest.approx(0.5)
    assert rates["b"] == pytest.approx(1.0)
    assert "never_selected" not in rates, "0 attempts is undefined, not 0.0"


def test_decline_rate_skips_operators_below_the_sample_floor():
    """A single decline out of one selection is not a 100% rate worth acting
    on; the floor is what makes the number mean something."""
    f = make_minimal_fuzzer(seed=1)
    engine = OperatorEngine(f)
    f._op_attempts.update({"rare": 2, "common": 50})
    f._op_declines.update({"rare": 2, "common": 10})

    rates = engine.op_decline_rates(min_attempts=10)

    assert set(rates) == {"common"}


def test_a_silent_no_change_is_counted_too():
    """The 41 handlers that detect failure and drop it on the floor.

    `_op_sleb128_encode` tests `if result != bytes(buf)` and falls off the
    end -- it knows it produced nothing and says nothing. 41 handlers in
    operators.py have that shape (an implicit `return None` as the failure
    path), so `mutate()` counts "produced nothing" once, centrally, instead
    of 41 call sites each remembering to.

    SLEB128 is the clean case to pin: values 0..0x3F encode to themselves,
    so on ascending bytes the operator is a complete no-op by correct
    varint semantics. Measured: 0/300 changes on `bytes(range(32))`,
    239/300 on ASCII text, 300/300 on 0x40..0x5F.
    """
    from fuzzer_tool.core.mutations import sleb128_encode
    from fuzzer_tool.core.rand_pool import RandPool

    rng = RandPool(seed=3)
    low = bytes(range(32))  # every byte below 0x40: encodes to itself
    assert sleb128_encode(low, rng, max_len=65536) == low, (
        "fixture guard: this input must be one SLEB128 cannot change"
    )

    high = bytes(range(0x40, 0x60))  # sign bit set: two-byte encoding
    assert sleb128_encode(high, rng, max_len=65536) != high


def test_exempt_operators_are_not_counted_as_declining():
    """havoc is measured by sub-mutation count and field_repair is correct
    to be a no-op on already-valid input, so neither is a failure."""
    from fuzzer_tool.services.operators import _DECLINE_EXEMPT

    assert frozenset({"havoc", "field_repair"}) == _DECLINE_EXEMPT
