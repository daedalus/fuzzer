"""Text/binary/mixed input mode as op-bandit context (AFL++ `input_mode`).

AFL++ keeps separate op weight tables for text and binary inputs. Here the
mode is a 3-way one-hot at the end of the shared LinUCB/C2UCB seed features.
"""

import pytest

from fuzzer_tool.core.mutations.generic import InputMode, input_mode
from fuzzer_tool.services.operators import CONTEXT_DIM
from tests.test_context_size_percentile import _engine_with_corpus

TEXT = b"GET /index.html HTTP/1.1\r\nHost: x\r\n\r\n"
BINARY = bytes(range(256))
MIXED = b"KEY=" + bytes(range(0, 8)) + b"value\n"  # 12 of 18 printable


@pytest.mark.parametrize(
    ("data", "mode"),
    [(TEXT, InputMode.TEXT), (BINARY, InputMode.BINARY), (MIXED, InputMode.MIXED)],
)
def test_classifies(data, mode):
    assert input_mode(data) is mode


def test_empty_is_binary():
    """Adversarial: no bytes, no evidence of text."""
    assert input_mode(b"") is InputMode.BINARY


def _mode_tail(vec):
    return vec[-len(InputMode) :]


def test_context_carries_one_hot():
    engine, _ = _engine_with_corpus([10, 20, 30])
    for data, mode in (
        (TEXT, InputMode.TEXT),
        (BINARY, InputMode.BINARY),
        (MIXED, InputMode.MIXED),
    ):
        tail = _mode_tail(engine._build_shared_context(data))
        assert tail == [1.0 if m is mode else 0.0 for m in InputMode]


def test_text_and_binary_contexts_differ():
    """Falsification: a builder ignoring the mode gives equal tails."""
    engine, _ = _engine_with_corpus([10, 20, 30])
    assert _mode_tail(engine._build_shared_context(TEXT)) != _mode_tail(
        engine._build_shared_context(BINARY)
    )


def test_dim_matches_vector():
    engine, _ = _engine_with_corpus([10, 20, 30])
    assert len(engine._build_shared_context(TEXT)) + 1 == CONTEXT_DIM  # + per-arm cost
