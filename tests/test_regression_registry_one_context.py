"""available() built a MutationContext per class-based mutator per call.

~22 snapshots of the fuzzer per mutant (238k per 10k default-mode execs,
2.5 s). The fuzzer does not change while the predicates run, so one
snapshot per call answers the same.
"""

from unittest.mock import patch

from fuzzer_tool.core import mutator_interface as mi
from fuzzer_tool.core.operator_registry import REGISTRY
from tests.support.operator_env import make_minimal_fuzzer as make_fuzzer


def _per_spec(fuzzer, data: bytes) -> list[str]:
    """The pre-change semantics: every predicate as registered."""
    return [n for n, s in REGISTRY._ops.items() if s.available is None or s.available(fuzzer, data)]


def test_regression_registry_one_context():
    f = make_fuzzer()
    real = mi.MutationContext.from_fuzzer
    calls = []
    with patch.object(mi.MutationContext, "from_fuzzer", lambda fz: calls.append(1) or real(fz)):
        REGISTRY.available(f, b"\x89PNG\r\n\x1a\n" + b"x" * 64)
    assert len(calls) == 1


def test_same_answer_as_per_spec_predicates():
    """Identically seeded fuzzers: the format bootstrap trickle draws rng per call."""
    for seed, data in enumerate(
        (b"", b"plain text", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, b"PK\x03\x04" + b"a" * 30)
    ):
        a, b = make_fuzzer(seed=seed), make_fuzzer(seed=seed)
        assert REGISTRY.available(a, data) == _per_spec(b, data)
