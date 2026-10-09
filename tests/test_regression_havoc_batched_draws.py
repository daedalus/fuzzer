"""havoc_mutate draws every sub-mutation's randoms in one batch.

Each sub-mutation used to call ``randint_list(0, 1 << 30, 4)``: ~3.6 us of
call overhead per sub-mutation, 2-16 per havoc. One ``4 * n`` batch is ~6 us
for eight.
"""

from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer
from tests.support.scripted_rng import ScriptedRng


class _CountingRng(ScriptedRng):
    """ScriptedRng that logs each multi-value randint_list count."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.batches: list[int] = []

    def randint_list(self, a, b, count):
        if count > 1:
            self.batches.append(count)
        return super().randint_list(a, b, count)


def _engine(rng) -> OperatorEngine:
    f = make_minimal_fuzzer(pool=rng)
    f._adaptive_havoc = False
    return OperatorEngine(f)


def test_one_batch_per_havoc():
    """Falsification: n sub-mutations -> one draw of 4 * n."""
    n = 5
    rng = _CountingRng(counts=[n], batch_value=0)
    buf = bytearray(b"\x10ABC")
    _engine(rng).havoc_mutate(buf)

    assert rng.batches == [4 * n]
    # batch_value 0 -> op 0 (bit flip) at offset 0, bit 0, n times.
    assert buf == bytes([0x10 ^ (n & 1)]) + b"ABC"


def test_single_mutation_still_draws_its_own():
    """Adversarial: the direct caller (``_op_havoc`` retry) keeps working."""
    rng = _CountingRng(batch_value=0)
    buf = bytearray(b"\x10ABC")
    _engine(rng)._apply_single_mutation(buf)

    assert rng.batches == [4]
    assert buf == b"\x11ABC"


def test_empty_buffer_fill_then_batch_continues():
    """Adversarial: an empty start consumes its own fill, not batch slots."""
    n = 3
    fill = b"\x01\x02"
    rng = _CountingRng(counts=[n], batch_value=0, randints=[len(fill)], randbytes=[fill])
    buf = bytearray()
    _engine(rng).havoc_mutate(buf)

    assert rng.batches == [4 * n]
    # Sub-mutation 1 fills; 2..n flip bit 0 of byte 0.
    assert buf == bytes([fill[0] ^ ((n - 1) & 1)]) + fill[1:]
