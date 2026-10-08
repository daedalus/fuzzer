"""hint_operands re-scanned the whole cmplog pool on every new seed.

The narrow (distinct byte) cap rarely fills, so each call read all ~8k
operands: 15 ms x 512 calls per 3k --hail-mary execs. The result depends on
the pool only, so it is cached per pool (owner + length, as
``scanner_for_pairs``).
"""

from fuzzer_tool.core import checksum_sites as cs


def _pool(n: int) -> list[tuple[bytes, bytes]]:
    return [(bytes([i % 251, 0, 0, 0]), (i % 65536).to_bytes(2, "big")) for i in range(1, n)]


def test_regression_cksum_hint_cache(monkeypatch):
    """Same pool object, same length: no second scan."""
    pool = _pool(500)
    first = cs.hint_operands(pool)
    calls = []
    real = cs._byte_operand
    monkeypatch.setattr(cs, "_byte_operand", lambda op: calls.append(op) or real(op))
    assert cs.hint_operands(pool) == first
    assert calls == []


def test_grown_pool_rescanned():
    """Adversarial: an append to the same list must be seen."""
    pool = [(b"\x07", b"\x00")]
    assert cs.hint_operands(pool) == [b"\x07"]
    pool.append((b"\x09\x00", b"\x00"))
    assert cs.hint_operands(pool) == [b"\x09\x00", b"\x07", b"\x09"]


def test_none_and_empty():
    """Falsification: no pool, no hints."""
    assert cs.hint_operands(None) == []
    assert cs.hint_operands([]) == []
