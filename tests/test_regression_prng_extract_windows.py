"""_extract_by_site scanned the whole input once per operand (input.find).

~3800 compare records x 2 operands per drain, each an O(len) scan: 4 s per
10k default-mode execs. The set of the input's 4- and 8-byte windows is
built once per call; membership answers the same question.
"""

from fuzzer_tool.core.analyzers import analyzer_prng_state_learner as psl
from fuzzer_tool.core.rand_pool import RandPool
from tests.test_prng_state_learner import _Cond, _learner


def _old_by_site(conds, input_data: bytes) -> dict:
    """The pre-change grouping loop, verbatim (history side effects aside)."""
    by_site: dict = {}
    seen: set = set()
    for cond in conds:
        base = cond.base
        for op in (base.op_a, base.op_b):
            width = len(op)
            if width not in psl._OPERAND_WIDTHS or input_data.find(op) >= 0:
                continue
            value = int.from_bytes(op, "little")
            site = (base.pc, width)
            bucket = by_site.setdefault(site, [])
            if (site, value) in seen:
                continue
            seen.add((site, value))
            bucket.append(value)
    return by_site


def _workload(seed: int):
    rng = RandPool(seed)
    data = rng.randbytes(rng.randint(0, 600))
    conds = []
    for _ in range(400):
        width = rng.choice([1, 2, 4, 8, 4, 8])
        if data and len(data) >= width and rng.random() < 0.4:
            at = rng.randint(0, len(data) - width)
            op = data[at : at + width]  # present in the input: dropped
        else:
            op = rng.randbytes(width)
        conds.append(_Cond(op, rng.randbytes(width), rng.choice([None, 0x10, 0x20])))
    return conds, data


def test_regression_prng_extract_windows():
    for seed in range(30):
        conds, data = _workload(seed)
        learner = _learner(conds)
        assert learner._extract_by_site(data) == _old_by_site(conds, data)


def test_empty_input_keeps_every_operand():
    """Adversarial: nothing is 'in' an empty input, so every width-4/8 op stays."""
    conds = [_Cond(b"\x01\x02\x03\x04", b"\x00" * 8, 0x10)]
    assert _learner(conds)._extract_by_site(b"") == _old_by_site(conds, b"")
