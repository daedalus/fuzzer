"""``cond_tuples_from_cmplog_text`` fast path matches the pre-change parser.

The parser stripped every line and tracked a stripped-line set, though
``_parse_lines`` already dedupes raw lines and the ``(op_a, op_b, width,
result, pc)`` key dedupes the rest. Lines that already start with ``"CMP "``
now skip ``strip`` and split once. Oracle: the pre-change body, frozen here.
"""

from fuzzer_tool.adapters.track_parser import cond_tuples_from_cmplog_text
from fuzzer_tool.core.rand_pool import RandPool

_SEED = 0xC0FFEE
_N_LINES = 2000
_WIDTHS = (1, 2, 4, 8)

# Lines a strict ``"CMP "`` prefix check must still reject or accept as before.
_ADVERSARIAL = [
    "CMP\t43 44 1 1",  # tab after CMP: rejected
    "CMPX 45 46 1 1",  # longer kind: rejected
    "  CMP 41 42 1 1 0x10  ",  # surrounding whitespace: accepted
    "\tCMP 41 42 0 1\n",  # leading tab: accepted
    "CMP 41 42 1 1 0x10\n",  # trailing newline, repeat of the above key
    "CMP  41   42  1  1",  # runs of spaces
    "CMP 41",  # too short
    "CMP ",
    "CMP",
    "",
    "\n",
    "CMP zz 42 1 1",  # bad hex
    "CMP 41 42 x 1",  # bad result
    "CMP 41 42 1 1 0xzz",  # bad pc
    "CMP  ",  # empty operands
    "CMP 41 42",  # no result/width/pc
    "CMP 41 42 1",  # no width
    "DIV 00010000 4 0x1",
    "GEP 0000000000010000 8 0x2",
]


def _oracle(lines):
    """Pre-change cond_tuples_from_cmplog_text, verbatim."""
    out = []
    seen_lines = set()
    seen = set()
    for line in lines:
        line = line.strip()
        if line in seen_lines:
            continue

        seen_lines.add(line)
        if not line.startswith("CMP "):
            continue
        parts = line[4:].split()
        if len(parts) < 2:
            continue
        try:
            op_a = bytes.fromhex(parts[0])
            op_b = bytes.fromhex(parts[1])
            result = int(parts[2]) if len(parts) > 2 else None
            width = int(parts[3]) if len(parts) > 3 else None
            pc = int(parts[4], 16) if len(parts) > 4 else None
        except (ValueError, IndexError):
            continue
        if not op_a and not op_b:
            continue
        k = (op_a, op_b, width, result, pc)
        if k in seen:
            continue

        seen.add(k)
        out.append(k)
    return out


def _drain(rng: RandPool) -> list[str]:
    """Shim-shaped CMP/DIV/GEP lines with repeats and newline variants."""
    lines = []
    for _ in range(_N_LINES):
        w = rng.choice(_WIDTHS)
        a = rng.randbytes(w).hex()
        b = rng.randbytes(w).hex()
        res = rng.randint(-1, 1)
        pc = rng.randrange(1 << 20)
        lines.append(f"CMP {a} {b} {res} {w} 0x{pc:x}\n")
    # Repeats, newline-stripped twins and adversarial lines interleaved.
    lines += [line.rstrip("\n") for line in lines[::7]]
    lines += lines[::5] + _ADVERSARIAL
    rng.shuffle(lines)
    return lines


def test_control_oracle_matches_itself():
    """Rule 46: the oracle is deterministic on identical input."""
    lines = _drain(RandPool(seed=_SEED))
    assert _oracle(lines) == _oracle(list(lines))


def test_regression_fastpath_matches_oracle():
    """Falsification: same tuples, same order, on a shuffled drain."""
    lines = _drain(RandPool(seed=_SEED))
    assert cond_tuples_from_cmplog_text(lines) == _oracle(lines)


def test_regression_fastpath_adversarial_lines():
    """Adversarial: whitespace and malformed variants, alone and repeated."""
    lines = _ADVERSARIAL + _ADVERSARIAL[::-1]
    got = cond_tuples_from_cmplog_text(lines)
    assert got == _oracle(lines)
    # Tab after CMP stays rejected; surrounding whitespace stays accepted.
    assert (b"A", b"B", 1, 1, 0x10) in got
    assert (b"C", b"D", 1, 1, None) not in got
    assert (b"E", b"F", 1, 1, None) not in got


def test_regression_fastpath_generator_input():
    """A one-shot iterable parses like a list."""
    lines = _drain(RandPool(seed=_SEED + 1))
    assert cond_tuples_from_cmplog_text(iter(lines)) == _oracle(lines)
