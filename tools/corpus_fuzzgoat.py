#!/usr/bin/env python3
"""Generate a JSON corpus for fuzzgoat calibration runs.

Mirrors the mix the edge-id handover measured on: 120 structured JSON
documents, 120 byte-level mutations of those, 10 hand-written edge cases.
A corpus with both well-formed and malformed inputs matters because the
parser's parse/fail split is one of the few signals the matrix analyses
recover (the handover's PC2/validity result), and a valid-only corpus
would hide it.

Randomness comes from ``core.rand_pool.RandPool`` (Hard Rule 16); one
seeded pool, so the corpus is byte-for-byte reproducible.

Usage::

    python3 tools/corpus_fuzzgoat.py [--out DIR] [--count N]

Output is a flat directory of ``.json`` files (the edge-matrix tool reads
every file under the corpus dir; no ``seeds/`` sub-layout is required).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fuzzer_tool.core.rand_pool import RandPool

_HAND_WRITTEN = [
    b"{}",
    b"[]",
    b'{"a": 1}',
    b"[1, 2, 3]",
    b'{"a": {"b": {"c": [true, false, null, [1.5, 2.5]]}}}',
    b'{"s": "esc\\"ape\\n\\t\\u00e9"}',
    b'{"n": -12345678901234567890}',
    b'{"d": 1e309}',
    b'{"u": "\\ud800"}',
    b"{]",
    b'{"a":}',
    b"[1, 2,]",
    b'{"a": 1,}',
    b'{1: "key must be string"}',
    b'{"a": 1',
    b"",
    b"   ",
    b"\x00\x01\x02",
    b'{"a": null}{"b": 1}',
    b'\\"not json\\"',
]


def _random_value(pool: RandPool, depth: int) -> str:
    kind = pool.choice([0, 1, 2, 3, 4, 5, 6])
    if depth > 3:
        kind = pool.choice([1, 2, 6, 6])
    if kind == 0:  # object
        n = pool.randrange(5)
        return (
            "{"
            + ", ".join(_random_key(pool) + ": " + _random_value(pool, depth + 1) for _ in range(n))
            + "}"
        )
    if kind == 1:  # array
        n = pool.randrange(5)
        return "[" + ", ".join(_random_value(pool, depth + 1) for _ in range(n)) + "]"
    if kind == 2:  # string
        chars = [
            pool.choice(["a", "b", "c", "Z", "0", "_", " ", '"', "\\\\", "/", "\\u00e9", "\\n"])
            for _ in range(pool.randrange(8))
        ]
        return '"' + "".join(chars) + '"'
    if kind == 3:  # integer
        return str(pool.randrange(10 ** (pool.randrange(6) + 1)) * pool.choice([-1, 1]))
    if kind == 4:  # float
        return f"{pool.choice([-1, 1]) * pool.random() * 10 ** (pool.randrange(3) + 1):.3f}"
    if kind == 5:  # boolean / null
        return pool.choice(["true", "false", "null"])
    return str(pool.randrange((1 << 63) - 1))  # big number


def _random_key(pool: RandPool) -> str:
    return '"k' + str(pool.randrange(10**6)) + '"'


def _valid_documents(count: int, pool: RandPool) -> list[bytes]:
    return [_random_value(pool, 0).encode() for _ in range(count)]


def _mutate(data: bytes, pool: RandPool) -> tuple[str, bytes]:
    """Return (operation, result) for one byte-level mutation of *data*."""
    if not data:
        return "insert", b"{}"
    op = pool.choice(["flip", "insert", "delete", "duplicate", "truncate", "grow"])
    b = bytearray(data)
    if op == "flip":
        i = pool.randrange(len(b))
        b[i] ^= 1 << pool.randrange(8)
    elif op == "insert":
        i = pool.randrange(len(b) + 1)
        b[i:i] = pool.randbytes(pool.choice([1, 1, 2, 4, 8]))
    elif op == "delete":
        i = pool.randrange(len(b))
        b[i : i + pool.randrange(min(8, len(b) - i)) + 1] = b""
    elif op == "duplicate":
        i = pool.randrange(len(b))
        b[i : i + 1] = b[i : i + 1] * pool.choice([2, 2, 8])
    elif op == "truncate":
        return "truncate", bytes(b[: pool.randrange(len(b))])
    else:  # grow: extend with junk bytes
        return "grow", bytes(b) + pool.randbytes(pool.randrange(15) + 1)
    return op, bytes(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        default="~/fuzzing/corpus/json",
        help="corpus directory (default ~/fuzzing/corpus/json)",
    )
    ap.add_argument("--count", type=int, default=250, help="total inputs (default 250)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    pool = RandPool(seed=args.seed)
    n_valid = args.count // 2
    docs = _valid_documents(n_valid, pool)
    seeds = [(f"valid_{i:04d}.json", d) for i, d in enumerate(docs)]
    for i, d in enumerate(docs[:n_valid]):
        op, mutated = _mutate(d, pool)
        seeds.append((f"mut_{op}_{i:04d}.json", mutated))
    for i, raw in enumerate(_HAND_WRITTEN):
        seeds.append((f"edge_{i:03d}.json", raw))

    seeds = seeds[: args.count] if args.count > 0 else seeds
    for name, data in seeds:
        with open(os.path.join(out, name), "wb") as f:
            f.write(data)
    print(f"[*] corpus ready: {len(seeds)} inputs in {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
