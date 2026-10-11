#!/usr/bin/env python3
"""Write one seed that contains every token of the given dictionaries.

Overlapping tokens are merged (GIF + IFA -> GIFA), so the seed is shorter
than their concatenation.  See ``core/token_superstring.py``.

Usage:
    python tools/gen_dict_seed.py -D dictionaries/gif.dict --out ~/corpus

Output is written as <out>/seeds/id_<hash>_dictseed.bin, matching the
layout load_corpus() reads.
"""

import argparse
import glob
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzzer_tool.core.mutations import load_dictionary  # noqa: E402
from fuzzer_tool.core.rand_pool import RandPool  # noqa: E402
from fuzzer_tool.core.token_superstring import (  # noqa: E402
    DEFAULT_KICKS,
    reduce_input,
    token_seed,
)

LABEL = "dictseed"
HASH_LEN = 16


def _load(patterns: list[str]) -> list[bytes] | None:
    """Tokens of every dictionary matching the patterns; None if one is missing."""
    tokens: list[bytes] = []
    for pattern in patterns:
        paths = glob.glob(pattern)
        if not paths:
            print(f"[!] Dictionary not found: {pattern}", file=sys.stderr)
            return None
        for path in sorted(paths):
            tokens.extend(load_dictionary(path))
    return tokens


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-D", "--dict", required=True, nargs="+", help="dictionary file(s) or globs")
    ap.add_argument(
        "--out", required=True, type=Path, help="corpus directory (writes <out>/seeds/)"
    )
    ap.add_argument(
        "--kicks", type=int, default=DEFAULT_KICKS, help="ILS kicks (default: %(default)s)"
    )
    ap.add_argument("--rng-seed", type=int, default=None, help="deterministic RNG seed")
    args = ap.parse_args()

    tokens = _load(args.dict)
    if tokens is None:
        return 1
    if not reduce_input(tokens):
        print("[!] No tokens in the given dictionaries.", file=sys.stderr)
        return 1

    rng = RandPool(seed=args.rng_seed) if args.rng_seed is not None else None
    data = token_seed(tokens, rng=rng, kicks=args.kicks)

    seeds_dir = args.out / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    name = f"id_{hashlib.sha256(data).hexdigest()[:HASH_LEN]}_{LABEL}.bin"
    (seeds_dir / name).write_bytes(data)

    total = sum(len(t) for t in set(tokens))
    print(
        f"[*] {len(set(tokens))} tokens, {total} B concatenated -> {len(data)} B "
        f"({100 * (1 - len(data) / total):.1f}% shorter)"
    )
    print(f"[*] Wrote {seeds_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
