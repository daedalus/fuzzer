#!/usr/bin/env python3
"""Generate seeds from a FormatLearner's inferred format fields.

Takes the "Inferred format fields" table a running fuzzer builds
(offset/width/type/confidence/observations/edges/sensitive-ops, see
``--format-learn`` and the "Format Structure Learning" report section) and
uses it to synthesize field-targeted seed variants of a base seed: length
fields get swept through boundary values, crc fields get cheap stress
values, data/unknown fields get replayed with whichever mutation operators
the learner already saw move coverage there, and magic/padding fields are
left untouched.

Input is a JSON dump of ``FormatLearner.get_state()`` (its hypotheses are
all this tool needs; the timeline is ignored). There's no existing
in-fuzzer hook that writes this dump, so produce it ad hoc, e.g. from a
live fuzzer instance ``f``:

    import json
    json.dump(f._format_learner.get_state(), open("format_state.json", "w"))

Usage:
    python tools/gen_format_seeds.py --state format_state.json \\
        --seed base_seed.bin --out corpus/ --count 32

Output is written as corpus/seeds/id_<hash>_<label>.bin, matching the
layout load_corpus() reads.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzzer_tool.core.format_seed_generator import FormatSeedGenerator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--state", required=True, type=Path, help="FormatLearner.get_state() JSON dump")
    ap.add_argument(
        "--seed", required=True, type=Path, help="base seed file to derive variants from"
    )
    ap.add_argument(
        "--out", required=True, type=Path, help="corpus directory (writes into <out>/seeds/)"
    )
    ap.add_argument("--count", type=int, default=32, help="max seeds to generate (default: 32)")
    ap.add_argument("--rng-seed", type=int, default=None, help="deterministic RNG seed")
    args = ap.parse_args()

    state = json.loads(args.state.read_text())
    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        print("[!] No field hypotheses in state file — nothing to target.", file=sys.stderr)
        return 1

    base_seed = args.seed.read_bytes()
    rng = random.Random(args.rng_seed) if args.rng_seed is not None else None
    gen = FormatSeedGenerator(hypotheses, rng=rng)
    results = gen.generate(base_seed, n_seeds=args.count)

    if not results:
        print(
            "[!] Generated 0 seeds (no non-skipped fields fit this seed's length).", file=sys.stderr
        )
        return 1

    seeds_dir = args.out / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        h = hashlib.sha256(r.data).hexdigest()[:16]
        path = seeds_dir / f"id_{h}_{r.label()}.bin"
        path.write_bytes(r.data)

    print(f"[*] Wrote {len(results)} seeds to {seeds_dir}")
    by_type: dict[str, int] = {}
    for r in results:
        by_type[r.field_type] = by_type.get(r.field_type, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"    {t:>8s}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
