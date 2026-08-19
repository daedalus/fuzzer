#!/usr/bin/env python3
"""Replay real-corpus sweep samples through LiveBitMaskEstimator.

Reads the sparse TSV produced by ``tools/sweep_jpeg_liveness.py`` and replays
each ``(region, diff_bits)`` observation through a fresh
``LiveBitMaskEstimator(n_bits=65536, switch_after=N)`` for every
``N in SWITCH_AFTERS``.

Outputs a Markdown table summarizing convergence behavior per region and
threshold.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from fuzzer_tool.core.live_bit_mask import LiveBitMaskEstimator

SWITCH_AFTERS = (50, 100, 200, 400, 800)
MAP_BITS = 65536


def _load_samples(path: Path) -> dict[int, list[int]]:
    """Return {region_idx: [diff_bits_int, ...]} from the sparse TSV."""
    by_region: dict[int, list[int]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        region = int(parts[0])
        if len(parts) == 1 or not parts[1].strip():
            diff_bits = 0
        else:
            diff_bits = 0
            for token in parts[1].strip().split(","):
                if token:
                    diff_bits |= 1 << int(token)
        by_region[region].append(diff_bits)
    return dict(by_region)


def replay(switch_after: int, samples: dict[int, list[int]]) -> dict:
    rows = []
    for region in sorted(samples):
        est = LiveBitMaskEstimator(n_bits=MAP_BITS, switch_after=switch_after)
        for diff_bits in samples[region]:
            est.observe(0, diff_bits)
        rows.append(
            {
                "region": region,
                "samples": est.samples_seen,
                "mask_bits": est.mask.bit_count(),
                "converged": est.is_converged,
            }
        )
    return {
        "switch_after": switch_after,
        "regions": rows,
        "any_converged": any(r["converged"] for r in rows),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay liveness samples through LiveBitMaskEstimator")
    ap.add_argument("samples", type=Path, help="Path to sparse TSV from sweep_jpeg_liveness.py")
    args = ap.parse_args(argv)

    samples = _load_samples(args.samples)
    results = [replay(sa, samples) for sa in SWITCH_AFTERS]

    print("# JPEG real-corpus liveness replay\n")
    print(
        f"Data: {args.samples} — {sum(len(v) for v in samples.values())} samples across {len(samples)} region(s)\n"
    )

    for r in results:
        print(f"## switch_after={r['switch_after']}")
        print(f"any_converged={r['any_converged']}")
        for row in r["regions"]:
            print(
                f"  region {row['region']}: samples={row['samples']} "
                f"mask_bits={row['mask_bits']} converged={row['converged']}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
