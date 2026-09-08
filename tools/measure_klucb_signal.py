#!/usr/bin/env python3
"""Measure whether KL-UCB buys tail share on the fuzzer's reward distribution.

The D-UCB and SW-UCB indexes both ship with a Gaussian confidence width
``B*sqrt(xi*log(n_t)/N_t(i))``.  That bound is valid for sub-Gaussian rewards,
but the cost-adjusted surprisal weights handed to ``record()`` are not
Gaussian: they are bounded in [0, 1] with a mass at zero (an operator that
found no new coverage gets exactly 0), so the true tail is heavier and the
Gaussian width under-covers.

KL-UCB (Cappé, Garivier, Maillard, Munos, Stoltz, 2013) replaces it with the
empirical-Bernoulli width, implemented as ``kl_ucb=True`` on both schedulers.
This script runs both forms on a Bernoulli environment -- the fuzzer's reward
distribution, since a coverage hit is binary -- and reports tail share on the
best arm.  If KL-UCB is not better on the measured distribution, leave the
flag off; if it is, enable it per-target rather than repo-wide, because the
convergence floors in ``tests/test_scheduler_convergence.py`` were pinned
against the Gaussian form.

Usage::

    python tools/measure_klucb_signal.py [--rounds 6000] [--seed 92]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

# E402 suppressed: the path manipulation above is required for these imports
# to resolve, and moving them above it would break the script.
from tests.support.bandit_env import StationaryBernoulli, run  # noqa: E402

from fuzzer_tool.core.rand_pool import RandPool  # noqa: E402
from fuzzer_tool.core.schedulers import DUCBScheduler, SWUCBScheduler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=6_000)
    ap.add_argument("--seed", type=int, default=92)
    args = ap.parse_args()

    env = StationaryBernoulli.build()
    print(f"env: {len(env.arms)} arms, best={env.best!r} p={env.probs[env.best]:.3f}")

    print(f"\n{'scheduler':<12} {'width':<12} {'tail share':>11}")
    print("-" * 39)
    for sched_name, factory in (
        ("DUCB", lambda: DUCBScheduler(rng=RandPool(args.seed))),
        ("SWUCB", lambda: SWUCBScheduler(rng=RandPool(args.seed))),
    ):
        for kl in (False, True):
            sched = factory()
            sched.kl_ucb = kl
            c = run(sched, env, seed=args.seed, rounds=args.rounds)
            print(
                f"{sched_name:<12} {'KL' if kl else 'Gaussian':<12} {c.tail_share(env.best):>11.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
