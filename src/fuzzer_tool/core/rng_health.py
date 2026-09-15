"""Quick, cheap sanity check for the fuzzer's PRNG stream at startup.

Not a cryptographic test suite -- fuzzing does not need one (see the
modulo-bias note in ``rand_pool.py``). This exists to catch a *broken* RNG
setup before a campaign burns hours on it: a seed wired to a constant, a
bit-generator that silently degenerates, a pool-refill bug that makes the
stream sticky, etc.

Rather than reimplementing statistical tests, this reuses three of the
NIST/dieharder-derived checks already in ``core/randomness.py`` (calibrated
there against ``os.urandom`` -- see ``test_randomness.py``):

  * ``monobit``     -- #1-bits vs #0-bits imbalance
  * ``byte_chisq``  -- byte-histogram uniformity
  * ``runs_test``   -- bit oscillation rate (catches RLE/padding-like output)

plus ``repeat_test`` on the raw draws, which is the one failure mode the
three bit/byte-level tests structurally can't see: a stream that repeats
its previous value more often than chance (e.g. a pool-refill boundary bug
or an EMA-style feedback loop feeding the RNG), since marginal byte
frequencies stay uniform under stickiness.

The four p-values are combined with ``fishers_method``. This never raises:
a suspect RNG is a warning, not a reason to abort a run that was otherwise
ready to go.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fuzzer_tool.core.randomness import byte_chisq, fishers_method, monobit, repeat_test, runs_test

_DEFAULT_N_BYTES = 4096
_P_THRESHOLD = 0.01  # standard NIST SP 800-22 significance level


@dataclass
class RngHealthResult:
    n_bytes: int
    combined_p: float
    pvalues: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.combined_p >= _P_THRESHOLD and not self.warnings

    def summary(self) -> str:
        if self.ok:
            detail = ", ".join(f"{name}={p:.3f}" for name, p in self.pvalues.items())
            return f"OK (n={self.n_bytes}B, {detail})"
        reasons = list(self.warnings)
        if self.combined_p < _P_THRESHOLD:
            detail = ", ".join(f"{name}={p:.4f}" for name, p in self.pvalues.items())
            reasons.append(f"combined p={self.combined_p:.4f} ({detail})")
        return "SUSPECT: " + "; ".join(reasons)


def quick_health_check(rng, n_bytes: int = _DEFAULT_N_BYTES) -> RngHealthResult:
    """Run a quick sanity check on *rng*'s output stream.

    Draws ``n_bytes`` bytes (default 4096 -- one ``RandPool`` refill's
    worth, negligible next to a fuzzing campaign) and runs monobit,
    byte_chisq, runs_test and repeat_test, then combines the four p-values
    with Fisher's method. Also flags an outright-constant stream, which a
    single small sample could in principle slip past the statistical tests.

    *rng* only needs a ``randint_list(a, b, count)`` method -- the public
    ``RandPool`` API -- so this does not reach into pool internals and
    works with any drop-in replacement (e.g. ``tests/support/scripted_rng.py``).
    """
    values = rng.randint_list(0, 255, n_bytes)
    data = bytes(values)

    warnings: list[str] = []
    distinct = len(set(values))
    if distinct <= 1:
        warnings.append("RNG stream is constant (a single repeated byte value)")

    pvalues = {
        "monobit": monobit(data),
        "byte_chisq": byte_chisq(data),
        "runs": runs_test(data),
        "repeat": repeat_test(values, alphabet=256),
    }
    combined_p = fishers_method(list(pvalues.values()))

    return RngHealthResult(
        n_bytes=len(data),
        combined_p=combined_p,
        pvalues=pvalues,
        warnings=warnings,
    )
