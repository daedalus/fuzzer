"""Cook-Mertz Tree Evaluation — GF(2^q) field layer.

Provides the finite-field arithmetic and primitive-element enumeration needed
for the multilinear extension (MLE) interpolation identity in the Cook-Mertz
protocol:

    f(x) = sum_{omega in F*} lambda_omega * f(omega)

where ``lambda_omega`` are Lagrange coefficients in ``GF(2^q)``.  The
primitive element ``omega`` must satisfy that ``{omega^1, ..., omega^m}``
enumerates ``F*`` exactly once, where ``m = |F| - 1 = 2^q - 1``.

The field arithmetic itself lives in :mod:`fuzzer_tool.core.gf2_common`; this
module wires it into the Cook-Mertz-specific API.
"""

from __future__ import annotations

from fuzzer_tool.core.gf2_common import GF2n, is_irreducible


class CookMertzField:
    """GF(2^q) field with a verified primitive element for MLE interpolation."""

    def __init__(self, q: int, modulus: int | None = None, seed: int = 0) -> None:
        if modulus is not None and not is_irreducible(modulus, q):
            raise ValueError(f"modulus {modulus:#x} is not irreducible of degree {q}")
        self.q = q
        self.field = GF2n(q, modulus=modulus, seed=seed)
        self.order = self.field.order  # 2^q
        self.m = self.field.m  # 2^q - 1
        self.modulus = self.field.mod
        self.omega = self.field.gen  # primitive element
        self._powers: dict[int, int] | None = None

    @classmethod
    def default(cls, q: int = 8, seed: int = 0) -> CookMertzField:
        """Convenience constructor: auto-select irreducible, deterministic seed."""
        return cls(q=q, seed=seed)

    def powers(self) -> dict[int, int]:
        """Return dict ``i -> omega^i`` for ``i = 0..m``.

        The caller uses indices ``1..m`` to enumerate every non-zero field
        element exactly once.  Cached after first call.
        """
        if self._powers is None:
            self._powers = self.field.omega_powers()
        return self._powers

    def interpolate(self, evaluations: dict[int, int]) -> dict[int, int]:
        """Lagrange interpolation in GF(2^q) over the omega power table.

        Args:
            evaluations: Mapping ``omega^i -> f(omega^i)`` for a subset of
                ``i`` values.

        Returns:
            Dict of all ``omega^i -> interpolated f(omega^i)`` for ``i`` in
            ``1..m``.

        Raises:
            ValueError: If *evaluations* is empty or contains duplicate ``i``
                values.
        """
        if not evaluations:
            raise ValueError("evaluations must be non-empty")

        powers = self.powers()
        idx: dict[int, int] = {}
        for i, w in powers.items():
            if i == 0:
                continue
            idx[w] = i

        seen_i: set[int] = set()
        for w in evaluations:
            if w not in idx:
                raise ValueError(f"evaluation point {w:#x} is not in the omega enumeration")
            i = idx[w]
            if i in seen_i:
                raise ValueError(f"duplicate evaluation at omega^{i}")
            seen_i.add(i)

        active_i = sorted(seen_i)
        result: dict[int, int] = {}
        for j in range(1, self.m + 1):
            target = powers[j]
            val = 0
            for i in active_i:
                num = 1
                den = 1
                for k in active_i:
                    if k == i:
                        continue
                    num = self.field.mul(num, target ^ powers[k])
                    den = self.field.mul(den, powers[i] ^ powers[k])
                val ^= self.field.mul(
                    self.field.mul(evaluations[powers[i]], num), self.field.inv(den)
                )
            result[target] = val
        return result

    def evaluate_mle(self, coefficients: list[int], point: int) -> int:
        """Evaluate a multilinear extension at a single field element.

        Args:
            coefficients: MLE coefficients in the standard basis
                (coefficient of ``prod_i x_i^{e_i}`` for each monomial).
            point: Field element to evaluate at.

        Returns:
            ``f(point)`` in ``GF(2^q)``.
        """
        result = 0
        for e_idx, coeff in enumerate(coefficients):
            if coeff == 0:
                continue
            term = coeff
            for bit in range(self.q):
                if (e_idx >> bit) & 1:
                    term = self.field.mul(term, point)
            result ^= term
        return result
