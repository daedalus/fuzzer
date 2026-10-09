"""collapse_correlated: no Gibbs sweeps over an all-zero coupling.

With J = 0 every sweep step redraws a bit from its own field, the same
marginal as the sweep-0 draw, so the sweeps changed nothing in distribution
and cost 40-50x a plain collapse (0.9 vs 0.018 ms at 512 bytes). Every
individual starts at J = 0 and only QEA-picked parents ever learn one.
"""

import numpy as np

from fuzzer_tool.core import qea as q
from fuzzer_tool.core.rand_pool import RandPool

SEED = 5


def _old_collapse_correlated(amplitudes, coupling, *, n_sweeps=q.CORRELATION_SWEEPS_DEFAULT, rng):
    """The pre-change sampler, verbatim (minus the default-pool fallback)."""
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    n_bits = len(amplitudes)
    if n_bits == 0:
        return b""
    num_bytes = n_bits // q.BITS_PER_BYTE
    coupling = np.asarray(coupling, dtype=np.float64)
    fields = q._alpha_to_field(amplitudes).reshape(num_bytes, q.BITS_PER_BYTE)
    p_plus0 = q._sigmoid(fields)
    state = np.where(rng.random_array((num_bytes, q.BITS_PER_BYTE)) < p_plus0, 1, -1)
    for _ in range(max(0, n_sweeps)):
        for bit_idx in range(q.BITS_PER_BYTE):
            j_row = coupling[:, bit_idx, :].copy()
            j_row[:, bit_idx] = 0.0
            local_field = fields[:, bit_idx] + np.einsum("bj,bj->b", j_row, state)
            p_plus = q._sigmoid(local_field)
            draw = rng.random_array(num_bytes) < p_plus
            state[:, bit_idx] = np.where(draw, 1, -1)
    bits = (state == -1).astype(np.uint8).reshape(-1)
    return bytes(np.packbits(bits).tobytes())


def _amps(n):
    rnd = np.random.default_rng(0)
    return rnd.uniform(q.ALPHA_MIN, q.ALPHA_MAX, 8 * n)


def _learned(n, seed):
    """A coupling after a few Hebbian updates, nonzero diagonal included."""
    c = q._zero_coupling(n)
    rnd = np.random.default_rng(seed)
    for _ in range(5):
        q.update_couplings(c, rnd.bytes(n), improved=bool(rnd.integers(2)), delta=0.1)
    c[0, 3, 3] = 0.7  # callers are not trusted to keep the diagonal zero
    return c


# ---------------------------------------------------------------------------


def test_regression_qea_zero_coupling():
    """Zero coupling: the result is the sweep-0 draw, one (n, 8) array of draws."""
    n = 64
    amps = _amps(n)
    got = q.collapse_correlated(amps, q._zero_coupling(n), rng=RandPool(seed=SEED))

    pool = RandPool(seed=SEED)
    fields = q._alpha_to_field(amps).reshape(n, 8)
    ones = pool.random_array((n, 8)) >= q._sigmoid(fields)  # s = -1 is bit 1
    assert got == bytes(np.packbits(ones.astype(np.uint8).reshape(-1)))


def test_zero_coupling_keeps_marginals():
    """Falsification: P(bit = 0) stays alpha^2 without the sweeps."""
    n = 2048
    amps = np.full(8 * n, 0.6)  # P(bit=0) = 0.36
    out = q.collapse_correlated(amps, q._zero_coupling(n), rng=RandPool(seed=SEED))
    bits = np.unpackbits(np.frombuffer(out, dtype=np.uint8))
    p0 = float((bits == 0).mean())
    sd = (0.36 * 0.64 / bits.size) ** 0.5
    assert abs(p0 - 0.36) < 5 * sd


def test_learned_coupling_matches_old_sampler():
    """Adversarial: any nonzero entry runs the sweeps, draw for draw as before."""
    for seed in range(12):
        n = 1 + seed * 7
        amps = _amps(n)
        c = _learned(n, seed)
        ctl = _old_collapse_correlated(amps, c, rng=RandPool(seed=seed))
        # Control (Hard Rule 46): the oracle against a second run of itself.
        assert _old_collapse_correlated(amps, c, rng=RandPool(seed=seed)) == ctl
        assert q.collapse_correlated(amps, c, rng=RandPool(seed=seed)) == ctl


def test_single_nonzero_entry_still_sweeps():
    """Adversarial: one coupled pair in the last byte is enough to sample jointly."""
    n = 16
    amps = _amps(n)
    c = q._zero_coupling(n)
    c[-1, 0, 1] = c[-1, 1, 0] = 1.5
    want = _old_collapse_correlated(amps, c, rng=RandPool(seed=SEED))
    assert q.collapse_correlated(amps, c, rng=RandPool(seed=SEED)) == want


def test_caller_coupling_is_not_modified():
    c = _learned(4, 1)
    before = c.copy()
    q.collapse_correlated(_amps(4), c, rng=RandPool(seed=SEED))
    np.testing.assert_array_equal(c, before)
