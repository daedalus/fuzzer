"""Regression for HIGH finding #16 from docs/bugreport_2026-08-21_merged.md.

Fails against the pre-fix source.
"""

from __future__ import annotations

import random

from fuzzer_tool.core.transfer_entropy import TransferEntropy


class TestTransferEntropyBiasCorrection:
    """#16 core/transfer_entropy.py:84-105 -- plug-in TE estimator reported
    ~2.6 bits for independent uniform byte streams (n=1500); no bias
    correction for context cardinality. byte_to_edge_flow/causal_chains
    produced spurious causal edges on essentially any input."""

    def test_independent_streams_report_near_zero_te(self):
        te = TransferEntropy(history_length=1, n_bins=256)
        vals = []
        for seed in range(8):
            rng = random.Random(seed)
            n = 1500
            source = [rng.randint(0, 255) for _ in range(n)]
            target = [rng.randint(0, 255) for _ in range(n)]
            vals.append(te.transfer_entropy(source, target))
        # Pre-fix this was ~2.6 bits; the bias-corrected estimate should
        # sit near the noise floor for genuinely independent signals.
        assert max(vals) < 0.5

    def test_real_causal_coupling_still_detected(self):
        """The bias correction must not wash out a genuine X->Y influence."""
        te = TransferEntropy(history_length=1, n_bins=256)
        rng = random.Random(1)
        n = 1500
        source = [rng.randint(0, 15) for _ in range(n)]
        target = [0]
        for t in range(n - 1):
            target.append((source[t] * 3 + rng.randint(0, 1)) % 16)

        te_forward = te.transfer_entropy(source, target)
        te_reverse = te.transfer_entropy(target, source)
        assert te_forward > 1.0
        assert te_reverse < te_forward * 0.1
