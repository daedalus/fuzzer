"""tools/lib/bench_crash_cluster.py: ARI scorer and fuzzgoat input generator."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

import bench_crash_cluster as B  # noqa: E402

from fuzzer_tool.core.rand_pool import RandPool  # noqa: E402

# An object or empty array in a noise-only document would plant a second bug.
_STRUCT_RE = re.compile(rb"\{|\[\s*\]")
_STRING_RE = re.compile(rb'"([^"]*)"')


class TestAri:
    def test_identity_and_relabel(self):
        t = np.array([0, 0, 1, 1, 2, 2])

        assert B.ari(t, t) == 1.0
        assert B.ari(t, (t + 1) % 3) == 1.0

    def test_falsification_crossed_split(self):
        # Every pair split differently: contingency all ones, n=4.
        # sij=0, sa=sb=2, expected=2*2/6, max=2 -> (0 - 2/3) / (2 - 2/3) = -0.5.
        assert B.ari(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) == pytest.approx(-0.5)

    def test_adversarial_one_cluster_scores_zero(self):
        t = np.array([0, 0, 1, 1, 2, 2])

        assert B.ari(t, np.zeros_like(t)) == 0.0


class TestGenerator:
    def test_one_trigger_per_document(self):
        rp = RandPool(11)

        for trig in B.TRIGGERS.values():
            for _ in range(50):
                doc = B._doc(rp, trig.encode().decode())
                # Exactly one trigger: removing it leaves pure noise.
                rest = doc.replace(trig.encode(), b"0", 1)
                assert trig.encode() in doc
                assert not _STRUCT_RE.search(rest)
                # Strings of length 0 or 1 are triggers too.
                assert all(len(x) >= 2 for x in _STRING_RE.findall(rest))
