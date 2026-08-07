"""Regression: ensure_poly() must not re-run recovery on every call.

``ensure_poly()`` runs once per fuzz iteration through the ``crc_learn``
availability gate (``REGISTRY.available`` -> ``build_ops`` -> ``mutate``).
When recovery cannot verify a polynomial it leaves ``_poly`` None, and the
old code re-executed the whole GCD/BM recovery on every call — measured
~56 s of a 103 s fuzz profile, collapsing eps to single digits. Recovery
must instead be re-attempted only when new pairs arrive.
"""

from __future__ import annotations

from fuzzer_tool.core.berlekamp_massey import compute_checksum
from fuzzer_tool.core.checksum_learner import ChecksumLearner

# Same data with different checksums: no single polynomial reproduces two of
# them, so recovery deterministically fails and never activates a model.
_UNVERIFIABLE = [(b"\x00\x00", crc) for crc in range(8, 16)]

_CUSTOM_POLY = 0x1D  # x^8 + x^4 + x^3 + x^2 + 1


class _FakeFuzzer:
    """Minimal stand-in for Fuzzer used by ChecksumLearner in tests."""

    def __init__(self):
        self._cmplog = None


def _counting_recover(monkeypatch, counter: list[int]):
    """Spy on _recover so tests can assert it is (not) re-invoked."""
    original = ChecksumLearner._recover

    def counting(self):
        counter[0] += 1
        return original(self)

    monkeypatch.setattr(ChecksumLearner, "_recover", counting)


def test_regression_ensure_poly_skips_rerecovery_on_unchanged_pairs(monkeypatch):
    f = _FakeFuzzer()
    learner = ChecksumLearner(f, min_pairs=8, poly_width=8)
    recover_calls = [0]
    _counting_recover(monkeypatch, recover_calls)

    # First attempt fails deterministically.
    learner.add_pairs(_UNVERIFIABLE)
    assert learner.ensure_poly() is None
    assert recover_calls[0] >= 1

    # Pairs unchanged: repeated ensure_poly() calls must not re-run recovery.
    calls_after_fail = recover_calls[0]
    for _ in range(100):
        assert learner.ensure_poly() is None
    assert recover_calls[0] == calls_after_fail

    # New evidence re-triggers a retry even though it will fail again.
    learner.add_pairs([(b"\x01\x02", crc) for crc in range(16, 24)])
    assert recover_calls[0] > calls_after_fail


def test_regression_ensure_poly_still_recovers_on_new_evidence(monkeypatch):
    from fuzzer_tool.core.crc32 import set_active_model

    f = _FakeFuzzer()
    learner = ChecksumLearner(f, min_pairs=8, poly_width=8)
    recover_calls = [0]
    _counting_recover(monkeypatch, recover_calls)

    good = [
        (bytes([i]), compute_checksum(bytes([i]), poly=_CUSTOM_POLY, width=8)) for i in range(8, 16)
    ]
    learner.add_pairs(good)
    poly = learner.ensure_poly()
    assert poly is not None
    assert recover_calls[0] >= 1
    assert compute_checksum(bytes([9]), poly=poly, width=8) == compute_checksum(
        bytes([9]), poly=_CUSTOM_POLY, width=8
    )

    # Once recovered, ensure_poly() must not re-run recovery either.
    calls_after_success = recover_calls[0]
    for _ in range(50):
        assert learner.ensure_poly() is poly
    assert recover_calls[0] == calls_after_success

    # Recovery activated the 8-bit model in the crc32 module; reset so later
    # tests see the standard (None) model.
    set_active_model(None)
