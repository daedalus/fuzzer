"""QEA state: an all-zero coupling persists as its byte count, not 64 floats a byte.

With ``--hail-mary`` (``use_correlation``) every individual carries a
(num_bytes, 8, 8) coupling and only QEA-picked parents ever learn one: 190 of
200 were all zero after 3k fuzzgoat execs, yet ``to_dict`` wrote each as
nested float lists. That was 82 of the state pickle's 102 MB, and the save
built, pickled and (``_verify_readable``) unpickled 8.6M Python floats:
peak RSS 0.65-1.45 GB.
"""

import json
import pickle

import numpy as np

from fuzzer_tool.core.qea import ALPHA_UNIFORM, QEAIndividual, _zero_coupling

N_BYTES = 300


def _ind(coupling):
    return QEAIndividual(amplitudes=[ALPHA_UNIFORM] * (8 * N_BYTES), coupling=coupling)


def test_regression_qea_zero_coupling_state():
    """A zero coupling serializes to a constant-size marker and restores exactly."""
    d = _ind(_zero_coupling(N_BYTES)).to_dict()
    assert len(pickle.dumps(d["coupling"])) < 64

    restored = QEAIndividual.from_dict(d)
    assert restored.coupling.shape == (N_BYTES, 8, 8)
    assert restored.coupling.dtype == np.float64
    assert not restored.coupling.any()


def test_learned_coupling_still_round_trips():
    """Falsification: one nonzero entry keeps the full tensor."""
    c = _zero_coupling(N_BYTES)
    c[-1, 2, 5] = c[-1, 5, 2] = -0.75
    restored = QEAIndividual.from_dict(_ind(c).to_dict())
    np.testing.assert_allclose(restored.coupling, c, atol=1e-6)


def test_zero_marker_survives_json():
    """Adversarial: the legacy JSON save path still round-trips the marker."""
    d = json.loads(json.dumps(_ind(_zero_coupling(3)).to_dict()))
    assert QEAIndividual.from_dict(d).coupling.shape == (3, 8, 8)


def test_legacy_list_zero_coupling_still_loads():
    """Adversarial: state written before the marker (nested zero lists) loads unchanged."""
    d = _ind(None).to_dict()
    d["coupling"] = _zero_coupling(2).tolist()
    restored = QEAIndividual.from_dict(d)
    np.testing.assert_array_equal(restored.coupling, _zero_coupling(2))


def test_none_coupling_stays_none():
    assert QEAIndividual.from_dict(_ind(None).to_dict()).coupling is None
