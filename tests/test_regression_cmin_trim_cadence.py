"""cmin replay trims the heap once per _TRIM_EVERY seeds, not every 10.

malloc_trim walks every arena; at 1 call per 10 seeds it ran ~2x per 10
seeds (progress line + periodic block) on in-process targets.
"""

import numpy as np

from fuzzer_tool.adapters import process
from fuzzer_tool.services import minimize


class _Shm:
    """ShmCoverage stand-in: every replay reports one edge per byte value."""

    current: bytes = b""

    def __init__(self):
        self.env_id = "0"
        self.num_entries = 16

    def reset_edge_map(self):
        pass

    def edge_id_array(self):
        return np.array(sorted(set(type(self).current)), dtype=np.uint32) + 1

    def cleanup(self):
        pass


def _fake_run(target, data, timeout, env=None, **_):
    _Shm.current = data
    return 0, "", 0


def _trims(tmp_path, monkeypatch, n: int) -> int:
    """trim_heap calls during a coverage cmin over *n* seeds."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    paths = []
    for i in range(n):
        p = corpus / f"s{i:05d}"
        p.write_bytes(bytes([i % 256, 7]))
        paths.append(p)

    calls = []
    monkeypatch.setattr(minimize, "ShmCoverage", _Shm)
    monkeypatch.setattr(process, "run_target_stdin", _fake_run)
    monkeypatch.setattr(minimize.libc_mem, "trim_heap", lambda: calls.append(1) or 0)
    minimize._minimize_with_coverage(paths, "t", 1.0, False, None, str(tmp_path / "out"), corpus)
    return len(calls)


def test_regression_cmin_trim_every_100(tmp_path, monkeypatch):
    assert minimize._TRIM_EVERY == 100
    n = 2 * minimize._TRIM_EVERY + minimize._TRIM_EVERY // 2
    assert _trims(tmp_path, monkeypatch, n) == n // minimize._TRIM_EVERY


def test_regression_cmin_no_trim_below_cadence(tmp_path, monkeypatch):
    """Adversarial: one seed short of the cadence never trims."""
    assert _trims(tmp_path, monkeypatch, minimize._TRIM_EVERY - 1) == 0
