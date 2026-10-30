"""cmin reports timed-out replays and keeps seeds over --max-seed-bytes unreplayed.

A 1 s default timeout on slow seeds is invisible otherwise: ffmpeg
minimize ran at 5 seeds/s with 96% of wall in the target call.
"""

import numpy as np

from fuzzer_tool.adapters import process
from fuzzer_tool.services import minimize

# Inputs starting with this byte "time out" in the fake target.
_SLOW = b"T"


class _Shm:
    """ShmCoverage stand-in: one edge per distinct byte of the last input."""

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


def _setup(tmp_path, monkeypatch, seeds: dict[str, bytes]):
    """Corpus of *seeds*; returns (paths, corpus dir, replayed inputs)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    paths = []
    for name, data in seeds.items():
        p = corpus / name
        p.write_bytes(data)
        paths.append(p)

    replayed: list[bytes] = []

    def fake_run(target, data, timeout, env=None, **_):
        replayed.append(data)
        _Shm.current = data
        rc = minimize._TIMEOUT_RC if data.startswith(_SLOW) else 0
        return rc, "", 0

    monkeypatch.setattr(minimize, "ShmCoverage", _Shm)
    monkeypatch.setattr(process, "run_target_stdin", fake_run)
    return paths, corpus, replayed


def test_regression_cmin_reports_timeouts(tmp_path, monkeypatch, capsys):
    seeds = {"a": b"Tab", "b": b"Tcd", "c": b"ef", "d": b"gh", "e": b"ij"}
    paths, corpus, _ = _setup(tmp_path, monkeypatch, seeds)
    minimize._minimize_with_coverage(paths, "t", 1.0, False, None, str(tmp_path / "o"), corpus)

    slow = sum(d.startswith(_SLOW) for d in seeds.values())
    assert f"Timed out: {slow}/{len(seeds)} seeds" in capsys.readouterr().out


def test_regression_cmin_no_timeout_line_when_none(tmp_path, monkeypatch, capsys):
    """Falsification: a clean run prints no timeout line."""
    paths, corpus, _ = _setup(tmp_path, monkeypatch, {"a": b"ab", "b": b"cd"})
    minimize._minimize_with_coverage(paths, "t", 1.0, False, None, str(tmp_path / "o"), corpus)
    assert "Timed out" not in capsys.readouterr().out


def test_regression_cmin_oversized_kept_in_place(tmp_path, monkeypatch):
    cap = 4
    seeds = {"small": b"ab", "edge": b"abcd", "big": b"abcdefghij"}
    paths, corpus, replayed = _setup(tmp_path, monkeypatch, seeds)
    minimize._minimize_with_coverage(paths, "t", 1.0, False, None, None, corpus, max_seed_bytes=cap)

    # Over the cap: never replayed, never pruned. At the cap: replayed.
    assert seeds["big"] not in replayed
    assert seeds["edge"] in replayed
    assert (corpus / "big").read_bytes() == seeds["big"]
    assert not (corpus / "pruned" / "big").exists()


def test_regression_cmin_oversized_copied_to_output(tmp_path, monkeypatch):
    """Adversarial: with -o, an unmeasured seed must still reach the output."""
    paths, corpus, _ = _setup(tmp_path, monkeypatch, {"a": b"ab", "big": b"x" * 64})
    out = tmp_path / "o"
    minimize._minimize_with_coverage(
        paths, "t", 1.0, False, None, str(out), corpus, max_seed_bytes=8
    )
    assert (out / "big").exists()


def test_regression_cmin_size_cap_off_replays_all(tmp_path, monkeypatch):
    seeds = {"a": b"ab", "big": b"y" * 1000}
    paths, corpus, replayed = _setup(tmp_path, monkeypatch, seeds)
    minimize._minimize_with_coverage(paths, "t", 1.0, False, None, str(tmp_path / "o"), corpus)
    assert sorted(replayed) == sorted(seeds.values())


def test_regression_cmin_oversized_with_edge_hist(tmp_path, monkeypatch):
    """Adversarial: --edge-hist reads kept keys from the edge book; oversized have none."""
    from fuzzer_tool.core.run_mode import Switch

    paths, corpus, _ = _setup(tmp_path, monkeypatch, {"a": b"ab", "big": b"z" * 64})
    out = tmp_path / "o"
    minimize._minimize_with_coverage(
        paths, "t", 1.0, False, None, str(out), corpus, edge_hist=Switch.ON, max_seed_bytes=8
    )
    assert (out / "big").exists()
