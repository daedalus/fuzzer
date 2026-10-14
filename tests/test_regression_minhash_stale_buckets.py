"""MinHashLSH must drop a seed's old band entries when its signature changes.

`EdgeTracker.record_edges` re-`add()`s a seed after every signature update.
`add()` used to overwrite the signature but keep the old band entries, so a
seed stayed a candidate for bands it no longer has, `remove()` left ghosts,
and the bucket map grew with every update (461k leftover entries after
removing all 50k seeds that each got three updates).
"""

import random
from array import array

from fuzzer_tool.core.edge_tracker import MinHashLSH

PERM = 4
BANDS = 2


def _lsh() -> MinHashLSH:
    return MinHashLSH(num_perm=PERM, num_bands=BANDS)


def _sig(*vals: int) -> array:
    return array("Q", vals)


def _oracle(sigs: dict[str, array], key: str) -> set[str]:
    """Seeds sharing any band with `key`, by direct band comparison (no buckets)."""
    size = PERM // BANDS
    q = sigs[key]
    bands = {(b, tuple(q[b * size : (b + 1) * size])) for b in range(BANDS)}
    out = set()
    for k, s in sigs.items():
        if k == key:
            continue
        if any((b, tuple(s[b * size : (b + 1) * size])) in bands for b in range(BANDS)):
            out.add(k)
    return out


def test_regression_minhash_stale_buckets_readd():
    lsh = _lsh()
    lsh.add("a", _sig(1, 2, 3, 4))
    lsh.add("b", _sig(1, 2, 9, 9))  # shares band 0 with a's first signature
    lsh.add("a", _sig(5, 6, 7, 8))  # a no longer shares any band with b

    assert lsh.find_similar("b", min_jaccard=0) == set()


def test_regression_minhash_stale_buckets_remove():
    lsh = _lsh()
    lsh.add("a", _sig(1, 2, 3, 4))
    lsh.add("a", _sig(5, 6, 7, 8))
    lsh.remove("a")
    lsh.add("b", _sig(1, 2, 3, 4))

    assert lsh.find_similar("b", min_jaccard=0) == set()


def test_readd_keeps_unchanged_band():
    # Falsification: an update that keeps band 0 must keep the band-0 match.
    lsh = _lsh()
    lsh.add("a", _sig(1, 2, 3, 4))
    lsh.add("b", _sig(1, 2, 0, 0))
    lsh.add("c", _sig(9, 9, 3, 4))
    lsh.add("a", _sig(1, 2, 5, 6))

    assert lsh.find_similar("b", min_jaccard=0) == {"a"}
    assert lsh.find_similar("c", min_jaccard=0) == set()


def test_readd_same_sig_idempotent():
    lsh = _lsh()
    lsh.add("a", _sig(1, 2, 3, 4))
    lsh.add("b", _sig(1, 2, 3, 4))
    lsh.add("a", _sig(1, 2, 3, 4))
    lsh.remove("a")

    assert lsh.find_similar("b", min_jaccard=0) == set()


def test_oracle_control():
    # Rule 46: the oracle must be able to report a match and a non-match.
    sigs = {"a": _sig(1, 2, 3, 4), "b": _sig(1, 2, 0, 0), "c": _sig(7, 7, 7, 7)}
    assert _oracle(sigs, "a") == {"b"}
    assert _oracle(sigs, "c") == set()


def test_churn_matches_band_oracle():
    # Adversarial: random add/update/remove churn over a tiny value space, so
    # bands collide often; every query must equal direct band comparison.
    rng = random.Random(20261014)
    lsh = _lsh()
    live: dict[str, array] = {}
    keys = [f"k{i}" for i in range(12)]
    for _ in range(400):
        k = rng.choice(keys)
        if k in live and rng.random() < 0.25:
            lsh.remove(k)
            del live[k]
            continue
        s = _sig(*(rng.randrange(3) for _ in range(PERM)))
        lsh.add(k, s)
        live[k] = s
        for q in live:
            assert lsh.find_similar(q, min_jaccard=0) == _oracle(live, q)

    # Memory bound: removing every live seed leaves no bucket behind.
    for k in list(live):
        lsh.remove(k)
    assert all(not band for band in lsh._bands)


def test_band_hashes_byte_order_paths_agree(monkeypatch):
    # The buffer fast path must hash exactly what the struct fallback packs.
    import fuzzer_tool.core.edge_tracker as et

    lsh = MinHashLSH(num_perm=64, num_bands=8)
    sig = lsh.compute_signature(set(range(0, 3000, 7)))
    fast = lsh._band_hashes(sig)
    monkeypatch.setattr(et, "_LITTLE_ENDIAN", False)

    assert lsh._band_hashes(sig) == fast
