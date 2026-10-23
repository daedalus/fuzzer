"""``OperatorEngine.parent_map`` extends on append instead of rebuilding.

It rebuilt over the whole corpus on every admission (~0.9% of ffmpeg
--lightweight wall time at 5k seeds). Same map as a full rebuild; an
in-place swap (trim) still forces one.
"""

from fuzzer_tool.services.corpus_manager import seed_key
from fuzzer_tool.services.operators import OperatorEngine
from tests.test_bit_cross import _Fuzzer


def _full(corpus, meta) -> dict:
    # Reference: the rebuild the cache replaces.
    return {seed_key(bytes(s)): (meta.get(bytes(s)) or {}).get("parent_key") for s in corpus}


def _engine(corpus, parents):
    fz = _Fuzzer(corpus, parents=parents)
    return OperatorEngine(fz), fz


SEEDS = [b"seed-%03d" % i for i in range(12)]
PARENTS = {SEEDS[i]: SEEDS[i - 1] for i in range(1, 12, 2)}


def test_regression_parent_map_control():
    # Control (Hard Rule 46).
    _e, fz = _engine(list(SEEDS), PARENTS)
    assert _full(fz.corpus, fz.seed_meta) == _full(fz.corpus, fz.seed_meta)


def test_regression_parent_map_appends_match_rebuild():
    corpus = SEEDS[:3]
    eng, fz = _engine(corpus, PARENTS)
    for s in SEEDS[3:]:
        assert eng.parent_map() == _full(corpus, fz.seed_meta)
        corpus.append(s)
    assert eng.parent_map() == _full(corpus, fz.seed_meta)


def test_regression_parent_map_keys_only_the_tail():
    # Falsification: an append hashes the new seeds, not the whole corpus.
    corpus = list(SEEDS[:10])
    eng, _fz = _engine(corpus, PARENTS)
    eng.parent_map()
    hashed = []
    eng._key_fn = lambda: lambda b: hashed.append(b) or seed_key(b)
    corpus.extend(SEEDS[10:])
    eng.parent_map()
    assert hashed == SEEDS[10:]


def test_regression_parent_map_swap_then_grow_rebuilds():
    # Adversarial: trim replaces a seed in place, then the corpus grows.
    corpus = list(SEEDS[:5])
    eng, fz = _engine(corpus, PARENTS)
    eng.parent_map()
    corpus[2] = b"trimmed"
    corpus.append(SEEDS[5])
    got = eng.parent_map()
    assert got == _full(corpus, fz.seed_meta)
    assert seed_key(SEEDS[2]) not in got


def test_regression_parent_map_new_list_rebuilds():
    # Adversarial: pruning rebinds the corpus to a shorter list.
    eng, fz = _engine(list(SEEDS), PARENTS)
    eng.parent_map()
    fz.corpus = SEEDS[6:]
    eng._ctx_cache = None  # mutate() refreshes ctx once per round
    assert eng.parent_map() == _full(fz.corpus, fz.seed_meta)
