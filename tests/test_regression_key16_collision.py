"""Forged key16 (XXH64, fixed seed) collisions must never lose a distinct input.

Load used to unlink any file whose ``id_<h>`` name was taken, and import
skipped any input whose hash was seen, without comparing bytes.
"""

from pathlib import Path

from fuzzer_tool.adapters.filesystem import hash_data, load_corpus
from fuzzer_tool.services.import_corpus import import_from_afl, import_from_libfuzzer
from tests.support.xxh64_collide import colliding

_REF = b"R" * 64 + b"shared-tail"
_PREFIXES = [b"orlp-xxh64-second-preimage-%04d" % i for i in range(1, 3)]
_PREFIXES = [p + b"!" * (32 - len(p)) for p in _PREFIXES]


def _forged(n: int) -> list[bytes]:
    return colliding(_REF, _PREFIXES[: n - 1])


def _stored(root: Path) -> list[bytes]:
    return sorted(f.read_bytes() for f in root.rglob("*") if f.is_file())


def test_forged_inputs_collide():
    """Control: the helper forges, and the oracle can tell distinct digests."""
    a, b, c = _forged(3)
    assert len({a, b, c}) == 3
    assert hash_data(a) == hash_data(b) == hash_data(c)
    assert hash_data(a) != hash_data(a + b"x")


def test_regression_load_keeps_colliding_seed(tmp_path):
    a, b = _forged(2)
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / f"id_{hash_data(a)}").write_bytes(a)
    (seeds / "imported_b").write_bytes(b)

    load_corpus(tmp_path, add_default=False)

    assert _stored(seeds) == sorted([a, b])


def test_load_still_removes_true_duplicate(tmp_path):
    """Falsification: identical bytes under a second name stay deduplicated."""
    a = _forged(1)[0]
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / f"id_{hash_data(a)}").write_bytes(a)
    (seeds / "copy").write_bytes(a)

    load_corpus(tmp_path, add_default=False)

    assert [f.name for f in seeds.iterdir()] == [f"id_{hash_data(a)}"]


def test_load_is_stable_over_collision_slots(tmp_path):
    """Adversarial: three-way collision, loaded twice; nothing moves or vanishes."""
    forged = _forged(3)
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    for i, data in enumerate(forged):
        (seeds / f"raw_{i}").write_bytes(data)

    load_corpus(tmp_path, add_default=False)
    names = sorted(f.name for f in seeds.iterdir())
    load_corpus(tmp_path, add_default=False)

    assert sorted(f.name for f in seeds.iterdir()) == names
    assert _stored(seeds) == sorted(forged)


def test_regression_import_keeps_colliding_seeds(tmp_path):
    forged = _forged(3)
    src = tmp_path / "src"
    src.mkdir()
    for i, data in enumerate(forged):
        (src / f"in_{i}").write_bytes(data)
    dest = tmp_path / "corpus"

    assert import_from_libfuzzer(str(src), str(dest)) == len(forged)
    assert _stored(dest) == sorted(forged)
    assert import_from_libfuzzer(str(src), str(dest)) == 0


def test_import_collision_against_legacy_name(tmp_path):
    """Adversarial: the held twin sits under a name unrelated to its hash."""
    a, b = _forged(2)
    dest = tmp_path / "corpus"
    dest.mkdir()
    (dest / "id_0000000000000000").write_bytes(a)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a").write_bytes(a)
    (src / "b").write_bytes(b)

    assert import_from_libfuzzer(str(src), str(dest)) == 1
    assert _stored(dest) == sorted([a, b])


def test_regression_crash_import_keeps_colliding_crash(tmp_path):
    a, b = _forged(2)
    crashes = tmp_path / "afl" / "crashes"
    crashes.mkdir(parents=True)
    (crashes / "crash_a").write_bytes(a)
    (crashes / "crash_b").write_bytes(b)
    (crashes / "crash_b.txt").write_text("meta-b")
    out = tmp_path / "crashes"

    _, n = import_from_afl(str(tmp_path / "afl"), str(tmp_path / "corpus"), str(out))

    assert n == 2
    bins = {f.read_bytes(): f for f in out.glob("*.bin")}
    assert sorted(bins) == sorted([a, b])
    assert bins[b].with_suffix(".txt").read_text() == "meta-b"
