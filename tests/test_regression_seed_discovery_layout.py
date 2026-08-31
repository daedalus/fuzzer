"""Corpus consumers must walk seeds/<hh>/, not list the corpus directory.

``save_to_corpus`` writes ``seeds/<hh>/id_<hash>``. Three separate modules
listed the corpus directory non-recursively instead, and because the only
top-level regular file a live corpus holds is ``state.pkl.gz``, each found
zero seeds and silently substituted the sidecar:

  - ``minimize.py``       printed "Corpus is empty" and exited 0
  - ``parallel.py``       imported gzip bytes into every sibling worker
  - ``root_cause.py``     reported state.pkl.gz as the "nearest corpus seed"
                          and diffed the crash against it

The first two were fixed separately; this file pins the third and the shared
``discover_seed_files`` all three now go through.

Every assertion here is on VALUES — which bytes came back — because the
defect produced a well-typed, non-empty, entirely wrong result. A test
asserting only ``len(seeds) > 0`` passes against the pre-fix code.
"""

import gzip
import pickle

import pytest

from fuzzer_tool.adapters.filesystem import (
    discover_seed_files,
    save_crashing_seed,
    save_irreplaceable,
    save_to_corpus,
)
from fuzzer_tool.services.minimize import _discover_corpus_files
from fuzzer_tool.services.root_cause import _load_corpus

SEEDS = (b"seed-alpha", b"seed-bravo", b"seed-charlie")


@pytest.fixture
def corpus(tmp_path):
    """A corpus in the shape save_to_corpus actually produces."""
    for payload in SEEDS:
        save_to_corpus(payload, tmp_path, set())
    save_irreplaceable(b"seed-irreplaceable", tmp_path, set(), set())
    save_crashing_seed(b"seed-crashing", tmp_path, set(), set())

    pruned = tmp_path / "seeds" / "pruned" / "ab"
    pruned.mkdir(parents=True)
    (pruned / "id_abcdef0123456789").write_bytes(b"seed-pruned")

    # The sidecars that a flat listing picks up instead of the seeds.
    with gzip.open(tmp_path / "state.pkl.gz", "wb") as fh:
        pickle.dump({"markov": {}}, fh)
    (tmp_path / "stats.txt").write_text("execs 1000\n")
    (tmp_path / "coverage.json").write_text("{}")
    return tmp_path


def _bytes(paths):
    return sorted(p.read_bytes() for p in paths)


class TestRootCauseBaselineDiscovery:
    def test_returns_the_seeds_not_the_state_file(self, corpus):
        got = sorted(data for _name, data in _load_corpus(str(corpus)))
        assert got == sorted([*SEEDS, b"seed-irreplaceable"])

    def test_state_file_is_never_offered_as_a_baseline(self, corpus):
        names = [name for name, _ in _load_corpus(str(corpus))]
        assert "state.pkl.gz" not in names
        assert "stats.txt" not in names
        assert "coverage.json" not in names

    def test_no_returned_baseline_is_gzip(self, corpus):
        # The pre-fix failure was specifically that gzip bytes were diffed
        # against the crash input, so assert on the magic directly.
        for _name, data in _load_corpus(str(corpus)):
            assert not data.startswith(b"\x1f\x8b")

    def test_crashing_seeds_excluded_a_baseline_must_not_crash(self, corpus):
        got = [data for _, data in _load_corpus(str(corpus))]
        assert b"seed-crashing" not in got

    def test_pruned_seeds_excluded(self, corpus):
        got = [data for _, data in _load_corpus(str(corpus))]
        assert b"seed-pruned" not in got

    def test_irreplaceable_seeds_are_valid_baselines(self, corpus):
        # They are ordinary entries that are merely exempt from pruning,
        # so unlike minimize, root_cause must keep them.
        got = [data for _, data in _load_corpus(str(corpus))]
        assert b"seed-irreplaceable" in got

    def test_flat_directory_of_loose_files_still_works(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"loose-one")
        (tmp_path / "b.bin").write_bytes(b"loose-two")
        got = sorted(data for _, data in _load_corpus(str(tmp_path)))
        assert got == [b"loose-one", b"loose-two"]

    def test_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert _load_corpus(str(tmp_path / "nope")) == []


class TestSharedDiscovery:
    def test_default_excludes_pruned_and_crashing_keeps_irreplaceable(self, corpus):
        assert _bytes(discover_seed_files(corpus)) == sorted([*SEEDS, b"seed-irreplaceable"])

    def test_opt_in_subtrees(self, corpus):
        got = _bytes(discover_seed_files(corpus, include_pruned=True, include_crashing=True))
        assert got == sorted([*SEEDS, b"seed-irreplaceable", b"seed-crashing", b"seed-pruned"])

    def test_sidecars_never_returned(self, corpus):
        for p in discover_seed_files(corpus, include_pruned=True, include_crashing=True):
            assert p.suffix not in (".gz", ".txt", ".json", ".tmp")

    def test_symlinks_are_not_followed(self, corpus):
        link = corpus / "seeds" / "zz"
        link.mkdir()
        target = corpus / "state.pkl.gz"
        try:
            (link / "id_0000000000000000").symlink_to(target)
        except OSError:  # pragma: no cover - platform without symlink perms
            pytest.skip("symlinks unavailable")
        assert b"seed-alpha" in _bytes(discover_seed_files(corpus))
        for p in discover_seed_files(corpus):
            assert not p.is_symlink()


class TestMinimizeUsesTheSameWalk:
    def test_minimize_excludes_irreplaceable_root_cause_does_not(self, corpus):
        """The two consumers share the walk but not the exclusions.

        minimize must not offer never-prune entries as prune candidates;
        root_cause has no such constraint. Collapsing them to one policy
        would be wrong in one direction or the other.
        """
        assert _bytes(_discover_corpus_files(corpus)) == sorted(SEEDS)
        assert b"seed-irreplaceable" in [d for _, d in _load_corpus(str(corpus))]

    def test_minimize_finds_the_seeds_at_all(self, corpus):
        assert _bytes(_discover_corpus_files(corpus)) == sorted(SEEDS)
