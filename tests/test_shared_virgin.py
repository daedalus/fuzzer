"""--shared-virgin: one novelty store across the per-target coverage tables.

Multi-target mode gives every target its own ShmCoverage (own SHM region, own
generation counter). By default each also keeps its own virgin map, so an edge
id one target already covered is new again the first time another target fires
it. With the gate the tables share the *novelty* state (seen ids, masked ids,
the (edge, bucket) virgin map) but keep per-target ``cumulative_edges``, which
the schedulers weigh targets by.
"""

import ctypes
import tempfile
from unittest.mock import patch

import pytest

from fuzzer_tool.adapters.shm import SharedVirgin, ShmCoverage


@pytest.fixture
def pair():
    a, b = ShmCoverage(), ShmCoverage()
    try:
        yield a, b
    finally:
        a.cleanup()
        b.cleanup()


def _fire(cov: ShmCoverage, edges: dict[int, int] | set[int]) -> bool:
    """One execution of *cov*'s target firing *edges* ({id: hit count} or ids)."""
    counts = edges if isinstance(edges, dict) else dict.fromkeys(edges, 1)
    cov.reset_edge_map()
    gen = cov.read_generation()
    for slot, (eid, n) in enumerate(sorted(counts.items())):
        cov._entries[slot].edge_id = eid
        cov._entries[slot].count = ((gen & 0xFF) << 24) | n
    # Header: path_hash must differ run to run or the fast path swallows the
    # scan; edge_count is the shim's new-slot counter.
    ctypes.c_uint64.from_address(cov._ptr + 8).value = hash(tuple(sorted(counts.items()))) | 1
    ctypes.c_uint64.from_address(cov._ptr + 16).value = len(counts)
    return cov.is_new_coverage()


class TestUnsharedBaseline:
    def test_each_table_judges_novelty_alone(self, pair):
        a, b = pair
        assert _fire(a, {1, 2, 3})
        assert _fire(b, {1, 2, 3})  # the control: B has never seen them


class TestSharedNovelty:
    def test_an_edge_seen_through_one_target_is_old_for_the_other(self, pair):
        a, b = pair
        sv = SharedVirgin()
        a.share_virgin(sv)
        b.share_virgin(sv)
        assert _fire(a, {1, 2, 3})
        assert not _fire(b, {1, 2, 3})
        assert len(sv) == 3

    def test_a_distinct_edge_is_added_to_the_union(self, pair):
        a, b = pair
        sv = SharedVirgin()
        a.share_virgin(sv)
        b.share_virgin(sv)
        _fire(a, {1, 2, 3})
        assert _fire(b, {2, 3, 4})
        assert b.last_new_ids == frozenset({4})
        assert len(sv) == 4

    def test_hit_count_buckets_are_shared_too(self, pair):
        a, b = pair
        sv = SharedVirgin()
        a.share_virgin(sv)
        b.share_virgin(sv)
        assert _fire(a, {5: 1})
        assert not _fire(b, {5: 1})  # same edge, same bucket
        assert _fire(b, {5: 2})  # new bucket, from B
        assert b.last_old_bucket_novel
        assert not _fire(a, {5: 2})  # ... which is now old for A

    def test_masking_through_one_target_masks_for_all(self, pair):
        a, b = pair
        sv = SharedVirgin()
        a.share_virgin(sv)
        b.share_virgin(sv)
        a.mask_edges({7})
        assert not _fire(b, {7})

    def test_late_join_keeps_what_the_table_already_held(self, pair):
        a, b = pair
        _fire(a, {1, 2})
        sv = SharedVirgin()
        a.share_virgin(sv)
        b.share_virgin(sv)
        assert not _fire(b, {1, 2})
        assert a.cumulative_edges == 2


class TestPerTargetAccounting:
    def test_cumulative_edges_counts_what_each_target_itself_reaches(self, pair):
        a, b = pair
        sv = SharedVirgin()
        a.share_virgin(sv)
        b.share_virgin(sv)
        _fire(a, {1, 2, 3})
        assert not _fire(b, {1, 2})
        # B found nothing new, but it does cover two edges: the scheduler's
        # gap signal must not read it as a target with no coverage.
        assert (a.cumulative_edges, b.cumulative_edges) == (3, 2)
        assert not _fire(b, {1, 2})
        assert b.cumulative_edges == 2  # not recounted

    def test_unshared_accounting_is_unchanged(self, pair):
        a, _ = pair
        _fire(a, {1, 2, 3})
        assert a.cumulative_edges == 3
        assert a._local_ids is None

    def test_phantom_rejection_still_retracts_the_count(self, pair):
        a, b = pair
        sv = SharedVirgin()
        a.share_virgin(sv)
        b.share_virgin(sv)
        _fire(a, {1, 2, 3})
        assert a.reject_phantoms({3}) == 1
        assert a.cumulative_edges == 2
        assert not _fire(b, {3})  # masked for everyone

    def test_maxima_stay_per_target(self, pair):
        a, b = pair
        sv = SharedVirgin()
        a.share_virgin(sv)
        b.share_virgin(sv)
        assert a._max_counts is not b._max_counts
        assert a._max_counts_wide is not b._max_counts_wide


class TestAdversarial:
    def test_ids_past_the_dense_range_are_shared_through_the_dict(self, pair):
        from fuzzer_tool.adapters.shm import VIRGIN_DENSE_MAX

        a, b = pair
        sv = SharedVirgin()
        a.share_virgin(sv)
        b.share_virgin(sv)
        big = VIRGIN_DENSE_MAX + 5
        assert _fire(a, {big: 1})
        assert not _fire(b, {big: 1})
        assert sv.wide[big] != 0
        assert a._virgin_wide is b._virgin_wide

    def test_a_join_after_both_tables_diverged_unions_them(self, pair):
        a, b = pair
        _fire(a, {1: 1, 2: 4})
        _fire(b, {2: 1, 3: 1})
        sv = SharedVirgin()
        a.share_virgin(sv)
        b.share_virgin(sv)
        assert len(sv) == 3
        assert not _fire(a, {3: 1})  # B's edge
        assert not _fire(b, {2: 4})  # A's bucket
        assert (a.cumulative_edges, b.cumulative_edges) == (3, 2)


class TestJoining:
    def test_joining_twice_is_a_noop_and_a_second_store_is_refused(self, pair):
        a, _ = pair
        sv = SharedVirgin()
        a.share_virgin(sv)
        a.share_virgin(sv)
        with pytest.raises(ValueError):
            a.share_virgin(SharedVirgin())

    def test_the_dense_array_is_full_size_so_no_sharer_can_detach(self):
        from fuzzer_tool.adapters.shm import VIRGIN_DENSE_MAX

        assert SharedVirgin().virgin.size == VIRGIN_DENSE_MAX


def _fuzzer(**kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmp = tempfile.mkdtemp(prefix="shared_virgin_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmp}/c",
            crashes_dir=f"{tmp}/x",
            max_len=64,
            timeout=1,
            mutations_per_input=2,
            use_coverage=True,
            **kw,
        )


class TestFuzzerGate:
    TARGETS = ["/bin/true", "/bin/false"]

    def test_off_by_default(self):
        f = _fuzzer(multi_targets=self.TARGETS)
        assert f._shared_virgin is None
        a, b = f._target_shm_covs.values()
        assert a._seen_edge_ids is not b._seen_edge_ids

    def test_on_shares_one_store_across_the_target_tables(self):
        f = _fuzzer(multi_targets=self.TARGETS, shared_virgin=True)
        a, b = f._target_shm_covs.values()
        assert f._shared_virgin is not None
        assert a._seen_edge_ids is b._seen_edge_ids is f._shared_virgin.seen
        assert a._virgin is b._virgin

    def test_single_target_ignores_it(self, capsys):
        f = _fuzzer(shared_virgin=True)
        assert f._shared_virgin is None
        assert "needs multi-target" in capsys.readouterr().out


class TestCli:
    def test_flag_is_exposed_and_passed_through(self):
        import inspect

        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        assert '"--shared-virgin"' in src
        assert 'shared_virgin=getattr(args, "shared_virgin", False)' in src
