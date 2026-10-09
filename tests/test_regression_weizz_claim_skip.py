"""build_tag_map_from_cmplog: claim free bytes through a skip list.

_claim_span re-scanned every byte of every operand occurrence to find free
ones, even when all were claimed: occurrences x operand size per pair. On
ffmpeg under --hail-mary (18 KB seeds, operands like b"\\x00\\x00" occurring
thousands of times) that was 54% of wall time (py-spy, 2026-10-09). Bytes
only go free -> claimed, so a path-compressed "next free byte" pointer skips
claimed runs; the tag map must not change.
"""

import pytest

from fuzzer_tool.core import aho_corasick as ac
from fuzzer_tool.core import weizz_tags as wt
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.weizz_tags import ByteTag

INPUT_LEN = 512
N_PAIRS = 120
N_CASES = 10
ALPHABET = b"\x00\x01\xffAB"  # low entropy: operands recur, spans collide


@pytest.fixture(autouse=True)
def _fresh_caches():
    ac._reset_scanner_cache()
    wt._plans.clear()
    yield
    ac._reset_scanner_cache()
    wt._plans.clear()


class _ScanClaims:
    """Oracle: the per-byte scan _claim_span did before the skip list."""

    def __init__(self, n: int):
        self.tags = [ByteTag() for _ in range(n)]
        self.dep_bytes: set[int] = set()

    def claim(self, off, size, cid, counter, flags) -> bool:
        end = off + size
        if not any(self.tags[i].cmp_id == 0 for i in range(off, end)):
            return False
        for i in range(off, end):
            if self.tags[i].cmp_id == 0:
                self.tags[i] = ByteTag(cmp_id=cid, parent=0, counter=counter, flags=flags)
                self.dep_bytes.add(i)
        return True

    def claim_each(self, offs, size, cid, counter, flags) -> int:
        first = -1
        for off in offs:
            if self.claim(off, size, cid, counter, flags) and first < 0:
                first = off
        return first


def _case(seed: int) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    rng = RandPool(seed)
    data = bytes(rng.choice(ALPHABET) for _ in range(INPUT_LEN))
    pairs = []
    for _ in range(N_PAIRS):
        size = rng.randint(1, 12)
        start = rng.randrange(INPUT_LEN - size)
        window = data[start : start + size]
        other = rng.randbytes(rng.randint(1, 12))
        pairs.append((window, other) if rng.randrange(2) else (other, window))
    return data, pairs


def _build(monkeypatch, claims_cls, data, pairs):
    monkeypatch.setattr(wt, "_TagClaims", claims_cls)
    wt._plans.clear()
    smap = wt.build_tag_map_from_cmplog(data, pairs)
    return smap.tags, smap


class TestClaimEquivalence:
    def test_regression_oracle_matches_itself(self, monkeypatch):
        """Control (Hard Rule 46)."""
        for seed in range(N_CASES):
            data, pairs = _case(seed)
            assert _build(monkeypatch, _ScanClaims, data, pairs) == _build(
                monkeypatch, _ScanClaims, data, pairs
            )

    def test_regression_same_tag_map_as_scan(self, monkeypatch):
        real = wt._TagClaims
        for seed in range(N_CASES):
            data, pairs = _case(seed)
            assert _build(monkeypatch, real, data, pairs) == _build(
                monkeypatch, _ScanClaims, data, pairs
            ), f"seed {seed}"

    def test_regression_cases_claim_and_collide(self, monkeypatch):
        """The cases must exercise refused claims, not only fresh ones."""
        refused = granted = 0
        real = wt._TagClaims

        class Counting(real):
            def claim(self, *args):
                nonlocal refused, granted
                ok = super().claim(*args)
                granted += ok
                refused += not ok
                return ok

        for seed in range(N_CASES):
            _build(monkeypatch, Counting, *_case(seed))
        assert granted > 0
        assert refused > 0

    def test_regression_adversarial_uniform_input(self, monkeypatch):
        """Every operand occurs at every offset: maximal re-claim pressure."""
        data = b"\x00" * 300
        pairs = [(b"\x00" * k, b"\x07" * k) for k in (1, 2, 3, 8, 64)]
        real = wt._TagClaims
        assert _build(monkeypatch, real, data, pairs) == _build(
            monkeypatch, _ScanClaims, data, pairs
        )

    def test_regression_adversarial_zero_cmp_id(self, monkeypatch):
        """cid 0 writes a tag but leaves the byte free (cmp_id stays 0)."""
        real_id = wt._stable_cmp_id
        monkeypatch.setattr(
            wt, "_stable_cmp_id", lambda a, b, pc: 0 if a[:1] == b"A" else real_id(a, b, pc)
        )
        data = b"AAB" * 40
        pairs = [(b"AA", b"zz"), (b"AB", b"yy"), (b"BA", b"xx"), (b"A", b"q")]
        real = wt._TagClaims
        assert _build(monkeypatch, real, data, pairs) == _build(
            monkeypatch, _ScanClaims, data, pairs
        )


class TestClaimCost:
    def test_regression_claimed_run_is_one_hop(self):
        """Falsification: refusing claimed spans must not walk them."""
        size = 4096

        class Reads(list):
            count = 0

            def __getitem__(self, i):
                Reads.count += 1
                return super().__getitem__(i)

        claims = wt._TagClaims(size)
        assert claims.claim(0, size, 7, 1, wt.TagFlags.NONE)
        claims._next = Reads(claims._next)
        for off in (0, 100, size - 64):
            Reads.count = 0
            assert not claims.claim(off, 64, 9, 2, wt.TagFlags.NONE)
            assert Reads.count <= 8, off  # constant: the span is 64, the run 4096

    def test_regression_tags_are_distinct_objects(self):
        """_assign_parents mutates tags in place: no shared ByteTag."""
        claims = wt._TagClaims(16)
        claims.claim(0, 16, 3, 1, wt.TagFlags.NONE)
        assert len({id(t) for t in claims.tags}) == 16

    def test_regression_saturated_occurrences_are_skipped(self):
        """Falsification: a claimed zero run is not visited per occurrence."""
        size, width = 4000, 64
        calls = 0

        class Counting(wt._TagClaims):
            def claim(self, *args):
                nonlocal calls
                calls += 1
                return super().claim(*args)

        claims = Counting(size)
        claims.claim(0, size, 7, 1, wt.TagFlags.NONE)
        calls = 0
        offs = list(range(size - width + 1))  # every occurrence of 64 zero bytes
        assert claims.claim_each(offs, width, 9, 2, wt.TagFlags.NONE) == -1
        assert calls == 1

    def test_regression_skip_resumes_at_free_bytes(self):
        """Adversarial: a free gap after a claimed run is still claimed."""
        claims = wt._TagClaims(32)
        claims.claim(0, 10, 7, 1, wt.TagFlags.NONE)
        assert claims.claim_each(list(range(0, 29)), 4, 9, 2, wt.TagFlags.NONE) == 7
        assert all(t.cmp_id for t in claims.tags)
