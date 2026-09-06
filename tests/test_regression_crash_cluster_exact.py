"""``cluster_crashes`` pruning must be exact, not approximate.

A2 first shipped with MinHash LSH sparsification. That lost roughly 80% of the
pairs it should have merged: against the dense implementation, 83 of 360
configurations disagreed and the LSH version always *under*-merged, frequently
returning ``n`` clusters. A direct probe -- 20 signatures with 5 pairs above
threshold by ground truth -- had the dense path find all 5 and LSH find 1.

The cause was a metric mismatch rather than bad tuning: MinHash approximates
**Jaccard** over token sets, while the clustering threshold is on
**Levenshtein** similarity. A pair can sit at Levenshtein 0.8 with a Jaccard far
below the band threshold and never become a candidate, and no banding parameter
reconciles the two orderings.

The replacement prunes with two *exact* lower bounds on edit distance, so it can
only discard pairs that provably cannot clear the threshold. That makes the
correct test an equivalence test against the dense O(n^2) comparison, which is
what this file is. The dense reference lives here rather than being imported, so
that changing the production code cannot silently change the oracle too.
"""

import random

import pytest

from fuzzer_tool.core.crash_metadata import cluster_crashes
from fuzzer_tool.core.similarity import (
    crash_signature_similarity,
    frame_sequence_similarity,
)

FAMILIES = [
    [f"{prefix}_fn{i}" for i in range(6)]
    for prefix in ("avcodec", "avformat", "swr", "zlib", "sqlite")
]
FRAME_POOL = [
    "malloc",
    "free",
    "memcpy",
    "av_read_frame",
    "decode_slice",
    "parse_hdr",
    "main",
    "__libc_start",
    "inflate",
    "png_read",
    "sqlite3Step",
]
KINDS = ["SEGV", "ASAN:heap-buffer-overflow", "ASAN:heap-use-after-free", "FPE"]


def dense_clusters(signatures, frame_lists=None, threshold=0.7):
    """The original all-pairs implementation, kept verbatim as the oracle."""
    if not signatures:
        return []
    n = len(signatures)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if frame_lists and i < len(frame_lists) and j < len(frame_lists):
                sim = frame_sequence_similarity(frame_lists[i], frame_lists[j])
            else:
                sim = crash_signature_similarity(signatures[i], signatures[j])
            if sim >= threshold:
                union(i, j)

    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def partition(clusters):
    """Order-independent canonical form, so cluster ordering cannot matter."""
    return frozenset(frozenset(c) for c in clusters)


def _make_crash(rng):
    kind = rng.choice(KINDS)
    frames = [rng.choice(FRAME_POOL) for _ in range(rng.randrange(1, 10))]
    frames = [f + (f"+0x{rng.randrange(4096):x}" if rng.random() < 0.4 else "") for f in frames]
    return kind + ":" + ":".join(frames), frames


@pytest.mark.parametrize("threshold", [0.5, 0.7, 0.85])
@pytest.mark.parametrize("frames_mode", ["none", "all", "partial", "empty"])
def test_matches_dense_oracle(threshold, frames_mode):
    """Pruned clustering must equal all-pairs clustering, exactly."""
    mismatches = []
    for trial in range(25):
        rng = random.Random(trial * 7 + int(threshold * 100))
        n = rng.randrange(2, 26)
        pairs = [_make_crash(rng) for _ in range(n)]
        sigs = [p[0] for p in pairs]
        frames = [p[1] for p in pairs]
        fl = {
            "none": None,
            "all": frames,
            "partial": frames[: max(0, n // 2)],
            "empty": [],
        }[frames_mode]

        expected = partition(dense_clusters(sigs, fl, threshold))
        actual = partition(cluster_crashes(sigs, fl, threshold))
        if expected != actual:
            mismatches.append(trial)
    assert not mismatches, (
        f"pruning changed the clustering for seeds {mismatches} "
        f"(threshold={threshold}, frames={frames_mode})"
    )


def test_recall_on_known_near_duplicates():
    """The direct probe the LSH version failed: 20 signatures, 5 true pairs.

    Stated as recall against ground truth rather than as a cluster count, so a
    failure says how many merges were lost rather than just that a number moved.
    """
    rng = random.Random(0)
    short_pool = ["malloc", "free", "memcpy", "decode_slice", "parse_hdr", "main"]
    sigs = ["SEGV:" + ":".join(rng.choice(short_pool) for _ in range(6)) for _ in range(20)]

    truth = {
        (i, j)
        for i in range(20)
        for j in range(i + 1, 20)
        if crash_signature_similarity(sigs[i], sigs[j]) >= 0.7
    }
    assert truth, "fixture produced no near-duplicates; the test would be vacuous"

    clusters = cluster_crashes(sigs, None, 0.7)
    member_of = {}
    for cid, c in enumerate(clusters):
        for i in c:
            member_of[i] = cid

    missed = [(i, j) for (i, j) in truth if member_of[i] != member_of[j]]
    assert not missed, (
        f"{len(missed)} of {len(truth)} above-threshold pairs were not merged: {missed}"
    )


def test_families_are_recovered():
    """Well-separated families must come back as themselves."""
    rng = random.Random(11)
    sigs = []
    for _ in range(120):
        base = list(FAMILIES[rng.randrange(len(FAMILIES))])
        if rng.random() < 0.5:
            base[rng.randrange(6)] = f"x{rng.randrange(50)}"
        sigs.append("SEGV:" + ":".join(base))
    clusters = cluster_crashes(sigs, None, 0.7)
    assert len(clusters) == len(FAMILIES), f"expected {len(FAMILIES)} families, got {len(clusters)}"


def test_degenerate_inputs():
    assert cluster_crashes([], None, 0.7) == []
    assert partition(cluster_crashes(["a"], None, 0.7)) == partition([[0]])
    # Identical signatures must all land together at any threshold.
    same = ["SEGV:main:parse"] * 6
    assert partition(cluster_crashes(same, None, 0.7)) == partition([[0, 1, 2, 3, 4, 5]])


def test_threshold_zero_merges_everything():
    """A zero threshold cannot be pruned away: every pair clears it."""
    rng = random.Random(5)
    sigs = [_make_crash(rng)[0] for _ in range(12)]
    clusters = cluster_crashes(sigs, None, 0.0)
    assert len(clusters) == 1, f"threshold 0 should give one cluster, got {len(clusters)}"
