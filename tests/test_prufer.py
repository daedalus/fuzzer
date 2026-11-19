"""Tests for Prüfer tree codec (core/prufer.py) and TLV reshape (tlv_reshape.py)."""

import itertools
from collections import Counter

import pytest

from fuzzer_tool.core.chi_squared import chi_squared_goodness_of_fit, chi_squared_test
from fuzzer_tool.core.mutations.tlv_reshape import tlv_reshape
from fuzzer_tool.core.prufer import ROOT, children, decode, edges, encode, random_tree, root_at
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.structural_constraints import build_tlv, parse_tlv
from tests.support.scripted_rng import ScriptedRng

ALPHA = 1e-3  # fixed seeds: deterministic, so this is a sanity bar, not a flake rate
CAYLEY_N = 5  # 5^3 = 125 labeled trees; 40 draws per tree keeps cells >= 5
DRAWS = 125 * 40


def _is_tree(parent):
    """Every node reaches the single root n-1 without revisiting a node."""
    n = len(parent)
    if parent[n - 1] != ROOT:
        return False
    for v in range(n):
        seen = 0
        while parent[v] != ROOT:
            v = parent[v]
            seen += 1
            if seen > n:
                return False
    return True


def _aldous_broder(n, rng):
    """Reference sampler: random walk on K_n -> uniform spanning tree.

    Independent of Prüfer codes, so it can serve as an oracle.
    """
    cur = rng.randint(0, n - 1)
    seen = {cur}
    tree = []
    while len(seen) < n:
        nxt = (cur + rng.randint(1, n - 1)) % n  # uniform over the other n-1
        if nxt not in seen:
            seen.add(nxt)
            tree.append((cur, nxt))
        cur = nxt
    return root_at(n, tree, n - 1)


def _counts(sampler, seed, keys):
    rng = RandPool(seed=seed)
    c = Counter(tuple(sampler(CAYLEY_N, rng)) for _ in range(DRAWS))
    return [c[k] for k in keys]


def _all_keys():
    return [tuple(decode(list(s))) for s in itertools.product(range(CAYLEY_N), repeat=CAYLEY_N - 2)]


# ── codec ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_roundtrip_bijection(n):
    """All n^(n-2) codes decode to distinct trees and encode back (Cayley)."""
    trees = set()
    for seq in itertools.product(range(n), repeat=n - 2):
        parent = decode(list(seq))
        assert _is_tree(parent)
        assert encode(parent) == list(seq)
        trees.add(tuple(parent))
    assert len(trees) == n ** (n - 2)


def test_decode_known_example():
    # Classic: code [3,3,3,4] on 6 nodes -> leaves 0,1,2 hang off 3, 3-4, 4-5.
    assert decode([3, 3, 3, 4]) == [3, 3, 3, 4, 5, ROOT]


def test_children_and_reroot():
    parent = decode([3, 3, 3, 4])
    assert children(parent) == [[], [], [], [0, 1, 2], [3], [4]]
    rerooted = root_at(6, edges(parent), 0)
    assert rerooted[0] == ROOT
    assert sorted(edges(rerooted)) != [] and len(edges(rerooted)) == 5
    assert {frozenset(e) for e in edges(rerooted)} == {frozenset(e) for e in edges(parent)}


# ── uniformity (Cayley) with Hard Rule 46 control ──────────────────────


def test_random_tree_uniform_with_control():
    keys = _all_keys()
    ref1 = _counts(_aldous_broder, 11, keys)
    ref2 = _counts(_aldous_broder, 12, keys)
    ours = _counts(random_tree, 13, keys)

    # Control first: reference against a second run of itself.
    _, p_ctrl, _ = chi_squared_test([ref1, ref2])
    assert p_ctrl > ALPHA

    _, p_ab, _ = chi_squared_test([ours, ref1])
    assert p_ab > ALPHA

    _, p_gof, _ = chi_squared_goodness_of_fit(ours)
    assert p_gof > ALPHA
    assert sum(1 for c in ours if c) == CAYLEY_N ** (CAYLEY_N - 2)


def test_falsification_biased_codes_detected():
    """Codes that never use label n-1 must fail the same oracle."""
    keys = _all_keys()

    def biased(n, r):
        return decode(r.randint_list(0, n - 2, n - 2))

    _, p_gof, _ = chi_squared_goodness_of_fit(_counts(biased, 14, keys))
    assert p_gof < ALPHA


# ── adversarial ────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [[-1], [5, 0, 0], [0, 99]])
def test_decode_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        decode(bad)


def test_encode_rejects_cycle_and_bad_root():
    with pytest.raises(ValueError):
        encode([1, 0, ROOT])  # 0<->1 cycle, 2 detached
    with pytest.raises(ValueError):
        encode([ROOT, 0, 0])  # root not n-1


def test_tiny_and_degenerate_trees():
    assert decode([]) == [1, ROOT]
    assert random_tree(1, RandPool(seed=1)) == [ROOT]
    assert random_tree(2, RandPool(seed=1)) == [1, ROOT]
    with pytest.raises(ValueError):
        random_tree(0, RandPool(seed=1))
    n = 50
    star = decode([7] * (n - 2))
    assert children(star)[7] == [v for v in range(n - 1) if v != 7]


def test_long_path_no_recursion():
    n = 20000
    path = decode(list(range(1, n - 1)))  # 0-1-2-...-(n-1)
    assert path == list(range(1, n)) + [ROOT]
    assert encode(path) == list(range(1, n - 1))


# ── TLV reshape ────────────────────────────────────────────────────────


def _frames(values, tag0=0x10):
    return [build_tlv(tag0 + i, v) for i, v in enumerate(values)]


def _covers_exactly(nodes, start, end):
    """Each frame's children tile its value exactly; siblings tile [start,end)."""
    pos = start
    for node in nodes:
        if node.tag_offset != pos:
            return False
        if node.children and not _covers_exactly(node.children, node.value_start, node.value_end):
            return False
        pos = node.value_end
    return pos == end


def _leaf_values(out, nodes):
    return [
        out[n.value_start : n.value_end] for root in nodes for n in root.walk() if not n.children
    ]


def test_reshape_exact_scripted():
    """Code [0] on 3 nodes = parent [2, 0, ROOT]: 1 under 0, 0 under 2.

    Internal nodes keep their tag and drop their value (constructed frame).
    """
    a, b, c = _frames([b"AA", b"B", b"CCC"])
    data = a + b + c
    rng = ScriptedRng(randints=[0, 0], counts=[0])
    out = tlv_reshape(data, rng=rng)

    inner = build_tlv(0x10, b)  # node 0 now contains node 1
    expected = build_tlv(0x12, inner)  # root (node 2) contains node 0
    assert out == expected


@pytest.mark.parametrize("seed", range(40))
def test_reshape_lengths_valid(seed):
    # 7 nodes: a path stays within parse_tlv's MAX_TLV_DEPTH (8).
    values = [bytes([i]) * (i % 5) for i in range(7)]
    data = b"".join(_frames(values))
    out = tlv_reshape(data, rng=RandPool(seed=seed))
    assert out is not None and out != data

    roots = parse_tlv(out, tag_width=1, length_width=2)
    assert len(roots) == 1
    assert _covers_exactly(roots, 0, len(out))
    assert sum(1 for _ in roots[0].walk()) == len(values)
    assert Counter(_leaf_values(out, roots)) <= Counter(values)  # leaves keep bytes


def test_reshape_inside_container_repairs_parent():
    """Sibling lists are ordered roots first, then containers in walk order."""
    kids = b"".join(_frames([b"x", b"yy", b"zzz"]))
    tail = build_tlv(0x02, b"t")
    data = build_tlv(0x01, kids) + tail
    rng = ScriptedRng(randints=[1, 0], counts=[0])  # list 1 = 0x01's children
    out = tlv_reshape(data, rng=rng)

    nested = build_tlv(0x12, build_tlv(0x10, build_tlv(0x11, b"yy")))
    assert out == build_tlv(0x01, nested) + tail


def test_falsification_validator_catches_bad_length():
    data = b"".join(_frames([b"a", b"bb", b"ccc"]))
    out = bytearray(tlv_reshape(data, rng=RandPool(seed=5)))
    out[2] += 1  # root length now overruns the buffer
    roots = parse_tlv(bytes(out), tag_width=1, length_width=2)
    assert not _covers_exactly(roots, 0, len(out))


@pytest.mark.parametrize("data", [b"", b"\x00", b"\x10\x00\x05ab", build_tlv(0x10, b"solo")])
def test_reshape_declines_without_siblings(data):
    assert tlv_reshape(data, rng=RandPool(seed=1)) is None


def test_reshape_declines_on_length_overflow():
    """Star at node 2: root holds two 40003-byte frames, 80006 > 0xFFFF."""
    big = b"\xaa" * 40000
    data = b"".join(_frames([big, big, b"r"]))
    rng = ScriptedRng(randints=[0, 0], counts=[2])
    assert tlv_reshape(data, rng=rng) is None


def test_reshape_caps_tree_size():
    """Long sibling runs are windowed to MAX_TREE_NODES frames."""
    from fuzzer_tool.core.mutations.tlv_reshape import MAX_TREE_NODES

    n = MAX_TREE_NODES * 3
    data = b"".join(_frames([b"v"] * n, tag0=0))
    out = tlv_reshape(data, rng=RandPool(seed=2))
    roots = parse_tlv(out, tag_width=1, length_width=2)
    assert len(roots) == n - MAX_TREE_NODES + 1
    assert _covers_exactly(roots, 0, len(out))
