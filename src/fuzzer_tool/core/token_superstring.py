"""One short seed carrying every token of a dictionary.

Tokens are merged where one's suffix equals another's prefix, so the seed is
shorter than their concatenation:

    GIF + IFA + 89a  ->  GIFA89a      (GIF/IFA share "IF")

Pipeline: drop tokens contained in others, then
  <= EXACT_MAX_LEN tokens: exact order (Held-Karp);
  more: greedy best-overlap-first, improved by iterated local search
        (or-opt moves plus double-bridge kicks) unless the overlap graph
        has over ILS_MAX_PAIRS pairs.
Port of the greedy/ILS code from improve.py.  Not ported: its lower-bound
check (scs.py).  Added: the exact path -- improve.py skips kicks below 8
tokens and its descent alone stalls above OPT on 3 tokens.

The seed is a new input, not a change to the dictionary or to any operator.
Whether one execution carrying many magic tokens hits several comparison
gates at once is not measured.
"""

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool

DEFAULT_KICKS = 300
PATIENCE = 60  # kicks without gain before ILS stops
EXACT_MAX_LEN = 10  # Held-Karp is 2^n * n^2: ~0.1 s at 10
MIN_KICK_LEN = 4  # a double bridge needs 3 distinct cuts
SEG_LENS = (1, 2, 3)  # or-opt segment lengths
ILS_MAX_PAIRS = 100_000  # denser: greedy only (ruby.dict: 846k pairs, 9 s/sweep, +1 byte)
MAX_OVERLAP = 256  # caps the prefix table at O(n * MAX_OVERLAP)
CUTS = 3


def reduce_input(tokens: list[bytes]) -> list[bytes]:
    """Unique non-empty tokens, minus those that sit inside another.

    Ties in length break by value: set order varies with the hash seed.
    """
    longest_first = sorted({t for t in tokens if t}, key=lambda t: (-len(t), t))
    out: list[bytes] = []
    for t in longest_first:
        if not any(t in kept for kept in out):
            out.append(t)
    return out


def overlap_map(toks: list[bytes]) -> dict[tuple[int, int], int]:
    """(i, j) -> longest k with toks[i][-k:] == toks[j][:k], i != j."""
    prefixes: dict[bytes, list[int]] = {}
    for j, t in enumerate(toks):
        for k in range(1, min(len(t), MAX_OVERLAP + 1)):
            prefixes.setdefault(t[:k], []).append(j)

    ov: dict[tuple[int, int], int] = {}
    for i, a in enumerate(toks):
        for k in range(min(len(a) - 1, MAX_OVERLAP), 0, -1):
            for j in prefixes.get(a[-k:], ()):
                if j != i and (i, j) not in ov:
                    ov[(i, j)] = k
    return ov


def greedy_order(toks: list[bytes], ov: dict[tuple[int, int], int]) -> list[int]:
    """Join best overlaps first; never close a cycle. Returns an order."""
    n = len(toks)
    nxt: list[int | None] = [None] * n
    has_prev = [False] * n
    root = list(range(n))

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for (i, j), _k in sorted(ov.items(), key=lambda e: -e[1]):
        if nxt[i] is not None or has_prev[j] or find(i) == find(j):
            continue
        nxt[i] = j
        has_prev[j] = True
        root[find(i)] = find(j)

    order: list[int] = []
    for head in range(n):
        if has_prev[head]:
            continue
        x: int | None = head
        while x is not None:
            order.append(x)
            x = nxt[x]
    return order


def exact_order(toks: list[bytes], ov: dict[tuple[int, int], int]) -> list[int]:
    """Order with the largest total overlap (Held-Karp over subsets)."""
    n = len(toks)
    full = (1 << n) - 1
    best = [[-1] * n for _ in range(1 << n)]
    prev = [[-1] * n for _ in range(1 << n)]
    for i in range(n):
        best[1 << i][i] = 0

    for mask in range(1 << n):
        for i in range(n):
            if best[mask][i] < 0:
                continue
            for j in range(n):
                if mask >> j & 1:
                    continue
                v = best[mask][i] + ov.get((i, j), 0)
                nm = mask | 1 << j
                if v > best[nm][j]:
                    best[nm][j], prev[nm][j] = v, i

    last = max(range(n), key=lambda i: best[full][i])
    order, mask = [], full
    while last >= 0:
        order.append(last)
        mask, last = mask ^ (1 << last), prev[mask][last]
    return order[::-1]


def build(order: list[int], toks: list[bytes], ov: dict[tuple[int, int], int]) -> bytes:
    """Concatenate along the order, dropping each overlap."""
    if not order:
        return b""
    out = bytearray(toks[order[0]])
    for prev, cur in zip(order, order[1:], strict=False):
        out += toks[cur][ov.get((prev, cur), 0) :]
    return bytes(out)


def _score(order: list[int], ov: dict[tuple[int, int], int]) -> int:
    """Total overlap along the order (bigger = shorter seed)."""
    return sum(ov.get(p, 0) for p in zip(order, order[1:], strict=False))


def _link(a: int | None, b: int | None, ov: dict) -> int:
    if a is None or b is None:
        return 0
    return ov.get((a, b), 0)


class _Net:
    """Overlap map plus its in/out neighbour lists."""

    def __init__(self, n: int, ov: dict[tuple[int, int], int]) -> None:
        self.ov = ov
        self.ins: list[list[int]] = [[] for _ in range(n)]
        self.outs: list[list[int]] = [[] for _ in range(n)]
        for i, j in ov:
            self.outs[i].append(j)
            self.ins[j].append(i)


def _lift_gain(cur: list[int], i: int, seg: int, ov: dict) -> int:
    """Overlap lost by lifting cur[i:i+seg] out and closing the gap."""
    prev = cur[i - 1] if i > 0 else None
    after = cur[i + seg] if i + seg < len(cur) else None
    return _link(prev, cur[i], ov) + _link(cur[i + seg - 1], after, ov) - _link(prev, after, ov)


def _rest_pos(pos: dict[int, int], x: int, i: int, n: int) -> int:
    """Index of `x` in the order once the n-long segment at i is lifted out."""
    return pos[x] - n if pos[x] >= i else pos[x]


def _slots(seg: list[int], net: _Net, pos: dict[int, int], i: int) -> list[int]:
    """Slots in `rest` where `seg` links to a neighbour, ascending.

    Dropping seg between a and b gains link(a,seg0) + link(segN,b) - link(a,b),
    positive only if one of the first two terms is, so no other slot can win.
    """
    inside, n = set(seg), len(seg)
    slots = {_rest_pos(pos, a, i, n) + 1 for a in net.ins[seg[0]] if a not in inside}
    slots |= {_rest_pos(pos, b, i, n) for b in net.outs[seg[-1]] if b not in inside}
    return sorted(slots)


def _best_slot(
    rest: list[int], seg: list[int], skip: int, net: _Net, pos: dict[int, int]
) -> tuple[int, int]:
    """(gain, slot) of the best place in `rest` to drop `seg`; slot -1 = none."""
    best, slot = 0, -1
    for p in _slots(seg, net, pos, skip):
        if p == skip:
            continue
        a = rest[p - 1] if p > 0 else None
        b = rest[p] if p < len(rest) else None
        gain = _link(a, seg[0], net.ov) + _link(seg[-1], b, net.ov) - _link(a, b, net.ov)
        if gain > best:
            best, slot = gain, p
    return best, slot


def _sweep(cur: list[int], net: _Net) -> int:
    """One or-opt pass over `cur` (edited in place). Returns overlap gained."""
    total = 0
    pos = {v: k for k, v in enumerate(cur)}
    for seg_len in SEG_LENS:
        i = 0
        while i + seg_len <= len(cur):
            seg = cur[i : i + seg_len]
            rest = cur[:i] + cur[i + seg_len :]
            lost = _lift_gain(cur, i, seg_len, net.ov)
            add, slot = _best_slot(rest, seg, i, net, pos)
            i += 1
            if slot < 0 or add <= lost:
                continue
            cur[:] = rest[:slot] + seg + rest[slot:]
            pos = {v: k for k, v in enumerate(cur)}
            total += add - lost
    return total


def _descend(cur: list[int], net: _Net) -> int:
    """Sweep until no move helps. Returns overlap gained."""
    total = 0
    while True:
        gain = _sweep(cur, net)
        if gain == 0:
            return total
        total += gain


def _kick(order: list[int], rng: RandPool) -> list[int]:
    """Double bridge: swap two adjacent middle blocks."""
    a, b, c = sorted(rng.sample(list(range(1, len(order))), CUTS))
    return order[:a] + order[b:c] + order[a:b] + order[c:]


def ils_order(order: list[int], ov: dict, rng: RandPool, kicks: int) -> list[int]:
    """Iterated local search; never returns worse than `order`."""
    net = _Net(len(order), ov)
    best = list(order)
    best_score = _score(best, ov)
    cur, cur_score = list(best), best_score
    stale = 0
    for t in range(kicks + 1):
        cur_score += _descend(cur, net)
        if cur_score > best_score:
            best, best_score, stale = list(cur), cur_score, 0
        else:
            stale += 1
        if t == kicks or stale >= PATIENCE or len(best) < MIN_KICK_LEN:
            break
        cur = _kick(best, rng)
        cur_score = _score(cur, ov)
    return best


def token_seed(
    tokens: list[bytes], rng: RandPool | None = None, kicks: int = DEFAULT_KICKS
) -> bytes:
    """Shortest seed found that contains every token."""
    toks = reduce_input(tokens)
    if not toks:
        return b""
    ov = overlap_map(toks)
    if len(toks) <= EXACT_MAX_LEN:
        return build(exact_order(toks, ov), toks, ov)

    if not ov:  # nothing overlaps: any order is the concatenation
        return b"".join(toks)

    order = greedy_order(toks, ov)
    if len(ov) <= ILS_MAX_PAIRS:
        order = ils_order(order, ov, rng or get_default_rand_pool(), kicks)
    return build(order, toks, ov)
