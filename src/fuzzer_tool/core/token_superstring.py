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

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool

DEFAULT_KICKS = 300
PATIENCE = 60  # kicks without gain before ILS stops
EXACT_MAX_LEN = 10  # Held-Karp is 2^n * n^2: ~0.1 s at 10
MIN_KICK_LEN = 4  # a double bridge needs 3 distinct cuts
SEG_LENS = (1, 2, 3)  # or-opt segment lengths
ILS_MAX_PAIRS = 100_000  # denser: greedy only (ruby.dict: 846k pairs, 9 s/sweep, +1 byte)
MAX_OVERLAP = 256  # caps the prefix table at O(n * MAX_OVERLAP)
CUTS = 3
_NEG = -(1 << 30)  # unreachable (mask, last) state


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
    """Order with the largest total overlap (Held-Karp over subsets).

    Vectorised per (popcount layer, last token): every mask of a layer that
    holds j is extended in one numpy step.  Ties resolve to the lowest
    predecessor, as the scalar loop did (argmax = first maximum).
    """
    n = len(toks)
    full = (1 << n) - 1
    w = np.zeros((n, n), dtype=np.int32)
    for (i, j), k in ov.items():
        w[i, j] = k
    best = np.full((1 << n, n), _NEG, dtype=np.int32)
    prev = np.full((1 << n, n), -1, dtype=np.int8)
    best[1 << np.arange(n), np.arange(n)] = 0

    masks = np.arange(1 << n, dtype=np.int64)
    pop = np.zeros(1 << n, dtype=np.int8)
    for j in range(n):
        pop += (masks >> j & 1).astype(np.int8)
    layers = [masks[pop == k] for k in range(n + 1)]

    for k in range(2, n + 1):
        for j in range(n):
            bit = 1 << j
            cur = layers[k][layers[k] & bit != 0]
            src = cur ^ bit
            cand = best[src] + w[:, j]  # (m, n): last token i, then j
            arg = cand.argmax(axis=1)
            val = cand[np.arange(len(cur)), arg]
            ok = best[src, arg] >= 0
            best[cur[ok], j] = val[ok]
            prev[cur[ok], j] = arg[ok]

    last = int(best[full].argmax())
    order, mask = [], full
    while last >= 0:
        order.append(last)
        mask, last = mask ^ (1 << last), int(prev[mask, last])
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


class _Net:
    """Overlap map as a dense matrix plus in/out neighbours in CSR form.

    Row/column `n` is a sentinel for "no neighbour" (overlap 0), so the ends
    of an order need no special case.
    """

    def __init__(self, n: int, ov: dict[tuple[int, int], int]) -> None:
        self.ov = ov
        self.n = n
        self.w = np.zeros((n + 1, n + 1), dtype=np.int32)
        for (i, j), k in ov.items():
            self.w[i, j] = k
        self.in_ptr, self.in_flat = self._csr(n, [(j, i) for i, j in ov])
        self.out_ptr, self.out_flat = self._csr(n, list(ov))

    @staticmethod
    def _csr(n: int, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        """(node -> neighbours) as (indptr, flat)."""
        ptr = np.zeros(n + 1, dtype=np.int64)
        if not pairs:
            return ptr, np.zeros(0, dtype=np.int64)
        arr = np.array(sorted(pairs), dtype=np.int64)
        np.cumsum(np.bincount(arr[:, 0], minlength=n), out=ptr[1:])
        return ptr, arr[:, 1].copy()


def _score(order: list[int], ov: dict[tuple[int, int], int]) -> int:
    """Total overlap along the order (bigger = shorter seed)."""
    return sum(ov.get(p, 0) for p in zip(order, order[1:], strict=False))


def _gather(ptr: np.ndarray, flat: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, neighbour) for every neighbour of every node in `nodes`."""
    counts = ptr[nodes + 1] - ptr[nodes]
    rows = np.repeat(np.arange(len(nodes)), counts)
    starts = np.repeat(ptr[nodes] - (np.cumsum(counts) - counts), counts)
    return rows, flat[starts + np.arange(len(rows))]


def _first_move(arr: np.ndarray, pos: np.ndarray, seg_len: int, start: int, net: _Net):
    """First or-opt move at a position >= start that gains overlap.

    Returns (position, slot, gain) or None.  Evaluates every position against
    the unchanged order in one numpy pass; the scalar sweep did the same one
    position at a time, and positions before the first accepted move cannot
    change, so the first accepted move is identical.
    """
    m, sent, w, L = len(arr), net.n, net.w, seg_len
    idx = np.arange(start, m - L + 1)
    if not len(idx):
        return None
    ap = np.concatenate(([sent], arr, [sent]))  # ap[k + 1] == arr[k]
    first, last = arr[idx], arr[idx + L - 1]
    prev, after = ap[idx], ap[idx + L + 1]
    lost = w[prev, first] + w[last, after] - w[prev, after]

    # Candidate slots: right after an in-neighbour of first, or right before
    # an out-neighbour of last (neighbours inside the segment are skipped);
    # once the segment is lifted out, an item at q >= i + L sits at q - L.
    rows_a, nb_a = _gather(net.in_ptr, net.in_flat, first)
    rows_b, nb_b = _gather(net.out_ptr, net.out_flat, last)
    qa, qb = pos[nb_a], pos[nb_b]
    ia, ib = idx[rows_a], idx[rows_b]
    ka, kb = (qa < ia) | (qa >= ia + L), (qb < ib) | (qb >= ib + L)
    slot_a = np.where(qa >= ia, qa - L, qa)[ka] + 1
    slot_b = np.where(qb >= ib, qb - L, qb)[kb]
    rows = np.concatenate((rows_a[ka], rows_b[kb]))
    slot = np.concatenate((slot_a, slot_b))
    keep = slot != idx[rows]
    rows, slot = rows[keep], slot[keep]

    i = idx[rows]
    ra, rb = slot - 1, slot  # `rest` index of the left and right neighbour
    left = np.where(ra < i, ap[ra + 1], ap[ra + L + 1])
    right = np.where(rb < i, ap[rb + 1], ap[rb + L + 1])
    gain = w[left, first[rows]] + w[last[rows], right] - w[left, right]
    pos_gain = gain > 0
    # Best slot per position: biggest gain, ties to the lowest slot.
    big = m + 3
    key = gain[pos_gain].astype(np.int64) * big + (big - 1 - slot[pos_gain])
    best = np.full(len(idx), -1, dtype=np.int64)
    np.maximum.at(best, rows[pos_gain], key)
    add = best // big
    ok = (best >= 0) & (add > lost)
    if not ok.any():
        return None
    j = int(ok.argmax())
    return int(idx[j]), int(big - 1 - best[j] % big), int(add[j] - lost[j])


def _sweep(cur: list[int], net: _Net) -> int:
    """One or-opt pass over `cur` (edited in place). Returns overlap gained."""
    total = 0
    arr = np.array(cur, dtype=np.int64)
    pos = np.empty(net.n, dtype=np.int64)
    for seg_len in SEG_LENS:
        start = 0
        while True:
            pos[arr] = np.arange(len(arr))
            move = _first_move(arr, pos, seg_len, start, net)
            if move is None:
                break
            i, slot, gain = move
            rest = np.concatenate((arr[:i], arr[i + seg_len :]))
            arr = np.concatenate((rest[:slot], arr[i : i + seg_len], rest[slot:]))
            total += gain
            start = i + 1
    cur[:] = arr.tolist()
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
