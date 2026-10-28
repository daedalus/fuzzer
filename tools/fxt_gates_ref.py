"""Python reference for targets/fxt_gates.c (gates after J. Arndt, "Matters Computational").

Independent of the C code: the test compares both on random and structured fields.
Each gate maps its input field to ``(valid, operand)``. A gate is solved iff it is valid
and ``operand == TARGETS[g]``. ``WITNESS`` holds one solving field per gate.
"""

from __future__ import annotations

import math
from collections.abc import Callable

SENTINEL = (1 << 64) - 1
M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1

# (name, book section, field bytes); gate g reads the next FIELD_LEN[g] input bytes.
GATES: tuple[tuple[str, str, int], ...] = (
    ("gray", "1.16", 4),
    ("popcount", "1.8", 8),
    ("negabinary", "1.22", 4),
    ("inv_mod_2n", "1.21", 4),
    ("perm_order", "2.3", 4),
    ("inversions", "3.1", 8),
    ("heap", "4.5", 8),
    ("comb_colex", "6.2", 4),
    ("composition", "7.3", 2),
    ("debruijn", "8.3", 4),
    ("mixed_radix", "9.1", 4),
    ("perm_lex", "10.2", 4),
    ("involution", "11.3", 4),
    ("kperm_lex", "12.1", 2),
    ("multiset_perm", "13.2", 2),
    ("fibonacci", "14.2", 4),
    ("dyck_lex", "15.4", 2),
    ("partition", "16.2", 4),
    ("set_part_rgs", "17.3", 4),
    ("necklace", "18.1", 4),
    ("hadamard_lfsr", "19.1", 2),
    ("rho_path", "20.2", 4),
    ("dft4", "21.2", 8),
    ("convolution", "22.1", 8),
    ("walsh", "23.1", 8),
    ("reed_muller", "23.11", 2),
    ("haar", "24.3", 8),
    ("hartley4", "25.2", 4),
    ("ntt4", "26.2", 4),
    ("wavelet_s", "27.2", 8),
    ("cast_out", "28.4", 8),
    ("isqrt", "29.1", 8),
    ("newton", "30.1", 4),
    ("agm", "31.1", 8),
    ("log2_q16", "32.1", 4),
    ("cordic", "33.2", 2),
    ("binary_split", "34.1", 1),
    ("chebyshev", "35.2", 3),
    ("vandermonde", "36.1", 2),
    ("euclid_steps", "37.3", 4),
    ("rudin_shapiro", "38.3", 2),
    ("modexp_sqrt", "39.9", 4),
    ("primitive_poly", "40.5", 2),
    ("crc16", "41.3", 6),
    ("gf256_trace", "42.3", 2),
    ("perm_sqrt", "2.3x", 4),  # extension of chapter 2, not a book section
)
GATE_COUNT = len(GATES)
FIELD_LEN = tuple(g[2] for g in GATES)
OFFSET = tuple(sum(FIELD_LEN[:g]) for g in range(GATE_COUNT))
INPUT_LEN = sum(FIELD_LEN)

Result = tuple[bool, int]


def _le(b: bytes) -> int:
    return int.from_bytes(b, "little")


def _s8(v: int) -> int:
    return v - 256 if v >= 128 else v


def _perm(x: int, n: int) -> list[int] | None:
    p = [(x >> (3 * i)) & 7 for i in range(n)]
    return p if sorted(p) == list(range(n)) else None


def _gray(b: bytes) -> Result:
    x = _le(b)
    return True, x ^ (x >> 1)


def _popcount(b: bytes) -> Result:
    return True, bin(_le(b)).count("1")


def _negabinary(b: bytes) -> Result:
    x = _le(b)
    return True, ((x + 0xAAAAAAAA) ^ 0xAAAAAAAA) & M32


def _inv_mod_2n(b: bytes) -> Result:
    x = _le(b)
    return (x & 1) == 1, pow(x, -1, 1 << 32) if x & 1 else 0


def _perm_order(b: bytes) -> Result:
    p = _perm(_le(b), 8)
    if p is None:
        return False, 0
    seen = [False] * 8
    order = 1
    for s in range(8):
        n, i = 0, s
        while not seen[i]:
            seen[i] = True
            i = p[i]
            n += 1
        order = math.lcm(order, max(n, 1))
    return True, order


def _inversions(b: bytes) -> Result:
    n = len(b)
    return True, sum(b[i] > b[j] for i in range(n) for j in range(i + 1, n))


def _heap(b: bytes) -> Result:
    ok = all(b[i] >= b[c] for i in range(8) for c in (2 * i + 1, 2 * i + 2) if c < 8)
    return ok, sum(b) & 0xFF


def _comb_colex(b: bytes) -> Result:
    x = _le(b)
    pos = [i for i in range(32) if x >> i & 1]
    if len(pos) != 6:
        return False, 0
    return True, sum(math.comb(p, i + 1) for i, p in enumerate(pos))


def _composition(b: bytes) -> Result:
    x = _le(b) & 0x7FFF
    marks = [i for i in range(15) if x >> i & 1]
    cuts = [-1, *marks, 15]
    parts = [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)]
    return True, math.prod(parts)


_DEBRUIJN = "0000100110101111"
_DB_POS = {int("".join(_DEBRUIJN[(t + k) % 16] for k in range(4)), 2): t for t in range(16)}


def _debruijn(b: bytes) -> Result:
    x = _le(b)
    return True, sum(_DB_POS[(x >> (4 * i)) & 15] << (4 * i) for i in range(8))


_RADIX = (2, 3, 4, 5, 6, 7)


def _mixed_radix(b: bytes) -> Result:
    x = _le(b)
    d = [(x >> (3 * i)) & 7 for i in range(6)]
    if any(d[i] >= _RADIX[i] for i in range(6)):
        return False, 0
    val, w = 0, 1
    for i in range(6):
        val += d[i] * w
        w *= _RADIX[i]
    return True, val


def _perm_lex(b: bytes) -> Result:
    p = _perm(_le(b), 8)
    if p is None:
        return False, 0
    rank = 0
    for i in range(8):
        smaller = sum(p[j] < p[i] for j in range(i + 1, 8))
        rank += smaller * math.factorial(7 - i)
    return True, rank


def _involution(b: bytes) -> Result:
    p = _perm(_le(b), 8)
    if p is None:
        return False, 0
    return True, int(all(p[p[i]] == i and p[i] != i for i in range(8)))


def _kperm_lex(b: bytes) -> Result:
    x = _le(b)
    a = [(x >> (3 * i)) & 7 for i in range(4)]
    if len(set(a)) != 4:
        return False, 0
    rank, used = 0, set()
    for i in range(4):
        c = sum(1 for v in range(a[i]) if v not in used)
        rank += c * math.perm(7 - i, 3 - i)
        used.add(a[i])
    return True, rank


def _multinomial(cnt: list[int]) -> int:
    r = math.factorial(sum(cnt))
    for c in cnt:
        r //= math.factorial(c)
    return r


def _multiset_perm(b: bytes) -> Result:
    x = _le(b)
    s = [(x >> (2 * i)) & 3 for i in range(8)]
    cnt = [s.count(c) for c in range(4)]
    if cnt != [2, 2, 2, 2]:
        return False, 0
    rank = 0
    for i in range(8):
        for c in range(s[i]):
            if cnt[c] == 0:
                continue
            cnt[c] -= 1
            rank += _multinomial(cnt)
            cnt[c] += 1
        cnt[s[i]] -= 1
    return True, rank


def _fibonacci(b: bytes) -> Result:
    x = _le(b)
    if x & (x >> 1):
        return False, 0
    w = [1, 2]
    for _ in range(30):
        w.append(w[-1] + w[-2])
    return True, sum(w[i] for i in range(32) if x >> i & 1)


def _ballot_table(n: int) -> list[list[int]]:
    """``t[rem][depth]``: ways to finish ``rem`` parentheses from ``depth`` to 0, never below 0."""
    t = [[0] * (n + 2) for _ in range(n + 1)]
    t[0][0] = 1
    for rem in range(1, n + 1):
        for d in range(rem + 1):
            t[rem][d] = t[rem - 1][d + 1] + (t[rem - 1][d - 1] if d else 0)
    return t


_BALLOT = _ballot_table(16)


def _ballot(rem: int, depth: int) -> int:
    return _BALLOT[rem][depth] if depth <= rem else 0


def _dyck_lex(b: bytes) -> Result:
    x = _le(b)
    ch = [(x >> i) & 1 for i in range(16)]  # 1 = '(' < ')' = 0
    depth, rank = 0, 0
    for i in range(16):
        if ch[i] == 0:
            rank += _ballot(15 - i, depth + 1)
        depth += 1 if ch[i] else -1
        if depth < 0:
            return False, 0
    return depth == 0, rank


def _partition(b: bytes) -> Result:
    x = _le(b)
    p = [(x >> (4 * i)) & 15 for i in range(8)]
    ok = all(p[i] >= p[i + 1] for i in range(7)) and sum(p) == 20
    return ok, sum(v * v for v in p)


def _bell_tail(rem: int, k: int) -> int:
    if rem == 0:
        return 1
    return k * _bell_tail(rem - 1, k) + _bell_tail(rem - 1, k + 1)


def _set_part_rgs(b: bytes) -> Result:
    x = _le(b)
    a = [(x >> (3 * i)) & 7 for i in range(8)]
    k, rank = 0, 0
    for i in range(8):
        if a[i] > k:
            return False, 0
        for v in range(a[i]):
            rank += _bell_tail(7 - i, k + 1 if v == k else k)
        if a[i] == k:
            k += 1
    return True, rank


def _necklace(b: bytes) -> Result:
    x = _le(b) & 0xFFFFFF
    return True, min(((x << r) | (x >> (24 - r))) & 0xFFFFFF for r in range(24))


def _hadamard16() -> list[list[int]]:
    s = [1, 0, 0, 0]
    while len(s) < 15:
        s.append(s[-3] ^ s[-4])
    h = [[1] * 16]
    for i in range(15):
        h.append([1] + [-1 if s[(i + j) % 15] else 1 for j in range(15)])
    return h


HADAMARD = _hadamard16()


def _hadamard_lfsr(b: bytes) -> Result:
    x = _le(b)
    v = [-1 if x >> j & 1 else 1 for j in range(16)]
    return True, max(abs(sum(r[j] * v[j] for j in range(16))) for r in HADAMARD)


def _rho_path(b: bytes) -> Result:
    x = _le(b)
    succ = [(x >> (3 * i)) & 7 for i in range(6)]
    if any(s >= 6 for s in succ):
        return False, 0
    first, v, t = {}, 0, 0
    while v not in first:
        first[v] = t
        v = succ[v]
        t += 1
    return True, (first[v] << 8) | (t - first[v])


def _dft4(b: bytes) -> Result:
    z = [complex(_s8(b[2 * j]), _s8(b[2 * j + 1])) for j in range(4)]
    X = [sum(z[j] * (-1j) ** (j * k) for j in range(4)) for k in range(4)]
    return True, ((int(X[1].real) & 0xFFFF) << 16) | (int(X[2].imag) & 0xFFFF)


def _convolution(b: bytes) -> Result:
    a, c = b[:4], b[4:]
    return True, sum(a[i] * c[(1 - i) % 4] for i in range(4))


def _walsh(b: bytes) -> Result:
    x = [_s8(v) for v in b]
    y = [sum((-1) ** bin(j & k).count("1") * x[j] for j in range(8)) for k in range(8)]
    return True, ((y[3] & 0xFFFF) << 16) | (y[6] & 0xFFFF)


def _reed_muller(b: bytes) -> Result:
    f = [(_le(b) >> i) & 1 for i in range(16)]
    anf = [0] * 16
    for m in range(16):
        s = 0
        sub = m
        while True:
            s ^= f[sub]
            if sub == 0:
                break
            sub = (sub - 1) & m
        anf[m] = s
    return True, sum(v << i for i, v in enumerate(anf))


def _haar(b: bytes) -> Result:
    x = list(b)
    h = [0] * 8
    n = 8
    while n > 1:
        s = [x[2 * i] + x[2 * i + 1] for i in range(n // 2)]
        d = [x[2 * i] - x[2 * i + 1] for i in range(n // 2)]
        h[n // 2 : n] = d
        x = s
        n //= 2
    h[0] = x[0]
    return True, (h[0] << 16) | (h[7] & 0xFFFF)


def _hartley4(b: bytes) -> Result:
    x = [_s8(v) for v in b]
    cas = (1, 1, -1, -1)
    h = [sum(x[j] * cas[(j * k) % 4] for j in range(4)) for k in range(4)]
    return True, (h[1] * h[2]) & M32


def _ntt4(b: bytes) -> Result:
    X = [sum(b[j] * pow(16, j * k, 257) for j in range(4)) % 257 for k in range(4)]
    return True, X[0] | (X[1] << 9) | (X[2] << 18) | (X[3] << 27)


def _wavelet_s(b: bytes) -> Result:
    x = list(b)
    d: list[list[int]] = []
    while len(x) > 1:
        d.append([x[2 * i] - x[2 * i + 1] for i in range(len(x) // 2)])
        x = [(x[2 * i] + x[2 * i + 1]) >> 1 for i in range(len(x) // 2)]
    d3, d2, d1 = d[2][0], d[1][0], d[0][3]
    return True, x[0] | ((d3 & 0xFF) << 8) | ((d2 & 0xFF) << 16) | ((d1 & 0xFF) << 24)


def _cast_out(b: bytes) -> Result:
    p = _le(b[:4]) * _le(b[4:])
    return True, ((p % 255) << 9) | (p % 257)


def _isqrt(b: bytes) -> Result:
    return True, math.isqrt(_le(b))


def _newton(b: bytes) -> Result:
    n = _le(b)
    if n == 0:
        return False, 0
    x = n
    for _ in range(3):
        x = (x + n // x) >> 1
    return True, x


def _agm(b: bytes) -> Result:
    a, c = _le(b[:4]), _le(b[4:])
    if a == 0 or c == 0:
        return False, 0
    for _ in range(64):
        if abs(a - c) <= 1:
            break
        a, c = (a + c) >> 1, math.isqrt(a * c)
    return True, (a + c) >> 1


def _log2_q16(b: bytes) -> Result:
    x = _le(b)
    if x == 0:
        return False, 0
    n = x.bit_length() - 1
    m = (x << (31 - n)) & M32
    frac = 0
    for _ in range(16):
        m = (m * m) >> 31
        bit = 0
        if m >= 1 << 32:
            m >>= 1
            bit = 1
        frac = (frac << 1) | bit
    return True, (n << 16) | frac


CORDIC_ATAN = tuple(round(math.atan(2.0**-i) * 65536) for i in range(16))
CORDIC_K = 39797


def _cordic(b: bytes) -> Result:
    z, x, y = _le(b), CORDIC_K, 0
    for i in range(16):
        d = 1 if z >= 0 else -1
        x, y = x - d * (y >> i), y + d * (x >> i)
        z -= d * CORDIC_ATAN[i]
    return True, ((x & 0xFFFF) << 16) | (y & 0xFFFF)


def _binary_split(b: bytes) -> Result:
    n = b[0] & 15
    return True, sum(math.factorial(n) // math.factorial(k) for k in range(n + 1))


def _chebyshev(b: bytes) -> Result:
    n, x = b[0], _le(b[1:3])
    t0, t1 = 1, x
    if n == 0:
        return True, 1
    for _ in range(n - 1):
        t0, t1 = t1, (2 * x * t1 - t0) & M32
    return True, t1 & M32


def _vandermonde(b: bytes) -> Result:
    n, m = b[0], b[1]
    if n > 33 or m > 33:
        return False, 0
    return True, math.comb(n + m, n)


def _euclid_steps(b: bytes) -> Result:
    p, q = _le(b[:2]), _le(b[2:])
    if q == 0:
        return False, 0
    steps = 0
    while q:
        p, q = q, p % q
        steps += 1
    return True, steps


def _rs_prefix() -> list[int]:
    """``p[n]``: sum of the Golay-Rudin-Shapiro sequence over k < n, for every 16-bit n."""
    p = [0] * (1 << 16)
    for k in range((1 << 16) - 1):
        p[k + 1] = p[k] + 1 - 2 * ((k & (k >> 1)).bit_count() & 1)
    return p


_RS_PREFIX = _rs_prefix()


def _rudin_shapiro(b: bytes) -> Result:
    return True, _RS_PREFIX[_le(b)] & M32


def _modexp_sqrt(b: bytes) -> Result:
    p = (1 << 31) - 1
    return True, pow(_le(b) % p, 1 << 29, p)


def _pmulmod(a: int, b: int, poly: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> 15 & 1:
            a ^= poly
    return r


def _ppow(e: int, poly: int) -> int:
    r, base = 1, 2
    while e:
        if e & 1:
            r = _pmulmod(r, base, poly)
        base = _pmulmod(base, base, poly)
        e >>= 1
    return r


def _primitive_poly(b: bytes) -> Result:
    poly = _le(b)
    if not (poly >> 15 & 1 and poly & 1):
        return False, 0
    if _ppow(32767, poly) != 1:
        return True, 0
    return True, int(all(_ppow(32767 // q, poly) != 1 for q in (7, 31, 151)))


def _crc16(b: bytes) -> Result:
    crc = 0xFFFF
    for v in b:
        crc ^= v << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return True, crc


def _gmul(a: int, b: int) -> int:
    r = 0
    for _ in range(8):
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= 0x11B
    return r


def _gf256_trace(b: bytes) -> Result:
    a, c = b[0], b[1]
    if a == 0:
        return False, 0
    tr, t = 0, a
    for _ in range(8):
        tr ^= t
        t = _gmul(t, t)
    return True, int(_gmul(a, c) == 1 and tr == 1)


def _perm_sqrt(b: bytes) -> Result:
    """Number of square roots of the permutation (cycle-type closed form)."""
    p = _perm(_le(b), 8)
    if p is None:
        return False, 0
    m: dict[int, int] = {}
    seen = [False] * 8
    for s in range(8):
        n, i = 0, s
        while not seen[i]:
            seen[i] = True
            i = p[i]
            n += 1
        if n:
            m[n] = m.get(n, 0) + 1
    total = 1
    for k, c in m.items():
        if k % 2 == 0:
            total *= 0 if c % 2 else math.prod(range(c - 1, 0, -2)) * k ** (c // 2)
        else:
            total *= sum(
                math.factorial(c) // (math.factorial(c - 2 * j) * math.factorial(j) * 2**j) * k**j
                for j in range(c // 2 + 1)
            )
    return True, total


_FNS: tuple[Callable[[bytes], Result], ...] = (
    _gray, _popcount, _negabinary, _inv_mod_2n, _perm_order, _inversions, _heap,
    _comb_colex, _composition, _debruijn, _mixed_radix, _perm_lex, _involution,
    _kperm_lex, _multiset_perm, _fibonacci, _dyck_lex, _partition, _set_part_rgs,
    _necklace, _hadamard_lfsr, _rho_path, _dft4, _convolution, _walsh, _reed_muller,
    _haar, _hartley4, _ntt4, _wavelet_s, _cast_out, _isqrt, _newton, _agm, _log2_q16,
    _cordic, _binary_split, _chebyshev, _vandermonde, _euclid_steps, _rudin_shapiro,
    _modexp_sqrt, _primitive_poly, _crc16, _gf256_trace, _perm_sqrt,
)  # fmt: skip


def gate(g: int, field: bytes) -> Result:
    """``(valid, operand)`` of gate ``g`` on its ``FIELD_LEN[g]``-byte field."""
    return _FNS[g](field[: FIELD_LEN[g]])


def fields(buf: bytes) -> list[bytes]:
    return [buf[OFFSET[g] : OFFSET[g] + FIELD_LEN[g]] for g in range(GATE_COUNT)]
