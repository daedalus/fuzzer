"""Byte-level HMM field segmentation, decoded with Viterbi.

Hidden states are anonymous field types. Viterbi gives each byte its most
likely state; a run of one state is a span, the unit the ``hmm_segment``
operator mutates::

    seed   89 50 4E 47 | 00 00 00 0D | 49 48 44 52 | ...
    path   s0 s0 s0 s0 | s1 s1 s1 s1 | s0 s0 s0 s0 | s2 ...
    spans  [(0, 4, 0), (4, 8, 1), (8, 12, 0), (12, .., 2)]

Training is hard EM (Viterbi training): decode, recount, renormalise with
pseudo-counts. That is coordinate ascent on the MAP objective
``sum((counts + alpha) * log(params))``, so the objective never decreases.
No RNG: the first path quantises byte values into K bands.

States carry a ``FieldKind`` only by majority vote of external labels
(Weizz tags); byte values alone cannot tell a length field from payload.
"""

import math
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any

import numpy as np

# Bytes decoded per input; the tail gets no spans (Hard Rule 54).
DECODE_MAX = 4096
# Bytes per seed used by fit(); bounds training cost.
TRAIN_LEN = 512
# Hard-EM rounds per fit(); stops early when the paths stop changing.
FIT_ITERS = 4
# Pseudo-counts: emission (per byte value) and transition / initial.
EMIT_ALPHA = 0.1
TRANS_ALPHA = 1.0
# Votes a state needs before it takes a FieldKind name.
NAME_MIN_VOTES = 8

_BYTE_VALUES = 256
# Backpointers are uint8.
_MAX_STATES = 256
# Rows decoded together: bounds the (rows, K, K) score tensor per step.
_BATCH_CELLS = 1 << 16
# Up to this K, pure-Python single-row decode beats numpy (per-step
# overhead dominates): n=4096, K=4 5.5 ms vs 16.7 ms; K=8 17.9 vs 15.7 ms.
_PY_MAX_STATES = 4


class FieldKind(Enum):
    """External label a state can take (0 = unlabelled)."""

    UNKNOWN = 0
    MAGIC = 1
    LENGTH = 2
    CHECKSUM = 3
    PAYLOAD = 4


_N_KINDS = len(FieldKind)


class FieldHMM:
    """K-state byte HMM: Viterbi decode, hard-EM training.

    Args:
        n_states: Hidden states, 1..256.

    Examples:
        >>> hmm = FieldHMM(n_states=2)
        >>> hmm.fit([b"\\x00" * 8 + b"\\xff" * 8])[-1] < 0
        True
        >>> [(s, e) for s, e, _ in hmm.spans(b"\\x00" * 8 + b"\\xff" * 8)]
        [(0, 8), (8, 16)]
    """

    def __init__(self, n_states: int = 8):
        if not 1 <= n_states <= _MAX_STATES:
            raise ValueError(f"n_states must be in 1..{_MAX_STATES}, got {n_states}")

        k = n_states
        self._k = k
        self._log_pi = np.full(k, -math.log(k))
        self._log_A = np.full((k, k), -math.log(k))
        self._log_B = np.full((k, _BYTE_VALUES), -math.log(_BYTE_VALUES))
        self._kinds = [FieldKind.UNKNOWN] * k
        # Decoder picked once per model (Hard Rule 57).
        self._decode_one = _viterbi_py if k <= _PY_MAX_STATES else _viterbi_one

    @property
    def n_states(self) -> int:
        return self._k

    def kind(self, state: int) -> FieldKind:
        """External label of *state* (UNKNOWN until named)."""
        return self._kinds[state]

    def decode(self, data: bytes) -> np.ndarray:
        """Most likely state per byte (uint8), first DECODE_MAX bytes."""
        obs = np.frombuffer(data[:DECODE_MAX], np.uint8)
        if obs.size == 0:
            return np.empty(0, np.uint8)

        return self._decode_one(obs, *self._params())

    def spans(self, data: bytes) -> list[tuple[int, int, int]]:
        """Run-length ``(start, end, state)`` of decode()."""
        path = self.decode(data)
        if path.size == 0:
            return []

        # Change points split the path into runs.
        cuts = np.flatnonzero(path[1:] != path[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [path.size]))
        return list(zip(starts.tolist(), ends.tolist(), path[starts].tolist(), strict=True))

    def fit(self, seeds: Sequence[bytes], iters: int = FIT_ITERS) -> list[float]:
        """Hard EM on the first TRAIN_LEN bytes of each seed.

        Returns:
            MAP objective after each round (non-decreasing); [] when there
            is nothing to train on.
        """
        rows = [s[:TRAIN_LEN] for s in seeds if s]
        if not rows:
            return []

        obs, lens = _pack(rows)
        paths = _quantise(obs, self._k)
        history: list[float] = []
        prev = None

        # M-step on the current paths, then E-step (Viterbi) under the new params.
        for _ in range(iters):
            self._m_step(obs, lens, paths)
            paths = self._decode_rows(obs, lens)
            history.append(self._objective(obs, lens, paths))
            if prev is not None and np.array_equal(paths, prev):
                break
            prev = paths
        return history

    def name_states(self, labelled: Iterable[tuple[bytes, np.ndarray]]) -> None:
        """Majority-vote a FieldKind per state from per-byte labels.

        Args:
            labelled: ``(data, kinds)`` pairs; ``kinds[i]`` is the
                ``FieldKind`` value of byte ``i`` (0 = unlabelled).
        """
        votes = np.zeros(self._k * _N_KINDS, np.int64)
        for data, kinds in labelled:
            path = self.decode(data)
            kv = np.asarray(kinds[: path.size], np.int64)
            path = path[: kv.size].astype(np.int64)
            hit = kv != 0
            votes += np.bincount(path[hit] * _N_KINDS + kv[hit], minlength=votes.size)

        # A name needs NAME_MIN_VOTES and a strict majority of labelled bytes.
        table = votes.reshape(self._k, _N_KINDS)
        top = table.argmax(axis=1)
        for s in range(self._k):
            best = table[s, top[s]]
            named = best >= NAME_MIN_VOTES and 2 * best > table[s].sum()
            self._kinds[s] = FieldKind(int(top[s])) if named else FieldKind.UNKNOWN

    def to_state(self) -> dict[str, list[Any]]:
        """Plain dict of lists (StateStore allowlist)."""
        return {
            "log_pi": self._log_pi.tolist(),
            "log_A": self._log_A.tolist(),
            "log_B": self._log_B.tolist(),
            "kinds": [k.value for k in self._kinds],
        }

    def load_state(self, state: dict[str, list[Any]]) -> None:
        """Restore to_state() output; ValueError on a shape mismatch."""
        k = self._k
        log_pi = np.asarray(state["log_pi"], np.float64)
        log_A = np.asarray(state["log_A"], np.float64)
        log_B = np.asarray(state["log_B"], np.float64)
        kinds = list(state["kinds"])
        shapes = (log_pi.shape, log_A.shape, log_B.shape, len(kinds))
        if shapes != ((k,), (k, k), (k, _BYTE_VALUES), k):
            raise ValueError(f"field_hmm state shapes {shapes} do not fit {k} states")

        self._log_pi, self._log_A, self._log_B = log_pi, log_A, log_B
        self._kinds = [FieldKind(v) for v in kinds]

    def _params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._log_pi, self._log_A, self._log_B

    def _decode_rows(self, obs: np.ndarray, lens: np.ndarray) -> np.ndarray:
        # Chunk rows so the per-step (rows, K, K) tensor stays bounded.
        step = max(1, _BATCH_CELLS // (self._k * self._k))
        parts = [
            _viterbi(obs[i : i + step], lens[i : i + step], *self._params())
            for i in range(0, obs.shape[0], step)
        ]
        return np.concatenate(parts)

    def _m_step(self, obs: np.ndarray, lens: np.ndarray, paths: np.ndarray) -> None:
        c_pi, c_A, c_B = _counts(obs, lens, paths, self._k)
        self._log_pi = _log_normalise(c_pi + TRANS_ALPHA)
        self._log_A = _log_normalise(c_A + TRANS_ALPHA)
        self._log_B = _log_normalise(c_B + EMIT_ALPHA)

    def _objective(self, obs: np.ndarray, lens: np.ndarray, paths: np.ndarray) -> float:
        # MAP objective: sum((counts + alpha) * log(params)).
        c_pi, c_A, c_B = _counts(obs, lens, paths, self._k)
        total = ((c_pi + TRANS_ALPHA) * self._log_pi).sum()
        total += ((c_A + TRANS_ALPHA) * self._log_A).sum()
        total += ((c_B + EMIT_ALPHA) * self._log_B).sum()
        return float(total)


def _pack(rows: Sequence[bytes]) -> tuple[np.ndarray, np.ndarray]:
    # Zero-padded (B, T) matrix plus true lengths.
    lens = np.array([len(r) for r in rows], np.int64)
    obs = np.zeros((len(rows), max(len(r) for r in rows)), np.uint8)
    for b, r in enumerate(rows):
        obs[b, : len(r)] = np.frombuffer(r, np.uint8)
    return obs, lens


def _quantise(obs: np.ndarray, k: int) -> np.ndarray:
    # Initial path: byte value band, e.g. K=4 -> 0x00-0x3F is state 0.
    return ((obs.astype(np.int64) * k) >> 8).astype(np.uint8)


def _counts(
    obs: np.ndarray, lens: np.ndarray, paths: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Initial, transition and emission counts over the unpadded cells.
    live = np.arange(obs.shape[1]) < lens[:, None]
    s = paths[live].astype(np.int64)
    c_B = np.bincount(s * _BYTE_VALUES + obs[live], minlength=k * _BYTE_VALUES)

    pair = live[:, 1:]
    prev = paths[:, :-1][pair].astype(np.int64)
    nxt = paths[:, 1:][pair].astype(np.int64)
    c_A = np.bincount(prev * k + nxt, minlength=k * k)

    c_pi = np.bincount(paths[:, 0], minlength=k)
    return c_pi.astype(np.float64), c_A.reshape(k, k), c_B.reshape(k, _BYTE_VALUES)


def _log_normalise(counts: np.ndarray) -> np.ndarray:
    # Row-wise log(p), p = counts / row sum.
    counts = np.asarray(counts, np.float64)
    return np.log(counts) - np.log(counts.sum(axis=-1, keepdims=True))


def _viterbi(
    obs: np.ndarray,
    lens: np.ndarray,
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
) -> np.ndarray:
    """Batched Viterbi over zero-padded rows.

    Args:
        obs: (B, T) uint8 bytes, row ``b`` valid up to ``lens[b]`` (>= 1).
        lens: (B,) row lengths.
        log_pi, log_A, log_B: Log initial (K,), transition (K, K) and
            emission (K, 256) probabilities.

    Returns:
        (B, T) uint8 state paths; cells past ``lens[b]`` repeat the last state.
    """
    n_rows, t_max = obs.shape
    k = log_pi.size

    # Longest rows first: the live rows at step t are a prefix.
    order = np.argsort(-lens, kind="stable")
    obs, lens = obs[order], lens[order]
    live_at = np.searchsorted(-lens, -np.arange(t_max), side="left")

    back = np.empty((t_max, n_rows, k), np.uint8)
    back[:] = np.arange(k, dtype=np.uint8)
    delta = log_pi + log_B[:, obs[:, 0]].T
    cols = np.arange(k)

    # Forward: best predecessor per (row, state); finished rows freeze.
    for t in range(1, t_max):
        n = live_at[t]
        scores = delta[:n, :, None] + log_A
        arg = scores.argmax(axis=1)
        back[t, :n] = arg
        delta[:n] = scores[np.arange(n)[:, None], arg, cols] + log_B[:, obs[:n, t]].T

    # Backtrack; frozen steps carry identity backpointers.
    paths = np.empty((n_rows, t_max), np.uint8)
    rows = np.arange(n_rows)
    cur = delta.argmax(axis=1).astype(np.uint8)
    paths[:, -1] = cur
    for t in range(t_max - 1, 0, -1):
        cur = back[t, rows, cur]
        paths[:, t - 1] = cur

    out = np.empty_like(paths)
    out[order] = paths
    return out


def _viterbi_one(
    obs: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray
) -> np.ndarray:
    """Single-row Viterbi, numpy over states; same ties as _viterbi."""
    n, k = obs.size, log_pi.size
    emit = log_B[:, obs].T
    back = np.empty((n, k), np.uint8)
    delta = log_pi + emit[0]
    cols = np.arange(k)

    for t in range(1, n):
        scores = delta[:, None] + log_A
        back[t] = scores.argmax(axis=0)
        delta = scores[back[t], cols] + emit[t]

    path = np.empty(n, np.uint8)
    path[-1] = delta.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _viterbi_py(
    obs: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray
) -> np.ndarray:
    """Single-row Viterbi in plain Python, for small K; same ties as _viterbi."""
    k = log_pi.size
    states = range(k)
    trans = log_A.tolist()
    emit_by_byte = log_B.T.tolist()
    seq = obs.tolist()

    first = emit_by_byte[seq[0]]
    delta = [p + first[s] for s, p in enumerate(log_pi.tolist())]
    back: list[list[int]] = []

    # Forward: strict '>' keeps the first maximum, like argmax.
    for o in seq[1:]:
        emit = emit_by_byte[o]
        arg = [0] * k
        nxt = [0.0] * k
        for j in states:
            best, bi = -math.inf, 0
            for i in states:
                v = delta[i] + trans[i][j]
                if v > best:
                    best, bi = v, i
            arg[j] = bi
            nxt[j] = best + emit[j]
        back.append(arg)
        delta = nxt

    cur = max(states, key=delta.__getitem__)
    path = [cur]
    for arg in reversed(back):
        cur = arg[cur]
        path.append(cur)
    return np.array(path[::-1], np.uint8)
