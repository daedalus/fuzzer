"""Tiny numpy MLP with a closed-form input gradient (NEUZZ-style program smoothing).

NEUZZ (She et al., S&P'19; Nicolae et al., FSE'23 "Neuzz++") fits a neural net from
seed bytes to the edges they cover and reads the gradient of one edge's logit with
respect to the input bytes as a byte-importance map. This module is that idea
re-derived for numpy, with no TensorFlow and no autograd, from the maths alone::

    h = x @ W1 + b1          (L inputs -> H hidden)
    a = relu(h)
    z = a @ W2 + b2          (H hidden -> K edge logits)
    p = sigmoid(z)

    dz_k/dx = W1 @ (1[h > 0] * W2[:, k])        one hidden layer: closed form

``saliency`` averages |dz_k/dx| over a few target columns ``k``. Fitting is
full-batch Adam on binary cross-entropy with the output bias started at the
column log-odds (so an untrained net already predicts base rates), bounded by
``epochs``; callers keep the sample and width small so a refit stays cheap.

Pure: no fuzzer imports, deterministic given the Generator passed to ``fit``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TinyMLP", "encode_inputs"]

_EPS = 1e-7


def encode_inputs(seeds: list[bytes], width: int) -> np.ndarray:
    """(n, width) float32 in [0, 1]: bytes / 255, truncated or zero-padded to *width*."""
    x = np.zeros((len(seeds), width), np.float32)
    for i, s in enumerate(seeds):
        v = np.frombuffer(s[:width], np.uint8)
        x[i, : len(v)] = v
    return x / 255.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class TinyMLP:
    """One hidden ReLU layer, sigmoid outputs, closed-form input gradient."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator) -> None:
        if min(n_in, n_hidden, n_out) < 1:
            raise ValueError("TinyMLP dimensions must be >= 1")
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.w1 = (rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, np.float32)
        self.w2 = (rng.standard_normal((n_hidden, n_out)) * np.sqrt(1.0 / n_hidden)).astype(
            np.float32
        )
        self.b2 = np.zeros(n_out, np.float32)

    # -- forward ----------------------------------------------------------------

    def logits(self, x: np.ndarray) -> np.ndarray:
        """(n, n_out) pre-sigmoid outputs for (n, n_in) inputs."""
        return np.maximum(x @ self.w1 + self.b1, 0.0) @ self.w2 + self.b2

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logits(x))

    # -- fit --------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 60,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
    ) -> float:
        """Full-batch Adam on BCE. Returns the final mean loss.

        Args:
            x: (n, n_in) float32 inputs.
            y: (n, n_out) 0/1 labels.
        """
        n = len(x)
        if n == 0 or y.shape != (n, self.n_out):
            raise ValueError("fit: x/y shape mismatch or empty")
        y = y.astype(np.float32)
        base = np.clip(y.mean(axis=0), 1e-3, 1.0 - 1e-3)
        self.b2 = np.log(base / (1.0 - base)).astype(np.float32)

        params = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1c, b2c = 0.9, 0.999
        loss = 0.0
        for t in range(1, epochs + 1):
            h = x @ self.w1 + self.b1
            a = np.maximum(h, 0.0)
            p = _sigmoid(a @ self.w2 + self.b2)
            loss = float(-np.mean(y * np.log(p + _EPS) + (1.0 - y) * np.log(1.0 - p + _EPS)))

            dz = (p - y) / (n * self.n_out)
            dw2 = a.T @ dz + weight_decay * self.w2
            db2 = dz.sum(axis=0)
            da = (dz @ self.w2.T) * (h > 0)
            dw1 = x.T @ da + weight_decay * self.w1
            db1 = da.sum(axis=0)

            for i, (param, g) in enumerate(zip(params, (dw1, db1, dw2, db2), strict=True)):
                m[i] = b1c * m[i] + (1 - b1c) * g
                v[i] = b2c * v[i] + (1 - b2c) * g * g
                mh = m[i] / (1 - b1c**t)
                vh = v[i] / (1 - b2c**t)
                param -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        return loss

    # -- gradient ---------------------------------------------------------------

    def input_grad(self, x: np.ndarray, targets: list[int] | np.ndarray) -> np.ndarray:
        """(n_in, T) d logit_k / d x for one input *x* (shape (n_in,)) and T target columns."""
        h = x @ self.w1 + self.b1
        gate = (h > 0).astype(np.float32)
        return self.w1 @ (gate[:, None] * self.w2[:, np.asarray(targets, dtype=np.int64)])

    def saliency(self, x: np.ndarray, targets: list[int] | np.ndarray) -> np.ndarray:
        """(n_in,) mean |d logit_k / d x| over *targets*: how much each byte moves them."""
        return np.abs(self.input_grad(x, targets)).mean(axis=1)
