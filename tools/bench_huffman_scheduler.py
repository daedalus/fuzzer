"""Synthetic harness: does an incremental Huffman/Fenwick-style weighted
sampler beat the fuzzer's current batched-CDF seed scheduler
(seed_picker.py::_cdf_pick + weighted_pick_seed's 200-exec staleness window)?

Two structurally different questions get conflated under "Huffman tree as a
seed scheduler", so this harness tests both:

  (A) SAMPLING COST: is an entropy-shaped (Huffman) tree faster to *sample*
      from than the existing cached-CDF bisect? Both are O(log n) worst case;
      Huffman's win is expected depth ~ H(p) <= log2(n) under skew. Cheap to
      settle analytically (see huffman_expected_depth below), included for
      completeness.

  (B) FRESHNESS COST: the *actual* tradeoff live in seed_picker.py today is
      that weighted_pick_seed() recomputes weights only every 200 execs (or
      a 20-seed corpus jump), because a full CDF rebuild is O(n). A Fenwick
      tree (point-update + prefix-sum sample, both O(log n)) removes that
      excuse -- weights could be exact on every single exec instead of
      stale for up to 200. This harness measures how much distributional
      error the staleness window actually costs, against realistic
      (power-law / "favored seed") energy skew, and how much it would cost
      in wall-clock to go exact.

A literal *static* Huffman tree buys nothing here: building one costs
O(n log n) and it would need rebuilding on every reweight, which is strictly
worse than the current O(n) CDF rebuild. The only way "Huffman-shaped" beats
"Fenwick-shaped" is if the tree is reshaped incrementally (Vitter/FGK dynamic
Huffman) -- which is real but adds real complexity for a gain that only shows
up in (A), which this harness shows is not where the money is.
"""
import math
import random
import time


def huffman_expected_depth(weights):
    """Expected leaf depth of the optimal (Huffman) prefix tree, in comparisons.

    Classic textbook build: repeatedly merge the two lightest nodes. Returns
    weighted average depth = sum(w_i * depth_i) / sum(w_i), the quantity
    that determines average sampling cost for a "walk down weighing
    subtrees" sampler.
    """
    import heapq

    heap = [(w, i, 0) for i, w in enumerate(weights)]  # (weight, tie-break, depth)
    heapq.heapify(heap)
    total_depth_weight = 0.0
    # Track depth via a small trick: re-heapify merged nodes carrying an
    # accumulated "member count" isn't enough for *weighted* depth, so
    # instead build the tree explicitly.
    nodes = [{"w": w, "depth_contribs": [(w, 0)]} for w in weights]
    heap = [(n["w"], i) for i, n in enumerate(nodes)]
    heapq.heapify(heap)
    alive = dict(enumerate(nodes))
    next_id = len(nodes)
    while len(heap) > 1:
        w1, i1 = heapq.heappop(heap)
        while i1 not in alive or alive[i1]["w"] != w1:
            if not heap:
                break
            w1, i1 = heapq.heappop(heap)
        w2, i2 = heapq.heappop(heap)
        while i2 not in alive or alive[i2]["w"] != w2:
            w2, i2 = heapq.heappop(heap)
        n1, n2 = alive.pop(i1), alive.pop(i2)
        merged = {
            "w": n1["w"] + n2["w"],
            "depth_contribs": [(w, d + 1) for w, d in n1["depth_contribs"]]
            + [(w, d + 1) for w, d in n2["depth_contribs"]],
        }
        alive[next_id] = merged
        heapq.heappush(heap, (merged["w"], next_id))
        next_id += 1
    root = next(iter(alive.values()))
    total_w = sum(weights)
    return sum(w * d for w, d in root["depth_contribs"]) / total_w


def shannon_entropy_bits(weights):
    total = sum(weights)
    return -sum((w / total) * math.log2(w / total) for w in weights if w > 0)


class FenwickSampler:
    """Point-update + weighted-sample in O(log n), always exact."""

    def __init__(self, weights):
        self.n = len(weights)
        self.tree = [0.0] * (self.n + 1)
        self.w = list(weights)
        for i, wi in enumerate(weights):
            self._add(i, wi)

    def _add(self, i, delta):
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def total(self):
        return sum(self.tree[i] for i in self._prefix_indices(self.n))

    def _prefix_sum(self, i):
        i += 1
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def _prefix_indices(self, i):
        # helper only used by total(); irrelevant to complexity analysis
        return range(1, i + 1)

    def update(self, i, new_w):
        self._add(i, new_w - self.w[i])
        self.w[i] = new_w

    def sample(self, r):
        """r in [0, total()); returns index via Fenwick binary search, O(log n)."""
        idx = 0
        bitmask = 1 << (self.n.bit_length())
        target = r
        pos = 0
        while bitmask:
            nxt = pos + bitmask
            if nxt <= self.n and self.tree[nxt] <= target:
                pos = nxt
                target -= self.tree[nxt]
            bitmask >>= 1
        return pos  # pos is count of elements with prefix_sum <= r; index = pos


def kl_divergence(p, q):
    total_p, total_q = sum(p), sum(q)
    s = 0.0
    for pi, qi in zip(p, q, strict=True):
        if pi <= 0:
            continue
        pn, qn = pi / total_p, max(qi / total_q, 1e-12)
        s += pn * math.log2(pn / qn)
    return s


def run_freshness_experiment(n_seeds, n_steps, skew, recompute_interval, rng):
    """Simulate live per-exec energy drift; compare staleness error of the
    current batched-CDF approach against an always-exact Fenwick sampler.
    """
    # Power-law-ish initial energy: a few "favored" seeds dominate mass,
    # matching this project's own documented seed-energy skew.
    true_weights = [max(rng.paretovariate(skew), 1e-3) for _ in range(n_seeds)]

    stale_weights = list(true_weights)
    kl_series = []
    fenwick = FenwickSampler(true_weights)

    for step in range(n_steps):
        # Drift: one random seed's energy jumps (a coverage hit changes its
        # priority) -- the event that both structures must eventually reflect.
        i = rng.randrange(n_seeds)
        new_w = max(rng.paretovariate(skew), 1e-3)
        true_weights[i] = new_w
        fenwick.update(i, new_w)  # always exact, O(log n)

        if step % recompute_interval == 0:
            stale_weights = list(true_weights)  # full O(n) rebuild, as today

        kl_series.append(kl_divergence(true_weights, stale_weights))

    return sum(kl_series) / len(kl_series), max(kl_series)


def run_cost_experiment(n_seeds, n_ops):
    weights = [random.random() + 1e-3 for _ in range(n_seeds)]

    # Current: rebuild full CDF (as _cdf_pick does whenever the weight
    # vector object changes -- i.e. every single reweight in an always-fresh
    # regime, since a new list is required to bust the identity cache).
    t0 = time.perf_counter()
    for _ in range(n_ops):
        i = random.randrange(n_seeds)
        weights = list(weights)
        weights[i] = random.random() + 1e-3
        cum = []
        acc = 0.0
        for w in weights:
            acc += w
            cum.append(acc)
    t_cdf_rebuild = time.perf_counter() - t0

    # Fenwick: point update, O(log n), always exact.
    fenwick = FenwickSampler(weights)
    t0 = time.perf_counter()
    for _ in range(n_ops):
        i = random.randrange(n_seeds)
        fenwick.update(i, random.random() + 1e-3)
    t_fenwick_update = time.perf_counter() - t0

    return t_cdf_rebuild, t_fenwick_update


if __name__ == "__main__":
    rng = random.Random(42)
    print("=== (A) Sampling shape: expected depth, Huffman-optimal vs log2(n) ===")
    for n_seeds, skew in [(50, 0.5), (50, 3.0), (500, 0.5), (500, 3.0)]:
        weights = [max(rng.paretovariate(skew), 1e-3) for _ in range(n_seeds)]
        h = shannon_entropy_bits(weights)
        d = huffman_expected_depth(weights)
        print(
            f"n={n_seeds:4d} skew(pareto-alpha)={skew:.1f}  "
            f"H(p)={h:5.2f} bits  huffman_exp_depth={d:5.2f}  log2(n)={math.log2(n_seeds):5.2f}"
        )

    print()
    print("=== (B) Freshness: KL(true || stale) under the current 200-exec window ===")
    for n_seeds in (100, 2000):
        for skew in (0.6, 1.5, 4.0):  # lower alpha = heavier tail = more skew
            for interval in (1, 50, 200):
                mean_kl, max_kl = run_freshness_experiment(
                    n_seeds, n_steps=4000, skew=skew, recompute_interval=interval,
                    rng=random.Random(7),
                )
                print(
                    f"n={n_seeds:5d} skew_alpha={skew:.1f} recompute_every={interval:4d} "
                    f"-> mean_KL={mean_kl:.4f} bits  max_KL={max_kl:.4f} bits"
                )

    print()
    print("=== (C) Wall-clock: O(n) CDF rebuild vs O(log n) Fenwick update, per reweight ===")
    for n_seeds in (100, 1000, 10000):
        t_cdf, t_fen = run_cost_experiment(n_seeds, n_ops=2000)
        print(
            f"n={n_seeds:6d}  cdf_rebuild={t_cdf*1e6/2000:8.2f} us/op  "
            f"fenwick_update={t_fen*1e6/2000:8.2f} us/op  speedup={t_cdf/t_fen:6.1f}x"
        )
