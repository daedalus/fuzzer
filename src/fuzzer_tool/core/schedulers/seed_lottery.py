"""SeedLotteryScheduler: lottery scheduling over the corpus (Waldspurger & Weihl 1994).

The randomized sibling of ``seed_stride``. Each seed holds tickets; a pick
draws a ticket and serves its holder, so the long-run share is proportional
to tickets with no pass state to repair when the corpus changes. A tree of
partial ticket sums (``core/fenwick_sampler.py``) makes insert, remove,
reweight and draw O(log n). Three of the paper's mechanisms are here::

    tickets   weight / cost                         (compensation, always on)
    currency  tickets are denominated per group     (--seed-lottery-currency)
    transfer  a successful seed credits its parent  (--seed-lottery-transfer)

**Compensation.** tickets = weight / cost, the caller's per-seed weight (AFL
favored 2x) over its mean exec time relative to the corpus mean, so equal
weight means equal target *time*. Flat cost and weight is a uniform random
pick -- the falsification condition.

**Currencies.** Seeds are grouped (the picker supplies the connected component
of the seed Gram matrix, ``core/edge_components.py``); a pick first draws a
*group* by its backing, then a seed inside it by tickets. ``off`` is one flat
currency. ``equal`` gives every group the same backing and ``sqrt`` gives it
sqrt(size); a group of near-duplicates then cannot take share from distinct
regions by being large, which a flat ticket pool lets it do::

    groups A(100 seeds) B(1 seed):  off P(B)=1/101   equal P(B)=1/2   sqrt P(B)=1/11

**Transfer.** When a seed's fuzzing succeeds, ``transfer`` times its reward is
added to its lineage parent's credit and the parent's tickets become
``base * (1 + credit)``. Credit is capped at TRANSFER_CAP and halves at every
refresh, so it is a short memory, not an accumulating bonus. Needs
``--lineage``: without a parent there is nobody to credit.

Weights, costs, groups and parents are read from the caller per pick for new
seeds and for every seed once per REFRESH_PICKS picks (a favored flag or a
component label can change between refreshes); in between, a seed keeps the
tickets it was given. ``record`` feeds the Elo match and, when ``transfer`` is
on, the parent credit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from typing import Protocol

from fuzzer_tool.core.fair_queue import neutral
from fuzzer_tool.core.fenwick_sampler import FenwickSampler
from fuzzer_tool.core.schedulers._arm_counts import ArmCounts, reward

#: Picks between full re-reads of every seed's weight, cost, group and parent.
REFRESH_PICKS = 256
#: Largest parent credit; tickets are at most (1 + TRANSFER_CAP) x base.
TRANSFER_CAP = 4.0
#: Credit kept across a refresh.
TRANSFER_DECAY = 0.5
#: Credit below this is dropped at a refresh.
CREDIT_EPS = 1e-3
CURRENCY_MODES = ("off", "equal", "sqrt")

_FLAT = 0


class _Random(Protocol):
    def random(self) -> float: ...


def _unit(_key: str) -> float:
    return 1.0


def _flat_group(_key: str) -> Hashable:
    return _FLAT


def _no_parent(_key: str):
    return None


class SeedLotteryScheduler(ArmCounts):
    """Lottery over seeds; compensated tickets, optional currencies and transfer."""

    #: No informative priors: selection reads tickets, never the posterior.
    supports_priors = False

    def __init__(self, rng: _Random, currency: str = "off", transfer: float = 0.0) -> None:
        super().__init__()
        if currency not in CURRENCY_MODES:
            raise ValueError(f"currency must be one of {CURRENCY_MODES}, got {currency!r}")
        if not (math.isfinite(transfer) and transfer >= 0.0):
            raise ValueError("transfer must be a finite number >= 0")
        self._rng = rng
        self._currency = currency
        self._transfer = float(transfer)
        self._picks = 0
        self._seen: list[str] | None = None
        self._base: dict[str, float] = {}
        self._group: dict[str, Hashable] = {}
        self._parent: dict[str, str | None] = {}
        self._credit: dict[str, float] = {}
        self._members: dict[Hashable, FenwickSampler] = {}
        self._top = FenwickSampler()

    @property
    def groups(self) -> int:
        """Number of live currencies (1 when currencies are off and the corpus is non-empty)."""
        return len(self._members)

    def select_seed(
        self,
        seed_ids: list[str],
        weight_fn: Callable[[str], float] = _unit,
        cost_fn: Callable[[str], float] = _unit,
        group_fn: Callable[[str], Hashable] = _flat_group,
        flow_fn: Callable[[str], str | None] = _no_parent,
    ) -> str:
        if not seed_ids:
            return ""
        if len(seed_ids) == 1:
            return seed_ids[0]
        self._trim(seed_ids)
        self._picks += 1
        args = (weight_fn, cost_fn, group_fn, flow_fn)
        if self._seen is None or self._picks % REFRESH_PICKS == 0:
            self._rebuild(seed_ids, *args)
        elif seed_ids != self._seen:
            self._sync(seed_ids, *args)

        group = _FLAT if self._currency == "off" else self._top.sample(self._rng.random())
        members = self._members.get(group)
        key = members.sample(self._rng.random()) if members is not None else None
        return key if key is not None else seed_ids[0]

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        super().record(name, success, weight)
        if self._transfer <= 0.0 or not success:
            return
        parent = self._parent.get(name)
        if parent is None or parent == name or parent not in self._base:
            return
        gained = self._transfer * reward(success, weight)
        self._credit[parent] = min(TRANSFER_CAP, self._credit.get(parent, 0.0) + gained)
        self._retarget(parent)

    # -- state ---------------------------------------------------------------

    def _tickets(self, key: str) -> float:
        return self._base[key] * (1.0 + self._credit.get(key, 0.0))

    def _backing(self, size: int) -> float:
        return math.sqrt(size) if self._currency == "sqrt" else 1.0

    def _admit(self, key: str, weight_fn, cost_fn, group_fn, flow_fn) -> Hashable:
        self._base[key] = neutral(weight_fn(key)) / neutral(cost_fn(key))
        group = _FLAT if self._currency == "off" else group_fn(key)
        self._group[key] = group
        self._parent[key] = flow_fn(key) if self._transfer > 0.0 else None
        self._members.setdefault(group, FenwickSampler()).set(key, self._tickets(key))
        return group

    def _set_backing(self, groups) -> None:
        if self._currency == "off":
            return
        for g in groups:
            sampler = self._members.get(g)
            if sampler is None or not len(sampler):
                self._members.pop(g, None)
                self._top.remove(g)
            else:
                self._top.set(g, self._backing(len(sampler)))

    def _rebuild(self, seed_ids, weight_fn, cost_fn, group_fn, flow_fn) -> None:
        """Re-read every seed; credit halves, departed seeds' credit is dropped."""
        live = set(seed_ids)
        self._credit = {
            k: v * TRANSFER_DECAY
            for k, v in self._credit.items()
            if k in live and v * TRANSFER_DECAY > CREDIT_EPS
        }
        self._base, self._group, self._parent = {}, {}, {}
        self._members, self._top = {}, FenwickSampler()
        for key in seed_ids:
            self._admit(key, weight_fn, cost_fn, group_fn, flow_fn)
        self._set_backing(list(self._members))
        self._seen = list(seed_ids)

    def _sync(self, seed_ids, weight_fn, cost_fn, group_fn, flow_fn) -> None:
        """Apply a membership change in O(changed log n): drop departed, admit joiners."""
        live = set(seed_ids)
        touched = set()
        for key in [k for k in self._base if k not in live]:
            group = self._group.pop(key)
            self._members[group].remove(key)
            self._base.pop(key)
            self._parent.pop(key, None)
            self._credit.pop(key, None)
            touched.add(group)
        for key in seed_ids:
            if key not in self._base:
                touched.add(self._admit(key, weight_fn, cost_fn, group_fn, flow_fn))
        self._set_backing(touched)
        self._seen = list(seed_ids)

    def _retarget(self, key: str) -> None:
        sampler = self._members.get(self._group[key])
        if sampler is not None:
            sampler.set(key, self._tickets(key))
