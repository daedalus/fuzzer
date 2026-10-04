"""Elo arena over position schedulers (``pos_<name>`` keys).

Third tournament beside operator and seed selection (see
``core/schedulers/pos_base.py``). ``OperatorEngine.select_position`` used
to pick uniformly among whatever tracker proposals existed; with the arena
on, Elo (Thompson over the ``pos_`` posteriors) picks which proposer
speaks, and uniform is a member: a proposer rated at or below it is
flagged (``Fuzzer._check_canary_inspection``).

Arms::

    uniform      baseline, always in the pool, first (Elo's cold-start pick)
    sensitivity  per-byte Lyapunov sensitivity
    te / phase   transfer-entropy map / record-stride phase lock
    mi           mutual-information map
    crash_mi     crash mutual-information map (after min_observations)
    region       statistical region profile
    field        FormatLearner field hypotheses (confirmed coverage-causal
                 offsets, --learn-format; joins once a hypothesis exists)
    burn_front   BurnFrontPositionScheduler (opt-in, --burn-front)
    kl_ducb      PositionKLDUCBScheduler, discounted KL-UCB over a seed's
                 offset bins -- a theoretically-grounded rival to
                 burn-front's heuristic (opt-in, --pos-kl-ducb; see
                 core/schedulers/pos_kl_ducb.py)
    canary       PositionCanaryScheduler, deliberately worst-in-class floor
                 (opt-in, --pos-canary; see core/schedulers/pos_canary.py)
    round_robin  PositionRoundRobinScheduler, deterministic cycling
                 (opt-in, --pos-round-robin; see
                 core/schedulers/pos_round_robin.py)
    fibonacci    PositionFibonacciScheduler, golden-ratio sweep, no
                 per-seed state (implied by --position-arena; see
                 core/schedulers/pos_fibonacci.py)
    fractal      PositionFractalScheduler, adaptive-resolution binary
                 tree that only refines where coverage-gain heat
                 justifies it (opt-in, --pos-fractal; see
                 core/schedulers/pos_fractal.py)
    cmplog       PositionCmplogScheduler, redqueen offsets + Weizz-flagged
                 spans (len/magic/checksum/input-to-state). Tracker-style
                 arm: joins the pool only while cmplog is live, never an
                 off-policy extra (opt-in, --pos-cmplog; see
                 core/schedulers/pos_cmplog.py)
    lineage      PositionLineageScheduler, the mutation sites recorded in
                 ``parent_sites`` seed metadata (+ geometric jitter).
                 Tracker-style arm: joins the pool only while ``--lineage``
                 is on (no other run records the sites), never an
                 off-policy extra (opt-in, --pos-lineage; see
                 core/schedulers/pos_lineage.py)
    context      PositionContextScheduler, cross-seed byte-context rates
                 (class, previous class, position decile) -- warm on a
                 brand-new seed of a familiar format (opt-in,
                 --pos-context; see core/schedulers/pos_context.py)
    levy         PositionLevyScheduler, a heavy-tailed jump around the
                 seed's last gain offset (opt-in, --pos-levy; see
                 core/schedulers/pos_levy.py)
    boundary     PositionBoundaryScheduler, content-derived field
                 boundaries (class changes, delimiters, entropy steps,
                 run edges), no state and no feedback signal (opt-in,
                 --pos-boundary; see core/schedulers/pos_boundary.py)
    effector     PositionEffectorScheduler, bytes the deterministic
                 byteflip pass saw move the trace. Tracker-style: joins
                 once a drained effector map exists (opt-in,
                 --pos-effector; see core/schedulers/pos_effector.py)
    finch        PositionFinchScheduler, bytes weighted by how many edges the
                 byteflip pass saw them move, plus a per-seed bonus learned
                 from gain rounds. Tracker-style like effector, but also fed
                 every settled round (opt-in, --pos-finch; see
                 core/schedulers/pos_finch.py)
    token        PositionTokenScheduler, occurrences of dictionary tokens
                 in the seed. Tracker-style: joins while the dictionary is
                 non-empty (opt-in, --pos-token; see
                 core/schedulers/pos_token.py)
    chunk        PositionChunkScheduler, container chunk headers from the
                 format parsers. Tracker-style: joins once a seed parsed
                 (opt-in, --pos-chunk; see core/schedulers/pos_chunk.py)
    changed      PositionChangedScheduler, pooled group testing on "did
                 the trace move?" -- sinks inert bytes (opt-in,
                 --pos-changed; see core/schedulers/pos_changed.py)
    rare_mask    PositionRareMaskScheduler, FairFuzz branch mask: bins
                 whose mutation keeps the seed's rarest edge (opt-in,
                 --pos-rare-mask; see core/schedulers/pos_rare_mask.py)
    good_turing  PositionGoodTuringScheduler, offset bins drawn by per-bin
                 Good-Turing discovery probability over edge identity
                 (opt-in, --pos-good-turing; see
                 core/schedulers/pos_good_turing.py)
    saliency     PositionSaliencyScheduler, offsets drawn by the input gradient
                 of a small net fitted corpus bytes -> edges (NEUZZ-style;
                 opt-in, --pos-saliency, NOT implied by --position-arena;
                 see core/schedulers/pos_saliency.py)
    consolidated PositionConsolidatedScheduler, uniform/boundary/levy/bin
                 candidates scored by context x per-seed bin rates
                 (opt-in, --pos-consolidated; see
                 core/schedulers/pos_consolidated.py)

Only arms whose feature is on join the pool, so nobody accrues phantom
matches. ``arms`` (``--pos-arena-arms``) narrows the pool further to a named
subset (uniform is always kept): an arm left out is neither proposed from nor
credited off-policy, so ``arena{uniform}`` vs ``arena{uniform, X}`` is a paired
A/B of one arm and ``arena{all}`` vs ``arena{all minus X}`` is its leave-one-out
ablation (``tools/lib/bench_paired.py``, ``pos-arena-*``). An arm that declines gets a uniform offset but is *charged under
its own name*: Elo picked it, so the round is its round. Charging the
decline to uniform instead made a declining arm unbeatable -- it never
served, so it only ever played as an opponent, and in a miss-dominated
campaign every opponent wins. Measured (20k rounds, 5% gains): an arm that
always declines rated 1725 against 1165 for uniform and 1627 for a real
proposer of uniform quality, and the uniform floor flagged nothing. Charged
to itself, a pure decliner is exactly uniform and rates as uniform.

Matches: a round's operators may land several positions. Every arm that
served one plays each pool member that did not, with the round score. Arms
that shared a round do not play each other.

``burn_front``, ``kl_ducb``, ``canary``, ``round_robin``, ``fibonacci``,
``fractal``, ``context``, ``levy``, ``boundary``, ``changed``, ``rare_mask``, ``good_turing``, ``saliency`` and ``consolidated`` are each
credited off-policy on every settled round, whoever served the positions, like
``seed_canary`` on the seed side.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from fuzzer_tool.core.analyzers.analyzer_elo import POS_STRATEGY_PREFIX
from fuzzer_tool.core.schedulers.pos_base import (
    CallablePosition,
    Outcome,
    PositionScheduler,
    UniformPosition,
)

UNIFORM = "uniform"
POSITION_STRATEGY_NAMES = (
    UNIFORM,
    "sensitivity",
    "te",
    "phase",
    "mi",
    "crash_mi",
    "region",
    "field",
    "burn_front",
    "kl_ducb",
    "canary",
    "round_robin",
    "fibonacci",
    "fractal",
    "cmplog",
    "lineage",
    "context",
    "levy",
    "boundary",
    "effector",
    "finch",
    "token",
    "chunk",
    "changed",
    "rare_mask",
    "good_turing",
    "saliency",
    "consolidated",
)


def parse_arena_arms(spec: str) -> tuple[str, ...]:
    """``"uniform,fractal"`` -> ``("uniform", "fractal")``; ``ValueError`` on a bad name.

    Order-preserving, de-duplicated. An empty spec is an error: ``None`` (flag
    absent) is how "every arm" is spelled, and an empty list silently meaning
    the same would hide a typo. Underscores and hyphens are interchangeable
    (``kl-ducb`` == ``kl_ducb``) to match the flag names.
    """
    names = [n.strip().replace("-", "_") for n in spec.split(",") if n.strip()]
    if not names:
        raise ValueError("--pos-arena-arms needs at least one arm name")
    unknown = [n for n in names if n not in POSITION_STRATEGY_NAMES]
    if unknown:
        raise ValueError(
            f"unknown position arena arm(s) {', '.join(unknown)}; "
            f"choose from {', '.join(POSITION_STRATEGY_NAMES)}"
        )
    return tuple(dict.fromkeys(names))


Gate = Callable[[], bool]
Arm = tuple[PositionScheduler, Gate]


class PositionArena:
    def __init__(
        self,
        f,
        region_fn,
        burn_front: PositionScheduler | None = None,
        kl_ducb: PositionScheduler | None = None,
        canary: PositionScheduler | None = None,
        round_robin: PositionScheduler | None = None,
        fibonacci: PositionScheduler | None = None,
        fractal: PositionScheduler | None = None,
        cmplog: PositionScheduler | None = None,
        lineage: PositionScheduler | None = None,
        context: PositionScheduler | None = None,
        levy: PositionScheduler | None = None,
        arms: Iterable[str] | None = None,
        boundary: PositionScheduler | None = None,
        effector: PositionScheduler | None = None,
        token: PositionScheduler | None = None,
        chunk: PositionScheduler | None = None,
        changed: PositionScheduler | None = None,
        rare_mask: PositionScheduler | None = None,
        consolidated: PositionScheduler | None = None,
        finch: PositionScheduler | None = None,
        good_turing: PositionScheduler | None = None,
        saliency: PositionScheduler | None = None,
    ) -> None:
        self._f = f
        # None = every arm whose feature is on; otherwise only these (+ uniform).
        self._enabled: frozenset[str] | None = (
            None if arms is None else frozenset(parse_arena_arms(",".join(arms))) | {UNIFORM}
        )
        self._uniform = UniformPosition(f._rng)
        # An arm left out of the subset is dropped here, once: it is neither
        # proposed from nor credited, so no later site can reach it.
        self._burn_front = burn_front if self.allows("burn_front") else None
        self._kl_ducb = kl_ducb if self.allows("kl_ducb") else None
        self._canary = canary if self.allows("canary") else None
        self._round_robin = round_robin if self.allows("round_robin") else None
        self._fibonacci = fibonacci if self.allows("fibonacci") else None
        self._fractal = fractal if self.allows("fractal") else None
        self._cmplog = cmplog if self.allows("cmplog") else None
        self._lineage = lineage if self.allows("lineage") else None
        self._context = context if self.allows("context") else None
        self._levy = levy if self.allows("levy") else None
        self._boundary = boundary if self.allows("boundary") else None
        # Passive arms gated on their own active(): wired in _add_trackers.
        self._gated: tuple[PositionScheduler, ...] = tuple(
            a for a in (effector, finch, token, chunk) if a is not None and self.allows(a.name)
        )
        # Gated arms that learn from outcomes: fed every settled round too.
        self._finch = finch if self.allows("finch") else None
        self._changed = changed if self.allows("changed") else None
        self._rare_mask = rare_mask if self.allows("rare_mask") else None
        self._consolidated = consolidated if self.allows("consolidated") else None
        self._good_turing = good_turing if self.allows("good_turing") else None
        self._saliency = saliency if self.allows("saliency") else None
        self._arms: dict[str, Arm] = {UNIFORM: (self._uniform, lambda: True)}
        self._add_trackers(region_fn)
        # Off-policy arms: fed every settled round whoever served. Single list
        # for the pool and for settle(), so a new arm cannot miss one of them.
        self._extras: tuple[PositionScheduler, ...] = tuple(
            e
            for e in (
                self._burn_front,
                self._kl_ducb,
                self._canary,
                self._round_robin,
                self._fibonacci,
                self._fractal,
                self._context,
                self._levy,
                self._boundary,
                self._changed,
                self._rare_mask,
                self._good_turing,
                self._saliency,
                self._consolidated,
            )
            if e is not None
        )
        for extra in self._extras:
            self._arms[extra.name] = (extra, lambda: True)
        self._used: list[str] = []
        self._seen_pool: list[str] = []

    def allows(self, name: str) -> bool:
        """Whether *name* may join the pool under ``arms`` (uniform always may)."""
        return self._enabled is None or name in self._enabled

    @property
    def enabled_arms(self) -> frozenset[str] | None:
        """The ``arms`` subset (uniform included), or None when unrestricted."""
        return self._enabled

    def _add_trackers(self, region_fn: Callable[[bytes, int], int | None]) -> None:
        f = self._f

        def on(flag: str, tracker: str) -> Gate:
            return lambda: bool(getattr(f, flag, False) and getattr(f, tracker, None))

        def crash_ready() -> bool:
            cm = getattr(f, "_crash_mi", None)
            return bool(cm and cm.total_execs >= cm.min_observations)

        def cmplog_ready() -> bool:
            # Passive arm: only meaningful while cmplog is collecting.
            return getattr(f, "_cmplog", None) is not None

        def lineage_ready() -> bool:
            # Passive arm: parent_sites exist only when lineage tracking is on.
            return bool(getattr(f, "_use_lineage", False))

        def field_ready() -> bool:
            fl = getattr(f, "_format_learner", None)
            return bool(fl and fl.clusters)

        def phase(data: bytes, n: int) -> int | None:
            meta = f.seed_meta.get(data)
            return f._get_phase_weighted_position(n, meta.get("record_stride") if meta else None)

        te_on = on("_use_transfer_entropy", "_te")
        specs: list[tuple[str, Callable[[bytes, int], int | None], Gate]] = [
            ("sensitivity", lambda d, n: f._sensitivity.get_weighted_position(d, n),
             on("_use_sensitivity", "_sensitivity")),
            ("te", lambda d, n: f._get_te_weighted_position(n), te_on),
            ("phase", phase, te_on),
            ("mi", lambda d, n: f._mi.weighted_position(n), on("_use_mi", "_mi")),
            ("crash_mi", lambda d, n: f._crash_mi.weighted_position(n), crash_ready),
            ("region", region_fn, lambda: bool(getattr(f, "_use_region_profile", False))),
            ("field", lambda d, n: f._format_learner.weighted_position(d, n), field_ready),
        ]  # fmt: skip
        if self._cmplog is not None:
            specs.append(("cmplog", self._cmplog.propose, cmplog_ready))
        if self._lineage is not None:
            specs.append(("lineage", self._lineage.propose, lineage_ready))
        for name, fn, gate in specs:
            if self.allows(name):
                self._arms[name] = (CallablePosition(name, fn), gate)
        # effector / finch / token / chunk: each knows when it has data.
        for arm in self._gated:
            self._arms[arm.name] = (arm, arm.active)

    def pool(self) -> list[str]:
        """Names of arms whose feature is on now; uniform first."""
        return [name for name, (_, gate) in self._arms.items() if gate()]

    def begin_round(self) -> None:
        """Forget selections from a mutant that will not be executed.

        ``OperatorEngine.mutate`` calls this first thing. ``_dedup_mutate``
        re-rolls a mutant the exec bloom has already seen, and each re-roll
        is a fresh ``mutate()``; without this the arms that served the
        discarded mutants stayed in ``_used`` and were credited with the
        outcome of the one that actually ran. ``settle`` still clears too.
        """
        self._used, self._seen_pool = [], []

    def used(self) -> list[str]:
        """Arms that served a position this round, in order."""
        return list(self._used)

    def select(self, data: bytes, buf_len: int) -> int:
        """Elo picks the arm; a declining arm gets a uniform offset.

        The arm stays charged for the round even when it declines (see the
        module docstring): a decline is the arm's choice, not uniform's.
        """
        pool = self.pool()
        self._seen_pool.extend(n for n in pool if n not in self._seen_pool)
        name = self._arbitrate(pool)

        pos = self._arms[name][0].propose(data, buf_len)
        if pos is None:
            pos = self._uniform.propose(data, buf_len)

        self._used.append(name)
        return min(max(pos, 0), buf_len - 1)

    def _arbitrate(self, pool: list[str]) -> str:
        if len(pool) == 1:
            return pool[0]

        picked = self._f._elo.select_strategy([POS_STRATEGY_PREFIX + n for n in pool])
        return picked.removeprefix(POS_STRATEGY_PREFIX)

    def settle(
        self,
        data: bytes,
        offsets: Sequence[int],
        outcome: Outcome,
        weight: float,
        score: float,
    ) -> None:
        """End of round: feed the off-policy arms, then play the Elo matches."""
        for extra in self._extras:
            extra.record(data, offsets, outcome, weight)
        if self._finch is not None:
            self._finch.record(data, offsets, outcome, weight)

        served = list(dict.fromkeys(self._used))
        pool = self._seen_pool
        self._used, self._seen_pool = [], []

        elo = getattr(self._f, "_elo", None)
        if not (getattr(self._f, "_use_elo", False) and elo):
            return

        for name in served:
            for other in pool:
                if other not in served:
                    elo.record_strategy_match(
                        POS_STRATEGY_PREFIX + name, POS_STRATEGY_PREFIX + other, score
                    )
