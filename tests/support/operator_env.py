"""Shared scaffolding for operator-level tests.

``OperatorEngine`` is constructed from a ``Fuzzer``, so exercising a single
byte-level operator nominally requires a full fuzzer -- and
``test_operator_smoke.py`` does exactly that, compiling a target binary to
reach ``_op_bit_flip``. The mock below is the cheap substitute two test
suites now share rather than duplicate.

Its size is the argument for the minimal-interface extraction: 29 attributes,
mostly None or False, all of them present only because the operator's declared
contract is the whole ``Fuzzer`` object. See ``docs/port-backlog.md``, item F1.
"""

from __future__ import annotations

# Every scheduler `OperatorEngine.select_op` puts on the Elo ballot, plus the
# two flags it reads alongside them. `select_op` reads each as a bare
# `f._use_X and f._X`, so a fake missing one pair raises AttributeError while
# the ballot is being built -- before any assertion in the test runs.
#
# One list, because there were six: this module and five hand-rolled fakes in
# test_regression_elo_all, test_invasion_elo_integration,
# test_regression_cmaes_elo_ballot, test_regression_round_robin_integration
# and test_new_operators. Adding C2UCB broke 21 tests across those files,
# none of them about C2UCB, and adding it to only some of them then surfaced
# kl_ducb, kl_swucb and cusum_ucb behind it -- the same omission three
# schedulers deep. test_regression_operator_env_covers_select_op parses the
# ballot out of `select_op` and fails by name when this list falls behind.
BALLOT_SCHEDULERS = (
    "c2ucb",
    "cmaes",
    "contextual",
    "cucb",
    "cusum_ucb",
    "ducb",
    "eps_greedy",
    "exp3",
    "fpl",
    "gp_ucb",
    "hierarchical",
    "kl_ducb",
    "kl_swucb",
    "mopt",
    "replicator",
    "round_robin",
    "swucb",
    "tang",
)


def install_scheduler_surface(obj, *, overwrite=False):
    """Give *obj* every ballot attribute, all off.

    ``overwrite=False`` leaves anything already set alone, so a fake that
    deliberately enables one scheduler can call this first and keep its own
    value. ``overwrite=True`` is for a test that pins how often the ballot is
    resolved: there the whole surface has to be pinned, not just the part the
    test cares about, or the ballot's contents vary with whatever the mock
    happens to build.

    Two notes carried over from the per-file lists this replaced, because
    they are the reason the surface has to be complete rather than
    convenient:

    - cmaes is on the ballot like every other scheduler, and the no-Elo
      fallback chain reads ``_use_cmaes`` unguarded. The fakes got away
      without it only while cmaes was *missing* from the ballot -- a bug
      fixed separately, which then broke every fake that had been relying
      on the omission.
    - invasion has no scheduler object of its own. It reads
      ``f.mc.bandit_stats()``, gated on ``_use_invasion`` and ``mc_bandit``
      alone, which is why those two are set here by name and not as a pair.
    """
    for sched in BALLOT_SCHEDULERS:
        for name in (f"_use_{sched}", f"_{sched}"):
            if overwrite or not hasattr(obj, name):
                setattr(obj, name, False if name.startswith("_use_") else None)
    for name, value in (
        ("_use_invasion", False),
        ("mc_bandit", False),
        ("_use_elo", False),
        ("_elo", None),
    ):
        if overwrite or not hasattr(obj, name):
            setattr(obj, name, value)
    return obj


def make_minimal_fuzzer(seed=None, pool=None):
    """Build a minimal fuzzer-like object for operator testing.

    Moved here verbatim from ``tests/test_new_operators.py`` so the
    exhaustive-enumeration harness can reuse it. It exists because
    ``OperatorEngine`` takes a whole ``Fuzzer``: ``test_operator_smoke.py``
    constructs a real one and therefore needs a compiled target binary on
    disk to call ``_op_bit_flip``. This mock is the 29-attribute shadow of
    that coupling, and it is what port item P1-4 exists to delete.

    Args:
        seed: Seed for a ``RandPool``. Left unseeded (the default, for
            tests that only assert type or length invariants) the pool is
            OS-seeded; tests that assert anything probabilistic must pass a
            seed so a failure is reproducible.
        pool: A pool object to install directly, taking precedence over
            ``seed``. This is the seam P1-5 needs: pass an
            ``ExhaustivePool`` and every operator that draws only bounded
            values becomes enumerable without touching the operator.
    """

    class _MockCorpus:
        _items = [b"AAAA", b"BBBB", b"CCCC", b"DDDD"]

        def __getitem__(self, idx):
            return self._items[idx]

        def __len__(self):
            return len(self._items)

    class _MockMarkov:
        order = 2

        def sample_byte(self, ctx):
            return 42

    class _MockMC:
        cem_fitted = False
        mc_bandit = False

    class _MockMI:
        def weighted_position(self, n):
            return None

    class _MockSensitivity:
        def get_weighted_position(self, data, n):
            return None

    class _MockElo: ...

    class _MockFrameshift:
        relations = []

    class _MockSeedMeta(dict):
        def get(self, key, default=None):
            return default

    class _MockCmplog:
        def __init__(self):
            self.pairs = []
            self.tokens = []

    class MinimalFuzzer:
        def __init__(self_):  # noqa: N805
            self_._cmplog = None
            self_._crash_mi = None
            self_._mi = _MockMI()
            self_._te = None
            self_._use_transfer_entropy = False
            self_._use_mi = False
            self_._sensitivity = _MockSensitivity()
            self_._elo = None
            self_._use_elo = False
            self_._replicator = None
            self_._use_replicator = False
            self_._mopt = None
            self_._use_mopt = False
            self_._prev_bandit_op = None
            self_._last_mopt_particles = []
            self_._last_ops_used = []
            # Mirrors Fuzzer.__init__: _apply_single_mutation reads both on
            # every call, and a mock missing them fails only inside havoc.
            self_._adaptive_havoc = True
            self_._last_havoc_subops = 0
            self_._meta_strategy = None
            self_._meta_strategy_cached = None
            self_._meta_strategy_used = set()
            self_._op_selector = None
            self_._stall_recovery_active = False
            self_._frameshift = _MockFrameshift()
            self_.markov = _MockMarkov()
            self_.markov_trained = False
            self_.mc = _MockMC()
            self_.mc_cem = False
            self_.grammar = None
            self_.dictionary = []
            self_.corpus = _MockCorpus()
            self_.max_len = 65536
            self_.seed_meta = _MockSeedMeta()
            self_.mutations_per_input = 1
            self_._wfc_enabled = False
            self_._smt_solver = None
            self_.enable_regex_bomb = False
            # Read on the mutate() path, which this mock did not previously
            # reach: test_operator_smoke and test_exhaustive_pool call the
            # _op_* handlers directly.
            self_._track_op_effect = False
            self_._det_execs = 0
            self_._op_time_ema = {}
            self_._last_op_costs = {}
            self_._op_attempts = {}
            self_._op_declines = {}
            self_._last_ops_applicable = []
            self_._last_ops_effective = []
            self_._last_ops_with_sites = []
            self_._current_context_shared = None
            self_._last_mutation_offset = -1
            self_._last_hamming_distance = -1
            self_._use_sensitivity = False
            install_scheduler_surface(self_)
            if pool is not None:
                self_._rng = pool
            else:
                from fuzzer_tool.core.rand_pool import RandPool

                self_._rng = RandPool(seed)
            self_._dict_scratch = []
            self_._dict_scratch_idx = 0

    return MinimalFuzzer()
