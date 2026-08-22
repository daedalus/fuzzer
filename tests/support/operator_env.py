"""Shared scaffolding for operator-level tests.

``OperatorEngine`` is constructed from a ``Fuzzer``, so exercising a single
byte-level operator nominally requires a full fuzzer -- and
``test_operator_smoke.py`` does exactly that, compiling a target binary to
reach ``_op_bit_flip``. The mock below is the cheap substitute two test
suites now share rather than duplicate.

Its size is the argument for port item P1-4: 29 attributes, mostly None or
False, all of them present only because the operator's declared contract is
the whole ``Fuzzer`` object. See ``docs/tigerbeetle_four_fuzzers_port.md``.
"""

from __future__ import annotations


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
            if pool is not None:
                self_._rand_pool = pool
            else:
                from fuzzer_tool.core.rand_pool import RandPool

                self_._rand_pool = RandPool(seed)
            self_._dict_scratch = []
            self_._dict_scratch_idx = 0

    return MinimalFuzzer()
