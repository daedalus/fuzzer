"""Regression: strict wfc_reorder_chunks retries a collapse that reproduces the parent.

asf returned its input unchanged on ~5% of calls (a wasted exec), sitting on
the handover's 95%-changed bar; seed 2 dropped to 187/200 when the RandPool
stream moved.
"""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.wfc_chunks import ASF_FORMAT, WfcChunkTableStore, wfc_reorder_chunks
from tests.test_wfc_chunks import _train, _try_asf, asf_sample

_CALLS = 200
_SEEDS = range(5)


def _changed(table, seed: int) -> int:
    data = asf_sample()
    chunks = ASF_FORMAT.parse(data)
    rng = RandPool(seed=seed)
    outs = [
        wfc_reorder_chunks(ASF_FORMAT, chunks, table, rng, mode="strict") for _ in range(_CALLS)
    ]
    assert all(_try_asf(o) is not None for o in outs)
    return sum(o != data for o in outs)


def test_regression_trained_asf_never_returns_parent():
    """Falsification: a trained table has other legal orders; every call uses one."""
    store = WfcChunkTableStore()
    _train(store, ASF_FORMAT, asf_sample())
    table = store.table_for("asf")

    assert [_changed(table, s) for s in _SEEDS] == [_CALLS] * len(_SEEDS)


def test_parent_only_table_keeps_parent():
    """Adversarial: the parent is the only legal order; it is returned, not broken."""
    store = WfcChunkTableStore()
    store.observe(ASF_FORMAT, asf_sample())
    table = store.table_for("asf")

    assert _changed(table, 0) == 0
