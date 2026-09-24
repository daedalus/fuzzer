"""Regression: the --schedule-ablation CSV had no operator column, so
per-operator reward vs own-pull-count (fatigue) was unmeasurable
(handover_non_ucb_schedulers_2026-09-13.md §5 step 1)."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer

_SIGNALS = {
    "seed_idx": 0,
    "seed_hash": "41414141",
    "fuzz_count": 3,
    "coverage_edges": 7,
    "age_s": "1.0",
    "temperature": "1.000",
    "base_w": "1.0000",
    "burst": "1.00",
    "penalty": "1.00",
    "subsumption": "1.0000",
    "diversity": "1.0000",
    "spatial": "1.0000",
    "mdl": "1.00",
    "final_w": "1.000000",
}


@pytest.fixture
def fuzzer(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "seed_a").write_bytes(b"AAAA")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        f = Fuzzer(
            target="/bin/true",
            corpus_dir=str(corpus),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=16,
            timeout=1,
            quiet_stats=True,
            schedule_ablation=str(tmp_path / "ablation.csv"),
        )
    yield f
    if f._ablation_file:
        f._ablation_file.close()


def _rows(f) -> list[dict[str, str]]:
    f._ablation_file.flush()
    with open(f._ablation_path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_regression_ablation_csv_has_operator_column(fuzzer):
    fuzzer._last_pick_signals = dict(_SIGNALS)
    fuzzer._last_ops_used = ["bit_flip", "havoc"]
    fuzzer._write_ablation_row(True, False)

    (row,) = _rows(fuzzer)
    assert row["operator"] == "bit_flip+havoc"
    assert row["new_coverage"] == "1"
    assert None not in row  # header and row agree on field count


def test_ablation_row_without_ops(fuzzer):
    """Adversarial: a round with no recorded op writes an empty cell."""
    fuzzer._last_pick_signals = dict(_SIGNALS)
    fuzzer._last_ops_used = []
    fuzzer._write_ablation_row(False, True)

    (row,) = _rows(fuzzer)
    assert row["operator"] == ""
    assert row["new_crash"] == "1"
