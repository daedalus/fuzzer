"""Input-hash-keyed coverage detection (AntiFuzz §4.1).

AntiFuzz hashes the input and calls a hash-chosen chain of fake functions,
so any byte change yields "new" edges. Two signals expose it: one-byte tail
variants of a seed all land on distinct edge sets, and nearly every
execution asks for corpus admission.
"""

import zlib
from unittest.mock import patch

from fuzzer_tool.core.coverage_noise import (
    NOISE_PROBE_VARIANTS,
    AdmissionMonitor,
    NoiseVerdict,
    classify_noise,
    tail_variants,
)
from fuzzer_tool.services.fuzzer import Fuzzer

_REAL = {1, 2, 3}


def _hash_keyed(data: bytes) -> set[int]:
    # Model of AntiFuzz: real edges plus a chain picked by an input hash.
    h = zlib.crc32(data) & 0xFFFF
    return _REAL | {1000 + h, 1000 + (h * 7) % 65536}


def test_tail_variants_share_prefix_and_differ() -> None:
    seed = b"PNG\x00body"
    variants = tail_variants(seed, NOISE_PROBE_VARIANTS)

    assert len(variants) == NOISE_PROBE_VARIANTS
    assert all(v[:-1] == seed for v in variants)
    assert len(set(variants)) == NOISE_PROBE_VARIANTS


def test_hash_keyed_coverage_suspected() -> None:
    seed = b"seed"
    base = _hash_keyed(seed)
    sets = [_hash_keyed(v) for v in tail_variants(seed, NOISE_PROBE_VARIANTS)]

    assert classify_noise([base, base], sets) is NoiseVerdict.SUSPECTED


def test_trailing_garbage_ignored_is_clean() -> None:
    # Falsification: a parser that ignores trailing bytes is not noise.
    sets = [set(_REAL) for _ in range(NOISE_PROBE_VARIANTS)]

    assert classify_noise([_REAL, _REAL], sets) is NoiseVerdict.CLEAN


def test_one_semantic_tail_branch_is_clean() -> None:
    # Falsification: a zero/non-zero check on the tail byte gives 2 sets, not N.
    sets = [_REAL | {9}] * NOISE_PROBE_VARIANTS

    assert classify_noise([_REAL, _REAL], sets) is NoiseVerdict.CLEAN


def test_unstable_base_abstains() -> None:
    # Adversarial: a nondeterministic target diverges on its own; the
    # probe must not blame the input hash for it.
    sets = [{i} for i in range(NOISE_PROBE_VARIANTS)]

    assert classify_noise([{1}, {2}], sets) is NoiseVerdict.UNMEASURED


def test_empty_measurement_abstains() -> None:
    assert classify_noise([], []) is NoiseVerdict.UNMEASURED
    assert classify_noise([_REAL, _REAL], []) is NoiseVerdict.UNMEASURED


def test_admission_monitor_fires_once_on_flood() -> None:
    monitor = AdmissionMonitor()
    monitor.observe(execs=100, admissions=5)  # baseline

    span = AdmissionMonitor.MIN_EXECS
    assert monitor.observe(execs=100 + span, admissions=5 + span)
    assert not monitor.observe(execs=100 + 2 * span, admissions=5 + 2 * span)


def test_admission_monitor_quiet_on_normal_rate() -> None:
    # Falsification: a healthy campaign admits a small fraction of execs.
    monitor = AdmissionMonitor()
    monitor.observe(execs=0, admissions=0)
    span = AdmissionMonitor.MIN_EXECS * 10

    assert not monitor.observe(execs=span, admissions=span // 100)


def test_admission_monitor_waits_for_min_execs() -> None:
    # Adversarial: the first few execs of a campaign admit almost everything.
    monitor = AdmissionMonitor()
    monitor.observe(execs=0, admissions=0)
    early = AdmissionMonitor.MIN_EXECS - 1

    assert not monitor.observe(execs=early, admissions=early)


class _Shm:
    """SHM stand-in reporting the edge set of the last executed input."""

    def __init__(self) -> None:
        self.last = b""

    def get_edge_ids(self) -> set[int]:
        return _hash_keyed(self.last)

    def dropped_edges_delta(self) -> int:
        return 0

    def read_path_hash(self) -> int:
        return 0


def _fuzzer(tmp_path) -> Fuzzer:
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=str(tmp_path / "corpus"),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=256,
            timeout=1,
        )


def test_regression_calibration_reports_hash_keyed_noise(tmp_path, capsys) -> None:
    f = _fuzzer(tmp_path)
    shm = _Shm()
    f.shm_cov = shm

    def run(data: bytes):
        shm.last = data
        return 0, ""

    f._run_target = run
    f._report_coverage_noise(b"seed")

    assert f._coverage_noise is NoiseVerdict.SUSPECTED
    assert "input-hash-keyed" in capsys.readouterr().out


def test_admission_flood_warns(tmp_path, capsys) -> None:
    f = _fuzzer(tmp_path)
    f._check_admission_rate()  # baseline at 0/0

    f.exec_count = AdmissionMonitor.MIN_EXECS
    f._total_corpus_attempts = AdmissionMonitor.MIN_EXECS
    f._check_admission_rate()

    assert "corpus admission" in capsys.readouterr().out
