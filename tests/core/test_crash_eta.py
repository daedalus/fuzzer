"""Tests for crash ETA estimation."""

from fuzzer_tool.core.crash_eta import CrashETA, estimate_risky_density
from fuzzer_tool.core.target_profiler import FunctionInfo, TargetProfile


def _empty_profile() -> TargetProfile:
    return TargetProfile(
        rodata_strings=[],
        interesting_strings=[],
        magic_bytes=[],
        functions={},
        hot_functions=[],
        entry_points=[],
        input_parsers=[],
        boundary_markers=[],
        format_signature=None,
        call_graph={},
        reverse_calls={},
    )


def test_estimate_risky_density_empty_profile():
    density = estimate_risky_density(_empty_profile())
    assert density == 0.0


def test_estimate_risky_density_with_error_keywords():
    profile = TargetProfile(
        rodata_strings=[(0x1000, "error: buffer overflow"), (0x2000, "invalid argument")],
        interesting_strings=[],
        magic_bytes=[],
        functions={
            "safe_func": FunctionInfo(addr=0x100, size=50, name="safe_func"),
            "error_handler": FunctionInfo(addr=0x200, size=30, name="error_handler"),
            "overflow_check": FunctionInfo(addr=0x300, size=40, name="overflow_check"),
        },
        hot_functions=[],
        entry_points=[],
        input_parsers=[],
        boundary_markers=[],
        format_signature=None,
        call_graph={},
        reverse_calls={},
    )
    density = estimate_risky_density(profile)
    assert density > 0.0
    assert density <= 1.0


def test_crash_eta_dataclass():
    eta = CrashETA(
        point_est=5000,
        low=1000,
        high=20000,
        confidence="low",
        reasoning="test",
    )
    assert eta.point_est == 5000
    assert eta.low < eta.point_est < eta.high


def test_estimate_risky_density_no_error_keywords():
    profile = TargetProfile(
        rodata_strings=[(0x1000, "hello world"), (0x2000, "foo bar")],
        interesting_strings=[],
        magic_bytes=[],
        functions={
            "init": FunctionInfo(addr=0x100, size=50, name="init"),
            "run": FunctionInfo(addr=0x200, size=30, name="run"),
        },
        hot_functions=[],
        entry_points=[],
        input_parsers=[],
        boundary_markers=[],
        format_signature=None,
        call_graph={},
        reverse_calls={},
    )
    density = estimate_risky_density(profile)
    assert density == 0.0


def test_estimate_risky_density_clamped():
    """Density should be clamped to 1.0 when all functions match."""
    profile = TargetProfile(
        rodata_strings=[],
        interesting_strings=[],
        magic_bytes=[],
        functions={
            "error_a": FunctionInfo(addr=0x100, size=50, name="error_a"),
            "invalid_b": FunctionInfo(addr=0x200, size=30, name="invalid_b"),
        },
        hot_functions=[],
        entry_points=[],
        input_parsers=[],
        boundary_markers=[],
        format_signature=None,
        call_graph={},
        reverse_calls={},
    )
    density = estimate_risky_density(profile)
    assert density == 1.0


def test_estimate_execs_basic():
    from fuzzer_tool.core.crash_eta import estimate_execs_to_first_crash

    profile = TargetProfile(
        rodata_strings=[(0x1000, "error handler")],
        interesting_strings=[],
        magic_bytes=[],
        functions={
            "main": FunctionInfo(addr=0x100, size=50, name="main"),
            "parse_input": FunctionInfo(addr=0x200, size=30, name="parse_input"),
            "error_check": FunctionInfo(addr=0x300, size=40, name="error_check"),
        },
        hot_functions=[],
        entry_points=[],
        input_parsers=[],
        boundary_markers=[],
        format_signature=None,
        call_graph={},
        reverse_calls={},
    )
    gt = {"n": 100, "n1": 10, "n2": 5, "confidence": "medium"}
    discovery = 5.0  # 5 edges per 1000 execs

    eta = estimate_execs_to_first_crash(profile, gt, discovery)
    assert eta.point_est > 0
    assert eta.low > 0
    assert eta.high >= eta.point_est
    assert eta.confidence in ("low", "medium", "high")


def test_estimate_execs_zero_density():
    from fuzzer_tool.core.crash_eta import estimate_execs_to_first_crash

    profile = TargetProfile(
        rodata_strings=[],
        interesting_strings=[],
        magic_bytes=[],
        functions={
            "main": FunctionInfo(addr=0x100, size=50, name="main"),
        },
        hot_functions=[],
        entry_points=[],
        input_parsers=[],
        boundary_markers=[],
        format_signature=None,
        call_graph={},
        reverse_calls={},
    )
    gt = {"n": 50, "n1": 2, "n2": 1, "confidence": "low"}
    discovery = 10.0

    eta = estimate_execs_to_first_crash(profile, gt, discovery)
    # Zero density -> infinity -> capped at large value
    assert eta.point_est > 1_000_000
