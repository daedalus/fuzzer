"""Regression: differential stderr mismatch was described but never flagged.

``services/differential.py``'s module docstring states the per-input contract as
an exact match on returncode, sanitizer report AND stderr. The stderr branch
appended ``"different stderr output"`` to ``reasons`` but left ``diverged``
False, so ``diff_run`` returned ``(False, "different stderr output")`` -- a
divergence, fully described, reported as a match. Every caller keys off the
boolean, so a target pair that differed only in stderr never registered.

The fix flags it, but only when neither side produced a valid sanitizer report.
When both crashed with the SAME error_type the branches above have already
adjudicated them as matching, and their stderr still differs in every run by
allocation addresses, pids and thread ids -- flagging on that would report a
divergence for every identical crash pair, which is worse than the original
bug.
"""

from unittest.mock import patch

from fuzzer_tool.services.differential import diff_run

ASAN_A = (
    "==1234==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000010\n"
    "    #0 0x4f1234 in main /src/a.c:10\n"
)
ASAN_B = (
    "==5678==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000998\n"
    "    #0 0x4f9876 in main /src/a.c:10\n"
)
ASAN_UAF = (
    "==5678==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000998\n"
    "    #0 0x4f9876 in main /src/a.c:22\n"
)


def _run(stderr_a, stderr_b, rc_a=0, rc_b=0):
    with patch(
        "fuzzer_tool.services.differential.run_target_stdin",
        side_effect=[(rc_a, stderr_a), (rc_b, stderr_b)],
    ):
        return diff_run("/bin/true", "/bin/false", b"input")


class TestStderrDivergenceIsFlagged:
    def test_plain_stderr_mismatch_diverges(self):
        diverged, desc = _run("hello\n", "goodbye\n")
        assert diverged is True
        assert "stderr" in desc

    def test_one_side_silent_diverges(self):
        diverged, desc = _run("", "unexpected warning\n")
        assert diverged is True
        assert "stderr" in desc

    def test_identical_stderr_does_not_diverge(self):
        diverged, desc = _run("same\n", "same\n")
        assert diverged is False
        assert desc == "identical"

    def test_both_silent_does_not_diverge(self):
        diverged, desc = _run("", "")
        assert diverged is False
        assert desc == "identical"


class TestFalsification:
    def test_reason_and_boolean_agree(self):
        # Falsification of the exact defect: the returned description said
        # "different stderr output" while the boolean said "no divergence".
        # Those two can never disagree again.
        diverged, desc = _run("a\n", "b\n")
        assert not (desc != "identical" and diverged is False), (
            f"described a divergence ({desc!r}) but reported diverged={diverged}"
        )

    def test_matching_sanitizer_reports_do_not_diverge_on_address_noise(self):
        # The regression the naive fix would introduce. Same error type, same
        # returncode, stderr differing only in pid/address/frame -- these are
        # the same crash and must NOT be reported as a divergence.
        diverged, desc = _run(ASAN_A, ASAN_B, rc_a=1, rc_b=1)
        assert diverged is False, f"address noise flagged as divergence: {desc}"

    def test_differing_sanitizer_reports_still_diverge(self):
        # And the carve-out must not suppress a real sanitizer divergence.
        diverged, desc = _run(ASAN_A, ASAN_UAF, rc_a=1, rc_b=1)
        assert diverged is True
        assert "heap-buffer-overflow" in desc and "heap-use-after-free" in desc


class TestAdversarial:
    def test_returncode_divergence_still_takes_precedence(self):
        # rc differs AND stderr differs: one divergence, both reasons, and the
        # stderr clause must not double-append.
        diverged, desc = _run("a\n", "b\n", rc_a=0, rc_b=1)
        assert diverged is True
        assert desc.count("different stderr output") <= 1
        assert "returncode" in desc

    def test_one_side_crashes_other_clean(self):
        diverged, desc = _run(ASAN_A, "", rc_a=1, rc_b=0)
        assert diverged is True

    def test_whitespace_only_difference_still_counts(self):
        # Adversarial: a trailing-newline-only difference is still a byte-level
        # mismatch under the stated contract. Asserting the actual behaviour
        # rather than assuming it is normalised away.
        diverged, _desc = _run("out", "out\n")
        assert diverged is True

    def test_large_stderr_does_not_change_semantics(self):
        big_a = "x" * 100_000
        big_b = "x" * 99_999 + "y"
        diverged, _ = _run(big_a, big_b)
        assert diverged is True
