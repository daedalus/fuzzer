"""``_modal_value`` must stay a plurality vote, not a strict majority.

Corrupt pairs scatter over ``[0, N)`` while honest pairs share one offset, so
the honest value wins on plurality even when it is a minority. Verification
(``min_matches``) then accepts it. Boyer–Moore majority returns nothing once
corrupt pairs reach half: measured 0/200 vs 200/200 at 8..12 bad of 16.
"""

from fuzzer_tool.core.int_checksum_solver import _modal_value

INIT = 1234
HONEST = 6
CORRUPT = 10


def test_regression_minority_honest_value_wins():
    values = [INIT] * HONEST + [INIT + 1 + i for i in range(CORRUPT)]
    assert len(values) > HONEST * 2
    assert _modal_value(values) == INIT


def test_adversarial_corrupt_values_first():
    """Order must not matter: corrupt pairs lead the stream."""
    values = [INIT + 1 + i for i in range(CORRUPT)] + [INIT] * HONEST
    assert _modal_value(values) == INIT


def test_falsify_no_repeat_returns_a_member():
    """All distinct: any member is returned; verification rejects it later."""
    values = [7, 8, 9]
    assert _modal_value(values) in values
