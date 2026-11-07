"""ASAN SEGV inside its own deallocator is a bad free.

fuzzgoat's ``""`` payload frees ``ptr - 1``. ASAN reports ``bad-free`` when
the chunk header there is readable and ``SEGV`` in
``__asan::Allocator::Deallocate`` when it is not: one root cause, two
signatures, two saved buckets.
"""

from fuzzer_tool.core.sanitizer import SanitizerReport

_HEAD = "==1==ERROR: AddressSanitizer: SEGV on unknown address 0x501fffffffff (pc 0x5c6 bp 0x0 sp 0x7f00 T0)\n"
_TAIL = (
    "    #1 0x55cfa63adadf in free (/b/fuzzgoat_read+0xcbadf)\n"
    "    #2 0x55cfa64258f4 in default_free /fuzz/vendor/fuzzgoat/fuzzgoat.c:85:4\n"
    "    #3 0x55cfa640ce61 in fuzz_shm_run /fuzz/src/targets/fuzzgoat_read.c:171:13\n"
)
_DEALLOC = (
    "    #0 0x55cfa63152c6 in __asan::Allocator::Deallocate(void*, unsigned long, unsigned long, "
    "__sanitizer::BufferedStackTrace*, __asan::AllocType) (/b/fuzzgoat_read+0x332c6)\n"
)
_BAD_FREE = (
    "==1==ERROR: AddressSanitizer: attempting free on address which was not malloc()-ed: "
    "0x502000000010 in thread T0\n"
    "    #0 0x55cfa63adadf in free (/b/fuzzgoat_read+0xcbadf)\n"
    "    #1 0x55cfa64258f4 in default_free /fuzz/vendor/fuzzgoat/fuzzgoat.c:85:4\n"
    "    #2 0x55cfa640ce61 in fuzz_shm_run /fuzz/src/targets/fuzzgoat_read.c:171:13\n"
)


def test_regression_dealloc_segv_is_bad_free():
    rep = SanitizerReport.parse(_HEAD + _DEALLOC + _TAIL)
    assert rep.error_type == "bad-free"


def test_dealloc_segv_matches_bad_free_signature():
    """Both ASAN spellings of the bug land in one bucket."""
    segv = SanitizerReport.parse(_HEAD + _DEALLOC + _TAIL)
    bad = SanitizerReport.parse(_BAD_FREE)
    assert segv.signature == bad.signature


def test_target_segv_stays_segv():
    """Adversarial: a SEGV in target code (free only lower in the stack) is not relabelled."""
    target_top = (
        "    #0 0x55cfa64258f4 in default_free /fuzz/vendor/fuzzgoat/fuzzgoat.c:85:4\n" + _TAIL
    )
    assert SanitizerReport.parse(_HEAD + target_top).error_type == "SEGV"
