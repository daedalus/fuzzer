"""Tests for core/format_generators.py -- the fmt-name -> seed-builder registry."""

from fuzzer_tool.core.format_generators import FORMATS, GENERATED_SEED_FORMATS, generate
from fuzzer_tool.core.minimal_seeds import MINIMAL_SEEDS
from fuzzer_tool.core.rand_pool import RandPool


class TestFormatsUnion:
    def test_formats_is_union_of_both_tables(self):
        assert set(FORMATS) == set(MINIMAL_SEEDS) | set(GENERATED_SEED_FORMATS)

    def test_no_overlap_between_the_two_tables(self):
        # Documented in generate()'s docstring as "MINIMAL_SEEDS wins on
        # overlap" -- true today only because there is no overlap. If this
        # ever fires, generate()'s priority comment needs to name the
        # formats it actually applies to.
        assert set(MINIMAL_SEEDS).isdisjoint(GENERATED_SEED_FORMATS)

    def test_sorted(self):
        assert list(FORMATS) == sorted(FORMATS)


class TestGenerateUnknownFormat:
    def test_returns_none(self):
        assert generate("not-a-real-format") is None

    def test_empty_string(self):
        assert generate("") is None


class TestGenerateEveryFormat:
    """Falsification: every registered format must produce non-empty bytes."""

    def test_every_format_produces_nonempty_bytes(self):
        rng = RandPool(seed=1234)
        for fmt in FORMATS:
            data = generate(fmt, max_len=2048, rng=rng)
            assert isinstance(data, (bytes, bytearray)), fmt
            assert len(data) > 0, fmt

    def test_every_generated_format_produces_nonempty_bytes_with_no_rng(self):
        # generate() must work standalone (rng=None) -- mutators fall back
        # to their own freshly-constructed RandPool in that case.
        for fmt in GENERATED_SEED_FORMATS:
            data = generate(fmt, max_len=2048, rng=None)
            assert isinstance(data, (bytes, bytearray)), fmt
            assert len(data) > 0, fmt


class TestMaxLenRespected:
    def test_minimal_seed_truncated_to_max_len(self):
        full = MINIMAL_SEEDS["png"]()
        assert len(full) > 4
        truncated = generate("png", max_len=4)
        assert truncated == full[:4]

    def test_max_len_zero_returns_full_minimal_seed(self):
        # Matches SeedPicker._format_aware_seed's own "limit > 0" guard.
        full = MINIMAL_SEEDS["gzip"]()
        assert generate("gzip", max_len=0) == full


class TestDeterminism:
    def test_generated_format_deterministic_for_same_seed(self):
        a = generate("flac", max_len=1024, rng=RandPool(seed=99))
        b = generate("flac", max_len=1024, rng=RandPool(seed=99))
        assert a == b

    def test_generated_format_varies_with_different_seed(self):
        a = generate("flac", max_len=1024, rng=RandPool(seed=1))
        b = generate("flac", max_len=1024, rng=RandPool(seed=2))
        assert a != b

    def test_minimal_seed_format_ignores_rng(self):
        # Constant builders: same output regardless of rng/seed.
        a = generate("png", rng=RandPool(seed=1))
        b = generate("png", rng=RandPool(seed=2))
        assert a == b


class TestPreviouslyUnreachableFormatsNowNamed:
    """The 24 generators that had no cold-start path before this module."""

    NEWLY_NAMED = (
        "adts",
        "arm",
        "asf",
        "av1",
        "avif",
        "cfhd",
        "der",
        "dvbsub",
        "ffconcat",
        "flac",
        "flv",
        "isobmff",
        "jpeg2000",
        "magicyuv",
        "mp3",
        "mpegts",
        "nal",
        "ogg",
        "pgs",
        "rasc",
        "shorten",
        "sqlite",
        "tiff",
        "x86",
    )

    def test_all_24_present_in_table(self):
        assert set(self.NEWLY_NAMED) <= set(GENERATED_SEED_FORMATS)

    def test_all_24_generate_successfully(self):
        rng = RandPool(seed=5)
        for fmt in self.NEWLY_NAMED:
            data = generate(fmt, max_len=4096, rng=rng)
            assert data, fmt
