"""Integration smoke test: every registered operator must fire without crashing.

This catches half-shipped features like gzip_chunk_mutate (registered in
MUTATIONS / dispatch table but module never created) that unit tests
miss because they never exercise the live dispatch path.
"""

import tempfile
from pathlib import Path

import pytest


def _build_fuzzer():
    """Build a minimal Fuzzer instance for operator dispatch testing."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus"
        crashes = Path(tmp) / "crashes"
        corpus.mkdir()
        crashes.mkdir()
        fuzzer = Fuzzer(
            target=str(Path(__file__).resolve().parent.parent / "targets" / "test_target"),
            corpus_dir=str(corpus),
            crashes_dir=str(crashes),
            max_len=4096,
        )
        return fuzzer


class TestOperatorDispatchSmoke:
    """Every operator in the dispatch table must execute without error."""

    def test_all_ops_fire(self):
        fuzzer = _build_fuzzer()
        dispatch = fuzzer._op_dispatch
        sample_buf = bytearray(b"\x00\x01\x02\x03\x04\x05\x06\x07" * 4)
        sample_data = bytes(sample_buf)

        failed = []
        for op_name, handler in dispatch.items():
            try:
                buf_copy = bytearray(sample_buf)
                result = handler(buf_copy, 0, sample_data)
                # Operators either mutate in-place (return None) or return new bytes
                if result is not None:
                    assert isinstance(result, (bytearray, bytes)), (
                        f"{op_name} returned {type(result).__name__}, expected bytearray/bytes"
                    )
            except Exception as exc:
                failed.append((op_name, exc))

        if failed:
            msg_lines = ["Operators that crashed:"]
            for name, exc in failed:
                msg_lines.append(f"  {name}: {type(exc).__name__}: {exc}")
            pytest.fail("\n".join(msg_lines))

    def test_dispatch_covers_mutations_list(self):
        """Every name in MUTATIONS must have a dispatch entry."""
        from fuzzer_tool.core.mutations import MUTATIONS

        fuzzer = _build_fuzzer()
        dispatch = fuzzer._op_dispatch
        missing = [op for op in MUTATIONS if op not in dispatch]
        assert not missing, f"Missing dispatch entries for: {missing}"

    def test_format_mutations_have_dispatch(self):
        """Every FORMAT_MUTATIONS entry must have a dispatch entry."""
        from fuzzer_tool.core.mutations import FORMAT_MUTATIONS

        fuzzer = _build_fuzzer()
        dispatch = fuzzer._op_dispatch
        missing = [op for op in FORMAT_MUTATIONS if op not in dispatch]
        assert not missing, f"Missing dispatch entries for format mutations: {missing}"

    def test_format_mutations_importable(self):
        """Every format mutation module must be importable."""
        from fuzzer_tool.core.mutations import FORMAT_MUTATIONS

        # Verify the modules behind lazy imports are actually loadable
        module_map = {
            "png_chunk_mutate": "fuzzer_tool.core.png_mutations",
            "jpeg_chunk_mutate": "fuzzer_tool.core.jpeg_mutations",
            "gzip_chunk_mutate": "fuzzer_tool.core.gzip_mutations",
            "bmp_chunk_mutate": "fuzzer_tool.core.bmp_mutations",
        }
        import importlib

        for op, mod_path in module_map.items():
            if op in FORMAT_MUTATIONS:
                mod = importlib.import_module(mod_path)
                assert hasattr(mod, "mutate") or any(
                    name.endswith("Mutator") for name in dir(mod)
                ), f"Module {mod_path} has no mutator class"
