"""Wiring for the filesystem, TCP/IP, LDPC and turbo-code operators.

Each must be registered in the format band, offered only when its gate
holds (sniffer for images/packets, CLI option for codeword streams), and
change its input through the real dispatch table.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.cli.commands import _build_parser
from fuzzer_tool.core.mutations.ldpc_fix import parse_alist
from fuzzer_tool.core.operator_registry import REGISTRY, format_gate_matches
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support import ntfs_build, tcpip_build, zfs_build
from .support.ext_build import build_ext2, build_ext4
from .support.fat_build import Kind, build_fat
from .support.ldpc_codes import HAMMING_N, HAMMING_ROWS, alist_text
from .support.operator_env import make_minimal_fuzzer

FORMAT_OPS = {
    "fat_field_mutate": build_fat(Kind.FAT12),
    "ntfs_record_mutate": ntfs_build.build_image(),
    "zfs_label_mutate": zfs_build.fragment(),
    "ext2_field_mutate": build_ext2(),
    "ext4_field_mutate": build_ext4(),
    "tcpip_packet_mutate": tcpip_build.ipv4(tcpip_build.UDP, tcpip_build.udp_dgram(b"query")),
}
GATED_OPS = ("ldpc_fix", "turbo_fix")
NEW_OPS = (
    *FORMAT_OPS,
    *GATED_OPS,
)
CODEWORDS = bytes(64)  # all-zero words are valid LDPC and turbo codewords


def _fuzzer(seed: int = 5):
    f = make_minimal_fuzzer()
    f._rng = RandPool(seed=seed)
    f.max_len = 1 << 20
    return f


def _hamming():
    return parse_alist(alist_text(HAMMING_N, HAMMING_ROWS))


@pytest.mark.parametrize("name", NEW_OPS)
def test_registered_in_format_band(name):
    assert name in REGISTRY.names()
    assert REGISTRY.category_of(name) == "format"


@pytest.mark.parametrize("name", list(FORMAT_OPS))
def test_format_op_gated_by_sniffer(name):
    assert format_gate_matches(name, FORMAT_OPS[name]) is True
    assert format_gate_matches(name, b"plain text, no structure" * 8) is False


@pytest.mark.parametrize("name", list(FORMAT_OPS))
def test_format_op_dispatch_changes_input(name):
    f = _fuzzer()
    data = FORMAT_OPS[name]
    dispatch = REGISTRY.dispatch(OperatorEngine(f))
    out = dispatch[name](bytearray(data), 0, data)
    assert bytes(out) != data


def test_gated_ops_offered_only_with_option():
    """Fresh fuzzer per check: the availability plan is cached per instance."""
    live = REGISTRY.available(_fuzzer(), CODEWORDS)
    assert not set(GATED_OPS) & set(live)

    f = _fuzzer()
    f.ldpc_code = _hamming()
    f.turbo_k = 40
    live = REGISTRY.available(f, CODEWORDS)
    assert set(GATED_OPS) <= set(live)


@pytest.mark.parametrize("name", GATED_OPS)
def test_gated_dispatch_keeps_length(name):
    f = _fuzzer()
    f.ldpc_code = _hamming()
    f.turbo_k = 40
    dispatch = REGISTRY.dispatch(OperatorEngine(f))
    out = bytes(dispatch[name](bytearray(CODEWORDS), 0, CODEWORDS))
    assert out != CODEWORDS and len(out) == len(CODEWORDS)


def test_cli_parses_ldpc_and_turbo(tmp_path):
    alist = tmp_path / "h.alist"
    alist.write_text(alist_text(HAMMING_N, HAMMING_ROWS))
    args = _build_parser().root.parse_args(
        ["fuzz", "t", "--ldpc-alist", str(alist), "--turbo-k", "48", "--turbo-qpp", "7,12"]
    )
    assert args.ldpc_alist.n == HAMMING_N
    assert (args.turbo_k, args.turbo_qpp) == (48, (7, 12))


@pytest.mark.parametrize("argv", [["--ldpc-alist", "/nonexistent.alist"], ["--turbo-qpp", "x"]])
def test_cli_rejects_bad_ecc_options(argv):
    """Adversarial: a missing alist or malformed QPP is a usage error, not a crash."""
    with pytest.raises(SystemExit):
        _build_parser().root.parse_args(["fuzz", "t", *argv])
