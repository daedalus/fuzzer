"""Mutation operators and dictionary handling."""

INTERESTING_8 = [0, 1, 0x7F, 0x80, 0xFF]
INTERESTING_16 = [0x7FFF, 0x8000, 0xFFFF, 0, 1]
INTERESTING_32 = [0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0, 1]

MUTATIONS = [
    "bit_flip",
    "byte_flip",
    "interesting_8",
    "interesting_16",
    "interesting_32",
    "random_bytes",
    "block_insert",
    "block_delete",
    "block_duplicate",
    "havoc",
]

DICT_MUTATIONS = [
    "dict_insert",
    "dict_replace",
]


def parse_dict_line(line: str) -> bytes | None:
    """Parse a single dictionary line.

    Args:
        line: Raw line from dictionary file.

    Returns:
        Parsed token bytes, or None if line is empty/comment.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("=", 1)
    token = parts[-1] if len(parts) == 2 else line
    return token.encode("raw_unicode_escape").decode("unicode_escape").encode("latin-1")


def load_dictionary(path: str) -> list[bytes]:
    """Load tokens from a dictionary file.

    Args:
        path: Path to dictionary file.

    Returns:
        List of token byte sequences.

    Raises:
        FileNotFoundError: If dictionary file does not exist.
    """
    d = []
    with open(path, errors="replace") as f:
        for line in f:
            tok = parse_dict_line(line)
            if tok is not None:
                d.append(tok)
    return d
