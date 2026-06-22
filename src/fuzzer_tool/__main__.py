"""CLI entry point for fuzzer-tool."""

from fuzzer_tool.cli.commands import main


def entry() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(entry())
