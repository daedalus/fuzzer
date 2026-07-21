#!/usr/bin/env python3
"""Wrapper to run fuzzer-tool fuzz under memray with ASAN."""
import os
import sys
sys.argv = [
    'fuzzer-tool', 'fuzz', 'targets/fuzz_regex_compile.so',
    '-d', '/tmp/fgrep_test_1', '-c', '-n', '100000',
]
from fuzzer_tool.cli.commands import main
main()
