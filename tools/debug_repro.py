#!/usr/bin/env python3
"""Debug reproducibility: run two Fuzzers with identical seeds in fresh temp dirs."""
import sys, os, tempfile, subprocess

script = """
import sys
sys.path.insert(0, '.')
import numpy as np
from fuzzer_tool.services.fuzzer import Fuzzer
from unittest.mock import patch
import random as _random

td = sys.argv[1]
with patch('os.path.isfile', return_value=True), patch('os.access', return_value=True):
    f = Fuzzer(target='/bin/true', corpus_dir=td + '/corpus', crashes_dir=td + '/crashes', max_len=256, timeout=1, mutations_per_input=2)
_random.seed(42)
for r in [f.mutate(b'AAAA') for _ in range(10)]:
    print(r.hex())
"""

results = []
for i in range(2):
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, '-c', script, td], capture_output=True, text=True, cwd='/home/dclavijo/my_code/fuzzer')
        hexes = [line.strip() for line in r.stdout.strip().split()]
        results.append(hexes)
        print(f'Run {i}: {[h[:16] for h in hexes]}')

match = results[0] == results[1]
print(f'\nMatch: {match}')
if not match:
    for i, (a, b) in enumerate(zip(results[0], results[1])):
        if a != b:
            print(f'  [{i}] {a[:20]} != {b[:20]}')
