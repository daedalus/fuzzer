#!/usr/bin/env python3
"""Coverage-guided binary fuzzer."""

import argparse
import hashlib
import os
import random
import signal
import subprocess
import struct
import sys
import time
from pathlib import Path


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
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("=", 1)
    token = parts[-1] if len(parts) == 2 else line
    return token.encode("raw_unicode_escape").decode("unicode_escape").encode("latin-1")


def load_dictionary(path: str) -> list[bytes]:
    d = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            tok = parse_dict_line(line)
            if tok is not None:
                d.append(tok)
    return d


class Fuzzer:
    def __init__(self, target, corpus_dir, crashes_dir, max_len=4096,
                 timeout=5, mutations_per_input=8, use_coverage=False,
                 dictionary=None):
        self.target = target
        self.corpus_dir = Path(corpus_dir)
        self.crashes_dir = Path(crashes_dir)
        self.max_len = max_len
        self.timeout = timeout
        self.mutations_per_input = mutations_per_input
        self.use_coverage = use_coverage
        self.dictionary = dictionary or []

        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir.mkdir(parents=True, exist_ok=True)

        self.corpus: list[bytes] = []
        self.seen_hashes: set[str] = set()
        self.crash_hashes: set[str] = set()
        self.exec_count = 0
        self.crash_count = 0
        self.start_time = time.time()

        self._load_corpus()

    def _hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    def _load_corpus(self):
        for f in self.corpus_dir.iterdir():
            if f.is_file():
                data = f.read_bytes()
                h = self._hash(data)
                if h not in self.seen_hashes:
                    self.seen_hashes.add(h)
                    self.corpus.append(data)
        if not self.corpus:
            self.corpus.append(b"AAAAAAAA")

    def _run_target(self, data: bytes) -> tuple[int, str]:
        """Execute target, return (returncode, stderr)."""
        try:
            env = os.environ.copy()
            if self.use_coverage:
                env["AFL_MAP_SIZE"] = "65536"

            proc = subprocess.Popen(
                [self.target],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid,
            )
            try:
                _, stderr = proc.communicate(input=data, timeout=self.timeout)
                return proc.returncode, stderr.decode(errors="replace")
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                return -1, "timeout"
        except Exception as e:
            return -2, str(e)

    SIGNAL_CRASH_CODES = {134, 135, 136, 139}  # SIGABRT, SIGBUS, SIGFPE, SIGSEGV

    def _is_interesting(self, returncode: int, stderr: str) -> bool:
        if returncode in self.SIGNAL_CRASH_CODES:
            return True
        if returncode < 0 and returncode != -1:
            return True
        if returncode in (-1, 0) and "ASAN" in stderr:
            return True
        if "Segmentation fault" in stderr:
            return True
        if "Aborted" in stderr:
            return True
        return False

    def _is_crash(self, returncode: int, stderr: str) -> bool:
        if returncode == -2:
            return False
        if returncode in self.SIGNAL_CRASH_CODES:
            return True
        if returncode < 0:
            return True
        if any(sig in stderr for sig in ["SIGSEGV", "SIGABRT", "SIGFPE", "SIGBUS",
                                         "Segmentation fault", "Aborted",
                                         "AddressSanitizer", "heap-buffer-overflow",
                                         "stack-buffer-overflow", "use-after-free"]):
            return True
        return False

    def mutate(self, data: bytes) -> bytes:
        buf = bytearray(data)
        if not buf:
            buf = bytearray(b"\x00" * random.randint(1, 32))

        ops = list(MUTATIONS)
        if self.dictionary:
            ops.extend(DICT_MUTATIONS)

        for _ in range(self.mutations_per_input):
            op = random.choice(ops)

            if op == "bit_flip" and buf:
                byte_idx = random.randint(0, len(buf) - 1)
                bit_idx = random.randint(0, 7)
                buf[byte_idx] ^= (1 << bit_idx)

            elif op == "byte_flip" and buf:
                byte_idx = random.randint(0, len(buf) - 1)
                buf[byte_idx] ^= 0xFF

            elif op == "interesting_8" and buf:
                idx = random.randint(0, len(buf) - 1)
                buf[idx] = random.choice(INTERESTING_8)

            elif op == "interesting_16" and len(buf) >= 2:
                idx = random.randint(0, len(buf) - 2)
                val = random.choice(INTERESTING_16)
                struct.pack_into("<H", buf, idx, val)

            elif op == "interesting_32" and len(buf) >= 4:
                idx = random.randint(0, len(buf) - 4)
                val = random.choice(INTERESTING_32)
                struct.pack_into("<I", buf, idx, val)

            elif op == "random_bytes" and buf:
                idx = random.randint(0, len(buf) - 1)
                buf[idx] = random.randint(0, 255)

            elif op == "block_insert" and len(buf) < self.max_len:
                idx = random.randint(0, len(buf))
                size = random.randint(1, min(32, self.max_len - len(buf)))
                buf[idx:idx] = bytes(random.randint(0, 255) for _ in range(size))

            elif op == "block_delete" and len(buf) > 1:
                idx = random.randint(0, len(buf) - 1)
                max_size = min(32, len(buf) - idx, len(buf) - 1)
                if max_size >= 1:
                    size = random.randint(1, max_size)
                    del buf[idx:idx + size]

            elif op == "block_duplicate" and len(buf) < self.max_len:
                idx = random.randint(0, len(buf) - 1)
                size = random.randint(1, min(16, len(buf) - idx))
                block = buf[idx:idx + size]
                ins = random.randint(0, len(buf))
                buf[ins:ins] = block

            elif op == "dict_insert" and self.dictionary:
                token = random.choice(self.dictionary)
                if len(buf) + len(token) <= self.max_len:
                    idx = random.randint(0, len(buf))
                    buf[idx:idx] = token

            elif op == "dict_replace" and self.dictionary and buf:
                token = random.choice(self.dictionary)
                idx = random.randint(0, len(buf) - 1)
                end = min(idx + len(token), len(buf))
                buf[idx:end] = token[:end - idx]

            elif op == "havoc":
                return bytes(self._havoc_mutate(buf))

        return bytes(buf)

    def _havoc_mutate(self, buf: bytearray) -> bytearray:
        for _ in range(random.randint(2, 8)):
            self._apply_single_mutation(buf)
        return buf

    def _apply_single_mutation(self, buf: bytearray):
        if not buf:
            buf.extend(random.randint(0, 255) for _ in range(random.randint(1, 16)))
            return
        op = random.randint(0, 4)
        if op == 0:
            idx = random.randint(0, len(buf) - 1)
            buf[idx] ^= (1 << random.randint(0, 7))
        elif op == 1:
            idx = random.randint(0, len(buf) - 1)
            buf[idx] = random.randint(0, 255)
        elif op == 2 and len(buf) > 1:
            i, j = random.sample(range(len(buf)), 2)
            buf[i], buf[j] = buf[j], buf[i]
        elif op == 3 and len(buf) < self.max_len:
            idx = random.randint(0, len(buf))
            buf.insert(idx, random.randint(0, 255))
        elif op == 4 and len(buf) > 1:
            idx = random.randint(0, len(buf) - 1)
            size = random.randint(1, min(len(buf) - 1, len(buf) - idx))
            del buf[idx:idx + size]

    def save_crash(self, data: bytes, returncode: int, stderr: str):
        h = self._hash(data)
        if h in self.crash_hashes:
            return
        self.crash_hashes.add(h)
        ts = int(time.time())
        crash_file = self.crashes_dir / f"crash_{ts}_{h}"
        crash_file.write_bytes(data)
        meta = crash_file.with_suffix(".txt")
        meta.write_text(f"returncode: {returncode}\nstderr:\n{stderr}\n")

    def save_to_corpus(self, data: bytes):
        h = self._hash(data)
        if h in self.seen_hashes:
            return
        self.seen_hashes.add(h)
        self.corpus.append(data)
        corpus_file = self.corpus_dir / f"id_{h}"
        corpus_file.write_bytes(data)

    def fuzz_one(self, data: bytes) -> bool:
        mutated = self.mutate(data)
        returncode, stderr = self._run_target(mutated)
        self.exec_count += 1

        if self._is_crash(returncode, stderr):
            self.crash_count += 1
            self.save_crash(mutated, returncode, stderr)
            return True

        if self._is_interesting(returncode, stderr):
            self.save_to_corpus(mutated)
            return True

        return False

    def print_stats(self):
        elapsed = time.time() - self.start_time
        eps = self.exec_count / elapsed if elapsed > 0 else 0
        dict_str = f" | dict: {len(self.dictionary)}" if self.dictionary else ""
        print(f"\r[*] execs: {self.exec_count} | corpus: {len(self.corpus)} | "
              f"crashes: {self.crash_count} | eps: {eps:.0f} | "
              f"time: {elapsed:.0f}s{dict_str}", end="", flush=True)

    def run(self, iterations=0):
        print(f"[*] Target: {self.target}")
        print(f"[*] Corpus: {self.corpus_dir} ({len(self.corpus)} seeds)")
        print(f"[*] Crashes: {self.crashes_dir}")
        print(f"[*] Max input length: {self.max_len}")
        print(f"[*] Timeout: {self.timeout}s")
        if self.dictionary:
            print(f"[*] Dictionary: {len(self.dictionary)} tokens")
        print(f"[*] Starting fuzzing...\n")

        i = 0
        try:
            while True:
                if iterations and i >= iterations:
                    break
                seed = random.choice(self.corpus)
                self.fuzz_one(seed)
                i += 1
                if i % 100 == 0:
                    self.print_stats()
        except KeyboardInterrupt:
            pass

        self.print_stats()
        print(f"\n\n[*] Fuzzing stopped. {self.crash_count} crashes found.")
        if self.crash_count:
            print(f"[*] Crash files in: {self.crashes_dir}")


def main():
    parser = argparse.ArgumentParser(description="Coverage-guided binary fuzzer")
    parser.add_argument("target", help="Path to target binary")
    parser.add_argument("-d", "--corpus", default="corpus", help="Corpus directory")
    parser.add_argument("-o", "--crashes", default="crashes", help="Crashes directory")
    parser.add_argument("-m", "--max-len", type=int, default=4096, help="Max input length")
    parser.add_argument("-t", "--timeout", type=float, default=5, help="Timeout in seconds")
    parser.add_argument("-n", "--iterations", type=int, default=0, help="Number of iterations (0=infinite)")
    parser.add_argument("-M", "--mutations", type=int, default=8, help="Mutations per input")
    parser.add_argument("-c", "--coverage", action="store_true", help="Enable coverage-guided mode")
    parser.add_argument("-D", "--dict", help="Dictionary file (one token per line, NAME=value or raw bytes)")
    args = parser.parse_args()

    if not os.path.isfile(args.target):
        print(f"[-] Target not found: {args.target}")
        sys.exit(1)

    if not os.access(args.target, os.X_OK):
        print(f"[-] Target not executable: {args.target}")
        sys.exit(1)

    dictionary = []
    if args.dict:
        if not os.path.isfile(args.dict):
            print(f"[-] Dictionary not found: {args.dict}")
            sys.exit(1)
        dictionary = load_dictionary(args.dict)
        print(f"[*] Loaded {len(dictionary)} tokens from {args.dict}")

    fuzzer = Fuzzer(
        target=args.target,
        corpus_dir=args.corpus,
        crashes_dir=args.crashes,
        max_len=args.max_len,
        timeout=args.timeout,
        mutations_per_input=args.mutations,
        use_coverage=args.coverage,
        dictionary=dictionary,
    )
    fuzzer.run(iterations=args.iterations)


if __name__ == "__main__":
    main()
