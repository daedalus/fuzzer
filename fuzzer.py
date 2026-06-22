#!/usr/bin/env python3
"""Binary fuzzer with ASAN/MSAN detection, dictionary and Markov mutations."""

import argparse
import collections
import hashlib
import math
import os
import random
import re
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


class MarkovChain:
    """Byte-level Markov chain for fuzz input generation."""

    def __init__(self, order=1, smoothing=1e-6):
        self.order = order
        self.smoothing = smoothing
        self.transitions = collections.defaultdict(lambda: collections.Counter())
        self._contexts_seen = 0

    def train(self, data: bytes):
        for i in range(len(data)):
            ctx = bytes(data[max(0, i - self.order):i]) if self.order else b""
            self.transitions[ctx][data[i]] += 1
            self._contexts_seen += 1

    def train_corpus(self, corpus: list[bytes]):
        for data in corpus:
            self.train(data)

    def generate(self, length: int) -> bytes:
        result = bytearray()
        ctx = b"\x00" * self.order
        for _ in range(length):
            counts = self.transitions.get(ctx)
            if counts is None or not counts:
                result.append(random.randint(0, 255))
                ctx = bytes(result[max(0, len(result) - self.order):])
                continue
            total = sum(counts.values()) + self.smoothing * 256
            r = random.random() * total
            cumulative = 0.0
            for byte_val, count in counts.items():
                cumulative += count + self.smoothing
                if r <= cumulative:
                    result.append(byte_val)
                    break
            else:
                result.append(random.randint(0, 255))
            ctx = bytes(result[max(0, len(result) - self.order):])
        return bytes(result)

    def sample_byte(self, ctx: bytes) -> int:
        counts = self.transitions.get(ctx)
        if counts is None or not counts:
            return random.randint(0, 255)
        total = sum(counts.values()) + self.smoothing * 256
        r = random.random() * total
        cumulative = 0.0
        for byte_val, count in counts.items():
            cumulative += count + self.smoothing
            if r <= cumulative:
                return byte_val
        return random.randint(0, 255)

    def is_trained(self) -> bool:
        return self._contexts_seen > 0


# Sanitizer error patterns
SANITIZER_PATTERNS = [
    (r"AddressSanitizer:\s*(heap-buffer-overflow|stack-buffer-overflow|heap-use-after-free"
     r"|global-buffer-overflow|stack-buffer-underflow|heap-buffer-overflow-|"
     r"dynamic-stack-buffer-overflow|stack-use-after-return|stack-use-after-scope"
     r"|allocation-size-too-big|double-free|invalid-malloc-size"
     r"|attempting-free-on-non-deallocated-memory|"
     r"negative-size-param|heap-use-after-scope", "ASAN"),
    (r"MemorySanitizer:\s*(use-of-uninitialized-value)", "MSAN"),
    (r"ThreadSanitizer:\s*(data-race|heap-use-after-race|lock-order-inversion", "TSAN"),
    (r"LeakSanitizer:\s*(leak)", "LSAN"),
    (r"UndefinedBehaviorSanitizer:\s*(undefined|shift-exponent|signed-integer-overflow"
     r"|null-pointer-use|integer-divide-by-zero)", "UBSAN"),
]

SANITIZER_ERROR_RE = re.compile(
    r"(AddressSanitizer|MemorySanitizer|ThreadSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer)"
    r":\s*(\S+(?:\s+\S+)?)",
    re.IGNORECASE,
)
SANITIZER_STACK_FRAME_RE = re.compile(
    r"#\d+\s+0x[0-9a-f]+\s+in\s+(\S+)\s+.*"
)
SANITIZER_FAULT_ADDR_RE = re.compile(
    r"(?:Address|Memory)Sanitizer.*(?:on|at) address\s+(0x[0-9a-f]+)",
    re.IGNORECASE,
)


class SanitizerReport:
    """Parse sanitizer output from a crashed process."""

    __slots__ = ("sanitizer", "error_type", "fault_addr", "frames", "raw", "signature")

    def __init__(self, sanitizer: str, error_type: str, fault_addr: str,
                 frames: list[str], raw: str):
        self.sanitizer = sanitizer
        self.error_type = error_type
        self.fault_addr = fault_addr
        self.frames = frames
        self.raw = raw
        self.signature = self._build_signature()

    def _build_signature(self) -> str:
        key = f"{self.sanitizer}:{self.error_type}"
        for f in self.frames[:6]:
            key += f"@{f}"
        return key

    @classmethod
    def parse(cls, stderr: str) -> "SanitizerReport | None":
        m = SANITIZER_ERROR_RE.search(stderr)
        if not m:
            return None
        sanitizer = m.group(1)
        error_type = m.group(2).strip()

        fault_addr = ""
        addr_m = SANITIZER_FAULT_ADDR_RE.search(stderr)
        if addr_m:
            fault_addr = addr_m.group(1)

        frames = SANITIZER_STACK_FRAME_RE.findall(stderr)
        return cls(sanitizer, error_type, fault_addr, frames, stderr)

    def is_valid(self) -> bool:
        return bool(self.sanitizer and self.error_type)


class Fuzzer:
    def __init__(self, target, corpus_dir, crashes_dir, max_len=4096,
                 timeout=5, mutations_per_input=8, use_coverage=False,
                 dictionary=None, file_mode=False, target_args=None,
                 markov_order=1, markov_generate=False):
        self.target = target
        self.corpus_dir = Path(corpus_dir)
        self.crashes_dir = Path(crashes_dir)
        self.max_len = max_len
        self.timeout = timeout
        self.mutations_per_input = mutations_per_input
        self.use_coverage = use_coverage
        self.dictionary = dictionary or []
        self.file_mode = file_mode
        self.target_args = target_args or []
        self._tmp_dir = Path("/tmp") / f"fuzzer_{os.getpid()}"
        if self.file_mode:
            self._tmp_dir.mkdir(parents=True, exist_ok=True)

        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir.mkdir(parents=True, exist_ok=True)

        self.corpus: list[bytes] = []
        self.seen_hashes: set[str] = set()
        self.crash_hashes: set[str] = set()
        self.crash_sigs: dict[str, int] = {}  # signature -> count
        self.exec_count = 0
        self.crash_count = 0
        self.start_time = time.time()
        self.last_report: SanitizerReport | None = None

        self.markov = MarkovChain(order=markov_order)
        self.markov_generate = markov_generate
        self.markov_trained = False

        self._load_corpus()
        if self.corpus:
            self.markov.train_corpus(self.corpus)
            self.markov_trained = self.markov.is_trained()

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

            if self.file_mode:
                tmp_file = self._tmp_dir / f"fuzz_{os.getpid()}"
                tmp_file.write_bytes(data)
                cmd = [self.target] + [
                    a.replace("{file}", str(tmp_file)) for a in self.target_args
                ]
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        env=env,
                        preexec_fn=os.setsid,
                    )
                    _, stderr = proc.communicate(timeout=self.timeout)
                    return proc.returncode, stderr.decode(errors="replace")
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait()
                    return -1, "timeout"
                finally:
                    try:
                        tmp_file.unlink()
                    except OSError:
                        pass
            else:
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
        self.last_report = None
        if returncode == -2:
            return False

        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            self.last_report = report
            return True

        if returncode in self.SIGNAL_CRASH_CODES:
            return True
        if returncode < 0:
            return True
        if any(sig in stderr for sig in ["SIGSEGV", "SIGABRT", "SIGFPE", "SIGBUS",
                                         "Segmentation fault", "Aborted"]):
            return True
        return False

    def mutate(self, data: bytes) -> bytes:
        buf = bytearray(data)
        if not buf:
            buf = bytearray(b"\x00" * random.randint(1, 32))

        ops = list(MUTATIONS)
        if self.dictionary:
            ops.extend(DICT_MUTATIONS)
        if self.markov_trained:
            ops.append("markov_bytes")

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

            elif op == "markov_bytes" and buf:
                idx = random.randint(0, len(buf) - 1)
                ctx = bytes(buf[max(0, idx - self.markov.order):idx]) if self.markov.order else b""
                buf[idx] = self.markov.sample_byte(ctx)

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

        report = SanitizerReport.parse(stderr)
        sig = report.signature if report and report.is_valid() else f"signal:{abs(returncode)}"
        self.crash_hashes.add(h)
        self.crash_sigs[sig] = self.crash_sigs.get(sig, 0) + 1

        ts = int(time.time())
        crash_file = self.crashes_dir / f"crash_{ts}_{h}"
        crash_file.write_bytes(data)

        meta = crash_file.with_suffix(".txt")
        lines = [f"returncode: {returncode}"]
        if report and report.is_valid():
            lines.extend([
                f"sanitizer: {report.sanitizer}",
                f"error: {report.error_type}",
                f"fault_addr: {report.fault_addr}",
                f"signature: {sig}",
                f"seen: {self.crash_sigs[sig]}x",
                "",
                "=== stack trace ===",
            ])
            for i, frame in enumerate(report.frames[:12]):
                lines.append(f"  #{i} {frame}")
            lines.extend(["", "=== raw stderr ===", report.raw])
        else:
            lines.extend(["", "=== stderr ===", stderr])
        meta.write_text("\n".join(lines))

    def save_to_corpus(self, data: bytes):
        h = self._hash(data)
        if h in self.seen_hashes:
            return
        self.seen_hashes.add(h)
        self.corpus.append(data)
        self.markov.train(data)
        self.markov_trained = self.markov.is_trained()
        corpus_file = self.corpus_dir / f"id_{h}"
        corpus_file.write_bytes(data)

    def _pick_seed(self) -> bytes:
        if self.markov_generate and self.markov_trained and random.random() < 0.15:
            length = random.randint(1, self.max_len)
            return self.markov.generate(length)
        if self.corpus:
            return random.choice(self.corpus)
        return b"AAAAAAAA"

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
        markov_str = " | markov: trained" if self.markov_trained else ""
        if self.markov_generate:
            markov_str += "+gen"
        sig_str = f" | sigs: {len(self.crash_sigs)}" if self.crash_sigs else ""
        print(f"\r[*] execs: {self.exec_count} | corpus: {len(self.corpus)} | "
              f"crashes: {self.crash_count}{sig_str} | eps: {eps:.0f} | "
              f"time: {elapsed:.0f}s{dict_str}{markov_str}", end="", flush=True)

    def run(self, iterations=0):
        print(f"[*] Target: {self.target}")
        print(f"[*] Corpus: {self.corpus_dir} ({len(self.corpus)} seeds)")
        print(f"[*] Crashes: {self.crashes_dir}")
        print(f"[*] Max input length: {self.max_len}")
        print(f"[*] Timeout: {self.timeout}s")
        if self.dictionary:
            print(f"[*] Dictionary: {len(self.dictionary)} tokens")
        if self.markov_trained:
            print(f"[*] Markov chain: order={self.markov.order}, "
                  f"transitions={len(self.markov.transitions)}")
        if self.markov_generate:
            print(f"[*] Markov generation: enabled (15% of seeds)")
        print(f"[*] Starting fuzzing...\n")

        i = 0
        try:
            while True:
                if iterations and i >= iterations:
                    break
                seed = self._pick_seed()
                self.fuzz_one(seed)
                i += 1
                if i % 100 == 0:
                    self.print_stats()
        except KeyboardInterrupt:
            pass

        self.print_stats()
        print(f"\n\n[*] Fuzzing stopped. {self.crash_count} crashes found "
              f"({len(self.crash_sigs)} unique signatures).")
        if self.crash_sigs:
            print("[*] Crash signatures:")
            for sig, count in sorted(self.crash_sigs.items(), key=lambda x: -x[1]):
                print(f"    {sig} ({count}x)")
            print(f"\n[*] Crash files in: {self.crashes_dir}")


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
    parser.add_argument("-F", "--file-mode", action="store_true",
                        help="Write input to temp file instead of stdin")
    parser.add_argument("-A", "--target-args", nargs=argparse.REMAINDER,
                        help="Target arguments (use {file} as placeholder for temp file)")
    parser.add_argument("--markov", action="store_true",
                        help="Enable Markov chain mutation (trained on corpus)")
    parser.add_argument("--markov-gen", action="store_true",
                        help="Enable Markov chain seed generation (15%% of seeds)")
    parser.add_argument("--markov-order", type=int, default=1,
                        help="Markov chain order (default: 1)")
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

    use_markov = args.markov or args.markov_gen

    fuzzer = Fuzzer(
        target=args.target,
        corpus_dir=args.corpus,
        crashes_dir=args.crashes,
        max_len=args.max_len,
        timeout=args.timeout,
        mutations_per_input=args.mutations,
        use_coverage=args.coverage,
        dictionary=dictionary,
        file_mode=args.file_mode,
        target_args=args.target_args,
        markov_order=args.markov_order if use_markov else 0,
        markov_generate=args.markov_gen,
    )
    fuzzer.run(iterations=args.iterations)


if __name__ == "__main__":
    main()
