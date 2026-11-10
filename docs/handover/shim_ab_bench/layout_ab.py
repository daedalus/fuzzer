"""layout_ab.py OUTDIR ROUNDS BASE NAME=shim.c [NAME=shim.c ...]
Layout-randomized paired wall A/B. OUTDIR holds t-*.o / h-*.o from run.sh.
Each arm is rebuilt at 8 code layouts (0..56 B of nop padding before the shim
in the same TU). Per layout: min of ROUNDS interleaved rounds, paired vs BASE.
Prints mean / median / min / max of the paired delta (%, negative = faster =
better) and in how many layouts the arm is worse. One layout is not evidence:
the same code moved wall time by -14%..+22% across layouts."""
import os, statistics, subprocess, sys
out, rounds, base = sys.argv[1], int(sys.argv[2]), sys.argv[3]
arms = dict(a.split("=", 1) for a in sys.argv[4:])
assert base in arms, "BASE must be one of the NAME= arms"
here = os.path.dirname(os.path.abspath(__file__)); main = f"{here}/main.c"
pads = [0, 8, 16, 24, 32, 40, 48, 56]
for n in pads:
    open(f"{out}/pad_{n}.h", "w").write(
        f'__attribute__((used,noinline)) static void __pad{n}(void){{ __asm__ volatile(".skip {n}, 0x90"); }}\n' if n else "")
    for a, src in arms.items():
        subprocess.check_call(["clang", "-O2", "-fno-omit-frame-pointer", "-w", "-include", f"{out}/pad_{n}.h",
                               "-include", os.path.abspath(src), "-c", main, "-o", f"{out}/ml-{a}-{n}.o"])
        for k in "th":
            for t in ("healthy", "const", "junk"):
                subprocess.check_call(["clang", f"{out}/ml-{a}-{n}.o", f"{out}/{k}-{t}.o", "-o",
                                       f"{out}/pl-{a}-{n}-{k}-{t}", "-ldl", "-lpthread", "-lrt"])
for a in arms:
    if a == base: continue
    print(f"{a} vs {base}: paired wall-min delta % over {len(pads)} layouts (negative = {a} faster, better)")
    for k, kn in (("t", "chain"), ("h", "hit")):
        for t in ("healthy", "const", "junk"):
            d = []
            for n in pads:
                res = {x: [] for x in (a, base)}
                for _ in range(rounds):
                    for x in res:
                        res[x].append(float(subprocess.check_output([f"{out}/pl-{x}-{n}-{k}-{t}", "20000"],
                                                                    stderr=subprocess.DEVNULL)))
                d.append(100 * (min(res[a]) / min(res[base]) - 1))
            print(f"  {kn}-{t:8s} mean {statistics.mean(d):+5.1f}%  median {statistics.median(d):+5.1f}%  "
                  f"min {min(d):+5.1f}%  max {max(d):+5.1f}%  worse-in {sum(x > 0 for x in d)}/{len(d)}")
