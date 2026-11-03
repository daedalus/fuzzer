"""wall.py OUTDIR ROUNDS NAME... : interleaved wall clock per case.
Prints min and median ns per run() (150 guards) and Mguards/s from the min."""
import subprocess, statistics, sys
out, rounds, names = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
print(f"{'case':14s}" + "".join(f"{n:>26s}" for n in names))
for k, kn in (("t", "chain"), ("h", "hit")):
    for t in ("healthy", "const", "junk"):
        res = {n: [] for n in names}
        for _ in range(rounds):
            for n in names:
                res[n].append(float(subprocess.check_output(
                    [f"{out}/p-{n}-{k}-{t}", "20000"], stderr=subprocess.DEVNULL)))
        row = f"{kn + '-' + t:14s}"
        for n in names:
            m = min(res[n])
            row += f"{m:7.0f}/{statistics.median(res[n]):5.0f} {150e3 / m:6.1f}Mg/s  "
        print(row)
