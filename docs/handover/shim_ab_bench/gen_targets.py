"""Write the four synthetic targets (150 guarded blocks each) into argv[1].
t_*: chain, 150 noinline leaf calls from 150 distinct call sites (every guard
     misses the ctx cache).
h_*: hit-heavy, one function with 150 branches, called from one site (every
     guard hits). asm barriers keep clang from folding the branches.
*_junk: run() first clobbers rbp (frame walk must bail)."""
import sys
N = 150
out = sys.argv[1]
JUNK = '__asm__ volatile("mov $0x1234, %%rbp" ::: "rbp");'

def chain(junk):
    p = ["volatile int sink;"]
    for i in range(N):
        p.append(f"__attribute__((noinline)) int f{i}(int x){{ if(x&1) sink+={i}; else sink-={i}; return sink; }}")
    p.append("void run(void){ " + (JUNK if junk else "") + " " + " ".join(f"f{i}(3);" for i in range(N)) + " }")
    return "\n".join(p) + "\n"

def hit(junk):
    body = "\n".join(
        f'  if(x&{1 << (i % 20)}) {{ sink+={i}; __asm__ volatile(""); }} else {{ sink-={i}; __asm__ volatile(""); }}'
        for i in range(N))
    return ("volatile int sink;\n__attribute__((noinline)) int g(int x){\n" + body + "\n return sink; }\n"
            "void run(void){ " + (JUNK if junk else "") + " g(0x5a5a5); }\n")

for name, src in (("t_plain.c", chain(False)), ("t_junk.c", chain(True)),
                  ("h_plain.c", hit(False)), ("h_junk.c", hit(True))):
    open(f"{out}/{name}", "w").write(src)
