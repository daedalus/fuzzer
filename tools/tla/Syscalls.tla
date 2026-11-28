---------------------------- MODULE Syscalls ----------------------------
(* Hand-written outcome model for the calls tools/tla_extract.py maps.     *)
(*                                                                         *)
(* Every operator takes the world w and two continuations:                 *)
(*   Ok(w2, v)     the call returned v                                     *)
(*   Raise(w2, e)  the call raised exception class e                       *)
(* A disjunction of continuations is a nondeterministic outcome.           *)
(*                                                                         *)
(* Handles (procs, readers, fds) are ints, NONE = 0. Other values are      *)
(* strings: "None", "" (b"", EOF), "line", "path".                         *)
(*                                                                         *)
(* Process status:                                                         *)
(*   "live"   running; may answer, stall or die                            *)
(*   "stopped" SIGSTOP'd (itself or by a signal); alive, not running       *)
(*   "dead"   exited or killed, not reaped (zombie)                        *)
(*   "reaped" waited for                                                   *)
(* pend[p]: replies p owes, oldest first. Consuming one while another is   *)
(* still owed means the caller read a stale reply: w.desync.               *)
(* stale[r]: a short read_exact left payload bytes in r; the next readline *)
(* reads them as a header: w.desync.                                       *)
(* files[f]: "free" | "open" | "closed" (TemporaryFile handles).           *)
(* segs[g]: SysV shm segment. "det" exists, unmapped; "att" mapped;        *)
(*   "rmatt" removed (IPC_RMID) but still mapped; "gone" freed.            *)
(*   A segment's id and attach address are both the handle g.              *)
(*                                                                         *)
(* Abstractions (each is a modeling choice, not a fact about the OS):      *)
(*   - wait() after kill always reaps (no D-state stalls).                 *)
(*   - a write to a live proc may still raise BrokenPipeError: the        *)
(*     loader closing its pipes on exit before it is reapable.             *)
(*   - Popen may raise OSError (EAGAIN/ENOMEM, missing binary).            *)
(*   - os.kill on a reaped pid sets w.stalekill: the number may already    *)
(*     name another process.                                               *)
(*   - A fatal signal (default action) raises the uncatchable "Killed".    *)
(*   - "Hang" / "Overflow" / "LoopBound" are uncatchable pseudo-exceptions.*)
EXTENDS Naturals

CONSTANT MaxProcs

NONE == 0
Truthy(v) == v \notin {"None", ""}
Status(w, p) == IF p = NONE THEN "none" ELSE w.procs[p]

\* subprocess.Popen(...): a fresh live loader. Beyond MaxProcs the path is pruned.
Spawn(w, Ok(_, _), Raise(_, _)) ==
    \/ Raise(w, "OSError")
    \/ IF w.nproc = MaxProcs THEN FALSE
       ELSE LET p == w.nproc + 1
            IN Ok([w EXCEPT !.nproc = p, !.procs[p] = "live"], p)

\* proc.poll(): None while live; reaps a zombie.
Poll(w, p, Ok(_, _), Raise(_, _)) ==
    CASE p = NONE              -> Raise(w, "AttributeError")
      [] w.procs[p] = "live"   -> Ok(w, "None")
      [] OTHER                 -> Ok([w EXCEPT !.procs[p] = "reaped"], "rc")

\* Pure poll() for invariants and guards.
PollValue(w, p) == IF Status(w, p) = "live" THEN "None" ELSE "rc"

\* proc.kill(): Popen.kill never raises on its own child.
Kill(w, p, Ok(_, _), Raise(_, _)) ==
    CASE p = NONE              -> Raise(w, "AttributeError")
      [] w.procs[p] = "live"   -> Ok([w EXCEPT !.procs[p] = "dead"], "None")
      [] OTHER                 -> Ok(w, "None")

\* os.killpg(proc.pid, sig): the group outlives the leader until it is reaped.
KillPg(w, p, Ok(_, _), Raise(_, _)) ==
    CASE p = NONE              -> Raise(w, "AttributeError")
      [] w.procs[p] = "live"   -> Ok([w EXCEPT !.procs[p] = "dead"], "None")
      [] w.procs[p] = "dead"   -> Ok(w, "None")
      [] OTHER                 -> Raise(w, "ProcessLookupError")

\* proc.wait(): blocks until exit; a live proc that never exits hangs us.
Wait(w, p, Ok(_, _), Raise(_, _)) ==
    CASE p = NONE              -> Raise(w, "AttributeError")
      [] w.procs[p] = "live"   -> Ok([w EXCEPT !.procs[p] = "reaped"], "rc")
                                  \/ Raise(w, "Hang")
      [] OTHER                 -> Ok([w EXCEPT !.procs[p] = "reaped"], "rc")

\* proc.wait(timeout=t)
WaitT(w, p, Ok(_, _), Raise(_, _)) ==
    CASE p = NONE              -> Raise(w, "AttributeError")
      [] w.procs[p] = "live"   -> Ok([w EXCEPT !.procs[p] = "reaped"], "rc")
                                  \/ Raise(w, "TimeoutExpired")
      [] OTHER                 -> Ok([w EXCEPT !.procs[p] = "reaped"], "rc")

\* proc.stdin.write(payload) / flush(); n = replies the payload asks for.
Write(w, p, n, Ok(_, _), Raise(_, _)) ==
    CASE p = NONE              -> Raise(w, "AttributeError")
      [] p \in w.closed        -> Raise(w, "ValueError")
      [] w.procs[p] = "live"   -> Ok([w EXCEPT !.pend[p] = @ + n], "None")
                                  \/ Raise(w, "BrokenPipeError")
      [] OTHER                 -> Raise(w, "BrokenPipeError")

Flush(w, p, Ok(_, _), Raise(_, _)) == Write(w, p, 0, Ok, Raise)

\* proc.stdin.close(): flushes what is buffered, so it may hit EPIPE.
CloseStdin(w, p, Ok(_, _), Raise(_, _)) ==
    CASE p = NONE              -> Raise(w, "AttributeError")
      [] p \in w.closed        -> Ok(w, "None")
      [] OTHER                 -> Ok(w, "None") \/ Raise(w, "BrokenPipeError")

\* proc.stdout.fileno(): the fd names the proc it reads from.
Fileno(w, p, Ok(_, _), Raise(_, _)) ==
    CASE p = NONE              -> Raise(w, "AttributeError")
      [] p \in w.closed        -> Raise(w, "ValueError")
      [] OTHER                 -> Ok(w, p)

\* PipeReader(fd)
NewReader(w, fd, Ok(_, _), Raise(_, _)) == Ok(w, fd)

\* One owed reply arrives; flags a stale read when more were owed.
Consume(w, r) ==
    [w EXCEPT !.pend[r] = @ - 1, !.desync = @ \/ w.pend[r] > 1]

\* Leftover payload bytes are read as a header.
StaleLine(w, r) == [w EXCEPT !.desync = TRUE, !.stale[r] = FALSE]

\* reader.readline(deadline): a line, None on timeout, "" at EOF.
ReadLineT(w, r, Ok(_, _), Raise(_, _)) ==
    CASE r = NONE              -> Raise(w, "AttributeError")
      [] r \in w.closed        -> Raise(w, "OSError")
      [] w.stale[r]            -> Ok(StaleLine(w, r), "line")
      [] w.procs[r] = "live" /\ w.pend[r] > 0
                               -> Ok(Consume(w, r), "line") \/ Ok(w, "None")
      [] w.procs[r] = "live"   -> Ok(w, "None")
      [] w.pend[r] > 0         -> Ok(Consume(w, r), "line") \/ Ok(w, "")
      [] OTHER                 -> Ok(w, "")

\* reader.readline(NO_DEADLINE): a silent live loader blocks us forever.
ReadLineBlock(w, r, Ok(_, _), Raise(_, _)) ==
    CASE r = NONE              -> Raise(w, "AttributeError")
      [] r \in w.closed        -> Raise(w, "OSError")
      [] w.stale[r]            -> Ok(StaleLine(w, r), "line")
      [] w.procs[r] = "live" /\ w.pend[r] > 0
                               -> Ok(Consume(w, r), "line") \/ Raise(w, "Hang")
      [] w.procs[r] = "live"   -> Raise(w, "Hang")
      [] w.pend[r] > 0         -> Ok(Consume(w, r), "line") \/ Ok(w, "")
      [] OTHER                 -> Ok(w, "")

\* reader.read_exact(n, deadline): n bytes, or None on timeout/EOF. A timeout
\* on a live loader may leave part of the payload buffered.
ReadExact(w, r, Ok(_, _), Raise(_, _)) ==
    CASE r = NONE              -> Raise(w, "AttributeError")
      [] r \in w.closed        -> Raise(w, "OSError")
      [] w.procs[r] = "live"   -> Ok(w, "bytes") \/ Ok([w EXCEPT !.stale[r] = TRUE], "None")
      [] OTHER                 -> Ok(w, "bytes") \/ Ok(w, "None")

\* tempfile.TemporaryFile(): bounded like Spawn.
FileOpen(w, Ok(_, _), Raise(_, _)) ==
    IF w.nfile = MaxProcs THEN FALSE
    ELSE LET f == w.nfile + 1
         IN Ok([w EXCEPT !.nfile = f, !.files[f] = "open"], f)

\* f.seek / read / write on a closed file raise ValueError.
FileUse(w, f, Ok(_, _), Raise(_, _)) ==
    CASE f = NONE              -> Raise(w, "AttributeError")
      [] w.files[f] = "open"   -> Ok(w, "None")
      [] OTHER                 -> Raise(w, "ValueError")

\* f.close(): idempotent.
FileClose(w, f, Ok(_, _), Raise(_, _)) ==
    IF f = NONE THEN Raise(w, "AttributeError")
    ELSE Ok([w EXCEPT !.files[f] = "closed"], "None")

\* _close_streams(proc)
CloseStreams(w, p, Ok(_, _), Raise(_, _)) ==
    IF p = NONE THEN Raise(w, "AttributeError")
    ELSE Ok([w EXCEPT !.closed = @ \cup {p}], "None")

\* A lookup that may come back empty (e.g. the loader binary).
Maybe(w, Ok(_, _), Raise(_, _)) == Ok(w, "path") \/ Ok(w, "None")

\* ── fork / waitpid / kill ───────────────────────────────────────────────

\* os.fork(), parent side: the child's pid (bounded like Spawn), or OSError.
Fork(w, Ok(_, _), Raise(_, _)) == Spawn(w, Ok, Raise)

\* Stop reports: job control, a ptrace breakpoint (SIGTRAP), a fault (SIGSEGV).
StopTags == {"stopped:SIGSTOP", "stopped:SIGTRAP", "stopped:SIGSEGV", "stopped:other"}

Reap(w, p) == [w EXCEPT !.procs[p] = "reaped"]

\* os.waitpid(pid, flags): (pid, status) as a status tag; the pid half is
\* derived by the caller: 0 for "none", else pid.
WaitPid(w, p, nohang, untraced, Ok(_, _), Raise(_, _)) ==
    CASE p = NONE                        -> Raise(w, "TypeError")
      [] w.procs[p] \in {"reaped", "free"} -> Raise(w, "ChildProcessError")
      [] w.procs[p] = "dead"             -> Ok(Reap(w, p), "signaled") \/ Ok(Reap(w, p), "exited")
      [] w.procs[p] = "live"             ->
            \/ Ok(Reap(w, p), "exited")
            \/ Ok(Reap(w, p), "signaled")
            \/ (untraced /\ \E g \in StopTags : Ok([w EXCEPT !.procs[p] = "stopped"], g))
            \/ (nohang /\ Ok(w, "none"))
            \/ (~nohang /\ Raise(w, "Hang"))
      [] OTHER                           ->   \* stopped: already reported
            IF nohang THEN Ok(w, "none") ELSE Raise(w, "Hang")

\* os.kill(pid, sig).
KillSig(w, p, sig, Ok(_, _), Raise(_, _)) ==
    CASE p = NONE                    -> Raise(w, "TypeError")
      [] w.procs[p] = "reaped"       -> Raise([w EXCEPT !.stalekill = TRUE], "ProcessLookupError")
      [] w.procs[p] = "dead"         -> Ok(w, "None")
      [] sig = "SIGKILL"             -> Ok([w EXCEPT !.procs[p] = "dead"], "None")
      [] sig = "SIGSTOP"             -> Ok([w EXCEPT !.procs[p] = "stopped"], "None")
      [] sig = "SIGCONT"             -> Ok([w EXCEPT !.procs[p] = "live"], "None")
      [] OTHER                       -> Ok(w, "None")

\* ── signal dispositions and interval timers ─────────────────────────────

SigNames == {"SIGALRM", "SIGPROF", "SIGVTALRM"}
\* Initial dispositions; "cforeign" is a handler installed outside Python,
\* which signal.signal / getsignal report as None.
Dispositions == {"default", "ignore", "other", "cforeign"}
Shown(d) == IF d = "cforeign" THEN "None" ELSE d

\* signal.signal(sig, handler): returns the previous disposition.
SigSet(w, s, h, Ok(_, _), Raise(_, _)) ==
    IF h \in {"None", "unset"} THEN Raise(w, "TypeError")
    ELSE Ok([w EXCEPT !.disp[s] = h], Shown(w.disp[s]))

SigGet(w, s, Ok(_, _), Raise(_, _)) == Ok(w, Shown(w.disp[s]))

\* signal.setitimer(which, seconds[, interval]); which is named by its signal.
\* Returns whether the timer was armed before (the saved value a caller restores).
SetItimer(w, s, arm, per, Ok(_, _), Raise(_, _)) ==
    Ok([w EXCEPT !.timer[s] = arm, !.periodic[s] = arm /\ per], w.timer[s])

\* ── threads (tools/tla_threads.py) ──────────────────────────────────────

CONSTANT MaxQueue

\* Work that may fail (e.g. a coverage measurement).
\* libc_shm wrappers: failures return None / False, never raise.
ShmGet(w, Ok(_, _), Raise(_, _)) ==
    \/ Ok(w, NONE)
    \/ IF w.nseg = MaxProcs THEN FALSE
       ELSE LET g == w.nseg + 1 IN Ok([w EXCEPT !.nseg = g, !.segs[g] = "det"], g)
ShmAt(w, g, Ok(_, _), Raise(_, _)) ==
    CASE g = NONE              -> Ok(w, NONE)
      [] w.segs[g] = "det"     -> Ok([w EXCEPT !.segs[g] = "att"], g) \/ Ok(w, NONE)
      [] w.segs[g] = "att"     -> Ok(w, g)
      [] OTHER                 -> Ok(w, NONE)
ShmDt(w, g, Ok(_, _), Raise(_, _)) ==
    CASE g = NONE              -> Ok(w, "False")
      [] w.segs[g] = "att"     -> Ok([w EXCEPT !.segs[g] = "det"], "True")
      [] w.segs[g] = "rmatt"   -> Ok([w EXCEPT !.segs[g] = "gone"], "True")
      [] OTHER                 -> Ok(w, "False")
ShmRmid(w, g, Ok(_, _), Raise(_, _)) ==
    CASE g = NONE              -> Ok(w, "False")
      [] w.segs[g] = "att"     -> Ok([w EXCEPT !.segs[g] = "rmatt"], "True")
      [] w.segs[g] = "det"     -> Ok([w EXCEPT !.segs[g] = "gone"], "True")
      [] OTHER                 -> Ok(w, "False")

\* libc.ptrace(req, pid, ...): errors come back as -1 (errno), never raised.
\* PTRACE_CONT/DETACH resume a stopped tracee; other requests leave it as is.
PtraceCont(w, p, Ok(_, _), Raise(_, _)) ==
    CASE p = NONE                 -> Raise(w, "ArgumentError")
      [] w.procs[p] = "stopped"   -> Ok([w EXCEPT !.procs[p] = "live"], "0")
      [] OTHER                    -> Ok(w, "-1")
Ptrace(w, p, Ok(_, _), Raise(_, _)) ==
    IF p = NONE THEN Raise(w, "ArgumentError") ELSE Ok(w, "rc")  \* 0 or -1: no state change

MayFail(w, Ok(_, _), Raise(_, _)) == Ok(w, "None") \/ Raise(w, "RuntimeError")

\* deque.append / popleft on a QUEUE field f (its length). Past MaxQueue the
\* path raises "Bound": the thread model freezes it (w.bound) rather than
\* pruning it, which would read as a deadlock.
QPush(w, f, Ok(_, _), Raise(_, _)) ==
    IF w.s[f] = MaxQueue THEN Raise(w, "Bound") ELSE Ok([w EXCEPT !.s[f] = @ + 1], "None")
QPop(w, f, Ok(_, _), Raise(_, _)) ==
    IF w.s[f] = 0 THEN Raise(w, "IndexError") ELSE Ok([w EXCEPT !.s[f] = @ - 1], "item")

\* Condition.notify / notify_all by thread t; the lock l must be held.
CNotify(w, c, l, t, Ok(_, _), Raise(_, _)) ==
    CASE w.lock[l] # t       -> Raise(w, "RuntimeError")
      [] w.waiters[c] = {}   -> Ok(w, "None")
      [] OTHER               -> \E u \in w.waiters[c] : Ok([w EXCEPT !.waiters[c] = @ \ {u}], "None")
CNotifyAll(w, c, l, t, Ok(_, _), Raise(_, _)) ==
    IF w.lock[l] # t THEN Raise(w, "RuntimeError") ELSE Ok([w EXCEPT !.waiters[c] = {}], "None")

\* Environment: a live loader may die between calls.
Died(w) ==
    {[w EXCEPT !.procs[p] = "dead"] : p \in {q \in 1..MaxProcs : w.procs[q] = "live"}}
=============================================================================
