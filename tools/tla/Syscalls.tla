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
(*   "dead"   exited or killed, not reaped (zombie)                        *)
(*   "reaped" waited for                                                   *)
(* pend[p]: replies p owes, oldest first. Consuming one while another is   *)
(* still owed means the caller read a stale reply: w.desync.               *)
(* stale[r]: a short read_exact left payload bytes in r; the next readline *)
(* reads them as a header: w.desync.                                       *)
(* files[f]: "free" | "open" | "closed" (TemporaryFile handles).           *)
(*                                                                         *)
(* Abstractions (each is a modeling choice, not a fact about the OS):      *)
(*   - wait() after kill always reaps (no D-state stalls).                 *)
(*   - a write to a live proc may still raise BrokenPipeError: the        *)
(*     loader closing its pipes on exit before it is reapable.             *)
(*   - Popen may raise OSError (EAGAIN/ENOMEM, missing binary).            *)
(*   - "Hang" / "Overflow" are uncatchable pseudo-exceptions.              *)
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

\* Environment: a live loader may die between calls.
Died(w) ==
    {[w EXCEPT !.procs[p] = "dead"] : p \in {q \in 1..MaxProcs : w.procs[q] = "live"}}
=============================================================================
