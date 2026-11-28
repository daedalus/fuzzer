/-
Runtime for Lean models emitted by tools/tla_lean.py.

  Val      TLA+ values: ints, strings, bools, sets, functions (records are
           functions over string keys). Sets and functions are kept sorted
           by key, so structural equality is TLA+ equality.
  ops      the operators the emitted fragment uses (app, EXCEPT, \in, ...).
  check    breadth-first state exploration, like TLC: every reachable state
           is checked against every invariant; the shortest trace to a
           violating state is printed in TLC's format.

Evaluation errors (a function applied outside its domain, a non-boolean
condition) panic: the checker reports them and exits 3.
-/
import Std.Data.HashMap

namespace Tla

inductive Val where
  | int (n : Int)
  | str (s : String)
  | bool (b : Bool)
  | set (xs : Array Val)
  | fn (kv : Array (Val × Val))
deriving Inhabited

/-- Actions return the successor states they allow; `[]` is FALSE. -/
abbrev Acts := List Val

def Val.tag : Val → Nat
  | .int _ => 0 | .str _ => 1 | .bool _ => 2 | .set _ => 3 | .fn _ => 4

-- Total order: sort sets and function domains into one canonical form.
partial def Val.cmp : Val → Val → Ordering
  | .int a, .int b => compare a b
  | .str a, .str b => compare a b
  | .bool a, .bool b => compare a.toNat b.toNat
  | .set a, .set b => Id.run do
      for i in [0:min a.size b.size] do
        let c := Val.cmp a[i]! b[i]!
        if c != .eq then return c
      return compare a.size b.size
  | .fn a, .fn b => Id.run do
      for i in [0:min a.size b.size] do
        let c := Val.cmp a[i]!.1 b[i]!.1
        if c != .eq then return c
        let d := Val.cmp a[i]!.2 b[i]!.2
        if d != .eq then return d
      return compare a.size b.size
  | a, b => compare a.tag b.tag

instance : BEq Val := ⟨fun a b => a.cmp b == .eq⟩

partial def Val.hash : Val → UInt64
  | .int n => mixHash 11 (Hashable.hash n)
  | .str s => mixHash 13 (Hashable.hash s)
  | .bool b => mixHash 17 (Hashable.hash b)
  | .set xs => xs.foldl (fun h x => mixHash h x.hash) 19
  | .fn kv => kv.foldl (fun h p => mixHash (mixHash h p.1.hash) p.2.hash) 23

instance : Hashable Val := ⟨Val.hash⟩

-- ── Constructors ────────────────────────────────────────────────

private def sortDedup (xs : Array Val) : Array Val :=
  let s := xs.qsort (fun a b => a.cmp b == .lt)
  s.foldl (fun acc x => if acc.size > 0 && acc.back! == x then acc else acc.push x) #[]

def setOf (xs : Array Val) : Val := .set (sortDedup xs)

def record (kv : Array (String × Val)) : Val :=
  .fn ((kv.map fun (k, v) => (Val.str k, v)).qsort (fun a b => a.1.cmp b.1 == .lt))

def elems : Val → Array Val
  | .set xs => xs
  | v => panic! s!"not a set (tag {v.tag})"

/-- `[x \in S |-> f(x)]`: S is sorted, so the keys are. -/
def fnOf (dom : Val) (f : Val → Val) : Val := .fn ((elems dom).map fun x => (x, f x))

def range (a b : Val) : Val :=
  match a, b with
  | .int lo, .int hi =>
      if hi < lo then .set #[]
      else .set ((List.range (hi - lo + 1).toNat).toArray.map fun i => .int (lo + Int.ofNat i))
  | _, _ => panic! "range over non-ints"

/-- `[S -> T]`: every function from S to T. -/
def funSet (dom cod : Val) : Val :=
  let ks := elems dom
  let vs := elems cod
  let maps := ks.foldl (fun acc k => acc.flatMap fun m => vs.map fun v => m.push (k, v)) #[#[]]
  .set (sortDedup (maps.map Val.fn))

-- ── Operators ───────────────────────────────────────────────────

def b : Val → Bool
  | .bool x => x
  | v => panic! s!"not a boolean (tag {v.tag})"

def app (f k : Val) : Val :=
  match f with
  | .fn kv => match kv.find? (·.1 == k) with
    | some (_, v) => v
    | none => panic! "function applied outside its domain"
  | _ => panic! "application of a non-function"

def dot (r : Val) (k : String) : Val := app r (.str k)

/-- `[r EXCEPT !path = f(@)]`; a key outside the domain leaves r unchanged. -/
partial def exc (r : Val) (path : List Val) (f : Val → Val) : Val :=
  match path, r with
  | [], _ => f r
  | k :: rest, .fn kv => match kv.findIdx? (·.1 == k) with
    | some i => .fn (kv.set! i (k, exc kv[i]!.2 rest f))
    | none => r
  | _, _ => panic! "EXCEPT on a non-function"

def mem (x s : Val) : Val := .bool ((elems s).contains x)
def cup (a c : Val) : Val := setOf (elems a ++ elems c)
def setMap (s : Val) (f : Val → Val) : Val := setOf ((elems s).map f)
def setFilter (s : Val) (p : Val → Bool) : Val := .set ((elems s).filter p)
def all (s : Val) (p : Val → Bool) : Val := .bool ((elems s).all p)
def any (s : Val) (p : Val → Bool) : Val := .bool ((elems s).any p)
def flatMapS (s : Val) (f : Val → Acts) : Acts := (elems s).toList.flatMap f

private def int2 [Inhabited α] (op : Int → Int → α) : Val → Val → α
  | .int x, .int y => op x y
  | _, _ => panic! "arithmetic on non-ints"

def add := int2 fun x y => Val.int (x + y)
def sub := int2 fun x y => Val.int (x - y)
def lt := int2 fun x y => Val.bool (x < y)
def gt := int2 fun x y => Val.bool (x > y)
def le := int2 fun x y => Val.bool (x ≤ y)
def ge := int2 fun x y => Val.bool (x ≥ y)

/-- TLC's ToString: strings keep their quotes. -/
def toStr : Val → Val
  | .int n => .str (toString n)
  | .bool x => .str (if x then "TRUE" else "FALSE")
  | .str s => .str s!"\"{s}\""
  | _ => panic! "ToString of a set or function"

-- ── Rendering (TLC trace syntax) ────────────────────────────────

private def isRecord (kv : Array (Val × Val)) : Bool :=
  kv.size > 0 && kv.all fun p => p.1.tag == 1

private def isTuple (kv : Array (Val × Val)) : Bool :=
  (kv.mapIdx fun i p => p.1 == .int (i + 1)).all id

partial def render : Val → String
  | .int n => toString n
  | .str s => s!"\"{s}\""
  | .bool x => if x then "TRUE" else "FALSE"
  | .set xs => "{" ++ ", ".intercalate (xs.map render).toList ++ "}"
  | .fn kv =>
    if isRecord kv then
      "[" ++ ", ".intercalate (kv.map fun (k, v) =>
        (match k with | .str s => s | _ => "") ++ " |-> " ++ render v).toList ++ "]"
    else if isTuple kv then
      "<<" ++ ", ".intercalate (kv.map fun p => render p.2).toList ++ ">>"
    else
      "(" ++ " @@ ".intercalate (kv.map fun (k, v) => render k ++ " :> " ++ render v).toList ++ ")"

-- ── Checker ─────────────────────────────────────────────────────

abbrev Invariant := String × (Val → Bool)

private structure Graph where
  seen : Std.HashMap Val Nat := {}
  states : Array Val := #[]
  parent : Array Nat := #[]

private def Graph.add (g : Graph) (s : Val) (from_ : Nat) : Graph × Bool :=
  if g.seen.contains s then (g, false)
  else ({ seen := g.seen.insert s g.states.size, states := g.states.push s,
          parent := g.parent.push from_ }, true)

/-- Root-first indices of the path to state i (roots are their own parent). -/
private partial def Graph.path (g : Graph) (i : Nat) (acc : List Nat := []) : List Nat :=
  if g.parent[i]! == i then i :: acc else g.path g.parent[i]! (i :: acc)

private def Graph.trace (g : Graph) (i : Nat) : String :=
  let lines := (g.path i).zipIdx.map fun (j, n) =>
    s!"State {n + 1}: <step>\nw = {render g.states[j]!}\n"
  "\n".intercalate lines

/-- Explore every state reachable from `init` under `next`; check every invariant.
    Prints `STATES n`, one `OK inv` / `FAIL inv` per invariant and each failure's
    shortest trace. Exit code: 0 clean, 1 violations. -/
def check (init : Acts) (next : Val → Acts) (invs : Array Invariant) : IO UInt32 := do
  let mut g : Graph := {}
  let mut bad : Array (Option Nat) := invs.map fun _ => none
  let mut frontier : Array Nat := #[]
  for s in init do
    let (g', fresh) := g.add s g.states.size
    g := g'
    if fresh then frontier := frontier.push (g.states.size - 1)
  let mut head := 0
  while head < frontier.size do
    let i := frontier[head]!
    head := head + 1
    let s := g.states[i]!
    for k in [0:invs.size] do
      if bad[k]!.isNone && !(invs[k]!.2 s) then bad := bad.set! k (some i)
    for t in next s do
      let (g', fresh) := g.add t i
      g := g'
      if fresh then frontier := frontier.push (g.states.size - 1)
  IO.println s!"STATES {g.states.size}"
  for k in [0:invs.size] do
    IO.println s!"{if bad[k]!.isSome then "FAIL" else "OK"} {invs[k]!.1}"
  for k in [0:invs.size] do
    if let some i := bad[k]! then
      IO.println s!"TRACE {invs[k]!.1}\n{g.trace i}END"
  return if bad.any Option.isSome then 1 else 0

end Tla
