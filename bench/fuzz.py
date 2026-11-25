"""Differential fuzzing: does the compiled function compute the same thing?

Generates random programs inside and just outside the compilable subset, runs
each under CPython and through the transpiler, and compares. Three kinds of
finding, in descending order of how much they matter:

  DISAGREE   both tiers ran and returned different answers. The worst possible
             bug in this project: the fast path is silently wrong.
  CRASH      the transpiler raised something that is not `Unsupported`, or the
             emitted Mojo did not compile. A refusal is fine — that is the
             documented contract — but it has to be a refusal, not a traceback,
             and the emitted source has to be valid Mojo whenever it is emitted.
  ACCEPTED   emitted and agreed. Counted so the generator can be judged: a
             fuzzer that only produces programs the transpiler refuses is
             testing the refusal path and nothing else.

    pixi run python bench/fuzz.py --programs 200
    pixi run python bench/fuzz.py --programs 500 --seed 7 --keep-going

Deterministic given `--seed`, so a finding can be replayed exactly.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import random
import sys
import traceback

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from mojosub.api import compile_function, type_of  # noqa: E402
from mojosub.transpile import Unsupported  # noqa: E402
from mojosub.types import NONE  # noqa: E402

try:
    import numpy as np
except ImportError:
    np = None


# --- program generation -----------------------------------------------------
#
# Grammar-directed rather than character-level: random bytes are rejected by
# `ast.parse` and test nothing. This emits programs that are plausibly in the
# subset, with a tunable chance of stepping just outside it — the boundary is
# where the interesting bugs are, because that is where a refusal and an
# acceptance are both defensible and only one is correct.

BIN_OPS = ["+", "-", "*", "/", "//", "%", "**"]
INT_OPS = ["+", "-", "*", "//", "%", "&", "|", "^", "<<", ">>"]
CMP_OPS = ["<", "<=", ">", ">=", "==", "!="]
MATH_1 = ["sqrt", "exp", "log", "sin", "cos", "tanh", "floor", "ceil", "erf"]


class Gen:
    def __init__(self, rng: random.Random, escape: float = 0.15):
        self.rng = rng
        self.escape = escape
        self.floats: list[str] = []
        self.ints: list[str] = []
        self.arrays: list[str] = []
        self.matrices: list[str] = []
        self.n = 0

    def fresh(self, prefix: str = "v") -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def float_expr(self, depth: int = 0) -> str:
        r = self.rng
        if depth > 2 or r.random() < 0.3:
            choices = ["const"]
            if self.floats:
                choices.append("var")
            if self.arrays:
                choices.append("index")
            if self.matrices:
                choices.append("mindex")
            if self.ints:
                choices.append("cast")
            what = r.choice(choices)
            if what == "const":
                # Small magnitudes: the point is to find miscompiles, not to
                # rediscover that 1e300 * 1e300 overflows in both tiers.
                return repr(round(r.uniform(-8.0, 8.0), 4))
            if what == "var":
                return r.choice(self.floats)
            if what == "index":
                arr = r.choice(self.arrays)
                return f"{arr}[{self.index_expr(arr)}]"
            if what == "mindex":
                m = r.choice(self.matrices)
                return (f"{m}[{self.row_expr(m)}, {self.col_expr(m)}]")
            return f"float({r.choice(self.ints)})"

        kind = r.random()
        if kind < 0.55:
            op = r.choice(BIN_OPS)
            left = self.float_expr(depth + 1)
            right = self.float_expr(depth + 1)
            if op in ("/", "//", "%"):
                # A zero divisor raises in CPython and is undefined in the
                # compiled form; that difference is a known one and not what is
                # being looked for here.
                right = f"({right} + 9.13)"
            if op == "**":
                # Fractional powers of negatives are complex in CPython.
                left = f"abs({left})"
                right = f"({self.float_expr(depth + 2)} % 3.0)"
            return f"({left} {op} {right})"
        if kind < 0.7:
            return f"{r.choice(MATH_1)}(abs({self.float_expr(depth + 1)}) + 1.0)"
        if kind < 0.8:
            return f"min({self.float_expr(depth + 1)}, {self.float_expr(depth + 1)})"
        if kind < 0.9:
            return f"abs({self.float_expr(depth + 1)})"
        return (f"({self.float_expr(depth + 1)} if {self.bool_expr(depth + 1)} "
                f"else {self.float_expr(depth + 1)})")

    def int_expr(self, depth: int = 0) -> str:
        r = self.rng
        if depth > 2 or r.random() < 0.4 or not self.ints:
            if self.ints and r.random() < 0.6:
                return r.choice(self.ints)
            # Kept well inside 2**63: Mojo's Int wraps where CPython promotes,
            # which is a documented difference rather than a bug to find.
            return str(r.randint(-1000, 1000))
        op = r.choice(INT_OPS)
        left = self.int_expr(depth + 1)
        right = self.int_expr(depth + 1)
        if op in ("//", "%"):
            right = f"(abs({right}) + 7)"
        if op in ("<<", ">>"):
            right = str(r.randint(0, 6))
        if op == "*":
            right = str(r.randint(-20, 20))
        return f"({left} {op} {right})"

    def bool_expr(self, depth: int = 0) -> str:
        r = self.rng
        a = self.float_expr(depth + 1)
        b = self.float_expr(depth + 1)
        expr = f"({a} {r.choice(CMP_OPS)} {b})"
        if r.random() < 0.25:
            other = self.bool_expr(depth + 1)
            expr = f"({expr} {r.choice(['and', 'or'])} {other})"
        if r.random() < 0.15:
            expr = f"(not {expr})"
        return expr

    def index_expr(self, arr: str) -> str:
        r = self.rng
        if r.random() < 0.7 and self.ints:
            return f"({r.choice(self.ints)} % len({arr}))"
        return f"({self.int_expr(1)} % len({arr}))"

    def row_expr(self, m: str) -> str:
        return f"({self.int_expr(1)} % {m}.shape[0])"

    def col_expr(self, m: str) -> str:
        return f"({self.int_expr(1)} % {m}.shape[1])"

    def statements(self, indent: int, depth: int, budget: int) -> list[str]:
        r = self.rng
        out: list[str] = []
        pad = "    " * indent
        for _ in range(budget):
            kind = r.random()
            if kind < 0.3 and self.floats:
                out.append(f"{pad}{r.choice(self.floats)} = {self.float_expr()}")
            elif kind < 0.42:
                name = self.fresh("f")
                out.append(f"{pad}{name} = {self.float_expr()}")
                self.floats.append(name)
            elif kind < 0.5 and self.ints:
                out.append(f"{pad}{r.choice(self.ints)} = {self.int_expr()}")
            elif kind < 0.6 and self.arrays:
                arr = r.choice(self.arrays)
                out.append(
                    f"{pad}{arr}[{self.index_expr(arr)}] = {self.float_expr()}")
            elif kind < 0.68 and self.matrices:
                m = r.choice(self.matrices)
                out.append(f"{pad}{m}[{self.row_expr(m)}, {self.col_expr(m)}] = "
                           f"{self.float_expr()}")
            elif kind < 0.78 and depth < 2:
                out.append(f"{pad}if {self.bool_expr()}:")
                out.extend(self.statements(indent + 1, depth + 1, max(1, budget // 2)))
                if r.random() < 0.4:
                    out.append(f"{pad}else:")
                    out.extend(
                        self.statements(indent + 1, depth + 1, max(1, budget // 2)))
            elif kind < 0.92 and depth < 2:
                idx = self.fresh("i")
                limit = r.randint(1, 9)
                out.append(f"{pad}for {idx} in range({limit}):")
                self.ints.append(idx)
                out.extend(self.statements(indent + 1, depth + 1, max(1, budget // 2)))
            elif kind < 0.96 and self.escape and r.random() < self.escape:
                # Deliberately outside the subset. The transpiler must *refuse*,
                # not crash, and must not emit something that fails to compile.
                out.append(r.choice([
                    f"{pad}pass",
                    f"{pad}{self.fresh('s')} = [1.0, 2.0]",
                    f"{pad}{self.fresh('s')} = {{}}",
                    f'{pad}{self.fresh("s")} = "text"',
                    f"{pad}{self.fresh('a')}, {self.fresh('b')} = 1.0, 2.0",
                ]))
            else:
                out.append(f"{pad}pass")
        return out or [f"{pad}pass"]

    def program(self) -> tuple[str, str, list]:
        """A function, its name, and concrete arguments to call it with."""
        r = self.rng
        name = "k"
        params: list[str] = []
        args: list = []
        for _ in range(r.randint(1, 3)):
            what = r.random()
            if what < 0.35:
                p = self.fresh("n")
                params.append(f"{p}: int")
                self.ints.append(p)
                args.append(r.randint(1, 12))
            elif what < 0.6:
                p = self.fresh("x")
                params.append(f"{p}: float")
                self.floats.append(p)
                args.append(round(r.uniform(-5, 5), 4))
            elif what < 0.85 or np is None:
                p = self.fresh("a")
                params.append(p)
                self.arrays.append(p)
                size = r.randint(2, 12)
                if np is not None:
                    args.append(np.array(
                        [round(r.uniform(-4, 4), 4) for _ in range(size)]))
                else:
                    args.append([round(r.uniform(-4, 4), 4) for _ in range(size)])
            else:
                p = self.fresh("m")
                params.append(p)
                self.matrices.append(p)
                rows, cols = r.randint(1, 4), r.randint(1, 4)
                args.append(np.ascontiguousarray(np.array(
                    [[round(r.uniform(-4, 4), 4) for _ in range(cols)]
                     for _ in range(rows)])))

        body = self.statements(1, 0, r.randint(2, 5))
        ret = self.float_expr()
        source = (f"import math\n\n\ndef {name}({', '.join(params)}) -> float:\n"
                  + "\n".join(body) + f"\n    return {ret}\n")
        # `math.` qualified, which is what the transpiler maps and what real code
        # writes; the generator emits bare names for brevity.
        for fn in MATH_1:
            source = source.replace(f" {fn}(", f" math.{fn}(")
            source = source.replace(f"({fn}(", f"(math.{fn}(")
        return source, name, args


# --- the differential check -------------------------------------------------

def snapshot(args):
    out = []
    for a in args:
        if hasattr(a, "copy"):
            out.append(a.copy())
        elif isinstance(a, list):
            out.append(list(a))
        else:
            out.append(a)
    return out


def restore(args, saved):
    for a, s in zip(args, saved):
        if hasattr(a, "__setitem__") and not isinstance(a, (int, float)):
            try:
                a[...] = s
            except (TypeError, IndexError):
                a[:] = s


def equalish(a, b, rtol=1e-6) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))
    return a == b


def buffers_equal(a, b) -> bool:
    fa = a.ravel() if hasattr(a, "ravel") else a
    fb = b.ravel() if hasattr(b, "ravel") else b
    if len(fa) != len(fb):
        return False
    return all(equalish(float(x), float(y)) for x, y in zip(list(fa), list(fb)))


def check(source: str, name: str, args) -> tuple[str, str]:
    """Returns (verdict, detail). Verdict is accepted / refused / DISAGREE / CRASH."""
    ns: dict = {"__mojosub_source__": source}
    try:
        exec(compile(source, "<fuzz>", "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return "skipped", f"generated program does not run: {exc}"
    fn = ns[name]

    before = snapshot(args)
    try:
        expected = fn(*args)
    except Exception as exc:  # noqa: BLE001
        return "skipped", f"CPython raised: {type(exc).__name__}: {exc}"
    after_python = snapshot(args)
    restore(args, before)

    try:
        argtypes = [type_of(a) for a in args]
    except Unsupported as exc:
        return "refused", str(exc)

    try:
        call, mojo_src = compile_function(fn, argtypes, type_of(expected)
                                         if expected is not None else NONE)
    except Unsupported as exc:
        return "refused", str(exc)
    except Exception as exc:  # noqa: BLE001
        return "CRASH", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    try:
        actual = call(*args)
    except Exception as exc:  # noqa: BLE001
        return "CRASH", f"dispatch raised {type(exc).__name__}: {exc}"

    if not equalish(expected, actual):
        return "DISAGREE", (f"return: python={expected!r} mojo={actual!r}\n"
                            f"--- python ---\n{source}\n--- mojo ---\n{mojo_src}")
    for i, (was, now) in enumerate(zip(after_python, args)):
        if hasattr(now, "__len__") and not buffers_equal(was, now):
            return "DISAGREE", (f"buffer {i} differs\npython={was!r}\nmojo={now!r}\n"
                                f"--- python ---\n{source}\n--- mojo ---\n{mojo_src}")
    return "accepted", ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--programs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--escape", type=float, default=0.15,
                    help="chance of emitting a statement outside the subset")
    ap.add_argument("--keep-going", action="store_true",
                    help="report every finding instead of stopping at the first")
    ap.add_argument("--build-timeout", type=float, default=90.0,
                    help="seconds a single mojo build may take before it is "
                         "killed and reported as a finding")
    args = ap.parse_args()
    # A real build is 3-5s. Some generated units make the compiler spin for
    # ever, which is a finding in itself - and without a cap it is a fuzzer that
    # stops on the first one and holds a core until somebody notices.
    os.environ["MOJOSUB_BUILD_TIMEOUT"] = repr(args.build_timeout)

    counts = {"accepted": 0, "refused": 0, "skipped": 0, "DISAGREE": 0, "CRASH": 0}
    reasons: dict[str, int] = {}
    findings = []
    for i in range(args.programs):
        rng = random.Random(args.seed * 1_000_003 + i)
        gen = Gen(rng, escape=args.escape)
        source, name, call_args = gen.program()
        verdict, detail = check(source, name, call_args)
        counts[verdict] += 1
        if verdict == "refused":
            reasons[detail[:60]] = reasons.get(detail[:60], 0) + 1
        if verdict in ("DISAGREE", "CRASH"):
            findings.append((i, verdict, detail, source))
            print(f"\n=== {verdict} at program {i} (seed {args.seed}) ===",
                  flush=True)
            print(detail[:4000], flush=True)
            if not args.keep_going:
                break
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.programs} "
                  + " ".join(f"{k}={v}" for k, v in counts.items() if v),
                  flush=True)

    print("\n" + " ".join(f"{k}={v}" for k, v in counts.items()))
    if counts["accepted"] == 0:
        print("WARNING: nothing was accepted — the generator is only testing "
              "the refusal path", file=sys.stderr)
    if reasons:
        print("\nmost common refusals:")
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {n:4}  {reason}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
