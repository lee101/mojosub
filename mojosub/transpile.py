"""Python subset -> Mojo source.

The subset is deliberately small and fully typed: scalars (int/float/bool),
flat float/int buffers, arithmetic, control flow, and calls between functions
defined in the same unit. Anything outside it raises `Unsupported`, and the
caller falls back to CPython.
"""

from __future__ import annotations

import ast
import math
import textwrap
from dataclasses import dataclass, field

from .types import (ARRAY_F64, ARRAY_I64, BOOL, FLOAT, INT, NONE, Array, Matrix,
                    Scalar, Ty, promote)

MAX_TYPE_PASSES = 4

VEC_WIDTH = 4
VEC_I = "__vi"
MAP_VECTOR_MIN_OPS = 3


# Every Python identifier is emitted into its own prefixed namespace, and the
# buffer lengths we synthesise into a second one. Nothing else is allowed to
# name a Mojo identifier.
#
# This is not cosmetic. Three separate bugs live in the obvious approach of
# emitting user names verbatim, and all three were found in one snippet of
# ordinary statistics code:
#
#   var = sq / n - mean * mean      -> `var var: Float64 = ...`, a syntax error.
#                                      `var`, `fn`, `alias`, `struct`, `ref`,
#                                      `owned` and `inout` are all legal Python
#                                      names and reserved Mojo ones. `var` for
#                                      a variance is not an unusual thing to
#                                      write; it cost five seconds of compiler
#                                      per submission and never once succeeded.
#   xs_len = 3                      -> silently reassigned the length parameter
#                                      synthesised for buffer `xs`, so every
#                                      later `len(xs)` returned 3. A WRONG
#                                      ANSWER, which is worse than a slow one.
#   Int = 4                         -> shadowed the type it is then used to cast
#                                      with.
#
# `u_` holds exactly the user's names and `n_` exactly the lengths of the user's
# buffers. Both maps are injective and their images are disjoint, so no user
# name can collide with a Mojo keyword, with a generated length, or with each
# other — including the awkward case of a program that has both `xs` and
# `xs_len`. Generated temporaries keep their leading underscores and cannot
# collide with either.
_USER = "u_"
_LEN = "n_"
_COLS = "c_"
# A fourth namespace, for the incoming value of a scalar parameter the body
# reassigns. Mojo function parameters are immutable bindings, so `n = n * 2` —
# entirely ordinary Python, and the first thing the fuzzer generated — emits
# `u_n = (u_n * 2)` and the compiler rejects it as a redefinition. The parameter
# arrives as `a_n` and the body gets a mutable `var u_n = a_n`.
_ARG = "a_"


def _u(name: str) -> str:
    """The Mojo identifier for a user's Python name."""
    return _USER + name


def _len_of(name: str) -> str:
    """The Mojo identifier for the synthesised length of user buffer `name`.

    For a 2-D buffer this is the row count, matching what `len()` means for a
    numpy array.
    """
    return _LEN + name


def _cols_of(name: str) -> str:
    """The Mojo identifier for the synthesised column count of matrix `name`."""
    return _COLS + name


def _arg_of(name: str) -> str:
    """The Mojo identifier a reassigned scalar parameter arrives under."""
    return _ARG + name


def _op_count(node) -> int:
    return sum(1 for n in ast.walk(node) if isinstance(n, (ast.BinOp, ast.UnaryOp)))

MATH_1 = {
    "sqrt": FLOAT, "exp": FLOAT, "log": FLOAT, "log2": FLOAT, "log10": FLOAT,
    "sin": FLOAT, "cos": FLOAT, "tan": FLOAT, "tanh": FLOAT, "atan": FLOAT,
    "asin": FLOAT, "acos": FLOAT, "sinh": FLOAT, "cosh": FLOAT, "floor": FLOAT,
    "ceil": FLOAT, "erf": FLOAT,
}
MATH_2 = {"atan2": FLOAT, "pow": FLOAT}
# `math.pi` is ordinary numeric code and was `Unsupported: expression
# Attribute`, which costs the whole kernel. Folded as a literal, like a module
# constant, so it is part of the unit's content hash.
MATH_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


class Unsupported(Exception):
    pass


@dataclass
class Signature:
    name: str
    params: list[tuple[str, Ty]]
    ret: Ty
    # Which buffer parameters the compiled code may store into. Filled in by
    # `transpile`, which is the only thing that can know: it has just decided
    # what to emit.
    #
    # It matters because a sequence that is not already a buffer has to be
    # copied into one for the call, and copied back out afterwards in case the
    # kernel wrote to it. For a reduction — `dot`, `sum`, a rolling statistic —
    # nothing is written, and the copy back is pure waste on the hot path. None
    # means "not analysed", which is read as "may write everything".
    writes: frozenset[str] | None = None

    def may_write(self, param: str) -> bool:
        return self.writes is None or param in self.writes

    @property
    def c_argtypes(self) -> list:
        out = []
        for _, ty in self.params:
            out.append(ty.ctype)
            if isinstance(ty, Matrix):
                out.append(INT.ctype)  # rows
                out.append(INT.ctype)  # cols
            elif isinstance(ty, Array):
                out.append(INT.ctype)  # length
        return out


@dataclass
class Unit:
    """One compilation unit: a set of functions plus their exported symbols."""

    source: str
    signatures: dict[str, Signature] = field(default_factory=dict)


def _fmt_const(v, ty: Ty) -> str:
    if ty is BOOL:
        return "True" if v else "False"
    if ty is FLOAT:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            raise Unsupported("non-finite literal")
        return repr(f) if ("." in repr(f) or "e" in repr(f)) else repr(f) + ".0"
    return repr(int(v))


def _zero(ty: Ty) -> str:
    """The value a hoisted declaration starts at.

    Never observable: Python cannot read a local before its first assignment
    either — it raises `UnboundLocalError` — so any program that would see this
    value is already broken under CPython and never reaches the compiled tier.
    """
    if ty is BOOL:
        return "False"
    if ty is FLOAT:
        return "0.0"
    return "0"


def _coerce(code: str, have: Ty, want: Ty) -> str:
    if have is want:
        return code
    if isinstance(have, (Array, Matrix)) or isinstance(want, (Array, Matrix)):
        raise Unsupported("cannot convert buffers")
    if want is FLOAT:
        return f"Float64({code})"
    if want is INT:
        return f"Int({code})"
    if want is BOOL:
        return f"({code} != 0)"
    raise Unsupported(f"cannot convert {have} to {want}")


def module_constants(tree: ast.Module) -> dict[str, tuple[str, Ty]]:
    """Module-level numeric constants, ready to fold into the emitted Mojo.

    `WINDOW = 64` at the top of a file and `for i in range(WINDOW)` inside the
    kernel is ordinary, good Python and was the single most common reason a
    numeric-looking function would not compile: to the transpiler `WINDOW` is
    an unbound name, because there is no environment beyond the function.

    Folding the *literal* rather than looking the value up at call time is what
    keeps this safe. The emitted Mojo contains the number, so the unit's content
    hash changes when the number does — a program that ships a new constant
    cannot be handed a library built for the old one.

    A name is only folded when the module binds it exactly once and nothing
    else anywhere in the module ever assigns to it. A constant that is really a
    variable would otherwise be frozen at whatever it was when we read it.
    """
    bound: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound[node.id] = bound.get(node.id, 0) + 1
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                bound[name] = bound.get(name, 0) + 2   # never fold these
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            bound[node.name] = bound.get(node.name, 0) + 2

    consts: dict[str, tuple[str, Ty]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name) or bound.get(target.id, 0) != 1:
            continue
        literal = _numeric_literal(value)
        if literal is not None:
            consts[target.id] = literal
    return consts


def _numeric_literal(node) -> tuple[str, Ty] | None:
    """`3`, `0.5`, `True`, `-1` — and nothing that could run code."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _numeric_literal(node.operand)
        if inner is None:
            return None
        code, ty = inner
        return (f"(-{code})" if isinstance(node.op, ast.USub) else code), ty
    if not isinstance(node, ast.Constant):
        return None
    v = node.value
    if isinstance(v, bool):
        return _fmt_const(v, BOOL), BOOL
    if isinstance(v, int):
        return _fmt_const(v, INT), INT
    if isinstance(v, float):
        return _fmt_const(v, FLOAT), FLOAT
    return None


class _Fn:
    """Emits one function body."""

    def __init__(self, unit_sigs: dict[str, Signature], sig: Signature,
                 vectorize: bool = True, consts: dict | None = None,
                 escaping: frozenset[str] = frozenset(),
                 pretyped: dict[str, Ty] | None = None):
        self.sigs = unit_sigs
        self.sig = sig
        self.vectorize = vectorize
        self.consts = consts or {}
        # A Python local has one type per assignment; a Mojo `var` has one for
        # ever. `t = 0` followed by `t += xs[i]` is the commonest shape there
        # is, and declaring `t` from its first assignment made every later
        # store `Int(...)` — a silently truncated answer, not a slow one. The
        # emitter records the join it needed (`widened`) and `emit_body` runs
        # it again with those types seeded, which is what `pretyped` carries.
        self.pretyped = pretyped or {}
        self.widened: dict[str, Ty] = {}
        # Names bound by a `for` target: their type is fixed at Int by the
        # loop, so a wider one cannot be given to them.
        self.loop_targets: set[str] = set()
        # Loop targets that are read after their loop has ended. Python leaves a
        # `for` target bound at its last value; Mojo scopes it to the loop body,
        # so `for i in range(n): ...` then `return xs[i]` compiles in Python and
        # is an unknown declaration in Mojo. These get a hoisted mutable local.
        self.escaping = escaping
        self._hoisted: set[str] = set()
        self._vec_index: str | None = None
        self._vec_acc: str | None = None
        self.env: dict[str, Ty] = {}
        self.lines: list[str] = []
        # Buffers this function may store through, by the name it knows them by.
        self.writes: set[str] = set()
        # Scalar parameters the body assigns to. Each needs a mutable local,
        # because a Mojo parameter cannot be rebound.
        self.reassigned: set[str] = set()
        # Locals in declaration order, with the type inferred at first
        # assignment. Declared at the top of the emitted function rather than
        # where they first appear, because Python scopes a local to the whole
        # function and Mojo scopes a `var` to its block — so `if c: t = 1.0`
        # followed by `return t` compiles in Python and is an unknown
        # declaration in Mojo. Hoisting is the only thing that matches Python's
        # scoping; emitting `var` at the assignment site cannot.
        self.locals: dict[str, Ty] = {}
        self.param_names = {name for name, _ in sig.params}
        for name, ty in sig.params:
            self.env[name] = ty

    # -- expressions ----------------------------------------------------
    def expr(self, node) -> tuple[str, Ty]:
        m = getattr(self, "e_" + type(node).__name__, None)
        if m is None:
            raise Unsupported(f"expression {type(node).__name__}")
        return m(node)

    def e_Constant(self, n) -> tuple[str, Ty]:
        v = n.value
        if isinstance(v, bool):
            return _fmt_const(v, BOOL), BOOL
        if isinstance(v, int):
            return _fmt_const(v, INT), INT
        if isinstance(v, float):
            return _fmt_const(v, FLOAT), FLOAT
        raise Unsupported(f"literal {v!r}")

    def e_Name(self, n) -> tuple[str, Ty]:
        ty = self.env.get(n.id)
        if ty is None:
            # A module-level constant, folded in as its literal. Locals and
            # parameters are checked first, so a name the function binds always
            # wins over one the module does.
            const = self.consts.get(n.id)
            if const is not None:
                return const
            raise Unsupported(f"unbound name {n.id}")
        return self.name(n.id), ty

    @staticmethod
    def name(python_name: str) -> str:
        """The Mojo identifier for a name the user wrote."""
        return _u(python_name)

    def e_Attribute(self, n) -> tuple[str, Ty]:
        if isinstance(n.value, ast.Name) and n.value.id in ("math", "np", "numpy"):
            value = MATH_CONSTS.get(n.attr)
            if value is not None:
                return _fmt_const(value, FLOAT), FLOAT
        raise Unsupported(f"attribute {n.attr}")

    def e_UnaryOp(self, n) -> tuple[str, Ty]:
        code, ty = self.expr(n.operand)
        if isinstance(n.op, ast.USub):
            return f"(-{code})", ty
        if isinstance(n.op, ast.UAdd):
            return code, ty
        if isinstance(n.op, ast.Not):
            return f"(not {_coerce(code, ty, BOOL)})", BOOL
        raise Unsupported("unary operator")

    def e_BinOp(self, n) -> tuple[str, Ty]:
        lc, lt = self.expr(n.left)
        rc, rt = self.expr(n.right)
        op = n.op
        if isinstance(op, ast.Div):
            t = FLOAT
            return f"({_coerce(lc, lt, t)} / {_coerce(rc, rt, t)})", t
        if isinstance(op, ast.Pow):
            t = promote(lt, rt)
            t = FLOAT if t is BOOL else t
            if t is INT:
                # `2 ** -1` is 0.5 in Python and 0 in Mojo, because CPython
                # switches to float for a negative exponent and an Int power
                # cannot. A literal says which it is; anything else is refused
                # rather than guessed, since guessing wrong is a wrong answer.
                exponent = _numeric_literal(n.right)
                if exponent is None:
                    raise Unsupported(
                        "integer ** a possibly negative exponent (cast to float)")
                if exponent[0].startswith("(-"):
                    t = FLOAT
            return f"({_coerce(lc, lt, t)} ** {_coerce(rc, rt, t)})", t
        sym = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//",
            ast.Mod: "%", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
            ast.LShift: "<<", ast.RShift: ">>",
        }.get(type(op))
        if sym is None:
            raise Unsupported(f"binary operator {type(op).__name__}")
        t = promote(lt, rt)
        if sym in ("&", "|", "^", "<<", ">>", "//", "%") and t is BOOL:
            t = INT
        if sym in ("&", "|", "^", "<<", ">>") and t is FLOAT:
            raise Unsupported("bitwise op on float")
        if t is BOOL:
            t = INT
        return f"({_coerce(lc, lt, t)} {sym} {_coerce(rc, rt, t)})", t

    def e_BoolOp(self, n) -> tuple[str, Ty]:
        parts = []
        for v in n.values:
            c, t = self.expr(v)
            parts.append(_coerce(c, t, BOOL))
        joiner = " and " if isinstance(n.op, ast.And) else " or "
        return "(" + joiner.join(parts) + ")", BOOL

    def e_Compare(self, n) -> tuple[str, Ty]:
        sym = {
            ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
            ast.Eq: "==", ast.NotEq: "!=",
        }
        code, ty = self.expr(n.left)
        out = []
        for op, comp in zip(n.ops, n.comparators):
            s = sym.get(type(op))
            if s is None:
                raise Unsupported("comparison operator")
            rc, rt = self.expr(comp)
            t = promote(ty, rt)
            out.append(f"({_coerce(code, ty, t)} {s} {_coerce(rc, rt, t)})")
            code, ty = rc, rt
        return "(" + " and ".join(out) + ")", BOOL

    def e_IfExp(self, n) -> tuple[str, Ty]:
        cc, ct = self.expr(n.test)
        ac, at = self.expr(n.body)
        bc, bt = self.expr(n.orelse)
        t = promote(at, bt)
        return (f"({_coerce(ac, at, t)} if {_coerce(cc, ct, BOOL)} "
                f"else {_coerce(bc, bt, t)})"), t

    def e_Subscript(self, n) -> tuple[str, Ty]:
        shape = self._shape_read(n)
        if shape is not None:
            return shape
        base, bt = self.expr(n.value)
        if isinstance(bt, Matrix):
            return self._matrix_index(n, base, bt), bt.elem
        if not isinstance(bt, Array):
            raise Unsupported("subscript of non-buffer")
        if isinstance(n.slice, ast.Tuple):
            raise Unsupported("2-D index into a 1-D buffer")
        ic, it = self.expr(n.slice)
        return f"{base}[{_coerce(ic, it, INT)}]", bt.elem

    def _matrix_index(self, n, base: str, bt: Matrix) -> str:
        """`m[i, j]` as a flat offset into row-major storage."""
        if not isinstance(n.slice, ast.Tuple):
            # `m[i]` would be a row — a view, which means carrying a length and
            # an origin around. Refused rather than half-supported.
            raise Unsupported("a matrix must be indexed as m[i, j]")
        if len(n.slice.elts) != 2:
            raise Unsupported("a matrix takes exactly two indices")
        if not isinstance(n.value, ast.Name):
            raise Unsupported("matrix index base must be a name")
        row_c, row_t = self.expr(n.slice.elts[0])
        col_c, col_t = self.expr(n.slice.elts[1])
        row = _coerce(row_c, row_t, INT)
        col = _coerce(col_c, col_t, INT)
        return f"{base}[({row}) * {_cols_of(n.value.id)} + ({col})]"

    def _shape_read(self, n) -> tuple[str, Ty] | None:
        """`m.shape[0]` and `m.shape[1]`, which is how the shape is spelled.

        Attribute access is otherwise entirely outside the subset, and stays so:
        this recognises the one exact pattern `<name>.shape[<0 or 1>]`, because
        that is what a caller who wrote their kernel against numpy has already
        written, and the alternative is asking them to change the program in
        order to make it fast.
        """
        value = n.value
        if not (isinstance(value, ast.Attribute) and value.attr == "shape"
                and isinstance(value.value, ast.Name)):
            return None
        name = value.value.id
        ty = self.env.get(name)
        if not isinstance(ty, (Array, Matrix)):
            raise Unsupported("shape of a non-buffer")
        if not (isinstance(n.slice, ast.Constant) and n.slice.value in (0, 1)):
            raise Unsupported("shape index must be the literal 0 or 1")
        axis = n.slice.value
        if isinstance(ty, Array):
            if axis != 0:
                raise Unsupported("a 1-D buffer has no axis 1")
            return _len_of(name), INT
        return (_len_of(name) if axis == 0 else _cols_of(name)), INT

    def e_Call(self, n) -> tuple[str, Ty]:
        if n.keywords:
            raise Unsupported("keyword arguments")
        fname = self._callee(n.func)
        args = [self.expr(a) for a in n.args]
        if fname in ("int", "float", "bool") and len(args) == 1:
            t = {"int": INT, "float": FLOAT, "bool": BOOL}[fname]
            return _coerce(args[0][0], args[0][1], t), t
        if fname == "len" and len(args) == 1:
            if not isinstance(n.args[0], ast.Name) or \
                    not isinstance(args[0][1], (Array, Matrix)):
                raise Unsupported("len of non-buffer")
            # For a matrix this is the row count, which is what len() means for
            # a 2-D numpy array.
            return _len_of(n.args[0].id), INT
        if fname == "abs" and len(args) == 1:
            return f"abs({args[0][0]})", args[0][1]
        if fname in ("min", "max") and len(args) >= 2:
            # Mojo's min/max are binary; Python's are n-ary, so they nest.
            t = args[0][1]
            for _, ty in args[1:]:
                t = promote(t, ty)
            out = _coerce(args[0][0], args[0][1], t)
            for code, ty in args[1:]:
                out = f"{fname}({out}, {_coerce(code, ty, t)})"
            return out, t
        if fname in MATH_1 and len(args) == 1:
            return f"{fname}({_coerce(args[0][0], args[0][1], FLOAT)})", FLOAT
        if fname in MATH_2 and len(args) == 2:
            a = _coerce(args[0][0], args[0][1], FLOAT)
            b = _coerce(args[1][0], args[1][1], FLOAT)
            return f"{fname}({a}, {b})", FLOAT
        sig = self.sigs.get(fname)
        if sig is not None:
            if len(args) != len(sig.params):
                raise Unsupported("arity mismatch")
            rendered = []
            for node, (code, ty), (_, want) in zip(n.args, args, sig.params):
                if isinstance(want, (Array, Matrix)):
                    if ty is not want:
                        raise Unsupported("buffer type mismatch")
                    if not isinstance(node, ast.Name):
                        # A buffer only ever arrives bound to a name, and its
                        # length is derived from that name — there is nothing
                        # to derive one from for any other expression.
                        raise Unsupported("buffer argument must be a name")
                    # A buffer handed to a helper is assumed written. Following
                    # it into the callee is a real escape analysis, and the
                    # error mode of getting that wrong is a silently stale array
                    # in the caller's hands — so this stays conservative and
                    # only costs a copy back that might not have been needed.
                    self.writes.add(node.id)
                    rendered.append(code)
                    rendered.append(_len_of(node.id))
                    if isinstance(want, Matrix):
                        rendered.append(_cols_of(node.id))
                else:
                    rendered.append(_coerce(code, ty, want))
            return f"{_u(fname)}({', '.join(rendered)})", sig.ret
        raise Unsupported(f"call to {fname}")

    @staticmethod
    def _callee(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in ("math", "np", "numpy"):
                return node.attr
        raise Unsupported("indirect call")

    # -- statements -----------------------------------------------------
    def block(self, body, depth: int) -> None:
        emitted = False
        for stmt in body:
            emitted |= self.stmt(stmt, depth)
        if not emitted:
            self.lines.append("    " * depth + "pass")

    def stmt(self, node, depth: int) -> bool:
        m = getattr(self, "s_" + type(node).__name__, None)
        if m is None:
            raise Unsupported(f"statement {type(node).__name__}")
        return m(node, depth)

    def out(self, depth: int, text: str) -> None:
        self.lines.append("    " * depth + text)

    def s_Pass(self, n, depth) -> bool:
        self.out(depth, "pass")
        return True

    def s_Break(self, n, depth) -> bool:
        self.out(depth, "break")
        return True

    def s_Continue(self, n, depth) -> bool:
        self.out(depth, "continue")
        return True

    def s_Expr(self, n, depth) -> bool:
        if isinstance(n.value, ast.Constant):  # docstring
            return False
        code, _ = self.expr(n.value)
        self.out(depth, "_ = " + code)
        return True

    def s_Return(self, n, depth) -> bool:
        if n.value is None:
            self.out(depth, "return")
            return True
        code, ty = self.expr(n.value)
        self.out(depth, "return " + _coerce(code, ty, self.sig.ret))
        return True

    def s_Assign(self, n, depth) -> bool:
        if len(n.targets) != 1:
            raise Unsupported("chained assignment")
        return self._assign(n.targets[0], n.value, depth)

    def s_AnnAssign(self, n, depth) -> bool:
        if n.value is None:
            raise Unsupported("bare annotation")
        return self._assign(n.target, n.value, depth)

    def _assign(self, target, value, depth) -> bool:
        code, ty = self.expr(value)
        if isinstance(target, ast.Subscript):
            base, bt = self.expr(target.value)
            if not isinstance(bt, (Array, Matrix)):
                raise Unsupported("store into non-buffer")
            if isinstance(target.value, ast.Name):
                self.writes.add(target.value.id)
            if isinstance(bt, Matrix):
                slot = self._matrix_index(target, base, bt)
            else:
                if isinstance(target.slice, ast.Tuple):
                    raise Unsupported("2-D index into a 1-D buffer")
                ic, it = self.expr(target.slice)
                slot = f"{base}[{_coerce(ic, it, INT)}]"
            self.out(depth, f"{slot} = {_coerce(code, ty, bt.elem)}")
            return True
        if not isinstance(target, ast.Name):
            raise Unsupported("assignment target")
        name = target.id
        known = self.env.get(name)
        if isinstance(known, (Array, Matrix)):
            # Rebinding a buffer parameter would leave its length behind,
            # pointing at the old one.
            raise Unsupported("cannot reassign a buffer parameter")
        if known is None:
            if isinstance(ty, (Array, Matrix)):
                # A second name for a buffer would need its length to follow it,
                # and nothing derives one from anything but a parameter name.
                # Emitting it anyway declares `var u_b: UnsafePointer[...] = 0`
                # and burns a compile to find that out.
                raise Unsupported("cannot bind a buffer to a local")
            declared = self.pretyped.get(name, ty)
            self.env[name] = declared
            self.locals[name] = declared
            self.out(depth, f"{self.name(name)} = {_coerce(code, ty, declared)}")
        else:
            wider = self._join(name, known, ty)
            if wider is not known:
                self.widened[name] = wider
                self.env[name] = wider
                known = wider
            if name in self.param_names:
                # Rebinding the parameter itself is a compile error; `render`
                # turns this into a mutable local seeded from the argument.
                self.reassigned.add(name)
            else:
                # A name known to the environment but not yet declared is a loop
                # target being assigned outside its loop; it needs a declaration
                # too. Rewriting the type of one already declared is what a
                # widening pass is for.
                self.locals[name] = known
            self.out(depth, f"{self.name(name)} = {_coerce(code, ty, known)}")
        return True

    def _join(self, name: str, known: Ty, ty: Ty) -> Ty:
        """The type a local needs to hold both what it has and what it is given.

        Python rebinds; Mojo does not. `t = 0` then `t = t + 1.5` used to emit
        `Int(...)` into an `Int` local and quietly return 2.0 where CPython
        returns 2.5 — a wrong answer, which is the one failure mode this
        project cannot have. Widening is only possible for a plain local: a
        parameter's type is fixed by the ABI and a loop counter's by its range.
        """
        if known is ty or not (isinstance(known, Scalar) and isinstance(ty, Scalar)):
            return known
        if known is NONE or ty is NONE:
            return known
        wider = promote(known, ty)
        if wider is known:
            return known
        if name in self.param_names:
            raise Unsupported(f"parameter {name} is reassigned to a wider type")
        if name in self.loop_targets:
            raise Unsupported(f"loop counter {name} is reassigned to a wider type")
        return wider

    def s_AugAssign(self, n, depth) -> bool:
        binop = ast.BinOp(left=n.target, op=n.op, right=n.value)
        ast.copy_location(binop, n)
        return self._assign(n.target, binop, depth)

    def s_If(self, n, depth) -> bool:
        cc, ct = self.expr(n.test)
        self.out(depth, f"if {_coerce(cc, ct, BOOL)}:")
        self.block(n.body, depth + 1)
        if n.orelse:
            self.out(depth, "else:")
            self.block(n.orelse, depth + 1)
        return True

    def s_While(self, n, depth) -> bool:
        if n.orelse:
            raise Unsupported("while/else")
        cc, ct = self.expr(n.test)
        self.out(depth, f"while {_coerce(cc, ct, BOOL)}:")
        self.block(n.body, depth + 1)
        return True

    # -- vectorization --------------------------------------------------
    def _vec_expr(self, node) -> str:
        """Render `node` as a float64 SIMD expression, or refuse.

        The only shapes allowed are the ones that vectorize cleanly: buffer
        loads at exactly the loop index, float scalars broadcast, and the four
        arithmetic operators. Anything else — a branch, a call, an index
        expression — bails out and the loop is emitted scalar.
        """
        if isinstance(node, ast.Subscript):
            base, bt = self.expr(node.value)
            if not isinstance(bt, Array) or bt.elem is not FLOAT:
                raise Unsupported("vector load of non-float buffer")
            if not (isinstance(node.slice, ast.Name) and node.slice.id == self._vec_index):
                raise Unsupported("vector load must be indexed by the loop variable")
            return f"{base}.load[width={VEC_WIDTH}]({VEC_I})"
        if isinstance(node, ast.Constant):
            code, ty = self.e_Constant(node)
            return f"SIMD[DType.float64, {VEC_WIDTH}]({_coerce(code, ty, FLOAT)})"
        if isinstance(node, ast.Name):
            if node.id == self._vec_index or node.id == self._vec_acc:
                raise Unsupported("loop-carried scalar in vector body")
            code, ty = self.e_Name(node)
            if ty not in (FLOAT, INT):
                raise Unsupported("vector operand must be numeric")
            return f"SIMD[DType.float64, {VEC_WIDTH}]({_coerce(code, ty, FLOAT)})"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            inner = self._vec_expr(node.operand)
            return f"(-{inner})" if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp):
            sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(
                type(node.op)
            )
            if sym is None:
                raise Unsupported("operator does not vectorize")
            return f"({self._vec_expr(node.left)} {sym} {self._vec_expr(node.right)})"
        raise Unsupported("expression does not vectorize")

    def _try_vectorize(self, n, depth) -> bool:
        """Emit a SIMD version of a one-statement map or reduce loop.

        Worth about 1.4x on a float reduction over a large buffer, which is the
        single most common shape in the code these functions come from.
        """
        if len(n.iter.args) != 1 or len(n.body) != 1:
            return False
        stmt = n.body[0]
        limit_code, limit_ty = self.expr(n.iter.args[0])
        limit = _coerce(limit_code, limit_ty, INT)
        self._vec_index = n.target.id
        self._vec_acc = None
        saved = list(self.lines)

        try:
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, ast.Add) \
                    and isinstance(stmt.target, ast.Name):
                acc = stmt.target.id
                if self.env.get(acc) is not FLOAT:
                    return False
                self._vec_acc = acc
                body = self._vec_expr(stmt.value)
                # Leading underscores: a generated temporary must not be able to
                # land in the user's namespace, whatever they called the
                # accumulator.
                partial = f"__v_{self.name(acc)}"
                self.out(depth, f"var {partial} = SIMD[DType.float64, {VEC_WIDTH}](0.0)")
                self.out(depth, f"var {VEC_I} = 0")
                self.out(depth, f"while {VEC_I} + {VEC_WIDTH} <= {limit}:")
                self.out(depth + 1, f"{partial} += {body}")
                self.out(depth + 1, f"{VEC_I} += {VEC_WIDTH}")
                self.out(depth, f"{self.name(acc)} += {partial}.reduce_add()")
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Subscript):
                target = stmt.targets[0]
                base, bt = self.expr(target.value)
                if not isinstance(bt, Array) or bt.elem is not FLOAT:
                    return False
                if not (isinstance(target.slice, ast.Name)
                        and target.slice.id == self._vec_index):
                    return False
                if _op_count(stmt.value) < MAP_VECTOR_MIN_OPS:
                    # A store loop this simple is memory-bound, and measurably
                    # slower vectorized (0.68ms -> 1.24ms on 1e6 elements) than
                    # what LLVM already does with the scalar form. Reductions
                    # have no such crossover, so only stores are gated.
                    return False
                body = self._vec_expr(stmt.value)
                self.out(depth, f"var {VEC_I} = 0")
                self.out(depth, f"while {VEC_I} + {VEC_WIDTH} <= {limit}:")
                self.out(depth + 1, f"{base}.store({VEC_I}, {body})")
                self.out(depth + 1, f"{VEC_I} += {VEC_WIDTH}")
            else:
                return False
        except Unsupported:
            self.lines = saved
            return False
        finally:
            self._vec_acc = None

        # the tail, scalar, for the elements the vector width did not cover
        self.env[self._vec_index] = INT
        self.loop_targets.add(self._vec_index)
        self.out(depth, f"for {self.name(self._vec_index)} in range({VEC_I}, {limit}):")
        self.block(n.body, depth + 1)
        return True

    def s_For(self, n, depth) -> bool:
        if n.orelse or not isinstance(n.target, ast.Name):
            raise Unsupported("for form")
        if not (isinstance(n.iter, ast.Call) and self._callee_safe(n.iter.func) == "range"):
            raise Unsupported("for over non-range")
        # A loop whose index outlives it is not vectorized: the vectorizer emits
        # its own scalar tail loop directly and would bypass the hoisting below.
        # Losing SIMD on a loop whose counter is read afterwards is a fair trade
        # for not emitting code that does not compile.
        if (self.vectorize and n.target.id not in self.escaping
                and self._try_vectorize(n, depth)):
            return True
        args = []
        for a in n.iter.args:
            c, t = self.expr(a)
            args.append(_coerce(c, t, INT))
        if not 1 <= len(args) <= 3:
            raise Unsupported("range arity")
        target = n.target.id
        self.env[target] = INT
        self.loop_targets.add(target)
        if target in self.escaping:
            # The name outlives the loop, so it cannot *be* the loop variable.
            # An iteration variable of our own drives the loop and copies into
            # it, which is what leaves the last value behind afterwards.
            #
            # The declaration goes in `locals`, not here: a `var` emitted at this
            # depth is still block-scoped, so a loop inside an `else:` whose
            # index is read after the `if` was exactly as broken as before.
            self.locals.setdefault(target, INT)
            iter_name = f"__lv_{self.name(target)}"
            self.out(depth, f"for {iter_name} in range({', '.join(args)}):")
            self.out(depth + 1, f"{self.name(target)} = {iter_name}")
            self.block(n.body, depth + 1)
            return True
        self.out(depth, f"for {self.name(target)} in range({', '.join(args)}):")
        self.block(n.body, depth + 1)
        return True

    @staticmethod
    def _callee_safe(node) -> str:
        return node.id if isinstance(node, ast.Name) else ""

    def render(self) -> str:
        params = []
        prologue = []
        for name, ty in self.locals.items():
            prologue.append(f"    var {self.name(name)}: {ty.mojo} = {_zero(ty)}")
        for name, ty in self.sig.params:
            if name in self.reassigned:
                # The body writes to this name, so it cannot be the parameter.
                params.append(f"{_arg_of(name)}: {ty.mojo}")
                prologue.append(
                    f"    var {_u(name)}: {ty.mojo} = {_arg_of(name)}")
            else:
                params.append(f"{_u(name)}: {ty.mojo}")
            if isinstance(ty, Matrix):
                params.append(f"{_len_of(name)}: Int")
                params.append(f"{_cols_of(name)}: Int")
            elif isinstance(ty, Array):
                params.append(f"{_len_of(name)}: Int")
        head = f"def {_u(self.sig.name)}({', '.join(params)})"
        if self.sig.ret is not NONE:
            head += f" -> {self.sig.ret.mojo}"
        head += ":"
        return head + "\n" + "\n".join(prologue + self.lines)


PRELUDE = (
    "from std.math import sqrt, exp, log, log2, log10, sin, cos, tan, tanh, "
    "atan, asin, acos, sinh, cosh, floor, ceil, erf, atan2\n"
)


def escaping_loop_targets(fn: ast.FunctionDef) -> frozenset[str]:
    """Loop targets this function reads outside the loop that binds them.

    Python's `for` target survives its loop; Mojo's does not. Computed per
    function so the emitted code only changes for the functions that need it —
    a hoisted variable in every loop would rewrite every unit's content hash and
    invalidate a cache full of correct libraries for no benefit.
    """
    out: set[str] = set()
    loops = [node for node in ast.walk(fn) if isinstance(node, ast.For)]
    for loop in loops:
        if not isinstance(loop.target, ast.Name):
            continue
        name = loop.target.id
        inside = {id(node) for stmt in loop.body for node in ast.walk(stmt)}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Name) or node.id != name:
                continue
            if isinstance(node.ctx, ast.Store) and node is not loop.target:
                # Python lets the body rebind the loop variable; a Mojo loop
                # variable is an immutable binding, so that has to become an
                # assignment to a hoisted local too.
                out.add(name)
                break
            if isinstance(node.ctx, ast.Load) and id(node) not in inside:
                out.add(name)
                break
    return frozenset(out)


def emit_body(make_emitter, body, depth: int) -> _Fn:
    """Emit `body`, re-emitting until every local's type stops widening.

    A local's Mojo type is fixed by its declaration, so it has to be the join of
    every value Python would put in it — which is only known once the body has
    been walked. The walk is pure (it builds strings), so it is simply run again
    with the wider types seeded. Two passes settle anything real; the cap is
    there so a pathological program cannot loop.
    """
    pretyped: dict[str, Ty] = {}
    for _ in range(MAX_TYPE_PASSES):
        emitter = make_emitter(pretyped)
        emitter.block(body, depth)
        if not emitter.widened:
            return emitter
        pretyped = {**pretyped, **emitter.widened}
    raise Unsupported("no stable type for a local")


def transpile(tree: ast.Module, sigs: dict[str, Signature], entry: str,
              vectorize: bool = True) -> Unit:
    """Render `tree`'s functions as Mojo, exporting `entry` under a C symbol."""
    bodies = []
    consts = module_constants(tree)
    fns = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    for name, sig in sigs.items():
        fn = fns.get(name)
        if fn is None:
            raise Unsupported(f"no such function {name}")
        if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.defaults:
            raise Unsupported("only positional parameters are supported")
        escaping = escaping_loop_targets(fn)
        emitter = emit_body(
            lambda pretyped: _Fn(sigs, sig, vectorize=vectorize, consts=consts,
                                 escaping=escaping, pretyped=pretyped),
            fn.body, 1)
        bodies.append(emitter.render())
        # Recorded on the signature object itself, which the caller is holding:
        # `Unit.signatures` and the caller's `sigs` share it, and the runtime
        # needs it to know whether a copied sequence has to be copied back.
        sig.writes = frozenset(emitter.writes)

    esig = sigs[entry]
    call_args = []
    wrap_params = []
    for name, ty in esig.params:
        if isinstance(ty, (Array, Matrix)):
            wrap_params.append(f"{_u(name)}_addr: Int")
            wrap_params.append(f"{_len_of(name)}: Int")
            if isinstance(ty, Matrix):
                wrap_params.append(f"{_cols_of(name)}: Int")
            call_args.append(
                f"UnsafePointer[{ty.elem.mojo}, AnyOrigin[mut=True]]"
                f"(unsafe_from_address={_u(name)}_addr)"
            )
            call_args.append(_len_of(name))
            if isinstance(ty, Matrix):
                call_args.append(_cols_of(name))
        else:
            wrap_params.append(f"{_u(name)}: {ty.mojo}")
            call_args.append(_u(name))
    ret = "" if esig.ret is NONE else f" -> {esig.ret.mojo}"
    kw = "" if esig.ret is NONE else "return "
    # The exported symbol keeps the Python name: it is a string the runtime
    # looks up by `sig.name`, not an identifier, so it is not part of any
    # namespace that could collide.
    wrapper = textwrap.dedent(f"""
        @export("mojosub_{entry}")
        def mojosub_wrap_{_u(entry)}({', '.join(wrap_params)}) abi("C"){ret}:
            {kw}{_u(entry)}({', '.join(call_args)})
    """).strip()

    src = PRELUDE + "\n" + "\n\n".join(bodies) + "\n\n" + wrapper + "\n"
    return Unit(source=src, signatures=dict(sigs))


def signature_from_annotations(fn: ast.FunctionDef) -> Signature | None:
    """Build a signature from annotations, or None if any are missing."""
    from .types import from_annotation

    params: list[tuple[str, Ty]] = []
    for arg in fn.args.args:
        if arg.annotation is None:
            return None
        params.append((arg.arg, from_annotation(ast.unparse(arg.annotation))))
    ret = NONE if fn.returns is None else from_annotation(ast.unparse(fn.returns))
    return Signature(fn.name, params, ret)
