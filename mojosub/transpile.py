"""Python subset -> Mojo source.

The subset is deliberately small and fully typed: scalars (int/float/bool),
flat float/int buffers, arithmetic, control flow, and calls between functions
defined in the same unit. Anything outside it raises `Unsupported`, and the
caller falls back to CPython.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field

from .types import ARRAY_F64, ARRAY_I64, BOOL, FLOAT, INT, NONE, Array, Scalar, Ty, promote

MATH_1 = {
    "sqrt": FLOAT, "exp": FLOAT, "log": FLOAT, "log2": FLOAT, "log10": FLOAT,
    "sin": FLOAT, "cos": FLOAT, "tan": FLOAT, "tanh": FLOAT, "atan": FLOAT,
    "asin": FLOAT, "acos": FLOAT, "sinh": FLOAT, "cosh": FLOAT, "floor": FLOAT,
    "ceil": FLOAT, "erf": FLOAT,
}
MATH_2 = {"atan2": FLOAT, "pow": FLOAT}


class Unsupported(Exception):
    pass


@dataclass
class Signature:
    name: str
    params: list[tuple[str, Ty]]
    ret: Ty

    @property
    def c_argtypes(self) -> list:
        out = []
        for _, ty in self.params:
            out.append(ty.ctype)
            if isinstance(ty, Array):
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


def _coerce(code: str, have: Ty, want: Ty) -> str:
    if have is want:
        return code
    if isinstance(have, Array) or isinstance(want, Array):
        raise Unsupported("cannot convert buffers")
    if want is FLOAT:
        return f"Float64({code})"
    if want is INT:
        return f"Int({code})"
    if want is BOOL:
        return f"({code} != 0)"
    raise Unsupported(f"cannot convert {have} to {want}")


class _Fn:
    """Emits one function body."""

    def __init__(self, unit_sigs: dict[str, Signature], sig: Signature):
        self.sigs = unit_sigs
        self.sig = sig
        self.env: dict[str, Ty] = {}
        self.lines: list[str] = []
        for name, ty in sig.params:
            self.env[name] = ty
            if isinstance(ty, Array):
                self.env[name + "_len"] = INT

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
            raise Unsupported(f"unbound name {n.id}")
        return n.id, ty

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
        base, bt = self.expr(n.value)
        if not isinstance(bt, Array):
            raise Unsupported("subscript of non-buffer")
        ic, it = self.expr(n.slice)
        return f"{base}[{_coerce(ic, it, INT)}]", bt.elem

    def e_Call(self, n) -> tuple[str, Ty]:
        if n.keywords:
            raise Unsupported("keyword arguments")
        fname = self._callee(n.func)
        args = [self.expr(a) for a in n.args]
        if fname in ("int", "float", "bool") and len(args) == 1:
            t = {"int": INT, "float": FLOAT, "bool": BOOL}[fname]
            return _coerce(args[0][0], args[0][1], t), t
        if fname == "len" and len(args) == 1:
            if not isinstance(n.args[0], ast.Name) or not isinstance(args[0][1], Array):
                raise Unsupported("len of non-buffer")
            return n.args[0].id + "_len", INT
        if fname == "abs" and len(args) == 1:
            return f"abs({args[0][0]})", args[0][1]
        if fname in ("min", "max") and len(args) == 2:
            t = promote(args[0][1], args[1][1])
            a = _coerce(args[0][0], args[0][1], t)
            b = _coerce(args[1][0], args[1][1], t)
            return f"{fname}({a}, {b})", t
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
            for (code, ty), (_, want) in zip(args, sig.params):
                if isinstance(want, Array):
                    if ty is not want:
                        raise Unsupported("buffer type mismatch")
                    rendered.append(code)
                    rendered.append(code + "_len")
                else:
                    rendered.append(_coerce(code, ty, want))
            return f"{fname}({', '.join(rendered)})", sig.ret
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
            if not isinstance(bt, Array):
                raise Unsupported("store into non-buffer")
            ic, it = self.expr(target.slice)
            self.out(depth, f"{base}[{_coerce(ic, it, INT)}] = "
                            f"{_coerce(code, ty, bt.elem)}")
            return True
        if not isinstance(target, ast.Name):
            raise Unsupported("assignment target")
        name = target.id
        known = self.env.get(name)
        if known is None:
            self.env[name] = ty
            self.out(depth, f"var {name}: {ty.mojo} = {code}")
        else:
            self.out(depth, f"{name} = {_coerce(code, ty, known)}")
        return True

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

    def s_For(self, n, depth) -> bool:
        if n.orelse or not isinstance(n.target, ast.Name):
            raise Unsupported("for form")
        if not (isinstance(n.iter, ast.Call) and self._callee_safe(n.iter.func) == "range"):
            raise Unsupported("for over non-range")
        args = []
        for a in n.iter.args:
            c, t = self.expr(a)
            args.append(_coerce(c, t, INT))
        if not 1 <= len(args) <= 3:
            raise Unsupported("range arity")
        self.env[n.target.id] = INT
        self.out(depth, f"for {n.target.id} in range({', '.join(args)}):")
        self.block(n.body, depth + 1)
        return True

    @staticmethod
    def _callee_safe(node) -> str:
        return node.id if isinstance(node, ast.Name) else ""

    def render(self) -> str:
        params = []
        for name, ty in self.sig.params:
            params.append(f"{name}: {ty.mojo}")
            if isinstance(ty, Array):
                params.append(f"{name}_len: Int")
        head = f"def {self.sig.name}({', '.join(params)})"
        if self.sig.ret is not NONE:
            head += f" -> {self.sig.ret.mojo}"
        head += ":"
        return head + "\n" + "\n".join(self.lines)


PRELUDE = (
    "from std.math import sqrt, exp, log, log2, log10, sin, cos, tan, tanh, "
    "atan, asin, acos, sinh, cosh, floor, ceil, erf, atan2\n"
)


def transpile(tree: ast.Module, sigs: dict[str, Signature], entry: str) -> Unit:
    """Render `tree`'s functions as Mojo, exporting `entry` under a C symbol."""
    bodies = []
    fns = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    for name, sig in sigs.items():
        fn = fns.get(name)
        if fn is None:
            raise Unsupported(f"no such function {name}")
        if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.defaults:
            raise Unsupported("only positional parameters are supported")
        emitter = _Fn(sigs, sig)
        emitter.block(fn.body, 1)
        bodies.append(emitter.render())

    esig = sigs[entry]
    call_args = []
    wrap_params = []
    for name, ty in esig.params:
        if isinstance(ty, Array):
            wrap_params.append(f"{name}_addr: Int")
            wrap_params.append(f"{name}_len: Int")
            call_args.append(
                f"UnsafePointer[{ty.elem.mojo}, AnyOrigin[mut=True]]"
                f"(unsafe_from_address={name}_addr)"
            )
            call_args.append(f"{name}_len")
        else:
            wrap_params.append(f"{name}: {ty.mojo}")
            call_args.append(name)
    ret = "" if esig.ret is NONE else f" -> {esig.ret.mojo}"
    kw = "" if esig.ret is NONE else "return "
    wrapper = textwrap.dedent(f"""
        @export("mojosub_{entry}")
        def mojosub_wrap_{entry}({', '.join(wrap_params)}) abi("C"){ret}:
            {kw}{entry}({', '.join(call_args)})
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
