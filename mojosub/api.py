"""The user-facing surface: `jit`, `compile_function`, and `execute`."""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading
import time
from dataclasses import dataclass, field

from . import cache
from .runtime import Callable as _Callable
from .transpile import (
    Signature,
    Unsupported,
    signature_from_annotations,
    transpile,
)
from .types import ARRAY_F64, ARRAY_I64, BOOL, FLOAT, INT, NONE, Array, Ty

__all__ = [
    "jit",
    "compile_function",
    "compile_source",
    "compile_batch",
    "execute",
    "Stats",
    "Unsupported",
]


def type_of(value) -> Ty:
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        return INT
    if isinstance(value, float):
        return FLOAT
    info = getattr(value, "__array_interface__", None)
    if info is not None:
        if len(info["shape"]) != 1:
            raise Unsupported("only 1-D arrays are supported")
        return ARRAY_F64 if info["typestr"].endswith("f8") else ARRAY_I64
    if isinstance(value, list) and value:
        return ARRAY_F64 if any(isinstance(v, float) for v in value) else ARRAY_I64
    if isinstance(value, list):
        return ARRAY_F64
    import array as _array

    if isinstance(value, _array.array):
        return ARRAY_F64 if value.typecode == "d" else ARRAY_I64
    raise Unsupported(f"unsupported argument type {type(value).__name__}")


_SCALAR_TYPES = {bool: BOOL, int: INT, float: FLOAT}


@dataclass
class Stats:
    python_calls: int = 0
    mojo_calls: int = 0
    compiles: int = 0
    compile_seconds: float = 0.0
    cache_hits: int = 0
    last_error: str | None = None


def _module_of(fn) -> ast.Module:
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    return tree


def _helpers(tree: ast.Module, entry: str, globalns: dict) -> dict[str, Signature]:
    """Signatures for annotated helper functions reachable from `entry`."""
    sigs: dict[str, Signature] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name != entry:
            sig = signature_from_annotations(node)
            if sig is not None:
                sigs[node.name] = sig
    return sigs


def compile_function(
    fn,
    argtypes: list[Ty] | None = None,
    ret: Ty | None = None,
    opt: int = 3,
    tree: ast.Module | None = None,
) -> tuple[_Callable, str]:
    """Compile `fn` for one concrete signature. Returns (callable, mojo source)."""
    tree = tree if tree is not None else _module_of(fn)
    entry = fn.__name__
    fndef = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry
    )
    annotated = signature_from_annotations(fndef)
    if argtypes is None:
        if annotated is None:
            raise Unsupported("no annotations and no argument types given")
        sig = annotated
    else:
        names = [a.arg for a in fndef.args.args]
        if len(names) != len(argtypes):
            raise Unsupported("argument count mismatch")
        chosen_ret = ret or (annotated.ret if annotated else None)
        if chosen_ret is None:
            raise Unsupported("unknown return type")
        sig = Signature(entry, list(zip(names, argtypes)), chosen_ret)

    sigs = _helpers(tree, entry, getattr(fn, "__globals__", {}))
    sigs[entry] = sig
    unit = transpile(tree, sigs, entry)
    lib = cache.compile_unit(unit.source, opt=opt)
    return _Callable(lib, sig), unit.source


class JitFunction:
    """A Python function that swaps itself for compiled Mojo once it can.

    The first call runs CPython — which is also how the return type is learned
    when the function has no annotation. Compilation then happens in the
    background (`mode="tiered"`, the default) or inline (`mode="blocking"`),
    and every later call with the same argument types goes native. Anything
    the transpiler rejects falls back to Python permanently and silently,
    unless `strict=True`.
    """

    def __init__(self, fn, mode: str = "tiered", opt: int = 3, strict: bool = False):
        self.fn = fn
        self.mode = mode
        self.opt = opt
        self.strict = strict
        self.stats = Stats()
        self.mojo_source: str | None = None
        self._variants: dict[tuple, _Callable] = {}
        self._pending: set[tuple] = set()
        self._failed: set[tuple] = set()
        self._lock = threading.Lock()
        self._tree: ast.Module | None = None
        functools_wraps(self, fn)

    # -- compilation ----------------------------------------------------
    def _tree_of(self) -> ast.Module:
        if self._tree is None:
            self._tree = _module_of(self.fn)
        return self._tree

    def _build(self, key: tuple, argtypes: list[Ty], ret: Ty) -> None:
        started = time.perf_counter()
        try:
            call, src = compile_function(
                self.fn, argtypes, ret, opt=self.opt, tree=self._tree_of()
            )
            self.mojo_source = src
            with self._lock:
                self._variants[key] = call.dispatch
                self.stats.compiles += 1
                self.stats.compile_seconds += time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001 - fallback is the contract
            with self._lock:
                self._failed.add(key)
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
            if self.strict:
                raise
        finally:
            with self._lock:
                self._pending.discard(key)

    def _signature_for(self, args) -> tuple[tuple, list[Ty]]:
        """Key the variant cache without paying for full inference each call.

        Scalars are keyed straight off `type(arg)`, which is a dict hit;
        only buffers, whose element type depends on their contents, go
        through `type_of`.
        """
        parts = []
        argtypes: list[Ty] = []
        for a in args:
            ty = _SCALAR_TYPES.get(a.__class__)
            if ty is None:
                ty = type_of(a)
            parts.append(ty.mojo)
            argtypes.append(ty)
        return tuple(parts), argtypes

    # -- dispatch -------------------------------------------------------
    def __call__(self, *args):
        try:
            key, argtypes = self._signature_for(args)
        except Unsupported as exc:
            if self.strict:
                raise
            self.stats.last_error = str(exc)
            self.stats.python_calls += 1
            return self.fn(*args)

        call = self._variants.get(key)
        if call is not None:
            self.stats.mojo_calls += 1
            return call(*args)
        if key in self._failed:
            self.stats.python_calls += 1
            return self.fn(*args)

        result = self.fn(*args)
        self.stats.python_calls += 1
        with self._lock:
            if key in self._pending or key in self._variants or key in self._failed:
                return result
            self._pending.add(key)
        try:
            ret = type_of(result) if result is not None else NONE
        except Unsupported:
            with self._lock:
                self._failed.add(key)
                self._pending.discard(key)
            return result
        if self.mode == "blocking":
            self._build(key, argtypes, ret)
        else:
            threading.Thread(
                target=self._build, args=(key, argtypes, ret), daemon=True
            ).start()
        return result

    def wait(self, timeout: float = 600.0) -> bool:
        """Block until every in-flight compile finishes. Returns True if idle."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._pending:
                    return True
            time.sleep(0.005)
        return False

    @property
    def compiled(self) -> bool:
        return bool(self._variants)


def functools_wraps(wrapper, fn) -> None:
    for attr in ("__name__", "__qualname__", "__doc__", "__module__"):
        try:
            setattr(wrapper, attr, getattr(fn, attr))
        except AttributeError:
            pass


def jit(fn=None, *, mode: str = "tiered", opt: int = 3, strict: bool = False):
    """Decorator: run `fn` in Mojo once it has been compiled for these types."""
    if fn is None:
        return lambda f: JitFunction(f, mode=mode, opt=opt, strict=strict)
    return JitFunction(fn, mode=mode, opt=opt, strict=strict)



def _rename_functions(tree: ast.Module, prefix: str) -> dict[str, str]:
    """Namespace every function in `tree` so units can be merged."""
    names = {
        n.name: prefix + n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.name = names[node.name]
        elif isinstance(node, ast.Name) and node.id in names:
            node.id = names[node.id]
    return names


def compile_batch(jobs, opt: int = 3):
    """Compile many snippets as ONE unit, then bind each entry separately.

    `mojo build` costs the same ~5 seconds whether it is handed one function
    or twenty, so batching is the single biggest lever on cold latency: twenty
    snippets compiled together cost about a quarter of a second each instead
    of five seconds each.

    `jobs` is a sequence of `(source, entry, args)`. Returns one callable per
    job, in order.
    """
    jobs = list(jobs)
    bodies: list[str] = []
    sigs_out: list[Signature] = []
    for idx, (source, entry, args) in enumerate(jobs):
        tree = ast.parse(textwrap.dedent(source))
        prefix = f"j{idx}_"
        _rename_functions(tree, prefix)
        renamed = prefix + entry
        fndef = next(
            (n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == renamed),
            None,
        )
        if fndef is None:
            raise Unsupported(f"no function named {entry}")
        argtypes = [type_of(a) for a in args]
        annotated = signature_from_annotations(fndef)
        ret = annotated.ret if annotated else None
        if ret is None or ret is NONE:
            namespace: dict = {}
            exec(compile(tree, "<mojosub>", "exec"), namespace)  # noqa: S102
            probe = namespace[renamed](*args)
            ret = type_of(probe) if probe is not None else NONE
        sigs = _helpers(tree, renamed, {})
        names = [a.arg for a in fndef.args.args]
        sigs[renamed] = Signature(renamed, list(zip(names, argtypes)), ret)
        unit = transpile(tree, sigs, renamed)
        bodies.append(unit.source.split("\n", 1)[1] if unit.source.startswith("from std.math") else unit.source)
        sigs_out.append(sigs[renamed])

    from .transpile import PRELUDE

    merged = PRELUDE + "\n" + "\n\n".join(bodies)
    lib = cache.compile_unit(merged, opt=opt)
    return [_Callable(lib, sig).dispatch for sig in sigs_out]


_EXEC_MEMO: dict[tuple, object] = {}


def compile_source(source: str, entry: str, args=(), opt: int = 3):
    """Compile `entry` from a source string for these argument types.

    Repeated calls with the same source and argument shapes hit an in-process
    memo, so nothing is re-parsed, re-transpiled or re-probed — the agent loop
    that runs the same snippet a thousand times pays for it once.
    """
    argtypes = [type_of(a) for a in args]
    memo_key = (source, entry, opt, tuple(t.mojo for t in argtypes))
    hit = _EXEC_MEMO.get(memo_key)
    if hit is not None:
        return hit

    tree = ast.parse(textwrap.dedent(source))
    fndef = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry),
        None,
    )
    if fndef is None:
        raise Unsupported(f"no function named {entry}")
    annotated = signature_from_annotations(fndef)
    ret = annotated.ret if annotated else None
    if ret is None or ret is NONE:
        namespace: dict = {}
        exec(compile(tree, "<mojosub>", "exec"), namespace)  # noqa: S102
        probe = namespace[entry](*args)
        ret = type_of(probe) if probe is not None else NONE
    sigs = _helpers(tree, entry, {})
    names = [a.arg for a in fndef.args.args]
    sigs[entry] = Signature(entry, list(zip(names, argtypes)), ret)
    unit = transpile(tree, sigs, entry)
    lib = cache.compile_unit(unit.source, opt=opt)
    call = _Callable(lib, sigs[entry]).dispatch
    _EXEC_MEMO[memo_key] = call
    return call


def execute(source: str, entry: str, args=(), opt: int = 3, strict: bool = True):
    """Compile and run a snippet of agent-generated Python through Mojo.

    This is the arbitrary-code path: the source is parsed, the named function
    is transpiled for the concrete argument types, and the compiled unit is
    cached by content on disk and by identity in this process, so an identical
    snippet never compiles — or even re-parses — twice.
    """
    try:
        call = compile_source(source, entry, args, opt=opt)
    except Exception:
        if strict:
            raise
        namespace: dict = {}
        exec(compile(textwrap.dedent(source), "<mojosub>", "exec"), namespace)  # noqa: S102
        return namespace[entry](*args)
    return call(*args)
