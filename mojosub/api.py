"""The user-facing surface: `jit`, `compile_function`, and `execute`."""

from __future__ import annotations

import ast
import inspect
import os
import statistics
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
from .gpu import transpile_gpu
from .types import (ARRAY_F64, ARRAY_I64, BOOL, FLOAT, INT, MATRIX_F64,
                    MATRIX_I64, NONE, Array, Matrix, Ty)
from .types import by_mojo as types_by_mojo

__all__ = [
    "jit",
    "compile_function",
    "compile_source",
    "compile_batch",
    "execute",
    "Stats",
    "Unsupported",
]


# How much slower than CPython a compiled variant may be before it is retired.
# Slightly over 1.0 because the compiled side pays a first-call page-in; a
# variant that is genuinely faster clears this by an order of magnitude, so the
# tolerance costs nothing real.
SLOWER_THAN_PYTHON = 1.1

# A call this short is not worth judging: the samples are dominated by timer
# noise, and a variant retired on a 50-microsecond sample never comes back -
# not even when the next call arrives with a million rows behind it.
MIN_TIMED_SECONDS = 0.001

# How many calls the two tiers run side by side before the slower one is
# dropped. One sample is not a measurement: the first native call pays a
# page-in and the first interpreted call may have warmed the caches for it,
# and a wrong verdict here is permanent and persisted. Three medians cost
# three interpreted calls, once per (source, signature, toolchain).
RACE_CALLS = max(1, int(os.environ.get("MOJOSUB_RACE_CALLS", "3")))

# A gap this wide is not noise. The race stops the moment either side wins by
# this factor, so the common case - native is 50x ahead - pays exactly one
# extra interpreted call, and only genuinely close calls run the full race.
RACE_DECISIVE = float(os.environ.get("MOJOSUB_RACE_DECISIVE", "4.0"))

# Set to 0 to dispatch natively as soon as a variant exists, without ever
# timing it against the interpreter.
RACE_ENABLED = os.environ.get("MOJOSUB_RACE", "1") != "0"


def type_of(value) -> Ty:
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        return INT
    if isinstance(value, float):
        return FLOAT
    info = getattr(value, "__array_interface__", None)
    if info is not None:
        dims = len(info["shape"])
        f64 = info["typestr"].endswith("f8")
        if dims == 1:
            return ARRAY_F64 if f64 else ARRAY_I64
        if dims == 2:
            return MATRIX_F64 if f64 else MATRIX_I64
        raise Unsupported(f"only 1-D and 2-D arrays are supported, got {dims}-D")
    if isinstance(value, list) and value:
        if any(isinstance(v, (list, tuple)) for v in value):
            # A list of lists is not one allocation. It used to be typed as a
            # flat int buffer, which failed later and confusingly at marshalling
            # time; a 2-D argument has to expose a buffer.
            raise Unsupported(
                "a 2-D argument must be a numpy array, not a list of lists")
        return ARRAY_F64 if any(isinstance(v, float) for v in value) else ARRAY_I64
    if isinstance(value, list):
        return ARRAY_F64
    import array as _array

    if isinstance(value, _array.array):
        return ARRAY_F64 if value.typecode == "d" else ARRAY_I64
    raise Unsupported(f"unsupported argument type {type(value).__name__}")


_SCALAR_TYPES = {bool: BOOL, int: INT, float: FLOAT}


# How far apart two tiers' floating-point answers may be before the compiled one
# is retired as wrong.
#
# 1e-9 was too tight, and the way that showed up is instructive: a logistic
# gradient over 20,000 rows disagreed in the 9th significant figure and was
# retired, so a correct 261x kernel silently fell back to CPython for ever.
#
# The error bound says it had to. Summing n terms in a different order differs by
# about `n * eps * sum(|x_i|)`, so the *relative* error is that over `|result|` —
# and a gradient is a sum of signed terms that mostly cancel, which makes that
# ratio large. For n = 5000 with three orders of cancellation the bound is around
# 1e-8. The measured difference was 6.2e-9: inside theory, outside the tolerance.
#
# 1e-6 keeps the guard's power because a real miscompile is not a rounding
# difference. Wrong index arithmetic reads different memory, a wrong operator
# computes a different function, and a wrapped integer is off by 2^64 — all of
# them O(1) relative, six orders clear of this line. The one thing loosened here
# is precisely the thing that is allowed to vary.
VERIFY_RTOL = float(os.environ.get("MOJOSUB_VERIFY_RTOL", "1e-6"))


def _equalish(a, b, rtol: float | None = None) -> bool:
    """Compare two results, tolerating float reassociation but nothing else."""
    rtol = VERIFY_RTOL if rtol is None else rtol
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:  # both NaN
            return True
        return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))
    return a == b


def _zero_copy(value) -> bool:
    """Whether this argument crosses the ABI as a pointer rather than a copy.

    Mirrors what `runtime.Buffer` actually does: numpy exposes an address, and
    so does an `array.array` - but only when its typecode already matches, and
    the caller cannot see the element type from here. Treating a mismatched
    `array` as copied keys it with the lists, which is the same cost bucket it
    lands in.
    """
    if getattr(value, "__array_interface__", None) is not None:
        return True
    import array as _array

    return isinstance(value, _array.array)


def _is_buffer(value) -> bool:
    """Whether this argument is one the compiled code can write through."""
    if isinstance(value, list):
        return True
    if getattr(value, "__array_interface__", None) is not None:
        return True
    import array as _array

    return isinstance(value, _array.array)


def _snapshot(value):
    return value.copy() if hasattr(value, "copy") else value[:]


def _restore(value, saved) -> None:
    value[:] = saved


def _flat(buf):
    """A flat element sequence, whatever the buffer's shape.

    `list()` on a 2-D array yields *rows*, so comparing them element by element
    hands `_equalish` two arrays and it returns an array — which `all()` then
    refuses with "the truth value of an array is ambiguous". The verify path is
    the one thing here that has to be trustworthy, so it flattens first.
    """
    ravel = getattr(buf, "ravel", None)
    if ravel is not None:
        return ravel()
    return buf


def _first_difference(a, b) -> str:
    """Where two buffers stop agreeing, and by how much.

    "these differ" is not a usable report when the variant is being retired for
    ever on the strength of it; the index and the magnitude are what say whether
    it is a reassociation rounding difference or a real miscompile.
    """
    fa, fb = list(_flat(a)), list(_flat(b))
    if len(fa) != len(fb):
        return f"length {len(fa)} vs {len(fb)}"
    for i, (x, y) in enumerate(zip(fa, fb)):
        if not _equalish(x, y):
            try:
                gap = abs(float(x) - float(y))
                rel = gap / max(1.0, abs(float(x)), abs(float(y)))
                return (f"element {i}: python={x!r} mojo={y!r} "
                        f"(absolute {gap:.3g}, relative {rel:.3g})")
            except (TypeError, ValueError):
                return f"element {i}: python={x!r} mojo={y!r}"
    return "no element differs (a shape or dtype mismatch)"


def _buffers_equal(a, b) -> bool:
    fa, fb = _flat(a), _flat(b)
    if len(fa) != len(fb):
        return False
    return all(_equalish(x, y) for x, y in zip(list(fa), list(fb)))


def _race_verdict(samples: list[tuple[float, float]]):
    """Decide between the two tiers from the samples so far.

    Returns `(verdict, python_seconds, native_seconds)` where verdict is
    `"keep"`, `"retire"`, or `None` for "not decided, sample again". Medians,
    not means: one descheduled call must not be able to move the answer.
    """
    python_seconds = statistics.median(s[0] for s in samples)
    native_seconds = statistics.median(s[1] for s in samples)

    if python_seconds < MIN_TIMED_SECONDS and native_seconds < MIN_TIMED_SECONDS:
        # Both tiers finish inside timer noise. There is nothing here worth
        # measuring and nothing worth losing either - keep the compiled one
        # and stop paying for interpreted calls.
        return "keep", python_seconds, native_seconds
    if native_seconds * RACE_DECISIVE < python_seconds:
        return "keep", python_seconds, native_seconds
    if (python_seconds >= MIN_TIMED_SECONDS
            and python_seconds * RACE_DECISIVE < native_seconds):
        return "retire", python_seconds, native_seconds
    if len(samples) < RACE_CALLS:
        return None, python_seconds, native_seconds
    if (python_seconds >= MIN_TIMED_SECONDS
            and native_seconds > python_seconds * SLOWER_THAN_PYTHON):
        return "retire", python_seconds, native_seconds
    return "keep", python_seconds, native_seconds


@dataclass
class Stats:
    python_calls: int = 0
    mojo_calls: int = 0
    compiles: int = 0
    warm_starts: int = 0
    compile_seconds: float = 0.0
    cache_hits: int = 0
    verify_failures: int = 0
    retired_slower: int = 0
    raced_calls: int = 0
    speedup: float | None = None
    last_error: str | None = None
    # Elements that had to be copied across the C ABI because they arrived as a
    # Python sequence rather than a buffer, and the parameters they arrived on.
    # This is the one cost in the whole pipeline that the *caller* can remove:
    # measured on this machine, the same reduction is 2.2x on a million-element
    # list and 147x on the same numbers in a numpy array. Nobody discovers that
    # by reading their own code, so it is reported rather than left implicit.
    copied_elements: int = 0
    copied_params: tuple = ()


def _module_of(fn, source: str | None = None) -> ast.Module:
    """The function's own source, as an AST.

    `inspect.getsource` cannot see a function defined by `exec` of a string -
    which is exactly how generated code arrives, so the most valuable functions
    to compile were the ones this used to refuse. Callers that hold the source
    can hand it over directly, tag the function with `__mojosub_source__`, or
    put the module text under that name in the namespace they exec into.
    """
    src = source
    if src is None:
        src = getattr(fn, "__mojosub_source__", None)
    if src is None:
        src = getattr(fn, "__globals__", {}).get("__mojosub_source__")
    if src is None:
        src = inspect.getsource(fn)
    tree = ast.parse(textwrap.dedent(src))
    # A whole-module source carries every other definition too. Keep them: the
    # helpers are what `_helpers` reads signatures from, and a unit that omits
    # them cannot emit `total += helper(x)` at all — it raises `Unsupported:
    # call to helper` and the kernel silently never compiles, however carefully
    # the caller annotated it.
    #
    # What does have to go is a *rival definition of the entry itself*, which
    # is what a notebook cell re-run leaves behind. The last one is the one the
    # caller is holding.
    entry_defs = [n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == fn.__name__]
    if len(entry_defs) > 1:
        latest = entry_defs[-1]
        tree.body = [n for n in tree.body
                     if n is latest or not (isinstance(n, ast.FunctionDef)
                                            and n.name == fn.__name__)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    return tree


def _helpers(tree: ast.Module, entry: str, globalns: dict) -> dict[str, Signature]:
    """Signatures for annotated helper functions **reachable from** `entry`.

    Reachability matters, and for a while this said so without doing it: every
    annotated sibling went into the unit whether the entry point called it or
    not. A unit is emitted whole or not at all, so one unrelated function using
    `sorted` made every function in the file uncompilable — and the reported
    reason was `call to sorted` for a kernel with no `sorted` in it, which is a
    hard thing to debug from the outside.

    Transitive, because a helper's own helpers are needed too.
    """
    defined = {node.name: node for node in tree.body
               if isinstance(node, ast.FunctionDef)}
    reachable: set[str] = set()
    queue = [entry]
    while queue:
        name = queue.pop()
        node = defined.get(name)
        if node is None:
            continue
        for called in ast.walk(node):
            if (isinstance(called, ast.Call) and isinstance(called.func, ast.Name)
                    and called.func.id in defined
                    and called.func.id not in reachable
                    and called.func.id != entry):
                reachable.add(called.func.id)
                queue.append(called.func.id)

    sigs: dict[str, Signature] = {}
    for name in reachable:
        sig = signature_from_annotations(defined[name])
        if sig is not None:
            sigs[name] = sig
    return sigs


def compile_function(
    fn,
    argtypes: list[Ty] | None = None,
    ret: Ty | None = None,
    opt: int = 3,
    tree: ast.Module | None = None,
    vectorize: bool = True,
    target: str = "cpu",
    only_if_cached: bool = False,
) -> tuple[_Callable, str]:
    """Compile `fn` for one concrete signature. Returns (callable, mojo source).

    `target="gpu"` emits a CUDA kernel instead, for the per-element loop shape
    only — see `mojosub.gpu` for the measurements that say when that is worth
    the transfer.

    `only_if_cached=True` returns `(None, source)` rather than starting a build
    when the library is not already on disk. Transpiling is milliseconds and
    compiling is seconds, so that is the difference between a caller that can
    afford to ask on the hot path and one that cannot.
    """
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

    if target == "gpu":
        return _compile_gpu(tree, entry, sig, opt)

    sigs = _helpers(tree, entry, getattr(fn, "__globals__", {}))
    sigs[entry] = sig
    unit = transpile(tree, sigs, entry, vectorize=vectorize)
    if only_if_cached:
        lib = cache.lookup(unit.source, opt=opt)
        if lib is None:
            return None, unit.source
    else:
        lib = cache.compile_unit(unit.source, opt=opt)
    return _Callable(lib, sig), unit.source


class GpuCallable:
    """A compiled GPU entry point. Results come back through the buffers."""

    def __init__(self, inner: _Callable):
        self._inner = inner
        self.sig = inner.sig
        self.lib_path = inner.lib_path

    @property
    def dispatch(self):
        return self

    def __call__(self, *args):
        status = self._inner(*args)
        if status != 0:
            raise RuntimeError(
                f"{self.sig.name}: GPU launch failed (is a device visible?)"
            )
        return None


def _compile_gpu(tree, entry: str, sig: Signature, opt: int):
    gpu_sig = Signature(entry, list(sig.params), NONE)
    unit = transpile_gpu(tree, gpu_sig, entry)
    lib = cache.compile_unit(unit.source, opt=opt)
    status_sig = Signature(entry, list(sig.params), INT)
    return GpuCallable(_Callable(lib, status_sig)), unit.source


class JitFunction:
    """A Python function that swaps itself for compiled Mojo once it can.

    The first call runs CPython — which is also how the return type is learned
    when the function has no annotation. Compilation then happens in the
    background (`mode="tiered"`, the default) or inline (`mode="blocking"`),
    and every later call with the same argument types goes native. Anything
    the transpiler rejects falls back to Python permanently and silently,
    unless `strict=True`.
    """

    def __init__(self, fn, mode: str = "tiered", opt: int = 3, strict: bool = False,
                 hot_calls: int = 1, hot_seconds: float = 0.0,
                 verify: bool = False, vectorize: bool = True, warm: bool = True,
                 source: str | None = None):
        self.fn = fn
        self.source = source
        self.mode = mode
        self.opt = opt
        self.strict = strict
        self.hot_calls = max(1, int(hot_calls))
        self.hot_seconds = float(hot_seconds)
        self.verify = verify
        self.vectorize = vectorize
        self.warm = warm
        self.stats = Stats()
        self.mojo_source: str | None = None
        self._variants: dict[tuple, _Callable] = {}
        self._pending: set[tuple] = set()
        self._failed: set[tuple] = set()
        # key -> [(python_seconds, native_seconds), ...] while the two tiers
        # are still being raced against each other. Absent means decided.
        self._trials: dict[tuple, list[tuple[float, float]]] = {}
        self._profile: dict[tuple, list] = {}   # key -> [calls, seconds]
        self._lock = threading.Lock()
        self._tree: ast.Module | None = None
        self._warm_tried: set[tuple] = set()
        self._src: str | None = None
        functools_wraps(self, fn)

    def _source(self) -> str:
        """Normalised source text, the key the signature memo is filed under."""
        if self._src is None:
            self._src = ast.unparse(self._tree_of())
        return self._src

    # -- compilation ----------------------------------------------------
    def _tree_of(self) -> ast.Module:
        if self._tree is None:
            self._tree = _module_of(self.fn, self.source)
        return self._tree

    def _build(self, key: tuple, argtypes: list[Ty], ret: Ty) -> None:
        started = time.perf_counter()
        try:
            call, src = compile_function(
                self.fn, argtypes, ret, opt=self.opt, tree=self._tree_of(),
                vectorize=self.vectorize,
            )
            self.mojo_source = src
            try:
                cache.remember_signature(self._source(), key, ret.key)
            except Exception:  # noqa: BLE001 - the memo is an optimisation
                pass
            with self._lock:
                self._variants[key] = call.dispatch
                if self.verify or RACE_ENABLED:
                    self._trials[key] = []
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

    def _warm_start(self, key: tuple, argtypes: list[Ty]):
        """Dispatch natively on the first call, when a previous run left both
        halves behind: the compiled library, and the return type it inferred.

        Without this the JIT is useless to anything that calls a function once
        - an agent's generated code runs, the profile records one call, the
        process exits, and the library it could have used sits in the cache
        unread. The cost of asking is a transpile (milliseconds) and a `stat`.
        """
        if key in self._warm_tried:
            return None
        self._warm_tried.add(key)
        try:
            verdict = cache.recall_verdict(self._source(), key)
            if verdict.get("retired"):
                # Another run already found this variant wrong or slower than
                # the interpreter. Re-learning that costs the same as learning
                # it did, in every worker, for ever.
                with self._lock:
                    self._failed.add(key)
                    self.stats.last_error = f"retired earlier: {verdict['retired']}"
                return None
            remembered = verdict.get("ret")
            if remembered is None:
                return None
            ret = types_by_mojo(remembered)
            call, _ = compile_function(
                self.fn, argtypes, ret, opt=self.opt, tree=self._tree_of(),
                vectorize=self.vectorize, only_if_cached=True,
            )
        except Exception as exc:  # noqa: BLE001 - warm start is never required
            self.stats.last_error = f"warm start: {type(exc).__name__}: {exc}"
            return None
        if call is None:
            return None
        with self._lock:
            self._variants[key] = call.dispatch
            # Verification and the timing race are both skipped when a previous
            # run already settled them: the library is the same bytes, compiled
            # from the same source by the same toolchain, so neither the answer
            # nor which tier is faster can have changed. This is the whole
            # point of persisting the verdict - a function called once per
            # request must not re-race itself on every request.
            needs_verify = self.verify and not verdict.get("verified")
            needs_race = RACE_ENABLED and not verdict.get("raced")
            if needs_verify or needs_race:
                self._trials[key] = []
            elif verdict.get("speedup"):
                self.stats.speedup = verdict["speedup"]
            self.stats.warm_starts += 1
        return call.dispatch

    def _signature_for(self, args) -> tuple[tuple, list[Ty]]:
        """Key the variant cache without paying for full inference each call.

        Scalars are keyed straight off `type(arg)`, which is a dict hit;
        only buffers, whose element type depends on their contents, go
        through `type_of`.

        A numpy array and a Python list of the same element type compile to the
        same Mojo, so they used to share a key - but one crosses the ABI as a
        pointer and the other is copied element by element, which is 0.6ms
        against 12ms on the same kernel. That is the difference between the
        compiled variant winning and losing, so the two get separate keys: the
        library is shared (same source, same content-addressed cache entry) and
        only the verdict about which tier is faster is kept apart.
        """
        parts = []
        argtypes: list[Ty] = []
        for a in args:
            ty = _SCALAR_TYPES.get(a.__class__)
            if ty is None:
                ty = type_of(a)
                parts.append(ty.key + ("#buf" if _zero_copy(a) else "#seq"))
            else:
                parts.append(ty.key)
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
        if call is None and self.warm and key not in self._failed:
            call = self._warm_start(key, argtypes)
        if call is not None:
            if key in self._trials:
                return self._trial_call(key, call, args)
            self.stats.mojo_calls += 1
            return call(*args)
        if key in self._failed:
            self.stats.python_calls += 1
            return self.fn(*args)

        started = time.perf_counter()
        result = self.fn(*args)
        elapsed = time.perf_counter() - started
        self.stats.python_calls += 1

        with self._lock:
            if key in self._pending or key in self._variants or key in self._failed:
                return result
            profile = self._profile.setdefault(key, [0, 0.0])
            profile[0] += 1
            profile[1] += elapsed
            hot = profile[0] >= self.hot_calls or (
                self.hot_seconds > 0.0 and profile[1] >= self.hot_seconds
            )
            if not hot:
                # not hot enough to be worth several seconds of compiler
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
            try:
                threading.Thread(
                    target=self._build, args=(key, argtypes, ret), daemon=True
                ).start()
            except RuntimeError as exc:
                # No thread to be had — a process limit, or a machine under
                # real pressure. Compiling is an optimisation and the caller is
                # already holding the right answer, so this must not be the
                # thing that ends their program. It used to be: the exception
                # unwound through `__call__` into user code, and a run that
                # would have finished on CPython died with a traceback into
                # this file.
                with self._lock:
                    self._pending.discard(key)
                    self.stats.last_error = f"could not start a compile: {exc}"
        return result

    def _trial_call(self, key, call, args):
        """Run both tiers, compare them, and keep whichever one deserves to win.

        Two separate questions are settled here, and both need an interpreted
        call, so they share one.

        Correctness: Mojo's Int is 64-bit and wraps where CPython promotes, and
        a vectorized reduction sums in a different order, so `verify=True` buys
        certainty for the price of one extra Python call per signature. A
        mismatch retires the variant permanently rather than raising - the
        answer the caller gets is always the Python one.

        Speed: native is not automatically faster. Crossing the C ABI copies any
        argument that is not already a buffer, and copying a 200,000-element
        Python list costs more than the loop it feeds - measured at 30ms native
        against 17ms interpreted, while the same function on a numpy array is
        0.2ms against 55ms. An accelerator that can lose is worse than none,
        because nobody profiles the thing that is supposed to be the fast path.

        The verdict comes from a race rather than a single sample: one pair of
        timings has the native side paying a first-call page-in and the
        interpreted side warming the caches for it, and the decision is
        permanent and written to disk. The race stops early the moment one side
        is ahead by `RACE_DECISIVE`, which is the usual case - so a 50x win
        costs one interpreted call, and only a genuinely close result pays for
        `RACE_CALLS` of them.
        """
        # A kernel that writes through its buffer argument must not see the
        # other tier's writes: running both back to back would apply the
        # transform twice, and the caller would get a doubled array out of what
        # is supposed to be a measurement. Each tier runs against the original
        # contents, and what the caller keeps is the native tier's result.
        buffers = [a for a in args if _is_buffer(a)]
        before = [_snapshot(b) for b in buffers]

        t0 = time.perf_counter()
        expected = self.fn(*args)
        t1 = time.perf_counter()
        python_seconds = t1 - t0

        after_python = [_snapshot(b) for b in buffers]
        for buf, saved in zip(buffers, before):
            _restore(buf, saved)

        t2 = time.perf_counter()
        actual = call(*args)
        native_seconds = time.perf_counter() - t2

        self.stats.python_calls += 1
        self.stats.raced_calls += 1

        disagreement = None
        if self.verify:
            # Which of the two disagreed has to be in the message. Reporting the
            # return value when a *buffer* was what differed sends the reader to
            # look at numbers that match, and the retirement is permanent.
            if not _equalish(expected, actual):
                disagreement = f"return value: python={expected!r} mojo={actual!r}"
            else:
                for idx, (was, now) in enumerate(zip(after_python, buffers)):
                    if not _buffers_equal(was, now):
                        disagreement = (
                            f"buffer argument {idx}: "
                            f"{_first_difference(was, now)}"
                        )
                        break
        agreed = disagreement is None
        if not agreed:
            with self._lock:
                self._trials.pop(key, None)
                self._variants.pop(key, None)
                self._failed.add(key)
                self.stats.verify_failures += 1
                self.stats.last_error = f"verification failed: {disagreement}"
            self._record(key, retired=f"disagreed with CPython — {disagreement}")
            # The buffers hold the native writes; the answer the caller gets is
            # the interpreted one, in every respect.
            for buf, saved in zip(buffers, after_python):
                _restore(buf, saved)
            return expected

        with self._lock:
            samples = self._trials.get(key)
            if samples is None:
                # Another thread settled this signature while this call was in
                # flight. Its verdict stands.
                self.stats.mojo_calls += 1
                return actual
            samples.append((python_seconds, native_seconds))
            verdict, median_python, median_native = _race_verdict(samples)
            if verdict is None:
                self.stats.mojo_calls += 1
                return actual
            self._trials.pop(key, None)
            speedup = (median_python / median_native) if median_native > 0 else None
            if verdict == "retire":
                reason = (f"native {median_native * 1e3:.2f}ms vs "
                          f"python {median_python * 1e3:.2f}ms "
                          f"over {len(samples)} calls")
                self._variants.pop(key, None)
                self._failed.add(key)
                self.stats.retired_slower += 1
                self.stats.last_error = f"retired: {reason}"
            else:
                self.stats.mojo_calls += 1
                self.stats.speedup = speedup

        if verdict == "retire":
            self._record(key, retired=reason)
            for buf, saved in zip(buffers, after_python):
                _restore(buf, saved)
            return expected
        # `verified` is only a fact when there was something to compare; with
        # verify off the race says which tier is faster and nothing more.
        self._record(
            key,
            verified=True if self.verify else None,
            raced=True,
            speedup=speedup,
        )
        return actual

    def _record(self, key: tuple, **verdict) -> None:
        """Persist a verdict, so no other process has to reach it again."""
        try:
            cache.remember_verdict(self._source(), key, **verdict)
        except Exception:  # noqa: BLE001 - the memo is an optimisation
            pass

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

    def copy_cost(self) -> tuple[int, tuple]:
        """Elements copied across the ABI so far, and the parameters they were.

        Summed from the variants rather than counted on the dispatch path: the
        bound `Callable` is the only thing that sees the arguments, and a
        scalar-only signature bypasses it entirely (and copies nothing).
        """
        total = 0
        params: set[str] = set()
        for call in self._variants.values():
            n = getattr(call, "copied_elements", 0)
            if n:
                total += n
                params |= getattr(call, "copied_params", set())
        self.stats.copied_elements = total
        self.stats.copied_params = tuple(sorted(params))
        return total, self.stats.copied_params


def functools_wraps(wrapper, fn) -> None:
    for attr in ("__name__", "__qualname__", "__doc__", "__module__"):
        try:
            setattr(wrapper, attr, getattr(fn, attr))
        except AttributeError:
            pass


def jit(fn=None, *, mode: str = "tiered", opt: int = 3, strict: bool = False,
        hot_calls: int = 1, hot_seconds: float = 0.0, verify: bool = False,
        warm: bool = True, source: str | None = None,
        vectorize: bool = True):
    """Decorator: run `fn` in Mojo once it has been compiled for these types.

    `hot_calls` / `hot_seconds` hold compilation back until the function has
    proved it is worth several seconds of compiler; `verify` checks the first
    native result against CPython before trusting the variant.
    """
    kwargs = dict(mode=mode, opt=opt, strict=strict, warm=warm, source=source,
                  hot_calls=hot_calls,
                  hot_seconds=hot_seconds, verify=verify, vectorize=vectorize)
    if fn is None:
        return lambda f: JitFunction(f, **kwargs)
    return JitFunction(fn, **kwargs)



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


def compile_source(source: str, entry: str, args=(), opt: int = 3,
                   target: str = "cpu"):
    """Compile `entry` from a source string for these argument types.

    Repeated calls with the same source and argument shapes hit an in-process
    memo, so nothing is re-parsed, re-transpiled or re-probed — the agent loop
    that runs the same snippet a thousand times pays for it once.
    """
    argtypes = [type_of(a) for a in args]
    memo_key = (source, entry, opt, target, tuple(t.mojo for t in argtypes))
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
    names = [a.arg for a in fndef.args.args]
    sig = Signature(entry, list(zip(names, argtypes)), ret)
    if target == "gpu":
        call = _compile_gpu(tree, entry, sig, opt)[0].dispatch
    else:
        sigs = _helpers(tree, entry, {})
        sigs[entry] = sig
        unit = transpile(tree, sigs, entry)
        lib = cache.compile_unit(unit.source, opt=opt)
        call = _Callable(lib, sig).dispatch
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
