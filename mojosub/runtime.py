"""Loading compiled units and dispatching calls into them."""

from __future__ import annotations

import array
import ctypes
import functools

from .transpile import Signature
from .types import (ARRAY_F64, ARRAY_I64, BOOL, FLOAT, INT, MATRIX_F64,
                    MATRIX_I64, NONE, Array, Matrix)

_TYPECODE = {ARRAY_F64: "d", ARRAY_I64: "q", MATRIX_F64: "d", MATRIX_I64: "q"}


@functools.lru_cache(maxsize=None)
def _dll(path: str) -> ctypes.CDLL:
    return ctypes.CDLL(path)


class _Buffer:
    """Adapts a Python sequence to an (address, length) pair."""

    __slots__ = ("addr", "length", "cols", "_src", "_copy", "copied")

    def __init__(self, obj, ty: Array, may_write: bool = True):
        """`may_write=False` promises the compiled code never stores through
        this parameter, so a copied sequence does not have to be copied back.

        The transpiler fills that in (`Signature.writes`) because it is the only
        thing that knows. It is worth knowing: for a reduction over a list the
        copy back is the single most expensive thing the call does and it cannot
        change anything.
        """
        code = _TYPECODE[ty]
        self._src = None
        self._copy = None
        # Elements copied to get across the ABI. Zero for anything that exposes
        # a buffer; the length of the sequence otherwise. Worth counting because
        # it is the one cost the *caller* can remove, by passing an array — and
        # they cannot know to unless somebody tells them.
        self.copied = 0
        want_dims = 2 if isinstance(ty, Matrix) else 1
        info = getattr(obj, "__array_interface__", None)
        if info is not None:  # numpy, without importing it
            if not info.get("data") or info["data"][1]:
                raise TypeError("read-only array")
            shape = info["shape"]
            if len(shape) != want_dims:
                raise TypeError(
                    f"expected a {want_dims}-D array, got {len(shape)}-D")
            want = "<f8" if code == "d" else "<i8"
            if info["typestr"] not in (want, want.replace("<", "|")):
                raise TypeError(f"expected {want} array, got {info['typestr']}")
            if want_dims == 2:
                # The emitted index arithmetic is `row * cols + col`, which is
                # only the right address for contiguous row-major storage. A
                # transposed or sliced array reports strides; indexing it as
                # though it were contiguous reads the wrong memory and returns a
                # plausible wrong answer, so it is refused here instead.
                strides = info.get("strides")
                itemsize = 8
                if strides is not None and strides != (
                        shape[1] * itemsize, itemsize):
                    raise TypeError(
                        "a 2-D argument must be C-contiguous — pass "
                        "np.ascontiguousarray(a)")
                self.addr = info["data"][0]
                self.length = shape[0]
                self.cols = shape[1]
                return
            self.addr = info["data"][0]
            self.length = shape[0]
            return
        if want_dims == 2:
            # A list of lists is not one allocation, so there is no address to
            # hand over and no cheap way to make one. Saying so beats copying
            # silently: the copy would dominate any kernel it fed.
            raise TypeError(
                "a 2-D argument must expose a buffer — pass a numpy array")
        if isinstance(obj, array.array) and obj.typecode == code:
            self.addr, self.length = obj.buffer_info()
            return
        buf = array.array(code, obj)
        self._copy = buf
        self._src = obj if (may_write and isinstance(obj, list)) else None
        self.addr, self.length = buf.buffer_info()
        self.copied = self.length

    def writeback(self) -> None:
        if self._src is not None and self._copy is not None:
            self._src[:] = self._copy.tolist()


class Callable:
    """A compiled entry point bound to its C signature."""

    def __init__(self, lib_path: str, sig: Signature):
        self.sig = sig
        self.lib_path = lib_path
        fn = getattr(_dll(lib_path), "mojosub_" + sig.name)
        fn.argtypes = sig.c_argtypes
        fn.restype = None if sig.ret is NONE else sig.ret.ctype
        self._fn = fn
        self._buffers = [isinstance(t, (Array, Matrix)) for _, t in sig.params]
        self._may_write = [sig.may_write(name) for name, _ in sig.params]
        # Running totals for the caller's benefit: how many elements had to be
        # copied, and which parameters they arrived on.
        self.copied_elements = 0
        self.copied_params: set[str] = set()
        # Scalar-only signatures need no marshalling at all: ctypes already
        # converts int/float/bool from the argtypes, so hand the call straight
        # to the FFI stub and save a few microseconds per call.
        self._simple = not any(self._buffers) and sig.ret is not NONE and sig.ret is not BOOL

    @property
    def dispatch(self):
        """The cheapest callable for this signature — the FFI stub, if possible."""
        return self._fn if self._simple else self

    def __call__(self, *args):
        if self._simple:
            return self._fn(*args)
        if len(args) != len(self.sig.params):
            raise TypeError(
                f"{self.sig.name}() takes {len(self.sig.params)} arguments, "
                f"got {len(args)}"
            )
        cargs = []
        held: list[_Buffer] = []
        for value, (name, ty), may_write in zip(args, self.sig.params, self._may_write):
            if isinstance(ty, (Array, Matrix)):
                buf = _Buffer(value, ty, may_write)
                held.append(buf)
                if buf.copied:
                    self.copied_elements += buf.copied
                    self.copied_params.add(name)
                cargs.append(buf.addr)
                cargs.append(buf.length)
                if isinstance(ty, Matrix):
                    cargs.append(buf.cols)
            elif ty is FLOAT:
                cargs.append(float(value))
            elif ty is INT:
                cargs.append(int(value))
            elif ty is BOOL:
                cargs.append(bool(value))
            else:
                raise TypeError(f"unsupported parameter type {ty}")
        result = self._fn(*cargs)
        for buf in held:
            buf.writeback()
        if self.sig.ret is NONE:
            return None
        if self.sig.ret is BOOL:
            return bool(result)
        return result
