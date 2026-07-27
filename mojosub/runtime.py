"""Loading compiled units and dispatching calls into them."""

from __future__ import annotations

import array
import ctypes
import functools

from .transpile import Signature
from .types import ARRAY_F64, ARRAY_I64, BOOL, FLOAT, INT, NONE, Array

_TYPECODE = {ARRAY_F64: "d", ARRAY_I64: "q"}


@functools.lru_cache(maxsize=None)
def _dll(path: str) -> ctypes.CDLL:
    return ctypes.CDLL(path)


class _Buffer:
    """Adapts a Python sequence to an (address, length) pair."""

    __slots__ = ("addr", "length", "_src", "_copy")

    def __init__(self, obj, ty: Array):
        code = _TYPECODE[ty]
        self._src = None
        self._copy = None
        info = getattr(obj, "__array_interface__", None)
        if info is not None:  # numpy, without importing it
            if not info.get("data") or info["data"][1]:
                raise TypeError("read-only array")
            if len(info["shape"]) != 1:
                raise TypeError("only 1-D arrays are supported")
            want = "<f8" if code == "d" else "<i8"
            if info["typestr"] not in (want, want.replace("<", "|")):
                raise TypeError(f"expected {want} array, got {info['typestr']}")
            self.addr = info["data"][0]
            self.length = info["shape"][0]
            return
        if isinstance(obj, array.array) and obj.typecode == code:
            self.addr, self.length = obj.buffer_info()
            return
        buf = array.array(code, obj)
        self._copy = buf
        self._src = obj if isinstance(obj, list) else None
        self.addr, self.length = buf.buffer_info()

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
        self._buffers = [isinstance(t, Array) for _, t in sig.params]
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
        for value, (_, ty) in zip(args, self.sig.params):
            if isinstance(ty, Array):
                buf = _Buffer(value, ty)
                held.append(buf)
                cargs.append(buf.addr)
                cargs.append(buf.length)
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
