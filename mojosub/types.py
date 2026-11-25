"""Type lattice for the Python subset mojosub compiles."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass


class Ty:
    mojo: str
    ctype: object

    def __repr__(self) -> str:
        # `key`, not `mojo`: a matrix and a flat array emit the same Mojo text,
        # and an error message that cannot tell them apart is the wrong message.
        return self.key

    @property
    def key(self) -> str:
        """This type's identity, for cache keys and the signature memo.

        Separate from `mojo` because two different types can compile to the same
        Mojo text: a 1-D buffer and a 2-D one are both a bare pointer, and only
        the number of shape arguments beside it differs. Keying on the emitted
        text would let a call with a matrix reuse the variant compiled for a
        flat array, which is not a slow answer, it is a wrong one.
        """
        return self.mojo


@dataclass(frozen=True, repr=False)
class Scalar(Ty):
    mojo: str
    ctype: object
    rank: int


INT = Scalar("Int", ctypes.c_int64, 2)
FLOAT = Scalar("Float64", ctypes.c_double, 3)
BOOL = Scalar("Bool", ctypes.c_bool, 1)
NONE = Scalar("None", None, 0)


@dataclass(frozen=True, repr=False)
class Array(Ty):
    """A contiguous buffer passed across the C ABI as (address, length)."""

    elem: Scalar

    @property
    def mojo(self) -> str:  # type: ignore[override]
        return f"UnsafePointer[{self.elem.mojo}, AnyOrigin[mut=True]]"

    @property
    def ctype(self):  # type: ignore[override]
        return ctypes.c_int64  # the address; length is a second argument


@dataclass(frozen=True, repr=False)
class Matrix(Ty):
    """A row-major 2-D buffer, passed as (address, rows, cols).

    Row-major and contiguous, which is what a plain `np.zeros((r, c))` is. It is
    the shape almost all of the interesting machine-learning work has — a
    feature matrix — and without it a kernel over one has to be written against
    a flattened array with the stride arithmetic done by hand, which is both
    unpleasant and a different program from the one the caller debugged in
    numpy.

    Deliberately not a general strided view: accepting a transpose or a slice
    means carrying strides through every index expression, and silently
    accepting a non-contiguous array while indexing it as though it were
    contiguous reads the wrong memory. Non-contiguous input is refused at the
    boundary instead.
    """

    elem: Scalar

    @property
    def mojo(self) -> str:  # type: ignore[override]
        return f"UnsafePointer[{self.elem.mojo}, AnyOrigin[mut=True]]"

    @property
    def ctype(self):  # type: ignore[override]
        return ctypes.c_int64  # the address; rows and cols follow

    @property
    def key(self) -> str:  # type: ignore[override]
        # Distinct from Array's, which deliberately keeps its old spelling so
        # signatures already on disk still match.
        return f"Matrix[{self.elem.mojo}]"


ARRAY_F64 = Array(FLOAT)
ARRAY_I64 = Array(INT)
MATRIX_F64 = Matrix(FLOAT)
MATRIX_I64 = Matrix(INT)

_NUMERIC = (INT, FLOAT, BOOL)


def promote(a: Ty, b: Ty) -> Ty:
    if a not in _NUMERIC or b not in _NUMERIC:
        raise TypeError(f"cannot combine {a} and {b}")
    return a if a.rank >= b.rank else b


def from_annotation(name: str) -> Ty:
    return {
        "int": INT,
        "float": FLOAT,
        "bool": BOOL,
        "None": NONE,
        "list[float]": ARRAY_F64,
        "list[int]": ARRAY_I64,
        "list[list[float]]": MATRIX_F64,
        "list[list[int]]": MATRIX_I64,
    }[name]


_ALL = (INT, FLOAT, BOOL, NONE, ARRAY_F64, ARRAY_I64, MATRIX_F64, MATRIX_I64)


def by_mojo(name: str) -> Ty:
    """The type whose identity is `name`.

    The inverse of `Ty.key`, used to rebuild a signature that a previous process
    inferred and wrote to the cache. Matched on `key` rather than `mojo` so that
    a matrix and a flat array are told apart — they emit identical Mojo text.
    `Array.key` is still its old `mojo` spelling, so memos already on disk
    resolve unchanged.
    """
    for ty in _ALL:
        if ty.key == name:
            return ty
    raise KeyError(name)
