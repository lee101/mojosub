"""Type lattice for the Python subset mojosub compiles."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass


class Ty:
    mojo: str
    ctype: object

    def __repr__(self) -> str:
        return self.mojo


@dataclass(frozen=True)
class Scalar(Ty):
    mojo: str
    ctype: object
    rank: int


INT = Scalar("Int", ctypes.c_int64, 2)
FLOAT = Scalar("Float64", ctypes.c_double, 3)
BOOL = Scalar("Bool", ctypes.c_bool, 1)
NONE = Scalar("None", None, 0)


@dataclass(frozen=True)
class Array(Ty):
    """A contiguous buffer passed across the C ABI as (address, length)."""

    elem: Scalar

    @property
    def mojo(self) -> str:  # type: ignore[override]
        return f"UnsafePointer[{self.elem.mojo}, AnyOrigin[mut=True]]"

    @property
    def ctype(self):  # type: ignore[override]
        return ctypes.c_int64  # the address; length is a second argument


ARRAY_F64 = Array(FLOAT)
ARRAY_I64 = Array(INT)

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
    }[name]
