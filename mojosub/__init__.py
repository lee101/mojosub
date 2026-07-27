"""mojosub — compile a Python subset to Mojo and call it at C speed.

    from mojosub import jit

    @jit
    def total(xs: list[float]) -> float:
        s = 0.0
        for i in range(len(xs)):
            s += xs[i]
        return s

The first call runs CPython, a background thread compiles a shared library,
and every later call dispatches into native code in under a microsecond.
"""

from .api import (
    JitFunction,
    Stats,
    Unsupported,
    compile_batch,
    compile_function,
    compile_source,
    execute,
    jit,
)
from .cache import clear as clear_cache
from .cache import compile_unit
from .toolchain import ToolchainError, mojo_version

__version__ = "0.1.0"
__all__ = [
    "jit",
    "execute",
    "compile_batch",
    "compile_function",
    "compile_source",
    "compile_unit",
    "clear_cache",
    "mojo_version",
    "JitFunction",
    "Stats",
    "Unsupported",
    "ToolchainError",
]
