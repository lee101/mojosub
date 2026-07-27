"""Throughput and latency benchmarks for mojosub.

    pixi run python bench/bench.py            # everything
    pixi run python bench/bench.py --latency  # just the dispatch path

Throughput compares one CPython call against one compiled call on the same
inputs. Latency measures the thing that actually decides whether a compiler is
usable for agent-generated code: how long it takes to get from a string of
Python to a first answer, cold and warm.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import textwrap
import time

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from mojosub import cache, execute, jit  # noqa: E402
from mojosub import api  # noqa: E402
from mojosub.api import compile_function, type_of  # noqa: E402


def timeit(fn, *args, repeat: int = 5, number: int = 1) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn(*args)
        best = min(best, (time.perf_counter() - t0) / number)
    return best


# ---------------------------------------------------------------- kernels
def loop_sum(n: int) -> int:
    s = 0
    for i in range(n):
        s += i * i % 7
    return s


def mandelbrot(w: int, h: int, limit: int) -> int:
    total = 0
    for py in range(h):
        y0 = float(py) / float(h) * 2.0 - 1.0
        for px in range(w):
            x0 = float(px) / float(w) * 3.0 - 2.0
            x = 0.0
            y = 0.0
            it = 0
            while x * x + y * y <= 4.0 and it < limit:
                xt = x * x - y * y + x0
                y = 2.0 * x * y + y0
                x = xt
                it += 1
            total += it
    return total


def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def monte_carlo_pi(n: int, seed: int) -> float:
    # Constants stay inside 64 bits on purpose: Mojo's Int wraps where
    # CPython's promotes, so a wider LCG would give two different answers.
    state = seed
    inside = 0
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        x = float(state % 100000) / 100000.0
        state = (state * 1103515245 + 12345) % 2147483648
        y = float(state % 100000) / 100000.0
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * float(inside) / float(n)


def dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def logistic_step(xs: list[float], r: float, steps: int) -> float:
    total = 0.0
    for i in range(len(xs)):
        x = xs[i]
        for _ in range(steps):
            x = r * x * (1.0 - x)
        xs[i] = x
        total += x
    return total


def sma_crossover(prices: list[float], fast: int, slow: int) -> float:
    """A realistic analytics kernel: two moving averages and a signal count."""
    signal = 0.0
    fsum = 0.0
    ssum = 0.0
    for i in range(len(prices)):
        fsum += prices[i]
        ssum += prices[i]
        if i >= fast:
            fsum -= prices[i - fast]
        if i >= slow:
            ssum -= prices[i - slow]
        if i >= slow:
            f = fsum / float(fast)
            s = ssum / float(slow)
            if f > s:
                signal += 1.0
            else:
                signal -= 1.0
    return signal


try:
    import numpy as np
except ImportError:  # numpy is optional; without it every buffer is copied
    np = None


def _floats(n: int, f):
    data = [f(i) for i in range(n)]
    return np.array(data) if np is not None else data


def _lists(n: int, f):
    return [f(i) for i in range(n)]


# (name, fn, CPython args, mojosub args). CPython is always given plain lists,
# because that is what the same code would be written against; mojosub is given
# whatever it would really be handed. The `list copy` row exists so the cost of
# not having numpy is visible rather than hidden.
KERNELS = [
    ("loop_sum(3e6)", loop_sum, lambda: (3_000_000,), None),
    ("mandelbrot(200x200,50)", mandelbrot, lambda: (200, 200, 50), None),
    ("fib(27) recursive", fib, lambda: (27,), None),
    ("monte_carlo_pi(1e6)", monte_carlo_pi, lambda: (1_000_000, 12345), None),
    ("dot(1e6)", dot,
     lambda: (_lists(1_000_000, float), _lists(1_000_000, lambda i: i * 0.5)),
     lambda: (_floats(1_000_000, float), _floats(1_000_000, lambda i: i * 0.5))),
    ("dot(1e6) list copy", dot,
     lambda: (_lists(1_000_000, float), _lists(1_000_000, lambda i: i * 0.5)),
     None),
    ("logistic(1e4x100)", logistic_step,
     lambda: (_lists(10_000, lambda i: 0.4 + 1e-6 * i), 3.7, 100),
     lambda: (_floats(10_000, lambda i: 0.4 + 1e-6 * i), 3.7, 100)),
    ("sma_crossover(1e6)", sma_crossover,
     lambda: (_lists(1_000_000, lambda i: 100.0 + math.sin(i / 50.0)), 20, 100),
     lambda: (_floats(1_000_000, lambda i: 100.0 + math.sin(i / 50.0)), 20, 100)),
]


def run_throughput() -> None:
    print(f"{'kernel':<26}{'CPython':>12}{'mojosub':>12}{'speedup':>10}  ok")
    print("-" * 72)
    for name, fn, mkpy, mkmo in KERNELS:
        mkmo = mkmo or mkpy
        args = mkmo()
        argtypes = [type_of(a) for a in args]
        ret = type_of(fn(*mkpy()))
        call, _ = compile_function(fn, argtypes, ret)
        py = timeit(fn, *mkpy(), repeat=3)
        mo = timeit(call, *mkmo(), repeat=3)
        a, b = fn(*mkpy()), call(*mkmo())
        ok = (abs(a - b) <= 1e-6 * max(1.0, abs(a))) if isinstance(a, float) else a == b
        print(f"{name:<26}{py * 1e3:>10.2f}ms{mo * 1e3:>10.2f}ms"
              f"{py / mo:>9.1f}x  {'yes' if ok else 'NO'}")


# ---------------------------------------------------------------- latency
SNIPPET = """
    def kernel(n):
        total = 0
        for i in range(n):
            if i % 3 == 0 or i % 5 == 0:
                total += i * {salt}
        return total
"""


def run_latency(samples: int = 5) -> None:
    print("\nlatency: string of Python -> first answer")
    print("-" * 72)

    # Salt the snippets so "cold" really means a cache miss even on a machine
    # that has run this benchmark before.
    base = random.randrange(1, 10**9) * 1000
    cold = []
    for k in range(samples):
        src = textwrap.dedent(SNIPPET.format(salt=base + k))
        t0 = time.perf_counter()
        execute(src, "kernel", (1000,))
        cold.append(time.perf_counter() - t0)

    api._EXEC_MEMO.clear()  # a disk hit must not be measured as a memo hit
    warm = []
    for k in range(samples):
        src = textwrap.dedent(SNIPPET.format(salt=base + k))
        t0 = time.perf_counter()
        execute(src, "kernel", (1000,))
        warm.append(time.perf_counter() - t0)

    memo = []
    src0 = textwrap.dedent(SNIPPET.format(salt=base))
    for _ in range(samples):
        t0 = time.perf_counter()
        execute(src0, "kernel", (1000,))
        memo.append(time.perf_counter() - t0)

    src = src0
    py_compile = timeit(lambda: compile(src, "<s>", "exec"), repeat=20, number=50)
    code = compile(src, "<s>", "exec")

    def py_exec():
        ns: dict = {}
        exec(code, ns)  # noqa: S102
        return ns["kernel"](1000)

    py_total = timeit(py_exec, repeat=20, number=50)

    @jit(mode="blocking")
    def tiny(a: int, b: int) -> int:
        return a + b

    tiny(1, 2)
    tiny.wait()
    per_call = timeit(tiny, 1, 2, repeat=20, number=20000)
    raw = next(iter(tiny._variants.values()))
    raw_call = timeit(raw, 1, 2, repeat=20, number=20000)

    print(f"{'CPython compile()':<38}{py_compile * 1e6:>10.1f} us")
    print(f"{'CPython compile()+exec+run':<38}{py_total * 1e6:>10.1f} us")
    print(f"{'mojosub cold (compile + run)':<38}{statistics.median(cold) * 1e3:>10.1f} ms")
    print(f"{'mojosub warm (disk cache + run)':<38}{statistics.median(warm) * 1e3:>10.2f} ms")
    print(f"{'mojosub warm (in-process memo + run)':<38}{statistics.median(memo) * 1e6:>10.1f} us")
    print(f"{'mojosub hot call (@jit dispatch)':<38}{per_call * 1e6:>10.3f} us")
    print(f"{'mojosub hot call (bound symbol)':<38}{raw_call * 1e6:>10.3f} us")
    print(f"\ncold/warm ratio: {statistics.median(cold) / statistics.median(warm):.0f}x"
          f"   cache: {cache.DEFAULT_ROOT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--throughput", action="store_true")
    args = ap.parse_args()
    both = not (args.latency or args.throughput)
    if both or args.throughput:
        run_throughput()
    if both or args.latency:
        run_latency()


if __name__ == "__main__":
    main()
