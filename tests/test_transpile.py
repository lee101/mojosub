import math

import pytest

from mojosub import Unsupported, execute, jit
from mojosub.api import compile_function
from mojosub.types import ARRAY_F64, FLOAT, INT


def test_scalar_roundtrip():
    @jit(mode="blocking")
    def poly(x: float, n: int) -> float:
        acc = 0.0
        for i in range(n):
            acc += x ** 2.0 + float(i) / 3.0
        return acc

    expected = poly.fn(1.5, 100)
    assert poly(1.5, 100) == pytest.approx(expected)
    assert poly.compiled
    assert poly(1.5, 100) == pytest.approx(expected)
    assert poly.stats.mojo_calls == 1


def test_buffer_read():
    @jit(mode="blocking")
    def dot(a: list[float], b: list[float]) -> float:
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    xs = [float(i) for i in range(64)]
    ys = [float(i) * 0.5 for i in range(64)]
    assert dot(xs, ys) == pytest.approx(dot.fn(xs, ys))
    assert dot(xs, ys) == pytest.approx(sum(x * y for x, y in zip(xs, ys)))


def test_buffer_write_back():
    @jit(mode="blocking")
    def scale(xs: list[float], k: float) -> float:
        t = 0.0
        for i in range(len(xs)):
            xs[i] = xs[i] * k
            t += xs[i]
        return t

    data = [1.0, 2.0, 3.0]
    scale(data, 2.0)          # first call runs Python
    data = [1.0, 2.0, 3.0]
    total = scale(data, 2.0)  # second call runs Mojo
    assert data == [2.0, 4.0, 6.0]
    assert total == pytest.approx(12.0)


def test_numpy_zero_copy():
    np = pytest.importorskip("numpy")

    @jit(mode="blocking")
    def add_one(xs: list[float]) -> float:
        for i in range(len(xs)):
            xs[i] = xs[i] + 1.0
        return xs[0]

    arr = np.zeros(8)
    add_one(arr)
    arr[:] = 0.0
    add_one(arr)
    assert (arr == 1.0).all()


def test_control_flow_and_math():
    @jit(mode="blocking")
    def collatz_energy(n: int) -> float:
        steps = 0
        x = n
        while x != 1:
            if x % 2 == 0:
                x = x // 2
            else:
                x = 3 * x + 1
            steps += 1
            if steps > 1000:
                break
        return math.sqrt(float(steps)) + math.log(float(n))

    for n in (7, 27, 97):
        assert collatz_energy(n) == pytest.approx(collatz_energy.fn(n))
    assert collatz_energy(27) == pytest.approx(collatz_energy.fn(27))


def test_helper_calls():
    def square(x: float) -> float:
        return x * x

    @jit(mode="blocking")
    def norm(xs: list[float]) -> float:
        s = 0.0
        for i in range(len(xs)):
            s += square(xs[i])
        return math.sqrt(s)

    # the helper must be visible in the same source unit
    norm.fn.__globals__["square"] = square
    xs = [3.0, 4.0]
    assert norm(xs) == pytest.approx(5.0)


def test_unsupported_falls_back():
    @jit(mode="blocking")
    def stringy(n: int) -> float:
        return float(len(str(n)))

    assert stringy(12345) == 5.0
    assert stringy(12345) == 5.0
    assert not stringy.compiled
    assert stringy.stats.python_calls == 2
    assert stringy.stats.last_error


def test_strict_raises():
    @jit(mode="blocking", strict=True)
    def stringy(n: int) -> float:
        return float(len(str(n)))

    with pytest.raises(Unsupported):
        stringy(1)


def test_execute_arbitrary_source():
    src = """
        def kernel(n):
            total = 0
            for i in range(n):
                if i % 3 == 0 or i % 5 == 0:
                    total += i
            return total
    """
    assert execute(src, "kernel", (1000,)) == sum(
        i for i in range(1000) if i % 3 == 0 or i % 5 == 0
    )


def test_compile_function_direct():
    def axpy(a: float, xs: list[float], ys: list[float]) -> float:
        t = 0.0
        for i in range(len(xs)):
            ys[i] = a * xs[i] + ys[i]
            t += ys[i]
        return t

    call, src = compile_function(axpy, [FLOAT, ARRAY_F64, ARRAY_F64], FLOAT)
    assert "@export" in src and "abi(\"C\")" in src
    xs = [1.0, 2.0]
    ys = [1.0, 1.0]
    assert call(2.0, xs, ys) == pytest.approx(8.0)
    assert ys == [3.0, 5.0]


def test_compile_batch_shares_one_compile():
    from mojosub import compile_batch

    jobs = [
        (f"def k(n):\n    t = 0\n    for i in range(n):\n        t += i * {j}\n    return t",
         "k", (10,))
        for j in range(1, 4)
    ]
    calls = compile_batch(jobs)
    for j, call in enumerate(calls, start=1):
        assert call(10) == sum(i * j for i in range(10))


def test_disk_cache_is_content_addressed():
    from mojosub import cache

    unit = (
        'def triple(n: Int) -> Int:\n'
        '    return n * 3\n\n'
        '@export("mojosub_triple")\n'
        'def mojosub_wrap_triple(n: Int) abi("C") -> Int:\n'
        '    return triple(n)\n'
    )
    first = cache.compile_unit(unit)
    assert cache.lookup(unit) == first
    assert cache.compile_unit(unit) == first  # second call must not rebuild
