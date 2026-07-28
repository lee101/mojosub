import math
import time

import pytest

from mojosub import Unsupported, execute, jit
from mojosub.api import compile_function
from mojosub.types import ARRAY_F64, FLOAT, INT


def test_scalar_roundtrip():
    # warm=False so this asserts the cold contract - first call interpreted,
    # later calls native. With warm start on, a library left in the cache by an
    # earlier run makes even the first call native, which `test_warm_start_*`
    # covers on purpose.
    @jit(mode="blocking", warm=False)
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


def test_hot_gating_delays_compilation():
    # The gate exists to avoid paying a compiler for a function that may never
    # run again, so it is only meaningful when there is a compile to pay for.
    @jit(mode="blocking", hot_calls=3, warm=False)
    def acc(n: int) -> int:
        t = 0
        for i in range(n):
            t += i
        return t

    acc(10)
    acc(10)
    assert not acc.compiled          # still proving itself
    assert acc.stats.python_calls == 2
    acc(10)                          # third call crosses the threshold
    assert acc.compiled
    assert acc(10) == 45


def test_verify_retires_a_wrong_variant():
    @jit(mode="blocking", verify=True)
    def overflowing(n: int) -> int:
        # CPython promotes past 2**63; Mojo's Int wraps, so these disagree
        t = 1
        for _ in range(n):
            t = t * 1103515245 + 12345
        return t

    truth = overflowing.fn(8)
    assert overflowing(8) == truth   # first call: python, schedules the build
    assert overflowing(8) == truth   # second: verified, mismatch, retired
    assert overflowing.stats.verify_failures == 1
    assert overflowing(8) == truth   # third: python again, permanently
    assert "verification failed" in overflowing.stats.last_error


def test_verify_passes_on_agreeing_code():
    @jit(mode="blocking", verify=True)
    def mean(xs: list[float]) -> float:
        s = 0.0
        for i in range(len(xs)):
            s += xs[i]
        return s / float(len(xs))

    xs = [float(i) for i in range(1000)]
    mean(xs)
    assert mean(xs) == pytest.approx(mean.fn(xs))
    assert mean.stats.verify_failures == 0
    assert mean.stats.mojo_calls >= 1


def test_reduction_is_vectorized():
    from mojosub.api import compile_function
    from mojosub.types import ARRAY_F64, FLOAT

    def total(xs: list[float]) -> float:
        s = 0.0
        for i in range(len(xs)):
            s += xs[i] * xs[i]
        return s

    call, src = compile_function(total, [ARRAY_F64], FLOAT)
    assert "SIMD[DType.float64, 4]" in src
    xs = [float(i) for i in range(1000)]
    assert call(xs) == pytest.approx(sum(x * x for x in xs))

    scalar, src2 = compile_function(total, [ARRAY_F64], FLOAT, vectorize=False)
    assert "SIMD" not in src2
    assert scalar(xs) == pytest.approx(call(xs))


def _has_gpu() -> bool:
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _has_gpu(), reason="no CUDA device")
def test_gpu_target_matches_the_cpu_result():
    np = pytest.importorskip("numpy")
    from mojosub.api import compile_function, type_of
    from mojosub.types import NONE

    def logistic(a: list[float], b: list[float], steps: int, r: float):
        for i in range(len(a)):
            x = a[i]
            for _ in range(steps):
                x = r * x * (1.0 - x)
            b[i] = x

    z = np.zeros(2)
    argtypes = [type_of(z), type_of(z), type_of(1), type_of(0.0)]
    try:
        gpu, src = compile_function(logistic, argtypes, NONE, target="gpu")
    except Exception as exc:  # MAX not installed in this environment
        pytest.skip(f"no GPU toolchain: {exc}")
    assert "__kernel_logistic" in src and "thread_idx" in src

    cpu, _ = compile_function(logistic, argtypes, NONE)
    a = np.ascontiguousarray(np.linspace(0.01, 0.99, 4096))
    on_gpu, on_cpu = np.zeros(4096), np.zeros(4096)
    gpu(a, on_gpu, 50, 3.7)
    cpu(a, on_cpu, 50, 3.7)
    assert np.allclose(on_gpu, on_cpu)


def test_gpu_rejects_shapes_it_cannot_map():
    from mojosub.api import compile_function, type_of
    from mojosub.types import ARRAY_F64, FLOAT, NONE

    def reduction(xs: list[float]) -> float:
        s = 0.0
        for i in range(len(xs)):
            s += xs[i]
        return s

    with pytest.raises(Unsupported):
        compile_function(reduction, [ARRAY_F64], FLOAT, target="gpu")


# -- warm start ------------------------------------------------------------
#
# The JIT used to be worthless to anything that calls a function once per
# process - generated code, a request handler, a script. It ran CPython, noted
# one call, and exited with the compiled library it could have used sitting
# unread in the cache. Warm start reads it.


def _agent_function(source: str, name: str):
    """A function defined the way generated code arrives: exec of a string."""
    ns = {"__mojosub_source__": source}
    exec(compile(source, "<agent>", "exec"), ns)  # noqa: S102
    return ns[name]


AGENT_SRC = """
def streak(xs: list[float], limit: float) -> int:
    best = 0
    run = 0
    for i in range(len(xs)):
        if xs[i] > limit:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best
"""


def test_source_can_be_supplied_for_code_that_has_no_file():
    """`inspect.getsource` cannot see an exec'd function.

    That refusal used to disqualify exactly the functions worth compiling.
    """
    fn = _agent_function(AGENT_SRC, "streak")
    with pytest.raises(OSError):
        import inspect

        inspect.getsource(fn)

    accelerated = jit(fn, mode="blocking", hot_calls=1, source=AGENT_SRC)
    xs = [float(i % 7) for i in range(500)]
    assert accelerated(xs, 3.0) == fn(xs, 3.0)
    assert accelerated.compiled


def test_source_is_found_in_the_namespace_it_was_exec_d_into():
    """A caller that cannot pass `source=` can leave it in the globals."""
    fn = _agent_function(AGENT_SRC, "streak")
    accelerated = jit(fn, mode="blocking", hot_calls=1)
    xs = [float(i % 5) for i in range(300)]
    assert accelerated(xs, 2.0) == fn(xs, 2.0)
    assert accelerated.compiled


def test_warm_start_dispatches_natively_on_the_first_call():
    fn = _agent_function(AGENT_SRC, "streak")
    xs = [float(i % 9) for i in range(400)]

    first = jit(fn, mode="blocking", hot_calls=1, source=AGENT_SRC)
    first(xs, 4.0)
    # Compiled here, or warm-started from a library an earlier test left in the
    # cache - either way the library now exists, which is this test's premise.
    assert first.compiled

    # A second wrapper is a second process as far as the cache is concerned:
    # no variants, no profile, only what is on disk.
    second = jit(fn, mode="blocking", hot_calls=99, source=AGENT_SRC)
    assert second(xs, 4.0) == fn(xs, 4.0)
    assert second.stats.warm_starts == 1
    assert second.stats.compiles == 0
    assert second.stats.python_calls == 0, "warm start must not need CPython first"


def test_warm_start_bypasses_the_hot_gate_only_because_nothing_is_compiled():
    """The gate protects against paying a compiler, not against going fast."""
    fn = _agent_function(AGENT_SRC, "streak")
    xs = [float(i % 3) for i in range(200)]
    jit(fn, mode="blocking", hot_calls=1, source=AGENT_SRC)(xs, 1.0)

    gated = jit(fn, mode="blocking", hot_calls=1000, source=AGENT_SRC)
    gated(xs, 1.0)
    assert gated.compiled, "a cached library should be used whatever the gate says"


def test_warm_start_off_restores_the_cold_path():
    fn = _agent_function(AGENT_SRC, "streak")
    xs = [float(i % 4) for i in range(120)]
    jit(fn, mode="blocking", hot_calls=1, source=AGENT_SRC)(xs, 1.0)

    cold = jit(fn, mode="blocking", hot_calls=1000, warm=False, source=AGENT_SRC)
    cold(xs, 1.0)
    assert not cold.compiled
    assert cold.stats.warm_starts == 0


def test_a_variant_slower_than_cpython_is_retired():
    """Crossing the ABI copies a list; for a cheap loop over a long one that
    costs more than it saves, and an accelerator that can lose is worse than
    none, because nobody profiles the thing that is meant to be the fast path.

    Both sides are stubbed with sleeps rather than raced for real: the decision
    under test is the comparison, and a test that depends on how loaded the
    machine is would be asserting something else.
    """
    from mojosub import api

    # Its own source: retiring is now written to the cache, and a verdict about
    # this test's stubbed timings must not follow the shared function around.
    src = AGENT_SRC.replace("def streak", "def streak_retire")
    fn = _agent_function(src, "streak_retire")
    accelerated = jit(fn, mode="blocking", hot_calls=1, verify=True, source=src)
    xs = [float(i % 6) for i in range(200)]
    accelerated(xs, 2.0)
    assert accelerated.compiled

    key = next(iter(accelerated._variants))
    native = accelerated._variants[key]
    expected = fn(xs, 2.0)

    def slow_native(*args):
        time.sleep(0.05)
        return native(*args)

    def quick_python(*args):
        time.sleep(0.005)     # over MIN_TIMED_SECONDS, so the guard judges
        return expected

    accelerated._variants[key] = slow_native
    accelerated._unverified.add(key)
    accelerated.fn = quick_python

    assert accelerated(xs, 2.0) == expected
    assert accelerated.stats.retired_slower == 1
    assert "retired" in accelerated.stats.last_error
    assert not accelerated.compiled

    # and it stays retired: the answer keeps coming from the interpreter
    assert accelerated(xs, 2.0) == expected
    assert api.SLOWER_THAN_PYTHON > 1.0


def test_a_short_call_is_not_judged_on_timing_noise():
    """A 50-microsecond sample cannot tell fast from slow, and a variant
    retired on one never comes back - not even for a million rows."""
    from mojosub import api

    assert api.MIN_TIMED_SECONDS > 0
    fn = _agent_function(AGENT_SRC, "streak")
    accelerated = jit(fn, mode="blocking", hot_calls=1, verify=True, source=AGENT_SRC)
    tiny = [1.0, 2.0, 3.0]
    accelerated(tiny, 1.5)
    accelerated(tiny, 1.5)
    assert accelerated.stats.retired_slower == 0
    assert accelerated.compiled
