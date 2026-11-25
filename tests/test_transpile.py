import ast
import math
import textwrap
import time

import pytest

from mojosub import Unsupported, execute, jit
from mojosub.api import RACE_CALLS as RACE_SETTLE_CALLS, compile_function
from mojosub.transpile import signature_from_annotations, transpile
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
    # The exported symbol keeps the Python name — the runtime looks it up by
    # that — while the kernel and every identifier inside it are namespaced.
    assert '@export("mojosub_logistic")' in src
    assert "__kernel_u_logistic" in src and "thread_idx" in src

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
    accelerated._trials[key] = []
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


def test_racing_a_kernel_that_writes_its_buffer_applies_it_once():
    """Both tiers run on the same call while the race is on. A kernel that
    writes through its argument would then be applied twice, and the caller
    would get a doubled array out of what is only supposed to be a
    measurement - so each tier runs against the original contents and the
    native one's writes are what survives."""
    np = pytest.importorskip("numpy")

    @jit(mode="blocking", verify=True)
    def double_in_place(xs: list[float], k: float) -> float:
        t = 0.0
        for i in range(len(xs)):
            xs[i] = xs[i] * k
            t += xs[i]
        return t

    for _ in range(RACE_SETTLE_CALLS + 2):
        arr = np.arange(1024, dtype=float)
        total = double_in_place(arr, 2.0)
        assert total == pytest.approx(float(np.arange(1024).sum() * 2))
        assert arr[1] == pytest.approx(2.0)   # not 4.0
    assert double_in_place.stats.verify_failures == 0


def test_race_verdict_needs_more_than_one_close_sample():
    """A close result is sampled again; a decisive one is not.

    The first native call pays a page-in that the interpreted call it is
    compared against has just warmed the caches for, so a single close sample
    is not evidence. A landslide is, and stopping early on it is what keeps the
    race down to one extra interpreted call in the case that actually matters.
    """
    from mojosub.api import RACE_CALLS, _race_verdict

    close = (0.010, 0.013)
    verdict, _, _ = _race_verdict([close])
    assert verdict is None
    verdict, _, _ = _race_verdict([close] * RACE_CALLS)
    assert verdict == "retire"

    # decisive either way, settled on the first sample
    assert _race_verdict([(0.100, 0.002)])[0] == "keep"
    assert _race_verdict([(0.002, 0.100)])[0] == "retire"


def test_race_takes_the_median_not_the_worst_sample():
    """One descheduled call must not retire a variant that is really faster."""
    from mojosub.api import RACE_CALLS, _race_verdict

    assert RACE_CALLS >= 3
    samples = [(0.010, 0.009), (0.010, 0.400), (0.010, 0.009)]
    verdict, python_seconds, native_seconds = _race_verdict(samples)
    assert verdict == "keep"
    assert native_seconds < python_seconds


def test_a_won_race_is_persisted_so_the_next_process_skips_it():
    """The race costs interpreted calls, which is what compiling was supposed
    to stop paying for. It is a fact about (source, signature, toolchain), so
    it belongs on disk next to the library it is about."""
    np = pytest.importorskip("numpy")
    from mojosub import cache

    src = AGENT_SRC.replace("def streak", "def streak_raced")
    fn = _agent_function(src, "streak_raced")
    accelerated = jit(fn, mode="blocking", hot_calls=1, verify=True, source=src)
    # numpy, not a list: a list is copied across the ABI and the race would
    # correctly conclude that the interpreter wins.
    xs = np.arange(200000, dtype=float) % 6.0
    for _ in range(RACE_SETTLE_CALLS + 1):
        accelerated(xs, 2.0)
    key = next(iter(accelerated._variants))
    assert key not in accelerated._trials

    verdict = cache.recall_verdict(accelerated._source(), key)
    assert verdict.get("raced") is True
    assert verdict.get("speedup", 0) > 0

    # a fresh wrapper over the same source warm-starts straight into native
    # dispatch, with no interpreted call spent re-deciding
    again = jit(_agent_function(src, "streak_raced"), mode="blocking",
                hot_calls=1, verify=True, source=src)
    again(xs, 2.0)

    assert again.stats.warm_starts == 1
    assert again.stats.python_calls == 0


def test_a_list_and_an_array_do_not_share_a_verdict():
    """They compile to the same Mojo, so they used to share a signature key -
    but one crosses the ABI as a pointer and the other is copied element by
    element, which is the difference between the compiled variant winning and
    losing. A verdict reached on one must not silence the race on the other."""
    np = pytest.importorskip("numpy")

    src = AGENT_SRC.replace("def streak", "def streak_keyed")
    fn = _agent_function(src, "streak_keyed")
    accelerated = jit(fn, mode="blocking", hot_calls=1, verify=True, source=src)

    accelerated(np.arange(4096, dtype=float) % 7.0, 2.0)
    accelerated(np.arange(4096, dtype=float) % 7.0, 2.0)
    array_keys = set(accelerated._variants) | set(accelerated._failed)

    accelerated([float(i % 7) for i in range(4096)], 2.0)
    accelerated([float(i % 7) for i in range(4096)], 2.0)
    all_keys = set(accelerated._variants) | set(accelerated._failed)

    assert len(all_keys) == len(array_keys) + 1


# --- identifier namespacing -------------------------------------------------
#
# Three bugs, all found by running a corpus of ordinary statistics code through
# the pipeline and looking at what the host compiler said. Each one is cheap to
# reintroduce and expensive to have: two are five wasted seconds of compiler per
# submission, and the third is a wrong answer.


def test_mojo_keywords_are_legal_python_names():
    """`var = sq/n - mean*mean` is what computing a variance looks like.

    It used to emit `var var: Float64 = ...` — a syntax error — so the kernel
    burned a compile on every submission and never once went native.
    """
    def variance(xs: list[float]) -> float:
        total = 0.0
        for i in range(len(xs)):
            total += xs[i]
        mean = total / float(len(xs))
        var = 0.0
        fn = 0.0
        for i in range(len(xs)):
            fn = xs[i] - mean
            var += fn * fn
        return var / float(len(xs))

    call, src = compile_function(variance, [ARRAY_F64], FLOAT)
    assert "var var" not in src and "var fn" not in src
    xs = [float(i % 17) for i in range(500)]
    mean = sum(xs) / len(xs)
    assert call(xs) == pytest.approx(sum((x - mean) ** 2 for x in xs) / len(xs))


def test_a_users_own_len_name_does_not_shadow_a_buffer_length():
    """`xs_len` is the name we synthesise for `len(xs)`.

    A program with its own variable of that name used to reassign it, so every
    later `len(xs)` returned whatever the user had stored — a wrong answer, in
    silence, which the verify pass can only catch after the fact.
    """
    def count_above(xs: list[float], cut: float) -> float:
        xs_len = 3.0
        hits = 0.0
        for i in range(len(xs)):
            if xs[i] > cut:
                hits += 1.0
        return hits + xs_len * 0.0

    call, src = compile_function(count_above, [ARRAY_F64, FLOAT], FLOAT)
    xs = [float(i) for i in range(100)]
    # 100 elements, not the 3 the user's variable would have claimed.
    assert call(xs, -1.0) == pytest.approx(100.0)
    assert "n_xs" in src and "u_xs_len" in src


def test_a_type_name_can_be_used_as_a_variable():
    def shadowy(n: int) -> int:
        Int = 0
        for i in range(n):
            Int += i
        return Int

    call, _src = compile_function(shadowy, [INT], INT)
    assert call(10) == 45


def test_namespacing_survives_helper_calls_and_buffer_lengths():
    """A helper has to receive the *caller's* length for the buffer it is given.

    The length is derived from the argument's name, so passing one across a call
    means naming it in the callee's namespace — the one place the two mappings
    have to agree. `compile_source` rather than a nested def: helpers are only
    reachable when they share a source unit with the entry point.
    """
    from mojosub.api import compile_source

    src = """
def inner(xs: list[float], var: float) -> float:
    out = 0.0
    for i in range(len(xs)):
        out += xs[i] * var
    return out

def outer(xs: list[float]) -> float:
    return inner(xs, 2.0) + float(len(xs))
"""
    xs = [1.0, 2.0, 3.0]
    call = compile_source(src, "outer", (xs,))
    assert call(xs) == pytest.approx(12.0 + 3.0)


def test_reassigning_a_buffer_parameter_is_refused_not_miscompiled():
    def rebind(xs: list[float]) -> float:
        xs = 1.0
        return xs

    with pytest.raises(Unsupported):
        compile_function(rebind, [ARRAY_F64], FLOAT)


# --- what the compiled code may write through -------------------------------


def test_a_read_only_list_argument_is_not_copied_back():
    """The copy back is 45-63% of the call for a reduction over a list.

    It is also pointless there: a reduction cannot have stored anything. The
    transpiler records which parameters it emitted stores for, and the runtime
    skips the write-back for the rest.
    """
    def dot(a: list[float], b: list[float]) -> float:
        total = 0.0
        for i in range(len(a)):
            total += a[i] * b[i]
        return total

    call, _src = compile_function(dot, [ARRAY_F64, ARRAY_F64], FLOAT)
    assert call.sig.writes == frozenset()
    assert not call.sig.may_write("a")
    xs, ys = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    assert call(xs, ys) == pytest.approx(32.0)
    # and the inputs are untouched, which is the only thing skipping it risks
    assert xs == [1.0, 2.0, 3.0] and ys == [4.0, 5.0, 6.0]


def test_a_written_list_argument_is_still_copied_back():
    def scale(xs: list[float], k: float) -> float:
        total = 0.0
        for i in range(len(xs)):
            xs[i] = xs[i] * k
            total += xs[i]
        return total

    call, _src = compile_function(scale, [ARRAY_F64, FLOAT], FLOAT)
    assert call.sig.writes == frozenset({"xs"})
    data = [1.0, 2.0, 3.0]
    assert call(data, 2.0) == pytest.approx(12.0)
    assert data == [2.0, 4.0, 6.0]


def test_a_buffer_passed_to_a_helper_is_assumed_written():
    """Following a buffer into a callee is a real escape analysis; getting it
    wrong hands the caller a silently stale array. So a buffer handed to a
    helper keeps its copy back, whatever the helper does with it."""
    from mojosub.api import compile_source

    src = """
def fill(xs: list[float], k: float) -> float:
    for i in range(len(xs)):
        xs[i] = k
    return k

def outer(xs: list[float]) -> float:
    return fill(xs, 7.0)
"""
    data = [1.0, 2.0, 3.0]
    call = compile_source(src, "outer", (data,))
    assert call(data) == pytest.approx(7.0)
    assert data == [7.0, 7.0, 7.0]


def test_an_unanalysed_signature_still_copies_back():
    """`writes=None` means nobody looked, which has to read as "may write" —
    a hand-built Signature must not silently lose a write-back."""
    from mojosub.transpile import Signature

    sig = Signature("f", [("xs", ARRAY_F64)], FLOAT)
    assert sig.writes is None
    assert sig.may_write("xs")


def test_cache_key_separates_machines_that_cannot_run_each_others_code():
    """`mojo build` targets the host CPU, and the cache is shared across a
    fleet through an object store. A unit built where AVX-512 exists and loaded
    where it does not runs until the first vector instruction and then dies, so
    the instruction set has to be part of the key."""
    from mojosub import cache

    unit = "def f(n: Int) -> Int:\n    return n\n"
    here = cache.key_for(unit, 3)

    real = cache.host_target
    try:
        cache.host_target = lambda: "0" * 12
        elsewhere = cache.key_for(unit, 3)
    finally:
        cache.host_target = real
    assert here != elsewhere, "same key on machines with different instruction sets"
    # and the same machine still shares with itself
    assert cache.key_for(unit, 3) == here


# --- module-level constants ---------------------------------------------
#
# `WINDOW = 64` at the top of a file and `range(WINDOW)` in the kernel is
# ordinary Python and was the most common reason a numeric function would not
# compile. The literal is folded into the emitted Mojo, so the unit's content
# hash moves when the constant does.

def _module(src):
    return ast.parse(textwrap.dedent(src))


def test_a_module_constant_is_folded_in():
    tree = _module("""
        WINDOW = 64

        def k(n: int) -> int:
            t = 0
            for i in range(n):
                t += WINDOW
            return t
    """)
    unit = transpile(tree, {"k": signature_from_annotations(tree.body[1])}, "k")
    assert "64" in unit.source
    assert "u_WINDOW" not in unit.source


def test_a_float_and_a_negative_constant_are_folded():
    tree = _module("""
        RATE = 0.25
        OFFSET = -3

        def k(n: int) -> float:
            t = 0.0
            for i in range(n):
                t += RATE * float(OFFSET)
            return t
    """)
    unit = transpile(tree, {"k": signature_from_annotations(tree.body[2])}, "k")
    assert "0.25" in unit.source and "-3" in unit.source


def test_a_name_the_module_rebinds_is_not_a_constant():
    """A constant that is really a variable would be frozen at whatever value
    it happened to have when we read the source."""
    tree = _module("""
        LIMIT = 10
        LIMIT = 20

        def k(n: int) -> int:
            t = 0
            for i in range(n):
                t += LIMIT
            return t
    """)
    with pytest.raises(Unsupported):
        transpile(tree, {"k": signature_from_annotations(tree.body[2])}, "k")


def test_a_name_a_function_assigns_is_not_a_constant():
    tree = _module("""
        COUNT = 5

        def bump():
            global COUNT
            COUNT = 6

        def k(n: int) -> int:
            return COUNT * n
    """)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "k"][0]
    with pytest.raises(Unsupported):
        transpile(tree, {"k": signature_from_annotations(fn)}, "k")


def test_a_computed_module_value_is_not_a_constant():
    """Only literals. Anything else is an expression we would have to run."""
    tree = _module("""
        SIZE = len([1, 2, 3])

        def k(n: int) -> int:
            return SIZE + n
    """)
    fn = tree.body[1]
    with pytest.raises(Unsupported):
        transpile(tree, {"k": signature_from_annotations(fn)}, "k")


def test_a_parameter_beats_a_constant_of_the_same_name():
    tree = _module("""
        SCALE = 2

        def k(SCALE: int) -> int:
            return SCALE + 1
    """)
    unit = transpile(tree, {"k": signature_from_annotations(tree.body[1])}, "k")
    assert "u_SCALE + 1" in unit.source.replace("(", "").replace(")", "")


def test_changing_a_constant_changes_the_emitted_source():
    """Which is what keeps the content-addressed cache honest."""
    def unit_for(value):
        tree = _module(f"""
            W = {value}

            def k(n: int) -> int:
                return W * n
        """)
        return transpile(tree, {"k": signature_from_annotations(tree.body[1])},
                         "k").source

    assert unit_for(3) != unit_for(4)


def test_a_compile_that_cannot_be_started_does_not_end_the_program(monkeypatch):
    """Compiling is an optimisation. A machine that cannot spare a thread must
    cost the caller their speedup, never their run."""
    import threading as _threading

    from mojosub import api as _api

    class Refusing:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(_api.threading, "Thread", Refusing)

    @jit(hot_calls=1, warm=False)
    def k(n: int) -> int:
        t = 0
        for i in range(n):
            t += i
        return t

    assert k(10) == 45
    assert k(10) == 45
    assert "could not start a compile" in (k.stats.last_error or "")
    assert _threading  # the real module is untouched outside the patch


def test_a_kernel_that_calls_a_helper_actually_compiles():
    """`_module_of` used to keep only the entry function, so a kernel calling
    an annotated helper raised `Unsupported: call to helper` every time — the
    filter offered it, the transpiler refused it, and the compile budget went
    on nothing. Helpers are ordinary code; this is the common case, not a
    corner."""
    source = textwrap.dedent("""
        def helper(x: float) -> float:
            return x * x + 1.0

        def outer(n: int) -> float:
            t = 0.0
            for i in range(n):
                t += helper(i * 0.5)
            return t
    """)
    ns: dict = {"__mojosub_source__": source}
    exec(source, ns)  # noqa: S102 - the point is a function with no file
    call, mojo = compile_function(ns["outer"])
    assert "u_helper" in mojo
    assert call(10) == ns["outer"](10)


# --- 2-D buffers ------------------------------------------------------------
#
# The shape almost all interesting machine-learning work has is a feature
# matrix. Without this it could not be expressed at all: a kernel over one had
# to be rewritten against a flat array with the stride arithmetic by hand, which
# is a different program from the one the caller debugged in numpy.


def test_a_matrix_is_indexed_row_major():
    np = pytest.importorskip("numpy")
    from mojosub.api import compile_source

    src = """
def rowsums(m, out) -> float:
    total = 0.0
    for i in range(m.shape[0]):
        s = 0.0
        for j in range(m.shape[1]):
            s += m[i, j]
        out[i] = s
        total += s
    return total
"""
    m = np.arange(12, dtype=np.float64).reshape(3, 4)
    out = np.zeros(3)
    call = compile_source(src, "rowsums", (m, out))
    out[:] = 0.0
    assert call(m, out) == pytest.approx(66.0)
    assert list(out) == [6.0, 22.0, 38.0]


def test_a_matrix_and_a_flat_array_do_not_share_a_variant():
    """Both compile to a bare pointer, so the emitted Mojo text is identical.
    Keying a variant on that text would let a 2-D call reuse the 1-D library —
    which reads the wrong addresses and returns a plausible wrong number."""
    from mojosub.types import ARRAY_F64, MATRIX_F64

    assert ARRAY_F64.mojo == MATRIX_F64.mojo
    assert ARRAY_F64.key != MATRIX_F64.key
    # Array keeps its old spelling so signature memos already on disk resolve.
    assert ARRAY_F64.key == ARRAY_F64.mojo

    from mojosub.types import by_mojo

    assert by_mojo(MATRIX_F64.key) is MATRIX_F64
    assert by_mojo(ARRAY_F64.key) is ARRAY_F64


def test_len_of_a_matrix_is_its_row_count():
    np = pytest.importorskip("numpy")
    from mojosub.api import compile_source

    src = """
def rows(m) -> int:
    n = 0
    for i in range(len(m)):
        n += 1
    return n
"""
    m = np.zeros((7, 3))
    assert compile_source(src, "rows", (m,))(m) == 7


def test_a_non_contiguous_matrix_is_refused_not_misread():
    """The emitted index arithmetic is `row * cols + col`. That is the wrong
    address for a transposed or sliced array, and reading the wrong memory
    returns a plausible number rather than an error — so it is refused."""
    np = pytest.importorskip("numpy")
    from mojosub.api import compile_source

    src = """
def total(m) -> float:
    t = 0.0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            t += m[i, j]
    return t
"""
    m = np.arange(12, dtype=np.float64).reshape(3, 4)
    call = compile_source(src, "total", (m,))
    assert call(m) == pytest.approx(66.0)
    with pytest.raises(TypeError, match="contiguous"):
        call(m[:, ::2])
    # and the fix the message names actually works
    assert call(np.ascontiguousarray(m[:, ::2])) == pytest.approx(
        float(m[:, ::2].sum()))


def test_a_list_of_lists_is_refused_with_a_useful_message():
    """It used to be typed as a flat int buffer and fail later at marshalling."""
    from mojosub.api import type_of

    with pytest.raises(Unsupported, match="numpy array"):
        type_of([[1.0, 2.0], [3.0, 4.0]])


def test_a_matrix_index_on_a_flat_buffer_is_refused():
    from mojosub.api import compile_function
    from mojosub.types import ARRAY_F64

    def bad(xs: list[float]) -> float:
        t = 0.0
        for i in range(len(xs)):
            t += xs[i, 0]
        return t

    with pytest.raises(Unsupported):
        compile_function(bad, [ARRAY_F64], FLOAT)


def test_a_matrix_crosses_into_a_helper_with_both_shape_arguments():
    np = pytest.importorskip("numpy")
    from mojosub.api import compile_source

    # The helper needs full annotations to be in the unit at all, and
    # `list[list[float]]` is how a 2-D buffer is spelled — the same way
    # `list[float]` means a flat one.
    src = """
def row_total(m: list[list[float]], i: int) -> float:
    t = 0.0
    for j in range(m.shape[1]):
        t += m[i, j]
    return t

def grand(m) -> float:
    t = 0.0
    for i in range(m.shape[0]):
        t += row_total(m, i)
    return t
"""
    m = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert compile_source(src, "grand", (m,))(m) == pytest.approx(66.0)


def test_shape_is_the_only_attribute_allowed():
    from mojosub.api import compile_function
    from mojosub.types import MATRIX_F64

    def bad(m) -> float:
        t = 0.0
        for i in range(m.size):
            t += 1.0
        return t

    with pytest.raises(Unsupported):
        compile_function(bad, [MATRIX_F64], FLOAT)


def test_verify_compares_a_2d_buffer_elementwise():
    """A 2-D buffer written by the kernel has to be comparable against CPython's.

    `list()` on a 2-D array yields rows, so the naive elementwise comparison
    hands two arrays to a scalar comparison and raises "the truth value of an
    array is ambiguous" — turning the verify pass, which exists to make the fast
    path trustworthy, into the thing that breaks the run.
    """
    np = pytest.importorskip("numpy")

    @jit(mode="blocking", verify=True, warm=False)
    def scale_rows(m, k: float) -> float:
        total = 0.0
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i, j] = m[i, j] * k
                total += m[i, j]
        return total

    want = float(np.arange(12, dtype=np.float64).sum() * 3.0)
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert scale_rows(a, 3.0) == pytest.approx(want)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert scale_rows(b, 3.0) == pytest.approx(want)   # the verified call
    assert scale_rows.stats.verify_failures == 0
    assert np.allclose(b, np.arange(12).reshape(3, 4) * 3.0)


def test_reassociation_is_tolerated_but_a_wrong_answer_is_not():
    """The verify tolerance has to sit between two things that are far apart.

    Summing n terms in a different order — which is what vectorising a reduction
    does — differs by about `n * eps * sum(|x_i|)`, and for a sum with heavy
    cancellation that relative error reaches 1e-8. A real miscompile is nothing
    like that small: wrong indexing reads different memory, a wrong operator
    computes a different function, a wrapped integer is off by 2^64. If the line
    is drawn too tight, correct kernels are retired for ever and fall back
    silently, which is what 1e-9 did to a 261x logistic gradient.
    """
    from mojosub.api import VERIFY_RTOL, _equalish

    # reassociation-scale disagreement: kept
    assert _equalish(1.7556090564380296, 1.755609045572715)
    assert 1e-9 < VERIFY_RTOL < 1e-3, VERIFY_RTOL

    # miscompile-scale disagreement: rejected
    assert not _equalish(1.0, 1.001)
    assert not _equalish(1.0, -1.0)
    assert not _equalish(0.0, 1e-3)
    # and integers are still compared exactly, which is what catches Int wrap
    assert not _equalish(2 ** 63, -(2 ** 63))


def test_an_unrelated_function_in_the_file_does_not_block_compilation():
    """A unit is emitted whole or not at all, so only what the entry point can
    actually reach may go into it.

    Every annotated sibling used to be included regardless. One function using
    `sorted` therefore made every function in the module uncompilable, and the
    reason reported for a clean kernel was `call to sorted` — pointing at code
    that is not in it.
    """
    from mojosub.api import compile_source

    src = """
def needs_sorting(xs: list[float]) -> float:
    return sorted(xs)[0]

def ewma(xs: list[float], alpha: float) -> float:
    acc = xs[0]
    for i in range(1, len(xs)):
        acc = alpha * xs[i] + (1.0 - alpha) * acc
    return acc
"""
    xs = [1.0, 2.0, 3.0, 4.0]
    call = compile_source(src, "ewma", (xs, 0.5))
    expected = xs[0]
    for x in xs[1:]:
        expected = 0.5 * x + 0.5 * expected
    assert call(xs, 0.5) == pytest.approx(expected)


def test_a_helper_the_entry_point_calls_is_still_included():
    from mojosub.api import compile_source

    src = """
def unrelated(xs: list[float]) -> float:
    return sorted(xs)[0]

def twice(x: float) -> float:
    return x * 2.0

def total(xs: list[float]) -> float:
    t = 0.0
    for i in range(len(xs)):
        t += twice(xs[i])
    return t
"""
    xs = [1.0, 2.0, 3.0]
    assert compile_source(src, "total", (xs,))(xs) == pytest.approx(12.0)


# --- Python scoping vs Mojo scoping -----------------------------------------
#
# Three bugs, all found by the first dozen programs out of bench/fuzz.py, all the
# same root cause: Python scopes a local to its function and Mojo scopes a `var`
# to its block. Every one of them made ordinary Python fail to compile — and in
# the service a failed compile is invisible, because the tiered design falls back
# to CPython and returns the right answer slowly.


def test_a_function_may_reassign_its_own_parameter():
    """`n = n * 2` is ordinary Python. A Mojo parameter is an immutable binding,
    so emitting it verbatim is `invalid redefinition` and the kernel never
    compiled — the first thing the fuzzer generated."""
    @jit(mode="blocking", warm=False)
    def doubled(n: int, x: float) -> float:
        n = n * 2
        x = x + 1.0
        total = 0.0
        for _ in range(n):
            total += x
        return total

    assert doubled(3, 1.0) == pytest.approx(12.0)
    assert doubled(3, 1.0) == pytest.approx(12.0)
    assert doubled.compiled
    assert doubled.stats.last_error is None, doubled.stats.last_error


def test_a_local_first_assigned_inside_a_block_survives_it():
    """Python scopes to the function; a Mojo `var` scopes to its block."""
    @jit(mode="blocking", warm=False)
    def conditional(n: int) -> float:
        total = 0.0
        for i in range(n):
            if i % 2 == 0:
                hit = 1.5
                total += hit
        return total + hit

    assert conditional(4) == pytest.approx(3.0 + 1.5)
    assert conditional(4) == pytest.approx(3.0 + 1.5)
    assert conditional.compiled


def test_a_loop_counter_can_be_read_after_its_loop():
    """Python leaves a `for` target bound at its last value; Mojo does not."""
    @jit(mode="blocking", warm=False)
    def last_index(n: int) -> float:
        for i in range(n):
            pass
        return float(i)

    assert last_index(5) == pytest.approx(4.0)
    assert last_index(5) == pytest.approx(4.0)
    assert last_index.compiled


def test_a_loop_counter_can_be_reassigned_inside_its_loop():
    """A Mojo loop variable is immutable; Python's is not."""
    @jit(mode="blocking", warm=False)
    def skipping(n: int) -> int:
        seen = 0
        for i in range(n):
            seen += 1
            i = i + 1
        return seen

    assert skipping(4) == 4
    assert skipping(4) == 4
    assert skipping.compiled


# --- a local holds one type in Mojo and many in Python -----------------------
#
# `t = 0` followed by `t += xs[i]` is the commonest shape in this whole subset,
# and declaring `t` from its first assignment made every later store `Int(...)`.
# That is a wrong answer rather than a slow one, and nothing was watching for it:
# the emitted Mojo compiled, ran, and returned a truncated number.


def test_a_local_widens_to_the_type_python_would_give_it():
    def widen(x: float) -> float:
        t = 0                 # an int here, a float after the loop
        for _ in range(3):
            t += x
        flag = True           # a bool, then an int
        flag = 5
        return float(t) + float(flag) + float(2 ** -1)

    call, src = compile_function(widen, [FLOAT], FLOAT)
    assert "var u_t: Float64" in src
    assert "var u_flag: Int" in src
    assert call(1.25) == pytest.approx(widen(1.25))
    assert call(1.25) == pytest.approx(3.75 + 5.0 + 0.5)


def test_an_integer_power_with_a_runtime_exponent_is_refused():
    """`2 ** n` is 0.5 in CPython and 0 in Mojo when n is -1, and which one it
    is cannot be known until the call. A refusal costs the speedup; guessing
    costs the answer."""
    def p(a: int, b: int) -> float:
        return float(a ** b)

    with pytest.raises(Unsupported):
        compile_function(p, [INT, INT], FLOAT)


def test_a_parameter_reassigned_to_a_wider_type_is_refused():
    """A parameter's type is fixed by the ABI, so it cannot be widened the way
    a local can."""
    def halve(n: int) -> float:
        n = n / 2
        return float(n)

    with pytest.raises(Unsupported):
        compile_function(halve, [INT], FLOAT)


def test_a_loop_counter_reassigned_to_a_float_is_refused():
    def bad(n: int) -> float:
        for i in range(n):
            i = 1.5
        return float(i)

    with pytest.raises(Unsupported):
        compile_function(bad, [INT], FLOAT)


def test_math_constants_and_n_ary_min_max():
    """`math.pi` was `Unsupported: expression Attribute`, and Mojo's min/max are
    binary where Python's take any number of arguments. Both cost the whole
    kernel for a line of ordinary arithmetic."""
    def k(x: float) -> float:
        return min(x, math.pi, math.e) + max(x, 1.0, 2.0, 3.0)

    call, src = compile_function(k, [FLOAT], FLOAT)
    assert "3.14159" in src
    assert call(2.0) == pytest.approx(k(2.0))


def test_a_buffer_cannot_be_bound_to_a_local():
    """Its length is derived from the parameter's name and would not follow it.
    Emitting it anyway declares a pointer initialised to 0 and burns a compile
    to discover that."""
    def alias(xs: list[float]) -> float:
        ys = xs
        return ys[0]

    with pytest.raises(Unsupported):
        compile_function(alias, [ARRAY_F64], FLOAT)


def test_a_build_that_never_finishes_is_a_refusal_not_a_hang():
    """Some units make the compiler spin - one program out of bench/fuzz.py ran
    `mojo build` for two days on 35 lines. The tiered design can only fall back
    to CPython if something comes back, so the build is bounded."""
    import os
    import tempfile

    from mojosub.toolchain import ToolchainError, build_shared_lib

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "unit.mojo")
        with open(src, "w") as fh:
            fh.write("def f(n: Int) -> Int:\n    return n\n")
        with pytest.raises(ToolchainError, match="did not finish"):
            build_shared_lib(src, os.path.join(tmp, "unit.so"), timeout=0.05)


def test_the_gpu_target_refuses_a_2d_buffer():
    """A matrix crosses the ABI as (address, rows, cols); the GPU wrapper only
    marshals (address, length), so emitting one builds a symbol whose arity
    does not match the call - undefined behaviour, not an error."""
    from mojosub.types import MATRIX_F64, NONE

    def rows(m, out: list[float]):
        for i in range(len(out)):
            out[i] = m[i, 0]

    with pytest.raises(Unsupported, match="2-D"):
        compile_function(rows, [MATRIX_F64, ARRAY_F64], NONE, target="gpu")


def test_a_loop_inside_a_branch_still_hoists_its_counter():
    """The hoisted declaration has to reach function scope. Emitted at the loop's
    own depth it is still inside the `else:` and just as unreachable."""
    @jit(mode="blocking", warm=False)
    def branchy(n: int, flag: bool) -> int:
        if flag:
            total = 1
        else:
            for i in range(n):
                total = 2
        i = 7
        return total + i

    assert branchy(3, False) == 9
    assert branchy(3, False) == 9
    assert branchy.compiled
