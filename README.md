# mojosub

Compile a subset of Python to [Mojo](https://www.modular.com/mojo), cache the
result by content, and call it through a C symbol in about a microsecond.

It exists to answer one question: **how cheaply can an AI coding agent execute
the code it just wrote?** Agents generate short numeric functions constantly —
scoring loops, backtests, feature transforms, simulations — and then run them
under CPython, which is the slowest part of the loop by two orders of
magnitude. mojosub takes that same string of Python and runs it natively.

```python
from mojosub import jit

@jit
def sma_crossover(prices: list[float], fast: int, slow: int) -> float:
    ...

sma_crossover(prices, 20, 100)   # first call: CPython, compile starts behind it
sma_crossover(prices, 20, 100)   # every later call: native, 112x faster
```

## Results

Measured on a Ryzen workstation, Mojo `1.0.0b3.dev2026072406`, CPython 3.13.
`pixi run python bench/bench.py` reproduces both tables.

| kernel | CPython | mojosub | speedup |
| --- | ---: | ---: | ---: |
| `loop_sum(3e6)` | 570.6 ms | 11.3 ms | **50x** |
| `mandelbrot(200x200, 50)` | 166.1 ms | 3.1 ms | **54x** |
| `fib(27)` recursive | 39.2 ms | 0.8 ms | **48x** |
| `monte_carlo_pi(1e6)` | 1119.9 ms | 15.9 ms | **70x** |
| `dot(1e6)` on numpy input | 87.6 ms | 2.4 ms | **36x** |
| `dot(1e6)` on a Python list | 95.1 ms | 302.9 ms | **0.3x** |
| `logistic(1e4 x 100)` | 71.6 ms | 3.0 ms | **24x** |
| `sma_crossover(1e6)` | 364.7 ms | 3.2 ms | **114x** |

The one loss is the point of the table. A `list[float]` has to be copied into a
contiguous buffer on every call, and for a million elements that copy costs more
than the arithmetic saves. Anything with `__array_interface__` — a numpy array,
an `array.array` — is passed by address with no copy at all.

Latency, which is what actually decides whether this is usable inside an agent
loop:

| path | time |
| --- | ---: |
| CPython `compile()` + `exec` + run | 206 us |
| mojosub cold (transpile + `mojo build` + run) | **4.5 s** |
| mojosub warm, disk cache hit | 0.64 ms |
| mojosub warm, in-process memo | 8.6 us |
| mojosub hot call, `@jit` dispatch | 2.0 us |
| mojosub hot call, bound C symbol | **1.2 us** |

`mojo build` costs 3–5 seconds and that cost is essentially **fixed** — one
function or twenty, the same wall clock. Three things follow, and all three are
implemented here:

1. **Never compile the same thing twice.** Units are content-addressed by the
   hash of their generated Mojo plus the compiler version, so the cache is safe
   to share across processes and machines.
2. **Never block on the first call.** `@jit` runs CPython immediately and
   compiles on a background thread; the swap happens when it is ready. The
   first call also teaches mojosub the return type, which is why unannotated
   functions work.
3. **Batch.** `compile_batch` merges many snippets into a single unit: eight
   snippets compiled together cost 0.98 s each instead of 5 s each, and twenty
   cost 0.25 s each.

### GPU

`target="gpu"` compiles a per-element loop into a CUDA kernel instead. On an
RTX 5090 over 1e6 float64 elements, varying the arithmetic per element:

| flops/byte | GPU | CPU (SIMD) | speedup |
| ---: | ---: | ---: | ---: |
| 0.2 | 4.2 ms | 2.6 ms | 0.6x |
| 1.9 | 3.6 ms | 8.9 ms | 2.5x |
| 18.8 | 6.2 ms | 220 ms | 35x |
| 187.5 | 9.1 ms | 2396 ms | **263x** |

The crossover is around 2 flops per byte, and it is a crossover in arithmetic
intensity, not in size: a single elementwise pass over host memory stayed
transfer-bound at every size tested up to 50M elements. That is why the GPU is
opt-in rather than automatic — a runtime trip count does not tell mojosub how
much arithmetic is coming.

```python
from mojosub import compile_function, type_of
from mojosub.types import NONE

def simulate(a: list[float], b: list[float], steps: int, r: float):
    for i in range(len(a)):          # one iteration per GPU thread
        x = a[i]
        for _ in range(steps):
            x = r * x * (1.0 - x)
        b[i] = x

kernel, _ = compile_function(simulate, argtypes, NONE, target="gpu")
kernel(a, b, 1000, 3.7)              # results come back in b
```

Needs the `max` package for the device runtime (it is in this repo's pixi
environment). Reductions and anything that is not a single element loop are
refused, not silently run on the CPU.

## Install

```bash
pixi install          # brings its own Mojo toolchain
pixi run test
pixi run bench
```

Outside pixi, point mojosub at a compiler:

```bash
export MOJOSUB_MOJO=/path/to/mojo        # or
export MOJOSUB_PIXI_PROJECT=/path/to/a/pixi/project
```

## API

```python
from mojosub import jit, execute, compile_source, compile_batch

@jit                                  # tiered: Python first, Mojo after
@jit(mode="blocking")                 # compile on the first call
@jit(strict=True)                     # raise instead of falling back

execute(source, "kernel", (n,))       # run a string of Python, natively
call = compile_source(source, "kernel", (n,))   # keep the callable
calls = compile_batch([(src, "kernel", args), ...])  # one compile, many entries
```

```python
@jit(hot_calls=4, hot_seconds=0.05)   # compile only what proves it is hot
@jit(verify=True)                     # check the first native result vs CPython
@jit(vectorize=False)                 # turn off SIMD codegen
```

`JitFunction.stats` reports how many calls went each way, how long compilation
took, how many variants failed verification, and why a function fell back.

### Being sure it is right

`verify=True` runs both tiers once per signature and compares before trusting
the compiled variant; a mismatch retires it permanently and the caller keeps
getting the CPython answer. That is what makes the 64-bit integer cliff below
safe to live with rather than a lurking wrong number.

### SIMD

Reduction loops (`s += f(a[i], b[i], ...)`) are emitted as width-4 `SIMD`
accumulations with a scalar tail, which is worth about 2x on a large float
reduction. Store loops are only vectorized when the expression carries at
least three arithmetic operations: below that they are memory-bound, and the
explicit vector form measured *slower* than what LLVM already does with the
scalar loop (0.68 ms vs 1.24 ms over 1e6 elements). The threshold is a
measurement, not a guess.

## The subset

Supported: `int` / `float` / `bool` scalars, flat `float64` and `int64` buffers
(numpy, `array.array`, or a list — a list is copied in and written back),
arithmetic and comparison and boolean operators, `if` / `while` / `for i in
range(...)` / `break` / `continue`, conditional expressions, indexed loads and
stores, `len()` of a buffer, `int()` / `float()` / `bool()` / `abs()` / `min()`
/ `max()`, the usual `math` functions, recursion, and calls to other annotated
functions defined alongside it.

Not supported, and falling back to CPython on sight: strings, dicts, classes,
closures, comprehensions, generators, exceptions, `None`-typed values, 2-D
arrays, and anything imported from outside the unit.

**Integers are 64-bit and they wrap.** CPython promotes to arbitrary precision;
Mojo does not. A kernel whose intermediate values exceed 2^63 will disagree with
Python, and mojosub will not warn you.

## How it works

```
python source ──ast──> typed Mojo source ──sha256──> ~/.cache/mojosub/<key>.so
                                                            │
                                        ctypes.CDLL ────────┘
                                                            │
                       @export("mojosub_f") ... abi("C") ───┴──> ~1 us dispatch
```

Buffers cross the ABI as an `(address, length)` pair of `Int`s and are rebuilt
inside the wrapper with `UnsafePointer[T, AnyOrigin[mut=True]]
(unsafe_from_address=...)`, which is the only form in Mojo 1.0 that gives a
non-parametric, mutable pointer — and `@export` rejects parametric functions.

See [`docs/research.md`](docs/research.md) for the measurements behind the
design, including what does *not* work.

## Related

- [mojo-sklearn](https://github.com/lee101/mojo-sklearn) — scikit-learn's core estimators in Mojo
- [mojo-arrow](https://github.com/lee101/mojo-arrow) — Arrow compute kernels
- [mojo-plotly](https://github.com/lee101/mojo-plotly) — plotly figures with Mojo kernels
- [mojo-notebook](https://github.com/lee101/mojo-notebook) — reactive Python + Mojo notebooks
- [mojo-wasm](https://github.com/lee101/mojo-wasm) — compile Mojo to WebAssembly

## License

MIT
